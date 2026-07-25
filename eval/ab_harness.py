"""Amplification-delta harness: generate conditions A/B/C/D per prompt.

The experiment that decides everything is one number per model size:

    delta_amp(M) = quality( swarm(M) )  -  quality( direct(M) )

measured blind. This module GENERATES the answers + their cost; `eval/judge.py`
scores them. Four conditions, all on the SAME prompt set:

  A  swarm(M)          run_swarm continuous pool -> answer.txt
  B  direct(M)         ONE call to M with a strong "think then answer" prompt.
                       This is the real baseline — make it genuinely good or you
                       beat a strawman (the most common self-deception).
  C  cheap-scaffold(M) a trivial trick with modest compute (verify-then-revise,
                       or best-of-N self-consistency). The "is the swarm worth
                       it vs five minutes of effort" control.
  D  direct(M+)        ONE call to a STRONGER model. The headline claim
                       ("small+swarm beats big-direct") lives here.
  E  synth-prompt(M)   ONE call to M given the swarm's own synthesis job as
                       a prompt: map the competing positions, stakeholders,
                       contexts where the answer flips, second-order effects,
                       then compose a thesis-led synthesis ending with the
                       conditions under which the thesis holds/fails. The
                       attribution control: if E matches A, the value was
                       the PROMPT, not the orchestration.

Five deltas fall out: A-B (amplify at all), A-C (beat the cheap trick),
A-D (the real goal), A-E (orchestration vs its own prompt), and the size
curve delta_amp(M) over M in {small, mid, frontier}.

Cost is recorded for every condition (latency, llm_calls, ~tokens) so the
judge can report quality next to the cost multiple — a +2% win at 300x the
calls is a practical loss.

MOCK-safe: with MOCK_LLM=1 everything runs (swarm + direct) and proves
plumbing only — never quote mock numbers as evidence (P0.1). For real numbers
set GROQ_API_KEY and pass --model / --strong-model Groq model names.

CLI
---
    # plumbing (mock), 8 prompts, conditions A+B+C:
    MOCK_LLM=1 python -m eval.ab_harness --mini 8 --conditions ABC

    # real minimal run on Groq (A vs B, 8 prompts):
    GROQ_API_KEY=... python -m eval.ab_harness --mini 8 --conditions AB \
        --model llama-3.1-8b-instant

    # real all-local run on one GPU (e.g. Blackwell) — A and B on the SAME
    # local weights (--model must be a valid HF id, not a Groq name):
    python -m eval.ab_harness --mini 8 --conditions AB --backend local \
        --model Qwen/Qwen2.5-7B-Instruct

    # full size-sweep arm (small model) with the strong-model D arm:
    GROQ_API_KEY=... python -m eval.ab_harness --conditions ABCD \
        --model llama-3.1-8b-instant --strong-model llama-3.3-70b-versatile

Writes eval/results/<exp>/conditions.jsonl (one row per (prompt, condition))
plus a run_meta.json. Feed the dir to eval/judge.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval import prompts as promptset  # noqa: E402


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    """Cheap, backend-agnostic token estimate (~4 chars/token). Good enough
    for a cost DENOMINATOR — we only ever report ratios, not billing."""
    return max(1, round(len(text or "") / 4))


@dataclass
class Cost:
    llm_calls: int = 0
    gen_tokens: int = 0       # generated (output) tokens, approx
    prompt_tokens: int = 0    # input tokens, approx
    latency_s: float = 0.0

    def add(self, prompt: str, output: str, dt: float) -> None:
        self.llm_calls += 1
        self.prompt_tokens += _approx_tokens(prompt)
        self.gen_tokens += _approx_tokens(output)
        self.latency_s += dt


# ---------------------------------------------------------------------------
# Direct model client (conditions B / C / D)
# ---------------------------------------------------------------------------

class DirectModel:
    """A single named model behind the shared `generate` contract.

    Resolution order:
      - MOCK_LLM=1            -> MockLLM (plumbing only)
      - GROQ_API_KEY + model  -> one GroqBackend pinned to that model name
      - otherwise             -> make_llm() local engine (model name ignored)

    This is deliberately NOT the swarm router: condition B/C/D must be a clean
    single model, not a per-role mix.
    """

    def __init__(self, model: str | None, force_local: bool = False):
        self.model = model
        self._mock = os.environ.get("MOCK_LLM", "").strip() not in ("", "0", "false", "False")
        self._engine = None
        if self._mock:
            from core.llm import MockLLM
            self._engine = MockLLM()
            self.label = f"mock({model or 'mock'})"
        elif force_local:
            # All-local parity test (e.g. Blackwell): condition B must run on the
            # SAME local weights as the swarm — not Groq — even when a
            # GROQ_API_KEY is present for the judge. Pin SWARM_MODEL *before*
            # importing core.llm so config resolves MODEL_NAME to M.
            if model:
                os.environ["SWARM_MODEL"] = model
            from core.llm import make_llm
            self._engine = make_llm()
            self.label = getattr(self._engine, "name", "local")
        elif os.environ.get("GROQ_API_KEY") and model:
            from core.llm_groq import GroqBackend
            self._engine = GroqBackend(model=model, api_key=os.environ["GROQ_API_KEY"])
            self.label = f"groq:{model}"
        else:
            from core.llm import make_llm
            self._engine = make_llm()
            self.label = getattr(self._engine, "name", "local")

    async def generate(self, prompt: str, max_tokens: int, temperature: float,
                       cost: Cost) -> str:
        t0 = time.monotonic()
        out = await self._engine.generate(
            prompt, role="synthesizer", max_tokens=max_tokens, temperature=temperature)
        cost.add(prompt, out or "", time.monotonic() - t0)
        out = (out or "").strip()
        if not out and not self._mock:
            raise EmptyAnswerError(
                f"{self.label} returned a blank answer for a direct-condition "
                f"call ({_approx_tokens(prompt)} prompt tokens, "
                f"max_tokens={max_tokens}) - treating this as a failed call, "
                f"not a valid empty row. Check the engine's stderr output just "
                f"above for the underlying API error.")
        return out


# ---------------------------------------------------------------------------
# Strong-prompt builders (B and D share the strong direct prompt)
# ---------------------------------------------------------------------------

_STRONG_DIRECT = (
    "You are an expert. Answer the question below as well as you possibly can.\n"
    "First think through the problem carefully and consider multiple angles, "
    "counterarguments, and any concrete evidence. Then write a single, "
    "well-structured, complete answer in clear prose. Do not hedge needlessly; "
    "be specific and grounded. There is no length limit — use the space the "
    "question deserves.\n\n"
    "QUESTION:\n{q}\n\nANSWER:"
)

_REVISE_CRITIQUE = (
    "Critique the draft answer below to the question. List its most important "
    "weaknesses: factual errors, unsupported claims, missing considerations, "
    "incoherence, or repetition. Be concrete and terse.\n\n"
    "QUESTION:\n{q}\n\nDRAFT:\n{draft}\n\nCRITIQUE:"
)

_REVISE_FINAL = (
    "Rewrite the draft answer into a single, improved, complete answer that "
    "fixes the weaknesses raised in the critique. Output only the improved "
    "answer in clear prose.\n\n"
    "QUESTION:\n{q}\n\nDRAFT:\n{draft}\n\nCRITIQUE:\n{crit}\n\nIMPROVED ANSWER:"
)

_BESTN_SELECT = (
    "Below are several candidate answers to the same question. Choose the ONE "
    "best answer (most correct, grounded, coherent, and complete). Reply with "
    "the full text of the best answer only, copied verbatim.\n\n"
    "QUESTION:\n{q}\n\n{candidates}\n\nBEST ANSWER:"
)

# Condition E: the swarm's own synthesis job, handed to ONE direct call.
# Mirrors what the swarm does end-to-end — scouts explore the answer-space
# (positions, stakeholders, contexts, second-order effects), then the
# composer writes a thesis-led synthesis with a conditions conclusion
# (requirements 1/3/6/8 of Synthesizer._compose_answer). If E matches A,
# the swarm's edge was the prompt, not the orchestration. Keep the two in
# sync: if the composer's writing requirements change, change this too.
_SYNTH_DIRECT = (
    "You are an expert analyst. Answer the question below by first mapping "
    "the space of competing positions, then synthesizing them.\n\n"
    "First, think through the answer-space carefully (do not include this "
    "in your answer):\n"
    "  - Enumerate the genuinely distinct positions on the question, "
    "including ones you disagree with. For each: its strongest case and "
    "the contexts where it holds.\n"
    "  - Identify where the answer is CONTEXT-DEPENDENT: which "
    "stakeholders, scales, places, or timeframes flip the answer.\n"
    "  - Identify second-order and displacement effects: what the obvious "
    "answer causes downstream that changes the assessment.\n\n"
    "Then write the synthesis as your answer:\n"
    "  1. Open with a 2-3 sentence direct answer — thesis first, before "
    "any supporting detail.\n"
    "  2. Order the argument logically — strongest material first, "
    "complications and trade-offs after.\n"
    "  3. Keep the strongest counter-positions IN the argument; engage "
    "them rather than omitting them.\n"
    "  4. Plain prose paragraphs. No markdown headers, no bullet lists.\n"
    "  5. End by stating the CONDITIONS under which your thesis holds and "
    "under which it fails (e.g. 'justified where X is already in place; "
    "counterproductive where Y'). Do NOT end with generic advice "
    "('policymakers must consider', 'careful planning is needed') — name "
    "the specific conditions.\n\n"
    "QUESTION:\n{q}\n\nSYNTHESIS:"
)


# Condition F: single-call RAG — ONE direct call to M given the SAME kind of
# retrieved evidence the swarm consumes. This is the missing attribution rung
# (critique loop 2026-07-06, iteration 3): A-vs-B confounds orchestration with
# retrieval access (the swarm searches the web mid-run; B answers from
# parametric memory), and E controls only for the synthesis prompt. F is the
# baseline any practitioner would deploy — if A cannot beat F, the
# orchestration adds nothing over plain RAG. The swarm's compression thesis
# predicts A>F only once the evidence pack outgrows what F's single context
# can use well; at small pack sizes A≈F is the expected (and acceptable) result.
_RAG_DIRECT = (
    "You are an expert analyst. Answer the question using the evidence "
    "excerpts below where they are relevant, plus your own knowledge. Weigh "
    "conflicting evidence rather than ignoring it; if the evidence is thin, "
    "say so.\n\n"
    "EVIDENCE:\n{evidence}\n\n"
    "QUESTION:\n{q}\n\nANSWER:"
)

# Cap on the evidence block handed to condition F (chars). Roughly matches
# the total evidence budget the swarm's scouts see across partitions.
_RAG_EVIDENCE_MAX_CHARS = 12000


def _retrieve_evidence(prompt_text: str,
                       max_chars: int = _RAG_EVIDENCE_MAX_CHARS) -> str:
    """Same retrieval stack the swarm uses (CompositeRetriever), rendered as
    one bounded evidence block. Falls back to the engineered placeholder
    corpus exactly like the swarm does — the comparison stays like-for-like."""
    from core.retrieval import CachedRetriever, CompositeRetriever
    try:
        chunks = CachedRetriever(CompositeRetriever()).retrieve(prompt_text)
    except Exception as exc:
        print(f"[ab] F retrieval failed ({type(exc).__name__}: {exc}); "
              f"empty evidence block")
        return "(no evidence retrieved)"
    if not chunks:
        return "(no evidence retrieved)"
    parts, used = [], 0
    for i, ch in enumerate(chunks):
        text = getattr(ch, "text", str(ch)).strip()
        if not text:
            continue
        tag = getattr(ch, "source_tag", "") or f"chunk_{i}"
        block = f"[{tag}]\n{text}"
        if used + len(block) > max_chars:
            block = block[: max(0, max_chars - used)]
        parts.append(block)
        used += len(block)
        if used >= max_chars:
            break
    return "\n\n".join(parts) if parts else "(no evidence retrieved)"


def _evidence_from_pack(pack_path: Path, max_chars: int) -> str:
    """Naive-fill evidence block from a pre-built eval.packs pack: concatenate
    chunk texts IN PACK ORDER up to max_chars. Deliberately dumb — condition F
    is the honest "practitioner just stuffed the pack into context" baseline,
    not a smart retriever over the pack (that's a separate future condition
    F+; see docs/OVERCONTEXT_EVAL_PLAN.md). This is what makes A (swarm over
    the same pack, via --corpus=pack:<path>) and F a fair over-context
    comparison — same evidence, different consumption strategy."""
    from eval.packs import load_pack
    chunks = load_pack(pack_path)
    if not chunks:
        return "(no evidence retrieved)"
    parts, used = [], 0
    for i, ch in enumerate(chunks):
        text = (ch.text or "").strip()
        if not text:
            continue
        tag = ch.source_tag or f"chunk_{i}"
        block = f"[{tag}]\n{text}"
        if used + len(block) > max_chars:
            block = block[: max(0, max_chars - used)]
        parts.append(block)
        used += len(block)
        if used >= max_chars:
            break
    return "\n\n".join(parts) if parts else "(no evidence retrieved)"


async def gen_direct(dm: DirectModel, text: str, max_tokens: int,
                     template: str = _STRONG_DIRECT) -> tuple[str, Cost]:
    """Condition B/D (strong direct) / E (synthesis-prompt): one call."""
    cost = Cost()
    ans = await dm.generate(template.format(q=text), max_tokens, 0.3, cost)
    return ans, cost


async def gen_direct_rag(dm: DirectModel, text: str, max_tokens: int,
                         pack_path: "Path | None" = None,
                         max_chars: int = _RAG_EVIDENCE_MAX_CHARS
                         ) -> tuple[str, Cost]:
    """Condition F: one call to M with retrieved evidence.

    Default (pack_path=None): the swarm's live retrieval stack, unchanged
    (`_retrieve_evidence`). When `pack_path` is given (the --pack-scale
    over-context eval mode), evidence instead comes from that fixed pack —
    the SAME pack condition A consumes via --corpus=pack:<path> — via a
    naive pack-order fill (`_evidence_from_pack`).
    """
    cost = Cost()
    if pack_path is not None:
        evidence = await asyncio.to_thread(_evidence_from_pack, pack_path, max_chars)
    else:
        evidence = await asyncio.to_thread(_retrieve_evidence, text, max_chars)
    ans = await dm.generate(
        _RAG_DIRECT.format(q=text, evidence=evidence), max_tokens, 0.3, cost)
    return ans, cost


async def gen_scaffold(dm: DirectModel, text: str, max_tokens: int,
                       kind: str, n: int) -> tuple[str, Cost]:
    """Condition C: cheap scaffold with modest compute."""
    cost = Cost()
    if kind == "revise":
        draft = await dm.generate(_STRONG_DIRECT.format(q=text), max_tokens, 0.3, cost)
        crit = await dm.generate(
            _REVISE_CRITIQUE.format(q=text, draft=draft[:6000]), max_tokens // 2, 0.3, cost)
        final = await dm.generate(
            _REVISE_FINAL.format(q=text, draft=draft[:6000], crit=crit[:3000]),
            max_tokens, 0.3, cost)
        return final, cost
    # best-of-N self-consistency
    drafts = []
    for _ in range(n):
        drafts.append(await dm.generate(
            _STRONG_DIRECT.format(q=text), max_tokens, 0.8, cost))
    block = "\n\n".join(f"CANDIDATE {i+1}:\n{d[:3000]}" for i, d in enumerate(drafts))
    best = await dm.generate(
        _BESTN_SELECT.format(q=text, candidates=block), max_tokens, 0.2, cost)
    return best, cost


# ---------------------------------------------------------------------------
# Condition A: the swarm (subprocess -> answer.txt)
# ---------------------------------------------------------------------------

def _is_mock() -> bool:
    return os.environ.get("MOCK_LLM", "").strip() not in ("", "0", "false", "False")


class ModelParityError(RuntimeError):
    """Raised when condition A (swarm) did not actually run on the pinned model M
    — the delta_amp comparison would be invalid. A real test must refuse this."""


class EmptyAnswerError(RuntimeError):
    """Raised when a direct-condition (B/C/D/E/F) call produced a blank answer.

    Before this guard, a permanently-failing API call (e.g. Groq 413 "request
    too large" on an oversized RAG evidence prompt) fell through to an empty
    string with no signal beyond a stderr print, and the harness happily wrote
    it to conditions.jsonl as an ordinary completed row — six such rows shipped
    to the judge as "the model produced no answer" (eval/results/overctx_16x,
    2026-07-24) rather than as the API failure they actually were. A blank
    answer is never a valid experimental result; surface it loudly instead."""


def _norm_model(s: str) -> str:
    """Loose model-id key: last path/colon segment, lowercased, alnum-only.
    'vLLM:Qwen/Qwen2.5-7B-Instruct' -> 'qwen257binstruct'."""
    s = (s or "").strip().lower()
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    if ":" in s:
        s = s.rsplit(":", 1)[-1]
    return re.sub(r"[^a-z0-9]", "", s)


def _swarm_output_dir(run_id: str) -> Path:
    root = "outputs_mock" if _is_mock() else "outputs"
    base_env = os.environ.get("SWARM_OUTPUTS_BASE_DIR")
    base = (Path(base_env) / root) if base_env else (_REPO / root)
    return base / run_id


def gen_swarm(task: str, text: str, run_id: str, extra: list[str],
              swarm_model: str | None, backend: str = "auto") -> tuple[str, Cost]:
    """Run the swarm once; return (answer_text, cost). Cost is read from the
    run's summary.json (wall_clock_s, total_llm_calls).

    `backend`: 'local' forces SWARM_BACKEND=local (one local model, no Groq
    routing); 'groq' forces groq-only; 'auto' inherits the ambient env.
    """
    cmd = [sys.executable, str(_REPO / "run_swarm.py"), task, text,
           f"--run-id={run_id}", *extra]
    env = os.environ.copy()
    if backend == "local":
        env["SWARM_BACKEND"] = "local"
    elif backend == "groq":
        env.pop("SWARM_BACKEND", None)
    # 'auto' leaves SWARM_BACKEND as inherited.
    if swarm_model:
        # Pin the swarm's base model to M so swarm(M) genuinely uses M.
        # Local engine reads SWARM_MODEL; Groq router reads per-role models —
        # pin every role to M via GROQ_ROLE_MODELS_JSON (but NOT in local mode,
        # where roles must not route to Groq even if a judge key is present).
        env["SWARM_MODEL"] = swarm_model
        if backend != "local" and env.get("GROQ_API_KEY"):
            roles = ["scout", "synthesizer", "forager", "developer",
                     "hater", "critic", "validator"]
            env["GROQ_ROLE_MODELS_JSON"] = json.dumps({r: swarm_model for r in roles})
    print(f"[ab] === A swarm: run-id={run_id} task={task} ===", flush=True)
    t0 = time.monotonic()
    rc = subprocess.run(cmd, cwd=str(_REPO), env=env).returncode
    dt = time.monotonic() - t0

    out_dir = _swarm_output_dir(run_id)
    answer = ""
    ans_path = out_dir / "answer.txt"
    if ans_path.exists():
        answer = ans_path.read_text(encoding="utf-8")
    elif rc != 0:
        print(f"[ab] WARNING: swarm exited {rc}, no answer.txt at {ans_path}",
              file=sys.stderr)

    cost = Cost(latency_s=dt)
    sum_path = out_dir / "summary.json"
    if sum_path.exists():
        try:
            s = json.loads(sum_path.read_text(encoding="utf-8"))
            # Continuous-pool summary reports total_iterations; the legacy
            # run_pipeline path reports total_llm_calls. Take whichever exists.
            cost.llm_calls = int(
                s.get("total_llm_calls") or s.get("total_iterations") or 0)
            cost.latency_s = float(s.get("wall_clock_s") or dt)
            tk = (s.get("llm_inflight") or {}).get("total_tokens_generated")
            if tk:
                cost.gen_tokens = int(tk)
        except Exception:
            pass
    if not cost.gen_tokens:
        cost.gen_tokens = _approx_tokens(answer)

    # --- Model-parity guard (real-test integrity) --------------------------
    # The vLLM load cascade silently substitutes a smaller model when the pinned
    # one can't load (e.g. a Groq name handed to the local engine -> falls back
    # to Qwen-3B). That makes condition A swarm(other) vs condition B direct(M):
    # different models, so delta_amp is meaningless. Refuse to continue.
    # Only a single-model local engine reports the loaded model in run_meta.json
    # ('vLLM:<id>' / 'RealLLM:<id>'); heterogeneous/Groq routers report
    # 'heterogeneous' and pin per-role explicitly (erroring loudly rather than
    # silently substituting), so the check is scoped to the single-model label.
    if swarm_model and not _is_mock():
        actual = ""
        meta_path = out_dir / "run_meta.json"
        if meta_path.exists():
            try:
                actual = str(json.loads(
                    meta_path.read_text(encoding="utf-8")).get("llm_backend") or "")
            except Exception:
                actual = ""
        if actual and actual.lower() != "heterogeneous" and ":" in actual:
            if _norm_model(swarm_model) not in _norm_model(actual):
                raise ModelParityError(
                    f"swarm ran on {actual!r} but condition A was pinned to "
                    f"{swarm_model!r} — the load cascade silently substituted a "
                    f"different model, so delta_amp would compare swarm(other) "
                    f"vs direct({swarm_model}). Fix: pass a model the engine can "
                    f"load — a valid HF id for '--backend local' (e.g. "
                    f"Qwen/Qwen2.5-7B-Instruct), or a Groq model name for "
                    f"'--backend groq' (e.g. llama-3.1-8b-instant)."
                )
        print(f"[ab] parity OK: swarm ran on {actual or '(unrecorded)'}",
              flush=True)
    return answer, cost


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class Row:
    pid: str
    task: str
    prompt: str
    condition: str       # A | B | C | D | E
    label: str           # model/engine label
    answer: str
    cost: dict = field(default_factory=dict)
    factual: bool = False
    must_include: list = field(default_factory=list)


async def run_experiment(args) -> Path:
    world = None
    if args.prompt_set == "synthetic":
        # --world required, enforced in main(); load (or fail loudly if the
        # world hasn't been built yet via `python -m eval.worlds --seed N`).
        from eval import worlds as worldmod
        world = worldmod.load_world(args.world)
        base_set = worldmod.synthetic_prompts_for_world(world)
    elif args.prompt_set == "overcontext":
        base_set = promptset.OVERCONTEXT_SET
    else:
        base_set = promptset.DEFAULT_SET
    plist = (promptset.mini(args.mini, base_set) if args.mini else base_set)
    conds = set(args.conditions.upper())
    out_root = _REPO / "eval" / "results" / args.name
    out_root.mkdir(parents=True, exist_ok=True)
    rows_path = out_root / "conditions.jsonl"

    _local = args.backend == "local"
    extra = list(args.swarm_flag or [])
    rows: list[Row] = []
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # --pack-scale: build/reuse a fixed evidence pack per prompt, then route
    # BOTH condition A (via --corpus=pack:<path>, appended to `extra` so it
    # wins over any --corpus swarm-flag) and condition F (via pack_path=) at
    # the SAME pack — the decisive over-context comparison (see module
    # docstring / docs/OVERCONTEXT_EVAL_PLAN.md).
    pack_paths: dict[str, Path] = {}
    if args.pack_scale and args.prompt_set == "synthetic":
        # Synthetic-world packs: same JSONL schema, built from the world's
        # rendered documents instead of live retrieval (docs/future/
        # STAGE1_SYNTHETIC_EVAL_SPEC.md Sec 2). One pack per prompt, but
        # every prompt in a synthetic run shares the same world, so this is
        # effectively one build reused across pids (build_pack_from_world's
        # own reuse-if-exists still applies per pid+scale).
        from eval import worlds as worldmod
        from eval.packs import build_pack_from_world
        packs_dir = _REPO / "eval" / "packs"
        world_docs = worldmod.load_rendered_docs(args.world)
        for p in plist:
            pack_paths[p.pid] = build_pack_from_world(
                world, world_docs, args.pack_scale, packs_dir, pid=p.pid,
                target_chars=args.target_chars)
            print(f"[ab] world pack ready: {p.pid} @ {args.pack_scale} -> "
                  f"{pack_paths[p.pid]}")
    elif args.pack_scale:
        from eval.packs import build_pack
        packs_dir = _REPO / "eval" / "packs"
        for p in plist:
            pack_paths[p.pid] = build_pack(
                p.text, args.pack_scale, packs_dir, pid=p.pid,
                target_chars=args.target_chars)
            print(f"[ab] pack ready: {p.pid} @ {args.pack_scale} -> "
                  f"{pack_paths[p.pid]}")

    # Direct clients (conditions B/C/D). On a single local GPU they must NOT be
    # resident while the swarm (A) runs: vLLM reserves most of VRAM, which would
    # starve the A subprocess and trip the parity guard. So in local mode we
    # phase-separate — all A first (GPU exclusive), then build the direct client
    # and run B/C/D. On Groq/mock there's no VRAM contention, so build up-front
    # and interleave for incremental output.
    dm: dict = {"m": None, "strong": None}

    def build_direct() -> None:
        if (conds & set("BCEF")) and dm["m"] is None:
            dm["m"] = DirectModel(args.model, force_local=_local)
        if "D" in conds and dm["strong"] is None:
            dm["strong"] = DirectModel(args.strong_model, force_local=_local)

    if not _local:
        build_direct()

    def _base(p) -> dict:
        return dict(pid=p.pid, task=p.task, prompt=p.text,
                    factual=p.factual, must_include=p.must_include)

    with rows_path.open("w", encoding="utf-8") as fh:
        def emit(r: Row) -> None:
            rows.append(r)
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            fh.flush()

        def do_swarm(p) -> None:
            run_id = f"delta_{args.name}_{p.pid}_{stamp}"
            swarm_extra = list(extra)
            if p.pid in pack_paths:
                # Wins over any --corpus swarm-flag: run_swarm.py takes the
                # LAST --corpus= flag, and this is appended after `extra`.
                swarm_extra.append(f"--corpus=pack:{pack_paths[p.pid]}")
            ans, cost = gen_swarm(p.task, p.text, run_id, swarm_extra, args.model,
                                  backend=args.backend)
            emit(Row(condition="A", label=f"swarm({args.model or 'M'})",
                     answer=ans, cost=asdict(cost), **_base(p)))

        async def do_direct(p) -> None:
            if "B" in conds:
                ans, cost = await gen_direct(dm["m"], p.text, args.max_tokens)
                emit(Row(condition="B", label=dm["m"].label, answer=ans,
                         cost=asdict(cost), **_base(p)))
            if "C" in conds:
                ans, cost = await gen_scaffold(
                    dm["m"], p.text, args.max_tokens, args.scaffold, args.best_of)
                emit(Row(condition="C", label=f"{dm['m'].label}+{args.scaffold}",
                         answer=ans, cost=asdict(cost), **_base(p)))
            if "D" in conds:
                ans, cost = await gen_direct(dm["strong"], p.text, args.max_tokens)
                emit(Row(condition="D", label=dm["strong"].label, answer=ans,
                         cost=asdict(cost), **_base(p)))
            if "E" in conds:
                ans, cost = await gen_direct(dm["m"], p.text, args.max_tokens,
                                             template=_SYNTH_DIRECT)
                emit(Row(condition="E", label=f"{dm['m'].label}+synthprompt",
                         answer=ans, cost=asdict(cost), **_base(p)))
            if "F" in conds:
                pack_path = pack_paths.get(p.pid)
                ans, cost = await gen_direct_rag(
                    dm["m"], p.text, args.max_tokens, pack_path=pack_path,
                    max_chars=args.rag_evidence_max_chars)
                label = f"{dm['m'].label}+rag" + (
                    f"+pack{args.pack_scale}" if pack_path else "")
                emit(Row(condition="F", label=label,
                         answer=ans, cost=asdict(cost), **_base(p)))

        if _local:
            # Phase 1: all swarm(A) runs — each subprocess owns the GPU and frees
            # it on exit. Phase 2: build the local direct client, run B/C/D.
            if "A" in conds:
                for p in plist:
                    print(f"\n[ab] ===== A swarm | prompt {p.pid} ({p.task}) =====")
                    do_swarm(p)
            if conds & set("BCDEF"):
                build_direct()
                for p in plist:
                    print(f"\n[ab] ===== direct | prompt {p.pid} ({p.task}) =====")
                    await do_direct(p)
        else:
            for p in plist:
                print(f"\n[ab] ===== prompt {p.pid} ({p.task}) =====")
                if "A" in conds:
                    do_swarm(p)
                await do_direct(p)

    meta = {
        "name": args.name,
        "timestamp": stamp,
        "mock": _is_mock(),
        "conditions": sorted(conds),
        "model_M": args.model,
        "strong_model": args.strong_model,
        "scaffold": args.scaffold,
        "best_of": args.best_of,
        "n_prompts": len(plist),
        "prompt_ids": [p.pid for p in plist],
        "swarm_flags": extra,
        "prompt_set": args.prompt_set,
        "world_seed": args.world if args.prompt_set == "synthetic" else None,
        "world_template": world.template_name if world is not None else None,
        "pack_scale": args.pack_scale,
        "target_chars": args.target_chars,
        "rag_evidence_max_chars": args.rag_evidence_max_chars,
        "pack_paths": {pid: str(p) for pid, p in pack_paths.items()},
        "warning": ("MOCK run — plumbing only, NOT empirical (P0.1)"
                    if _is_mock() else None),
    }
    (out_root / "run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[ab] wrote {len(rows)} rows -> {rows_path}")
    print(f"[ab] next: python -m eval.judge {out_root}")
    return out_root


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Swarm amplification-delta harness")
    ap.add_argument("--name", default="delta",
                    help="experiment name (results dir under eval/results/)")
    ap.add_argument("--conditions", default="AB",
                    help="subset of ABCDEF to generate (default AB). E = the "
                         "synthesis-prompt attribution control: one call to M "
                         "given the swarm's own synthesis instruction. F = the "
                         "retrieval attribution control: one call to M given "
                         "the same retrieved evidence stack the swarm uses "
                         "(single-call RAG — the practitioner baseline).")
    ap.add_argument("--mini", type=int, default=0,
                    help="use only the first N pre-registered prompts (0 = full set)")
    ap.add_argument("--model", default=None,
                    help="model M for A/B/C. For --backend groq/auto: a Groq "
                         "model name (e.g. llama-3.1-8b-instant). For "
                         "--backend local: a valid HF id (e.g. "
                         "Qwen/Qwen2.5-7B-Instruct). Blank = local default/mock.")
    ap.add_argument("--backend", choices=("auto", "local", "groq"), default="auto",
                    help="where conditions run. 'local' = one GPU model for BOTH "
                         "swarm(A) and direct(B) (true parity, e.g. Blackwell); "
                         "'groq' = groq-only; 'auto' = inherit ambient env.")
    ap.add_argument("--strong-model", default=None,
                    help="stronger model M+ for condition D")
    ap.add_argument("--scaffold", choices=("revise", "best_of_n"), default="revise",
                    help="condition C scaffold (default verify-then-revise)")
    ap.add_argument("--best-of", type=int, default=5,
                    help="N for best_of_n scaffold")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="max output tokens for direct conditions")
    ap.add_argument("--swarm-flag", action="append", default=[],
                    help="extra flag forwarded to run_swarm.py (repeatable)")
    ap.add_argument("--prompt-set", choices=("default", "overcontext", "synthetic"),
                    default="default",
                    help="prompt set to run: 'default' (DEFAULT_SET, unchanged "
                         "behavior), 'overcontext' (OVERCONTEXT_SET — synthesis-"
                         "with-conflict prompts for the --pack-scale eval; see "
                         "docs/OVERCONTEXT_EVAL_PLAN.md), or 'synthetic' (Stage 1 "
                         "invented-fact world; requires --world and --pack-scale; "
                         "see docs/future/STAGE1_SYNTHETIC_EVAL_SPEC.md)")
    ap.add_argument("--world", type=int, default=None,
                    help="synthetic-world seed (required with "
                         "--prompt-set synthetic; build/validate it first via "
                         "`python -m eval.worlds --seed N --template T`)")
    ap.add_argument("--pack-scale", choices=("1x", "4x", "16x"), default=None,
                    help="build/reuse a fixed evidence pack per prompt at this "
                         "scale (eval.packs.build_pack) and route condition A "
                         "(--corpus=pack:<path>) and condition F (naive pack-"
                         "order fill) through the SAME pack. Default None = "
                         "existing live-retrieval behavior, unchanged. This is "
                         "the decisive A-vs-F over-context comparison.")
    ap.add_argument("--target-chars", type=int, default=None,
                    help="override the pack char budget for --pack-scale "
                         "(default: 1x=24000 4x=96000 16x=384000, see "
                         "eval/packs.py SCALE_CHARS)")
    ap.add_argument("--rag-evidence-max-chars", type=int, default=_RAG_EVIDENCE_MAX_CHARS,
                    help="max evidence chars fed to condition F (default matches "
                         "the historical live-RAG cap; raise this for --pack-scale "
                         "4x/16x runs so F isn't truncated far below what the pack "
                         "actually contains — an unfair-to-F cap would bias the "
                         "A-vs-F comparison toward A)")
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.prompt_set == "synthetic":
        if args.world is None:
            raise ValueError("--prompt-set synthetic requires --world <seed> "
                             "(build one first: python -m eval.worlds --seed N "
                             "--template T)")
        if not args.pack_scale:
            # Per spec Sec 5.2: without a pack, condition A falls through to
            # live retrieval/placeholder corpus for an invented world
            # (nonsensical), and condition F's _retrieve_evidence hits a real
            # search backend with a fictional name and gets zero/garbage
            # results silently, corrupting the comparison without an error.
            raise ValueError("--prompt-set synthetic requires --pack-scale "
                             "{1x,4x,16x} — without a pack there is nothing "
                             "sensible for condition A/F to retrieve against "
                             "an invented world.")
    if "D" in args.conditions.upper() and not args.strong_model and not _is_mock():
        print("[ab] note: condition D requested without --strong-model; "
              "D will use the local/default engine.", file=sys.stderr)
    if args.backend == "local" and args.model and "/" not in args.model \
            and not _is_mock():
        print(f"[ab] WARNING: --backend local with --model {args.model!r}: a "
              f"local engine needs a Hugging Face id (org/name). A bare Groq "
              f"name will fail the model-parity guard.", file=sys.stderr)
    try:
        asyncio.run(run_experiment(args))
    except ModelParityError as exc:
        print(f"\n[ab] ABORT (model parity): {exc}", file=sys.stderr)
        return 2
    except EmptyAnswerError as exc:
        print(f"\n[ab] ABORT (empty answer): {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
