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

Four deltas fall out: A-B (amplify at all), A-C (beat the cheap trick),
A-D (the real goal), and the size curve delta_amp(M) over M in
{small, mid, frontier}.

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

    def __init__(self, model: str | None):
        self.model = model
        self._mock = os.environ.get("MOCK_LLM", "").strip() not in ("", "0", "false", "False")
        self._engine = None
        if self._mock:
            from core.llm import MockLLM
            self._engine = MockLLM()
            self.label = f"mock({model or 'mock'})"
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
        return (out or "").strip()


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


async def gen_direct(dm: DirectModel, text: str, max_tokens: int) -> tuple[str, Cost]:
    """Condition B/D: one strong direct call."""
    cost = Cost()
    ans = await dm.generate(_STRONG_DIRECT.format(q=text), max_tokens, 0.3, cost)
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


def _swarm_output_dir(run_id: str) -> Path:
    root = "outputs_mock" if _is_mock() else "outputs"
    base_env = os.environ.get("SWARM_OUTPUTS_BASE_DIR")
    base = (Path(base_env) / root) if base_env else (_REPO / root)
    return base / run_id


def gen_swarm(task: str, text: str, run_id: str, extra: list[str],
              swarm_model: str | None) -> tuple[str, Cost]:
    """Run the swarm once; return (answer_text, cost). Cost is read from the
    run's summary.json (wall_clock_s, total_llm_calls)."""
    cmd = [sys.executable, str(_REPO / "run_swarm.py"), task, text,
           f"--run-id={run_id}", *extra]
    env = os.environ.copy()
    if swarm_model:
        # Pin the swarm's base model to M so swarm(M) genuinely uses M.
        # Local engine reads SWARM_MODEL; Groq router reads per-role models —
        # pin every role to M via GROQ_ROLE_MODELS_JSON.
        env["SWARM_MODEL"] = swarm_model
        if env.get("GROQ_API_KEY"):
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
    return answer, cost


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class Row:
    pid: str
    task: str
    prompt: str
    condition: str       # A | B | C | D
    label: str           # model/engine label
    answer: str
    cost: dict = field(default_factory=dict)
    factual: bool = False
    must_include: list = field(default_factory=list)


async def run_experiment(args) -> Path:
    plist = (promptset.mini(args.mini) if args.mini else promptset.DEFAULT_SET)
    conds = set(args.conditions.upper())
    out_root = _REPO / "eval" / "results" / args.name
    out_root.mkdir(parents=True, exist_ok=True)
    rows_path = out_root / "conditions.jsonl"

    # Build direct clients once (reused across prompts).
    dm_m = DirectModel(args.model) if (conds & set("BC")) else None
    dm_strong = DirectModel(args.strong_model) if "D" in conds else None

    extra = list(args.swarm_flag or [])
    rows: list[Row] = []
    stamp = time.strftime("%Y%m%d_%H%M%S")

    with rows_path.open("w", encoding="utf-8") as fh:
        def emit(r: Row) -> None:
            rows.append(r)
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            fh.flush()

        for p in plist:
            print(f"\n[ab] ===== prompt {p.pid} ({p.task}) =====")
            base = dict(pid=p.pid, task=p.task, prompt=p.text,
                        factual=p.factual, must_include=p.must_include)

            if "A" in conds:
                run_id = f"delta_{args.name}_{p.pid}_{stamp}"
                ans, cost = gen_swarm(p.task, p.text, run_id, extra, args.model)
                emit(Row(condition="A", label=f"swarm({args.model or 'M'})",
                         answer=ans, cost=asdict(cost), **base))

            if "B" in conds:
                ans, cost = await gen_direct(dm_m, p.text, args.max_tokens)
                emit(Row(condition="B", label=dm_m.label, answer=ans,
                         cost=asdict(cost), **base))

            if "C" in conds:
                ans, cost = await gen_scaffold(
                    dm_m, p.text, args.max_tokens, args.scaffold, args.best_of)
                emit(Row(condition="C",
                         label=f"{dm_m.label}+{args.scaffold}", answer=ans,
                         cost=asdict(cost), **base))

            if "D" in conds:
                ans, cost = await gen_direct(dm_strong, p.text, args.max_tokens)
                emit(Row(condition="D", label=dm_strong.label, answer=ans,
                         cost=asdict(cost), **base))

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
                    help="subset of ABCD to generate (default AB)")
    ap.add_argument("--mini", type=int, default=0,
                    help="use only the first N pre-registered prompts (0 = full set)")
    ap.add_argument("--model", default=None,
                    help="model M for A/B/C (Groq name; blank = local/mock)")
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
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if "D" in args.conditions.upper() and not args.strong_model and not _is_mock():
        print("[ab] note: condition D requested without --strong-model; "
              "D will use the local/default engine.", file=sys.stderr)
    asyncio.run(run_experiment(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
