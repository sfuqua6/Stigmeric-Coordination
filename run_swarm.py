"""Pipeline orchestrator for the cleaned stigmergic swarm.

Usage:
    python run_swarm.py debate "Climate action is necessary"
    python run_swarm.py analysis "What causes innovation?"
    python run_swarm.py creative "Write a haiku about emergence"
    python run_swarm.py problem_solving "How can cities reduce traffic?"

Flags:
    --mode={stigmergic,baseline}   Default: stigmergic. Use baseline to run an
                                   independent-agent comparison (no signal store).
    --corpus={real,placeholder}    Default: real. Use placeholder to skip retrieval.
    --ignore-kb                    Disable knowledge base for this run.
    --reset-kb                     Quarantine existing KB entries before run.
    --small                        Use smaller 3B-class models. Selects the
                                   'small'/'small_coding' bundle (continuous pool),
                                   configs/heterogeneous_small.json (--heterogeneous),
                                   or Qwen2.5-3B-Instruct (--legacy-rounds).

Set MOCK_LLM=1 to run without loading the model.

Per round:
    1. Build/refresh the corpus and partition it across scouts.
    2. Phase A: scouts and validators run concurrently.
       (P2.3 / R6: validators must run BEFORE the agents whose deposits
       benefit from the provenance boost; otherwise the boost cannot fire.)
    3. Phase B: foragers, critics, haters run concurrently — they now see
       any VERIFICATION signals Phase A produced and the boost can engage.
    4. Decay all signals; prune those below the threshold.
    5. Compute and log the diversity metric over agent context records.

After all rounds, the synthesizer produces a single answer from the
surviving signals.

Outputs:
    outputs/RUN_TIMESTAMP/        — real-model runs
    outputs_mock/RUN_TIMESTAMP/   — MOCK_LLM=1 runs (P0.1: kept separate so
                                     mock artifacts cannot be confused with
                                     empirical evidence)
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Silence verbose load output from transformers / sentence-transformers /
# huggingface_hub before any of those libraries get imported. These print
# multi-line "BertModel LOAD REPORT" tables and duplicated "Loading weights:
# 100%" tqdm bars on every embedder load, which on Colab obscures the actual
# pipeline output. Override SWARM_QUIET_LIBS=0 to re-enable the chatter.
if os.environ.get("SWARM_QUIET_LIBS", "1") != "0":
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import transformers as _tf
        _tf.logging.set_verbosity_error()
        if hasattr(_tf.utils, "logging") and hasattr(_tf.utils.logging, "disable_progress_bar"):
            _tf.utils.logging.disable_progress_bar()
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))

# --depth N: synthesis net width (how many claim clusters the read-out surfaces).
# Pre-parsed HERE, before `from core import config`, because config binds
# RENDER_K at import time. Higher --depth surfaces more of the swarm's niche /
# long-tail coverage (the point of the random scout/partition split) instead of
# only the obvious high-consensus claims. Wide by default (config RENDER_K=8);
# this just lets a run override it. argparse declares it formally below.
for _di, _da in enumerate(sys.argv):
    if _da == "--depth" and _di + 1 < len(sys.argv):
        os.environ["SWARM_RENDER_K"] = sys.argv[_di + 1]
    elif _da.startswith("--depth="):
        os.environ["SWARM_RENDER_K"] = _da.split("=", 1)[1]

from core import config
from core.config import (
    NUM_SCOUTS, NUM_FORAGERS, NUM_CRITICS, NUM_HATERS, NUM_VALIDATORS,
    NUM_ROUNDS, ITERATIONS_PER_ROUND, USE_MOCK_LLM,
    N_FACETS, WEB_PARTITION_COUNT,
)
from core.signal_store import SignalStore
from core.clean_answer import write_answer_files
from core.intake import (
    chunk_corpus, partition_for_scouts, trivial_corpus_from_thesis, ScoutPartition,
)
from core.sampling import (
    strategy_for_forager, strategy_for_critic, strategy_for_validator,
)
from core.diversity import (
    AgentContextRecord,
    _role_partition_overlap, _overall_partition_overlap, format_partition_overlap_report,
    role_diversity, overall_diversity, format_report,  # backward-compat aliases
)
from core.output_diversity import format_output_diversity_report
from core.llm import make_llm
from core.llm_router import HeterogeneousRouter, make_router, make_bundle_router
from core.knowledge_base import KnowledgeBase

from agents.scout import Scout, ScoutConfig
from agents.developer import Developer
from agents.forager import Forager  # backward-compat alias for Developer
from agents.critic import Critic
from agents.hater import Hater
from agents.validator import Validator
from agents.synthesizer import Synthesizer
from core.role_registry import get_role_classes
from core.worker_pool import run_pool, decay_loop
from core.convergence import ConvergenceDetector
from core.topology import generate_topology, assign_topology_cells, format_cell_for_prompt


# ---------------------------------------------------------------------------
# Task prompt templates + role activation (P2.6 / R14 partial)
# ---------------------------------------------------------------------------

TASK_PROMPTS = {
    "debate": "Argue both sides of the following thesis with evidence: {prompt}",
    "analysis": "Provide a structured analysis of the following question: {prompt}",
    "creative": "Generate creative content addressing: {prompt}",
    "problem_solving": "Propose solutions to the following problem: {prompt}",
    "coding": "Implement a Python solution for the following programming task: {prompt}",
}

# Which roles each task type actually wants. Suppresses semantically wrong
# pairings (e.g., Hater on a creative task asking for a "shared assumption"
# in a haiku cluster). Roles always include scout, forager, synthesizer.
ROLES_FOR_TASK = {
    "debate":          {"critic", "hater", "validator"},
    "analysis":        {"critic", "hater", "validator"},
    "creative":        {"critic", "hater"},               # hater with craft-focused prompt; no validator (no external facts)
    "problem_solving": {"critic", "hater"},              # no validator
    "coding":          {"critic", "hater", "validator"}, # all roles, domain-specific classes
}


# ---------------------------------------------------------------------------
# Phase-isolated execution (split-away-from-parallelism mode)
# ---------------------------------------------------------------------------
# In phase-isolated mode, ONE subprocess invocation runs ONE role's agents
# for ONE round (or runs decay, or synth), persists the SignalStore to disk,
# and exits. The next subprocess loads that checkpoint and runs the next
# phase. This keeps exactly one model resident in memory at any time —
# the canonical workaround for llama-cpp-python's incomplete model unload
# on Windows / 16 GB-RAM hardware.
#
# Subprocess sequence for task=debate (all 5 roles active):
#   R1: scouts -> validators -> foragers -> critics -> haters -> decay
#   R2: same
#   R3: same
#   synth
#
# The orchestrator that emits these subprocess calls is tools/run_isolated.py.

VALID_PHASES = (
    "scouts", "validators", "foragers", "critics", "haters", "decay", "synth",
)

# Per-task ordered list of phases that should run each round.
# 'decay' always closes a round; 'synth' is the final non-round phase.
def _round_phases_for_task(task_type: str) -> list[str]:
    active = ROLES_FOR_TASK.get(task_type, {"critic", "hater", "validator"})
    seq: list[str] = ["scouts"]
    if "validator" in active:
        seq.append("validators")
    seq.append("foragers")
    if "critic" in active:
        seq.append("critics")
    if "hater" in active:
        seq.append("haters")
    seq.append("decay")
    return seq


def _phase_role(phase: str) -> str:
    """Map a phase name to the role name used by the heterogeneous router."""
    return {
        "scouts": "scout",
        "validators": "validator",
        "foragers": "forager",
        "critics": "critic",
        "haters": "hater",
        "synth": "synthesizer",
    }.get(phase, phase)


def build_task_prompt(task_type: str, user_prompt: str) -> str:
    template = TASK_PROMPTS.get(task_type)
    if template is None:
        raise SystemExit(
            f"Unknown task type {task_type!r}. Choose from: "
            f"{', '.join(TASK_PROMPTS)}"
        )
    # P1.6 / m5: replace() avoids KeyError if user_prompt contains {curly}
    return template.replace("{prompt}", user_prompt)


# ---------------------------------------------------------------------------
# Progress bar (tqdm if available, fallback otherwise)
# ---------------------------------------------------------------------------

def _make_progress(total: int, desc: str):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="iter",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    except ImportError:
        class _Fallback:
            def __init__(self, total, desc):
                self.total = total; self.n = 0; self.desc = desc
                self.start = time.time()
            def update(self, k=1):
                self.n += k
                pct = 100 * self.n / max(1, self.total)
                elapsed = time.time() - self.start
                rate = self.n / max(0.1, elapsed)
                remaining = (self.total - self.n) / max(0.01, rate)
                bar = ("#" * int(pct / 4)).ljust(25, "-")
                print(f"\r[{self.desc}] [{bar}] {self.n}/{self.total} "
                      f"({pct:.0f}%) elapsed={elapsed:.0f}s eta={remaining:.0f}s",
                      end="", flush=True)
                if self.n >= self.total:
                    print()
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): self.close()
        return _Fallback(total, desc)


_LLM_INFLIGHT_METRICS = {"peak": 0, "total_calls": 0, "total_tokens_generated": 0,
                         "total_generate_s": 0.0}


def _print_bundle_banner(router) -> None:
    """Emit the Phase 5 startup banner for a MultiEngineRouter."""
    from core.config import (
        ROLE_TO_ENGINE, SPECULATIVE_DRAFT_MODEL, SPECULATIVE_TARGET_ENGINE,
        MODEL_BUNDLES,
    )
    bundle = router.bundle_name
    print(f"[router] bundle: {bundle}")
    bundle_spec = MODEL_BUNDLES.get(bundle, {})
    for engine_name, llm in router.engines.items():
        cfg = bundle_spec.get(engine_name, {})
        model_id = cfg.get("model", getattr(llm, "name", "?"))
        quant = cfg.get("quantization", cfg.get("dtype", "?"))
        max_seqs = cfg.get("max_num_seqs", "?")
        print(f"[router] engine {engine_name:>10s}: {model_id} "
              f"({quant}, max_seqs={max_seqs})")
    routing = ROLE_TO_ENGINE.get(bundle, {})
    routing_pairs = ", ".join(
        f"{r}->{('|'.join(e) if isinstance(e, list) else (e or 'disabled'))}"
        for r, e in routing.items()
    )
    print(f"[router] role routing: {routing_pairs}")
    if router.speculative_enabled and SPECULATIVE_DRAFT_MODEL:
        print(f"[router] speculative: {SPECULATIVE_TARGET_ENGINE} engine "
              f"uses {SPECULATIVE_DRAFT_MODEL} draft model")
    else:
        print(f"[router] speculative: disabled")
    print(f"[router] prefix_caching=enabled on all engines")


def _wrap_llm_with_progress(llm, progress):
    """Wrap llm.generate so each call ticks the progress bar.

    Idempotent: re-wrapping the same llm (e.g. when the router reuses a
    cached model across phases) returns it untouched so the progress bar
    doesn't double-count.

    Also tracks peak in-flight concurrency on the wrapped object so
    summary.json can report concurrent_calls_peak and token throughput.
    """
    if getattr(llm, "_progress_wrapped", False):
        return llm
    original = llm.generate
    if not hasattr(llm, "_inflight_state"):
        llm._inflight_state = {"count": 0, "peak": 0, "total_calls": 0, "total_tokens_generated": 0}
    state = llm._inflight_state

    async def generate(prompt, role="agent", max_tokens=120, temperature=0.7):
        state["count"] += 1
        if state["count"] > state["peak"]:
            state["peak"] = state["count"]
            _LLM_INFLIGHT_METRICS["peak"] = max(_LLM_INFLIGHT_METRICS["peak"], state["peak"])
        state["total_calls"] += 1
        _LLM_INFLIGHT_METRICS["total_calls"] += 1
        _gen_t0 = time.time()
        try:
            result = await original(prompt, role=role, max_tokens=max_tokens, temperature=temperature)
            token_count = 0
            if isinstance(result, str):
                token_count = len(result.split())
            elif isinstance(result, (list, tuple)):
                token_count = sum(len(str(item).split()) for item in result)
            state["total_tokens_generated"] += token_count
            _LLM_INFLIGHT_METRICS["total_tokens_generated"] += token_count
            return result
        finally:
            state["count"] -= 1
            _LLM_INFLIGHT_METRICS["total_generate_s"] += time.time() - _gen_t0
            try:
                progress.update(1)
            except Exception:
                pass

    llm.generate = generate
    llm._progress_wrapped = True
    return llm


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _reset_kb(kb_dir: Path) -> None:
    """Move all *.json files in kb_dir to kb_dir/.quarantine_<timestamp>/.

    Called when --reset-kb is passed. Preserves the contaminated files so
    they can be inspected or migrated, but removes them from the active KB.
    """
    if not kb_dir.exists():
        print(f"[pipeline] --reset-kb: {kb_dir} does not exist, nothing to quarantine")
        return
    quarantine = kb_dir / f".quarantine_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    quarantine.mkdir(parents=True, exist_ok=True)
    moved = 0
    for f in kb_dir.glob("*.json"):
        shutil.move(str(f), str(quarantine / f.name))
        moved += 1
    if moved:
        print(f"[pipeline] --reset-kb: moved {moved} file(s) to {quarantine}")
    else:
        print(f"[pipeline] --reset-kb: no *.json files found in {kb_dir}")


async def run_pipeline(task_type: str, user_prompt: str, output_dir: Path,
                       ignore_kb: bool = False,
                       reset_kb: bool = False,
                       corpus_mode: str = "real",
                       show_partition_overlap: bool = False,
                       cloud_provider: str = "none",
                       prune_kb_before: Optional[str] = None) -> dict:
    # Phase M: wall-clock from pipeline start (covers retrieval, all rounds,
    # AND synthesis — round_logs only sum each round's compute time).
    pipeline_start = time.time()
    from core.search_tool import reset_search_stats as _reset_search_stats
    _reset_search_stats()  # per-run retrieval timing (Tier-0 instrumentation)
    task_prompt = build_task_prompt(task_type, user_prompt)
    if config.USE_HETEROGENEOUS:
        # make_router picks LoRAHeterogeneousRouter on L4/A100 (vLLM + LoRA)
        # and HeterogeneousRouter on laptop (GGUF sequential). T4 raises —
        # config._TIER='t4' should have flipped USE_HETEROGENEOUS off in
        # config.py already (B1), so reaching this with tier='t4' is a bug.
        router = make_router(config._TIER)
        print(f"[pipeline] heterogeneous routing enabled "
              f"(router={type(router).__name__}, tier={config._TIER}).")
        print(f"[pipeline] router manifest: {json.dumps(router.manifest(), indent=2)}")
        llm = None  # phase loops will acquire per-model below
    else:
        router = None
        llm = make_llm()
    store = SignalStore()

    active_roles = ROLES_FOR_TASK.get(task_type, {"critic", "hater", "validator"})
    n_critics = NUM_CRITICS if "critic" in active_roles else 0
    n_haters = NUM_HATERS if "hater" in active_roles else 0
    n_validators = NUM_VALIDATORS if "validator" in active_roles else 0

    # Knowledge base. Default: OFF (--use-kb opts in; --reset-kb wipes prior entries).
    from core.knowledge_base import _topic_hash as _kb_topic_hash
    current_topic_hash = _kb_topic_hash(user_prompt)
    kb = KnowledgeBase()
    if reset_kb:
        _reset_kb(kb.kb_dir)
    if ignore_kb:
        print("[kb] disabled (--use-kb to enable)")
    else:
        kb.load()
        if prune_kb_before:
            kb.prune_before(prune_kb_before)
        n_priors = (len(kb.prior_consensus(current_topic_hash))
                    + len(kb.prior_rejections(current_topic_hash)))
        print(f"[kb] loaded {n_priors} priors — set --no-kb to disable")

    # Corpus retrieval (§1): real retriever by default, placeholder opt-in via --corpus=placeholder
    if task_type == "coding":
        # Coding scouts read from the task prompt directly; no corpus needed.
        chunks = []
        partitions = partition_for_scouts(chunks, NUM_SCOUTS)
    elif corpus_mode == "placeholder":
        print(
            "[retrieval] WARNING: falling back to engineered placeholder corpus "
            "— partition diversity numbers from this run are not evidence"
        )
        corpus_text = trivial_corpus_from_thesis(user_prompt)
        chunks = chunk_corpus(corpus_text, source_tag="placeholder_corpus")
        partitions = partition_for_scouts(chunks, NUM_SCOUTS)
    else:
        # Fix S — FacetPlanner + web partitions + corpus retrieval
        from core.retrieval import CachedRetriever, CompositeRetriever
        from core.facet_planner import FacetPlanner
        from core.search_tool import search as _web_search_fn

        # Generate N_FACETS query angles. In heterogeneous mode llm is None;
        # FacetPlanner falls back to deterministic suffixes gracefully.
        _facet_planner = FacetPlanner(llm=llm, task_type=task_type)
        facets = await _facet_planner.generate(user_prompt, n=N_FACETS)

        # Web partitions: first WEB_PARTITION_COUNT scouts get live web results,
        # each stamped with partition_id="web_<slug>" via custom_partition_id.
        _n_web = min(WEB_PARTITION_COUNT, NUM_SCOUTS, len(facets))
        web_partitions: list = []
        for _wi in range(_n_web):
            _facet = facets[_wi]
            _slug = FacetPlanner.facet_slug(_facet)
            _pid = f"web_{_slug}"
            _wchunks = _web_search_fn(_facet, task_type=task_type)
            if not _wchunks:
                print(
                    f"[RETRIEVAL] *** WARNING *** 0 web chunks for facet {_wi} "
                    f"{_facet!r} — scout {_wi} will see an empty web partition."
                )
            for _ch in _wchunks:
                _ch.partition_id = _pid
            web_partitions.append(ScoutPartition(
                scout_index=_wi,
                chunks=_wchunks,
                custom_partition_id=_pid,
            ))
            print(f"[RETRIEVAL] web partition {_wi}: {len(_wchunks)} chunks "
                  f"pid={_pid!r}")

        # Corpus partitions: remaining scouts use the composite retriever.
        _n_corpus = NUM_SCOUTS - _n_web
        if _n_corpus > 0:
            retriever = CachedRetriever(CompositeRetriever())
            retrieved_chunks = retriever.retrieve(user_prompt)
            if retrieved_chunks:
                combined_text = "\n\n".join(c.text for c in retrieved_chunks)
                chunks = chunk_corpus(combined_text, source_tag="retrieved_corpus")
                for orig in retrieved_chunks:
                    for ch in chunks:
                        if ch.source_tag == "retrieved_corpus":
                            ch.source_tag = orig.source_tag
                            break
            else:
                print(
                    "[RETRIEVAL] *** ALL CORPUS BACKENDS FAILED *** — "
                    "falling back to placeholder corpus."
                )
                corpus_text = trivial_corpus_from_thesis(user_prompt)
                chunks = chunk_corpus(corpus_text, source_tag="placeholder_corpus")
            corpus_parts = partition_for_scouts(chunks, _n_corpus)
            # Re-index so corpus scout_index values don't collide with web scouts.
            for _j, _cp in enumerate(corpus_parts):
                _cp.scout_index = _n_web + _j
                _new_pid = _cp.partition_id  # "partition_{_n_web + _j}"
                for _ch in _cp.chunks:
                    _ch.partition_id = _new_pid
        else:
            corpus_parts = []
            chunks = []

        partitions = web_partitions + corpus_parts
        assert len(partitions) == NUM_SCOUTS, (
            f"[FIX S] partition count {len(partitions)} != NUM_SCOUTS={NUM_SCOUTS} "
            f"(web={len(web_partitions)}, corpus={len(corpus_parts)})"
        )

    # Topology generation (bounds-first exploration): one LLM call before any
    # scout runs, producing an AnswerSpaceTopology whose cells are pre-assigned
    # to scouts. Scouts carry topology_coords in INITIAL metadata for coverage
    # tracking in build_projection. Returns None gracefully on parse failure.
    topology = None
    topology_cell_assignments: list = [None] * NUM_SCOUTS
    if task_type != "coding" and llm is not None:
        topology = await generate_topology(task_prompt, task_type, llm)
        topology_cell_assignments = assign_topology_cells(topology, NUM_SCOUTS)

    # Estimate total LLM calls for the progress bar.
    # Per round: scouts + validators (phase A) + foragers + critics + haters (phase B).
    # Plus 1 synthesizer call at the end.
    agents_per_round = NUM_SCOUTS + NUM_FORAGERS + n_critics + n_haters + n_validators
    total_calls = agents_per_round * ITERATIONS_PER_ROUND * NUM_ROUNDS + 1

    print(f"\n[pipeline] task: {task_type!r} prompt: {user_prompt!r}")
    print(f"[pipeline] active roles: scout, forager, " + ", ".join(sorted(active_roles)) + ", synthesizer")
    print(f"[pipeline] population: scouts={NUM_SCOUTS} foragers={NUM_FORAGERS} "
          f"critics={n_critics} haters={n_haters} validators={n_validators}")
    print(f"[pipeline] rounds={NUM_ROUNDS} iter/round={ITERATIONS_PER_ROUND}")
    print(f"[pipeline] expected LLM calls: {total_calls}")
    print(f"[pipeline] corpus chunks: {len(chunks)} | partitions: {[len(p.chunks) for p in partitions]}")
    if router is not None:
        print(f"[pipeline] llm backend: heterogeneous (per-role routing)\n")
    else:
        print(f"[pipeline] llm backend: {llm.name}\n")

    progress = _make_progress(total_calls, desc="swarm")
    if llm is not None:
        llm = _wrap_llm_with_progress(llm, progress)

    async def _run_phase(agents):
        """Run a phase's agents concurrently, optionally grouped by model.

        Homogeneous (router is None): straight asyncio.gather over the shared
        llm. Heterogeneous (router not None): group by assigned model, load
        each unique model once, run the subset, then move on. Stats are
        reordered to match the input agent order so downstream zip() with
        all_agents stays correct.
        """
        if router is None:
            return await asyncio.gather(
                *(a.run(store, ITERATIONS_PER_ROUND) for a in agents)
            )
        groups = router.group_agents(agents)
        stats_by_id: dict = {}
        for model_path, agent_subset in groups.items():
            async with router.acquire(model_path) as model_llm:
                wrapped = _wrap_llm_with_progress(model_llm, progress)
                for a in agent_subset:
                    a.llm = wrapped
                stats_subset = await asyncio.gather(
                    *(a.run(store, ITERATIONS_PER_ROUND) for a in agent_subset)
                )
                for a, s in zip(agent_subset, stats_subset):
                    stats_by_id[a.agent_id] = s
        return [stats_by_id[a.agent_id] for a in agents]

    round_logs = []
    # pipeline_start is captured at function entry (Phase M wall-clock) —
    # don't reset it here or the wall_clock_s would exclude retrieval time.
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n=== Round {round_num} / {NUM_ROUNDS} ===")
        round_start = time.time()

        # Look up domain-specific role classes (coding tasks override defaults)
        role_cls = get_role_classes(task_type)
        ScoutClass = role_cls["scout"]
        DeveloperClass = role_cls["developer"]
        CriticClass = role_cls["critic"]
        HaterClass = role_cls["hater"]
        ValidatorClass = role_cls["validator"]

        if task_type == "coding":
            # Coding scouts don't use corpus partitions
            scouts = [
                ScoutClass(
                    agent_id=f"scout_R{round_num}_{i}",
                    llm=llm,
                    task_prompt=task_prompt,
                )
                for i in range(NUM_SCOUTS)
            ]
        else:
            scouts = []
            for i in range(NUM_SCOUTS):
                _cell = topology_cell_assignments[i] if topology_cell_assignments else None
                _cell_desc = (
                    format_cell_for_prompt(topology, _cell)
                    if topology is not None and _cell is not None
                    else ""
                )
                scouts.append(ScoutClass(
                    agent_id=f"scout_R{round_num}_{i}",
                    llm=llm,
                    config=ScoutConfig(
                        task_prompt=task_prompt,
                        partition=partitions[i],
                        # Phase 2C: agentic search by default; --corpus=placeholder
                        # disables it so the regression path still exercises the
                        # contiguous-partition diversity engine.
                        use_search=(corpus_mode != "placeholder"),
                        user_prompt=user_prompt,
                        topology_cell=_cell,
                        topology_cell_desc=_cell_desc,
                    ),
                ))
        validators = []
        for i in range(n_validators):
            name, strat = strategy_for_validator(i)
            validators.append(ValidatorClass(
                agent_id=f"validator_R{round_num}_{i}_{name}",
                llm=llm, strategy=strat, strategy_name=name,
                task_prompt=task_prompt,
                cloud_provider=cloud_provider,
            ))
        foragers = []
        for i in range(NUM_FORAGERS):
            name, strat = strategy_for_forager(i)
            foragers.append(DeveloperClass(
                agent_id=f"forager_R{round_num}_{i}_{name}",
                llm=llm, strategy=strat, strategy_name=name,
                task_prompt=task_prompt,
            ))
        critics = []
        for i in range(n_critics):
            name, strat = strategy_for_critic(i)
            critics.append(CriticClass(
                agent_id=f"critic_R{round_num}_{i}_{name}",
                llm=llm, strategy=strat, strategy_name=name,
                task_prompt=task_prompt,
            ))
        haters = [
            HaterClass(agent_id=f"hater_R{round_num}_{i}", llm=llm, task_prompt=task_prompt,
                       task_type=task_type)
            for i in range(n_haters)
        ]

        # Phase A: scouts + validators. Scouts deposit INITIAL signals;
        # validators (running on the previous round's INITIALs, if any)
        # deposit VERIFICATIONs that will inform Phase B's provenance boost.
        phase_a_agents = [*scouts, *validators]
        phase_a_stats = await _run_phase(phase_a_agents)
        assert len(phase_a_stats) == len(phase_a_agents), \
            f"phase A stats count {len(phase_a_stats)} != agents {len(phase_a_agents)}"

        # Phase B: foragers, critics, haters. By now Phase A's VERIFICATIONs
        # exist in the store, so any deposits here under verified lineage
        # pick up the provenance boost.
        phase_b_agents = [*foragers, *critics, *haters]
        phase_b_stats = await _run_phase(phase_b_agents)
        assert len(phase_b_stats) == len(phase_b_agents), \
            f"phase B stats count {len(phase_b_stats)} != agents {len(phase_b_agents)}"

        all_stats = list(phase_a_stats) + list(phase_b_stats)
        records = [s.context_record for s in all_stats]

        store.decay_all()
        pruned = store.prune_weak()
        # Corrective re-cluster once per round (see continuous-pool note): split
        # incohesive blobs so projection/synth see distinct positions, not a blob.
        if config.CLUSTER_RECLUSTER_EVERY:
            _n_split = store.recluster("INITIAL")
            if _n_split:
                print(f"[recluster] round {round_num}: split {_n_split} incohesive cluster(s)")

        elapsed = time.time() - round_start

        # Output diversity: collect all deposited texts from this round
        all_deposit_texts = [
            c for rec in records for c in rec.deposit_contents if c
        ]
        print()
        print(format_output_diversity_report(all_deposit_texts, round_num))
        if show_partition_overlap:
            print(format_partition_overlap_report(records))
        print(f"[round {round_num}] store stats: {store.stats()}")
        print(f"[round {round_num}] pruned this round: {pruned}")
        print(f"[round {round_num}] elapsed: {elapsed:.1f}s")

        # Build per-agent stats for the round log
        all_agents = [*scouts, *validators, *foragers, *critics, *haters]
        agent_stats_log = []
        for agent, stats in zip(all_agents, all_stats):
            agent_stats_log.append({
                "agent_id": agent.agent_id,
                "role": agent.ROLE,
                "deposits": stats.deposits,
                "rejected_dup": stats.rejected_dup,
                "iterations": stats.iterations,
                "novelty_per_iter": stats.novelty_per_iter,
            })

        # D3: Partition coverage logging — count INITIAL deposits per partition
        # in THIS round (scouts are named scout_R{round}_{partition_index}).
        partition_deposits: dict[str, int] = {}
        for agent, stats in zip(all_agents, all_stats):
            if agent.ROLE == "scout" and stats.deposits > 0:
                parts = agent.agent_id.split("_")
                tag = f"partition_{parts[2]}" if len(parts) >= 3 else "partition_unknown"
                partition_deposits[tag] = partition_deposits.get(tag, 0) + stats.deposits

        from core.output_diversity import (
            centroid_cosine_distance, self_bleu as _self_bleu,
            output_diversity_by_model,
        )
        # Cross-model decomposition: how much of the output spread is between
        # models (heterogeneity) vs. within-model. None for homogeneous runs.
        model_assignment_for_log = (
            router.manifest() if router is not None
            else {"all": (llm.name if llm is not None else "unknown")}
        )
        cross_model_delta = output_diversity_by_model(
            records, getattr(store, "_embedder", None), model_assignment_for_log,
        )
        round_logs.append({
            "round": round_num,
            "partition_overlap": {
                "by_role": _role_partition_overlap(records),
                "overall": _overall_partition_overlap(records),
            },
            "output_diversity": {
                "centroid_cosine_dist": centroid_cosine_distance(all_deposit_texts),
                "self_bleu": _self_bleu(all_deposit_texts),
                "n_deposits": len(all_deposit_texts),
            },
            "diversity": {
                "cross_model_delta": cross_model_delta,
            },
            "stats": store.stats(),
            "pruned": pruned,
            "elapsed_s": elapsed,
            "agents": agent_stats_log,
            "partition_deposits_this_round": partition_deposits,
        })

    # Persist FIRST so a synthesizer crash can't lose the run.
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_dump = [
        {
            "id": s.id, "type": s.type, "strength": round(s.strength, 4),
            "depositor": s.depositor, "parent_id": s.parent_id,
            "cluster_id": s.cluster_id,
            "visits": s.visits,
            "metadata": s.metadata,
            "content": s.content,
        }
        for s in store.all()
    ]
    (output_dir / "signals.json").write_text(
        json.dumps(signal_dump, indent=2), encoding="utf-8"
    )
    (output_dir / "round_log.json").write_text(
        json.dumps(round_logs, indent=2), encoding="utf-8"
    )
    backend_label = "heterogeneous" if router is not None else llm.name
    model_assignment = router.manifest() if router is not None else {"all": llm.name}
    # Phase M: record migration-relevant context so cross-run analysis can
    # distinguish laptop / Colab / heterogeneous / phase-isolated runs.
    model_loaded_once = (router is None and not config.USE_HETEROGENEOUS)
    (output_dir / "run_meta.json").write_text(json.dumps({
        "task_type": task_type,
        "user_prompt": user_prompt,
        "llm_backend": backend_label,
        "is_mock": USE_MOCK_LLM,
        "timestamp": os.environ.get("SWARM_RUN_TIMESTAMP", datetime.now().isoformat()),
        "colab_tier": config._TIER,
        "vllm_concurrency": int(config.LLM_CONCURRENCY),
        "vllm_dtype": str(config.VLLM_DTYPE),
        "population_scaled_for": config._TIER if config._TIER else "laptop",
        "model_loaded_once": model_loaded_once,
        "model_name": config.MODEL_NAME,
        "active_roles": sorted(active_roles),
        "population": {
            "scouts": NUM_SCOUTS, "foragers": NUM_FORAGERS,
            "critics": n_critics, "haters": n_haters, "validators": n_validators,
        },
        "model_assignment": model_assignment,
    }, indent=2), encoding="utf-8")
    print(f"\n[pipeline] signals + round log saved before synthesis: {output_dir}")

    # Synthesis (best-effort)
    print("=== Synthesizing final output ===")
    run_meta_dict = {
        "task_type": task_type,
        "user_prompt": user_prompt,
        "timestamp": datetime.now().isoformat(),
    }
    SynthesizerClass = get_role_classes(task_type).get("synthesizer", Synthesizer)
    try:
        if router is not None:
            async with router.acquire(router.model_for_role("synthesizer")) as synth_llm:
                synth = SynthesizerClass(
                    _wrap_llm_with_progress(synth_llm, progress), task_prompt
                )
                final_answer, citations, lineage_dot = await synth.synthesize(
                    store,
                    has_validators=(n_validators > 0),
                    prior_rejections=kb.prior_rejections(current_topic_hash) if not ignore_kb else None,
                    prior_consensus=kb.prior_consensus(current_topic_hash) if not ignore_kb else None,
                    output_dir=output_dir,
                    task_type=task_type,
                )
        else:
            synth = SynthesizerClass(llm, task_prompt)
            final_answer, citations, lineage_dot = await synth.synthesize(
                store,
                has_validators=(n_validators > 0),
                prior_rejections=kb.prior_rejections(current_topic_hash) if not ignore_kb else None,
                prior_consensus=kb.prior_consensus(current_topic_hash) if not ignore_kb else None,
                output_dir=output_dir,
                task_type=task_type,
            )
    except Exception as exc:
        print(f"[pipeline] synthesis failed ({type(exc).__name__}: {exc})")
        print(f"[pipeline] signals.json is still on disk — run synthesize.py against it.")
        final_answer = f"(synthesis failed: {type(exc).__name__}: {exc})"
        citations = {}
        lineage_dot = ""

    if router is not None:
        await router.teardown()

    write_answer_files(output_dir, final_answer)
    if citations:
        (output_dir / "citations.json").write_text(
            json.dumps(citations, indent=2), encoding="utf-8"
        )
    if lineage_dot:
        (output_dir / "lineage.dot").write_text(lineage_dot, encoding="utf-8")

    # Save to knowledge base (best-effort)
    kb_diff: dict = {"new": 0, "matched": 0, "archived": 0}
    if not ignore_kb:
        try:
            from core.projection import build_projection
            final_projection = build_projection(store, has_validators=(n_validators > 0), topology=topology)
            diff = kb.save(final_projection, store, run_meta_dict, output_dir=output_dir)
            kb_diff = {
                "new": len(diff.get("new_entries", [])),
                "matched": len(diff.get("matched_entries", [])),
                "archived": len(diff.get("decayed_to_archive", [])),
            }
        except Exception as exc:
            print(f"[pipeline] kb save failed ({type(exc).__name__}: {exc})")

    try:
        progress.close()
    except Exception:
        pass

    # D1: summary.json — compact run scorecard for empirical cross-run comparison.
    total_elapsed = sum(r.get("elapsed_s", 0) for r in round_logs)
    total_llm_calls = 0
    total_deposits = 0
    scout_rejected = 0
    scout_iters = 0
    non_scout_rejected = 0
    non_scout_iters = 0
    junk_rejections = 0
    for rlog in round_logs:
        for ag in rlog.get("agents", []):
            d = ag.get("deposits", 0)
            r = ag.get("rejected_dup", 0)
            its = ag.get("iterations", 0)
            total_llm_calls += its
            total_deposits += d
            if ag.get("role") == "scout":
                scout_rejected += r
                scout_iters += its
            else:
                non_scout_rejected += r
                non_scout_iters += its

    # Grab audit flag count from the last written audit. Sentinel semantics
    # (Bug 3): 0 = audit ran clean, N>=1 = N flags, -2 = audit crashed
    # (renderer_audit.json carries audit_error). Default to 0 when the file
    # is missing — the synthesizer now writes it on every branch including
    # no_consensus, so absence means the synth itself crashed before audit;
    # that's already captured by the (synthesis failed: ...) string in
    # answer.txt and we shouldn't add a separate ambiguous -1 sentinel.
    try:
        audit_data = json.loads((output_dir / "renderer_audit.json").read_text(encoding="utf-8"))
        audit_flags = audit_data.get("total_flags", 0)
    except Exception:
        audit_flags = 0

    # Final projection for cluster counts.
    try:
        from core.projection import build_projection as _bp
        _final_proj = _bp(store, has_validators=(n_validators > 0))
        ver_scores = [cp.verification_score for cp in _final_proj.surviving + _final_proj.contested]
        # Phase 3C: max SUPPORT chain depth across all clusters. Visible
        # measure of how deeply the swarm developed any single claim.
        support_depth_max = max(
            (cp.support_depth for cp in (
                _final_proj.surviving + _final_proj.contested
                + _final_proj.weakly_supported + _final_proj.rejected_by_field
                + _final_proj.unverified
            )),
            default=0,
        )
        wall_clock_s = round(time.time() - pipeline_start, 1)
        total_tokens_generated = _LLM_INFLIGHT_METRICS.get("total_tokens_generated", 0)
        peak_concurrency = _LLM_INFLIGHT_METRICS.get("peak", 0)
        tokens_per_second = total_tokens_generated / wall_clock_s if wall_clock_s > 0 else 0.0

        # Tier-0 timing breakdown: where did wall-clock go? `*_cumulative_s`
        # sums across concurrency so they can exceed wall_clock_s; the ratios
        # use wall-clock and are the actionable latency signal. Retrieval is
        # the historical sink (blocking HTTP), now run off the event loop.
        from core.search_tool import search_stats_snapshot
        _search_stats = search_stats_snapshot()
        _llm_cum_s = round(_LLM_INFLIGHT_METRICS.get("total_generate_s", 0.0), 1)
        timing = {
            "wall_clock_s": float(wall_clock_s),
            "search": _search_stats,
            "llm_generate_cumulative_s": _llm_cum_s,
            "search_cumulative_s": _search_stats.get("total_s", 0.0),
            "page_fetch_cumulative_s": _search_stats.get("fetch_total_s", 0.0),
            "search_fraction_of_wallclock": (
                round(_search_stats.get("total_s", 0.0) / wall_clock_s, 3)
                if wall_clock_s > 0 else 0.0
            ),
        }
        summary = {
            "task_type": task_type,
            "user_prompt": user_prompt,
            "llm_backend": backend_label,
            "total_llm_calls": total_llm_calls,
            "total_elapsed_s": round(total_elapsed, 1),
            "wall_clock_s": float(wall_clock_s),
            "total_tokens_generated": int(total_tokens_generated),
            "tokens_per_second": float(round(tokens_per_second, 4)),
            "concurrent_calls_peak": int(peak_concurrency),
            "scout_reject_rate": round(scout_rejected / max(1, scout_iters), 4),
            "non_scout_reject_rate": round(non_scout_rejected / max(1, non_scout_iters), 4),
            "junk_rejections": junk_rejections,
            "n_clusters": {
                "surviving": len(_final_proj.surviving),
                "contested": len(_final_proj.contested),
                "weakly_supported": len(_final_proj.weakly_supported),
                "rejected_by_field": len(_final_proj.rejected_by_field),
            },
            "max_verification_score": round(max(ver_scores), 4) if ver_scores else 0.0,
            "avg_verification_score": round(sum(ver_scores) / max(1, len(ver_scores)), 4) if ver_scores else 0.0,
            "support_depth_max": int(support_depth_max),
            "audit_flags": audit_flags,
            "kb_diff": kb_diff,
            "timing": timing,
        }
    except Exception as exc:
        summary = {"error": str(exc)}

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[pipeline] summary.json:")
    print(json.dumps(summary, indent=2))

    print(f"\n[pipeline] outputs: {output_dir}")
    if citations:
        print(f"[pipeline] citations.json: {len(citations)} signal entries")
    if lineage_dot:
        print(f"[pipeline] lineage.dot: written")
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(final_answer)
    print("=" * 60)

    return {"answer": final_answer, "rounds": round_logs}


# ---------------------------------------------------------------------------
# Continuous (action-pool) pipeline — Phase 1 of the emergent-swarm rewrite
# ---------------------------------------------------------------------------

_CONTINUOUS_DEFAULT_WORKERS = 24


async def run_continuous_pipeline(
    task_type: str, user_prompt: str, output_dir: Path,
    ignore_kb: bool = True,
    reset_kb: bool = False,
    n_workers: int = _CONTINUOUS_DEFAULT_WORKERS,
    cloud_provider: str = "none",
    bundle: Optional[str] = None,
    prune_kb_before: Optional[str] = None,
) -> dict:
    """Continuous worker pool replacing the round/phase scheduler.

    Workers pick actions per iteration against the live store; the
    ConvergenceDetector decides when to halt. No fixed NUM_ROUNDS,
    no Phase A / Phase B serialization.

    `bundle` (optional): name of a MODEL_BUNDLES entry. When provided, the
    pipeline loads every engine in the bundle and routes Worker.generate()
    calls per-role via MultiEngineRouter. When None and config.USE_MODEL_BUNDLES
    is set, falls back to TASK_TO_BUNDLE[task_type].
    """
    pipeline_start = time.time()
    from core.search_tool import reset_search_stats as _reset_search_stats
    _reset_search_stats()  # per-run retrieval timing (Tier-0); critical for
    #                        multi-run-per-process A/B so stats don't bleed.
    task_prompt = build_task_prompt(task_type, user_prompt)

    # Bundle path: load every engine declared in the bundle and route per-role.
    # Single-engine path: traditional make_llm() backed by vLLM cascade / HF.
    router = None
    use_bundle = bundle is not None or config.USE_MODEL_BUNDLES

    # Guard: bundles require vLLM. If vLLM failed to import (e.g. CUDA version
    # mismatch — libcudart.so.13 on a CUDA-12 host), skip the bundle and fall
    # through to make_llm() which handles HF transformers + bitsandbytes 4-bit.
    if use_bundle:
        from core.llm_vllm import _VLLM_AVAILABLE as _vllm_ok
        if not _vllm_ok:
            print(
                "[pipeline] WARNING: vLLM is unavailable (CUDA version mismatch or "
                "missing library). Skipping bundle and falling back to single-model "
                "HF transformers + bitsandbytes 4-bit path. The pipeline will still "
                "run; throughput will be lower than vLLM batch-mode.\n"
                "[pipeline] To fix: re-run Cell 4 in the Colab notebook — it detects "
                "your CUDA version and installs the matching vLLM wheel."
            )
            use_bundle = False
            bundle = None
            config.USE_MODEL_BUNDLES = False

    if use_bundle:
        router = make_bundle_router(task_type, override_bundle=bundle)
        config.ACTIVE_BUNDLE = router.bundle_name
        await router.load()
        # Preflight any Groq-backed bundle so unavailable model names swap to
        # the fallback before the run instead of zeroing a role mid-run.
        if hasattr(router, "preflight"):
            await router.preflight()
        _print_bundle_banner(router)
        # Warm every engine in parallel (cheap once they're loaded).
        await router.warmup(n=3)
        llm = next(iter(router.engines.values())) if router.engines else None
    else:
        # Groq API path: when GROQ_API_KEY is set, route each role to a
        # different Groq-hosted model family. This is the correct solution for
        # T4 (or any single-GPU environment): a genuine architectural diversity
        # without sequential loading. Workers still run via run_pool() with
        # router=router; engine_for(role) returns the right GroqBackend.
        _groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        _backend = os.environ.get("SWARM_BACKEND", "").strip().lower()
        # SWARM_BACKEND='local' is an explicit override: run every role on the
        # local engine even when a GROQ_API_KEY is present (e.g. kept around for
        # the eval judge). Without this, a present key silently forced GroqRouter
        # below and 'local' was ignored — so on a big GPU you'd still hit Groq's
        # rate limits. 'local' now means local.
        if _backend == "local" and _groq_key and not config.USE_MOCK_LLM:
            print("[pipeline] SWARM_BACKEND=local: ignoring GROQ_API_KEY, "
                  "running all roles on the local engine.")
        _use_groq = bool(_groq_key) and not config.USE_MOCK_LLM and _backend != "local"
        if _backend == "hybrid" and _use_groq:
            # Hybrid: local GPU model for high-volume roles (scout/dev/critic/
            # validator) + Groq for the few high-value roles (synth/hater). Best
            # fit for FREE Groq — tiny quota burn, full model-family diversity.
            from core.llm_hybrid import HybridRouter
            router = HybridRouter(api_key=_groq_key)
            # Preflight: ping each Groq model so a bad/gated name (e.g. the
            # llama-4-maverick hater that 404'd for a whole run) swaps to the
            # fallback now, not silently mid-run.
            await router.preflight()
            llm = router.engine_for("scout")   # local model; used for topology gen
            # Warm up the local model (the plain-local path does this) so the
            # first real generations don't each pay the JIT/slot-mapping cost —
            # critical on the HF backend, where many cold workers otherwise stall.
            if hasattr(llm, "warmup"):
                try:
                    await llm.warmup(n=5)
                except Exception as exc:
                    print(f"[pipeline] hybrid local warmup raised "
                          f"{type(exc).__name__}: {exc}")
            print(f"[pipeline] hybrid backend (local + Groq): {router.manifest()}")
        elif _use_groq:
            from core.llm_groq import GroqRouter
            router = GroqRouter(api_key=_groq_key)
            # Preflight: swap any unavailable model to the fallback before the
            # run rather than zeroing a role mid-run.
            await router.preflight()
            # llm is the scout backend (used for topology generation and any
            # path that expects a single llm before run_pool is called).
            llm = router.engine_for("scout")
        else:
            llm = make_llm()
            # Phase 0: JIT warmup so the slot-mapping kernel doesn't stall the
            # first real iteration. Only meaningful on vLLM; MockLLM/HF skip.
            if hasattr(llm, "warmup"):
                try:
                    await llm.warmup(n=5)
                except Exception as exc:
                    print(f"[pipeline] warmup raised {type(exc).__name__}: {exc}")

    store = SignalStore()

    # KB — default off, opt-in via --use-kb (handled in main()).
    from core.knowledge_base import _topic_hash as _kb_topic_hash
    current_topic_hash = _kb_topic_hash(user_prompt)
    kb = KnowledgeBase()
    if reset_kb:
        _reset_kb(kb.kb_dir)
    if ignore_kb:
        print("[kb] disabled (--use-kb to enable)")
    else:
        kb.load()
        if prune_kb_before:
            kb.prune_before(prune_kb_before)
        n_priors = (len(kb.prior_consensus(current_topic_hash))
                    + len(kb.prior_rejections(current_topic_hash)))
        print(f"[kb] loaded {n_priors} priors — set --no-kb to disable")
        # KB-aware novelty: register prior-consensus claims as novelty
        # references so scout multi-claim selection steers AWAY from what
        # the KB already holds — successive runs explore unclaimed regions
        # instead of re-deriving the same field for the dedup to merge.
        # (Scalar similarities only; no prior content reaches any prompt.)
        prior_texts = [
            e.get("representative_content", "")
            for e in kb.prior_consensus(current_topic_hash)
        ]
        n_refs = store.set_novelty_references(prior_texts)
        if n_refs:
            print(f"[kb] {n_refs} prior-consensus claims registered as "
                  f"scout novelty references")

    output_dir.mkdir(parents=True, exist_ok=True)
    validator_raw_path = output_dir / "validator_raw.log"
    validator_raw_path.write_text("", encoding="utf-8")  # truncate prior log
    from core.worker_pool import set_validator_raw_log
    set_validator_raw_log(validator_raw_path)

    # vLLM internal batcher means we want n_workers >= max_num_seqs. For
    # MockLLM / HF, the asyncio.Semaphore inside the LLM throttles real
    # concurrency to 1 — workers still queue cleanly.
    print(f"\n[pipeline] continuous pool — task={task_type!r} prompt={user_prompt!r}")
    if router is not None:
        _rb = router.bundle_name
        _label = (f"groq ({len(router.engines)} models)" if _rb == "groq"
                  else f"bundle:{_rb} ({len(router.engines)} engines)")
        print(f"[pipeline] llm backend: {_label}")
    else:
        print(f"[pipeline] llm backend: {llm.name}")
    print(f"[pipeline] n_workers={n_workers}")

    detector = ConvergenceDetector(store, task_type=task_type)
    stop_event = asyncio.Event()

    # Genome cache: cluster_id -> ClusterGenome. Shared with all workers via
    # reference. Refreshed periodically in the tick loop so workers see
    # up-to-date atom-targeted prompts as clusters develop.
    _genome_cache: dict = {}

    pool_task = asyncio.create_task(run_pool(
        store, llm, task_prompt, user_prompt, stop_event,
        n_workers=n_workers,
        task_type=task_type,
        router=router,
        genome_cache=_genome_cache,
    ))
    decay_task = asyncio.create_task(decay_loop(store, stop_event, interval_s=30.0))

    # Orchestrator tick loop — every 2s, snapshot state, log, check halt.
    _last_genome_refresh_iter = 0
    _GENOME_REFRESH_EVERY = 25   # refresh genome cache every N worker iterations
    _last_recluster_iter = 0
    while not stop_event.is_set():
        await asyncio.sleep(2.0)
        elapsed = time.time() - pipeline_start
        # Iterate counter is owned by the pool; peek via Task's pool_state.
        # We read it from the pool's state by polling its private accessor —
        # run_pool stores PoolState in a local. Read from store snapshot
        # via projection instead; iteration count surfaces via shares window.
        # Use a global tap: stash iteration_counter on the loop via pool_task.
        # Cleaner: ConvergenceDetector takes iteration & elapsed from us.
        # We pull iteration from PoolState via attribute on pool_task.
        ps = _peek_pool_state(pool_task)
        iter_n = ps.iteration_counter if ps else 0
        detector.tick(iter_n, elapsed)
        _log_progress(iter_n, elapsed, detector, ps, store, task_type=task_type)

        # Periodic divisive re-cluster (corrective): break up incohesive blobs so
        # the synthesizer is handed distinct positions, not one vague mega-cluster
        # + dust. Pure-Python; gated like the genome refresh. Runs BEFORE it so the
        # genome cache rebuilds on the cleaned partition.
        if config.CLUSTER_RECLUSTER_EVERY and (iter_n - _last_recluster_iter) >= config.CLUSTER_RECLUSTER_EVERY:
            try:
                n_split = store.recluster("INITIAL")
                if n_split:
                    print(f"[recluster] iter={iter_n}: split {n_split} incohesive cluster(s)")
            except Exception as exc:
                print(f"[recluster] failed: {type(exc).__name__}: {exc}")
            _last_recluster_iter = iter_n

        # Genome cache refresh: rebuild projection and update the shared dict
        # so workers get up-to-date atom-targeted prompts. Done every
        # _GENOME_REFRESH_EVERY iterations, not every tick (build_projection
        # is pure-Python but not free). Pure dict update is GIL-safe in async.
        if iter_n - _last_genome_refresh_iter >= _GENOME_REFRESH_EVERY:
            try:
                from core.projection import build_projection as _bp, FitnessTrajectory
                _proj = _bp(store, has_validators=True, task_type=task_type)
                new_cache: dict = {}
                for _cp in (_proj.surviving + _proj.contested
                            + _proj.weakly_supported + _proj.rejected_by_field):
                    if _cp.genome is None:
                        continue
                    _g = _cp.genome
                    # Dual-key: (1) representative_id for synthesizer/planner lookups,
                    # (2) ClusterRegistry cluster_id for worker pool lookups via
                    # Signal.cluster_id. Without key (2), _build_prompt and the
                    # targets_atom stamping always miss (different ID spaces).
                    _rep_sig = store.get(_cp.representative_id)
                    if _rep_sig and _rep_sig.cluster_id:
                        new_cache[_rep_sig.cluster_id] = _g
                    # Merge trajectory history from previous snapshot so
                    # FitnessTrajectory.fitness_history accumulates across refreshes
                    # and _trajectory_score() can return non-zero values.
                    _prev = _genome_cache.get(_cp.representative_id)
                    if _prev is not None and _prev.trajectory.fitness_history:
                        old_hist = _prev.trajectory.fitness_history
                        new_hist = old_hist + [(iter_n, _g.composite_fitness)]
                        # monotone_growth: fitness never decreased
                        _mono = all(
                            b >= a
                            for (_, a), (_, b) in zip(new_hist[:-1], new_hist[1:])
                        )
                        # consolidation: last 3 snapshots within 0.02
                        _consol = None
                        if len(new_hist) >= 3:
                            _recent = [f for _, f in new_hist[-3:]]
                            if max(_recent) - min(_recent) < 0.02:
                                _consol = new_hist[-3][0]
                        _g.trajectory = FitnessTrajectory(
                            formation_iteration=_prev.trajectory.formation_iteration,
                            fitness_history=new_hist,
                            strength_history=_prev.trajectory.strength_history,
                            member_count_history=_prev.trajectory.member_count_history,
                            monotone_growth=_mono,
                            consolidation_iteration=_consol,
                        )
                    else:
                        # First snapshot for this cluster
                        _g.trajectory.fitness_history = [(iter_n, _g.composite_fitness)]
                    new_cache[_cp.representative_id] = _g
                _genome_cache.clear()
                _genome_cache.update(new_cache)
                _last_genome_refresh_iter = iter_n
            except Exception:
                pass   # genome cache stays stale; workers fall back to standard prompts

        if detector.satisfied(iter_n, elapsed):
            break

    stop_event.set()
    # Give worker_loop one yield to notice the event, then wait.
    try:
        await asyncio.wait_for(pool_task, timeout=15.0)
    except asyncio.TimeoutError:
        pool_task.cancel()
        try:
            await pool_task
        except Exception:
            pass
    try:
        await asyncio.wait_for(decay_task, timeout=5.0)
    except asyncio.TimeoutError:
        decay_task.cancel()
        try:
            await decay_task
        except Exception:
            pass

    ps = _peek_pool_state(pool_task)
    total_iterations = ps.iteration_counter if ps else 0
    final_elapsed = time.time() - pipeline_start
    final_shares = ps.shares() if ps else {}

    print(f"\n[pipeline] pool halted: reason={detector.state.reason!r} "
          f"iterations={total_iterations} elapsed={final_elapsed:.1f}s")

    # Persist signals before synthesis.
    signal_dump = [
        {
            "id": s.id, "type": s.type, "strength": round(s.strength, 4),
            "depositor": s.depositor, "parent_id": s.parent_id,
            "cluster_id": s.cluster_id,
            "visits": s.visits,
            "metadata": s.metadata,
            "content": s.content,
        }
        for s in store.all()
    ]
    (output_dir / "signals.json").write_text(
        json.dumps(signal_dump, indent=2), encoding="utf-8"
    )

    _rb = router.bundle_name if router is not None else None
    backend_label = (
        ("groq:" + ":".join(sorted(set(router._role_models.values()))))
        if _rb == "groq"
        else (("bundle:" + _rb) if _rb else (llm.name if llm else "unknown"))
    )
    model_assignment = router.manifest() if router is not None else (
        {"all": llm.name} if llm is not None else {}
    )
    (output_dir / "run_meta.json").write_text(json.dumps({
        "task_type": task_type,
        "user_prompt": user_prompt,
        "llm_backend": backend_label,
        "is_mock": USE_MOCK_LLM,
        "timestamp": os.environ.get("SWARM_RUN_TIMESTAMP", datetime.now().isoformat()),
        "colab_tier": config._TIER,
        "model_name": config.MODEL_NAME,
        "execution_mode": "continuous_pool",
        "n_workers": n_workers,
        "bundle": (router.bundle_name if router is not None else None),
        # Single-model run; keep the heterogeneous-routing manifest shape
        # so downstream tooling that diffs runs across modes doesn't have to
        # branch.
        "model_assignment": model_assignment,
    }, indent=2), encoding="utf-8")

    # Synthesis
    run_meta_dict = {
        "task_type": task_type,
        "user_prompt": user_prompt,
        "timestamp": datetime.now().isoformat(),
    }
    SynthesizerClass = get_role_classes(task_type).get("synthesizer", Synthesizer)
    try:
        synth_llm = router.engine_for("synthesizer") if router is not None else llm
        synth = SynthesizerClass(synth_llm, task_prompt)
        final_answer, citations, lineage_dot = await synth.synthesize(
            store,
            has_validators=True,
            prior_rejections=kb.prior_rejections(current_topic_hash) if not ignore_kb else None,
            prior_consensus=kb.prior_consensus(current_topic_hash) if not ignore_kb else None,
            output_dir=output_dir,
            task_type=task_type,
        )
    except Exception as exc:
        print(f"[pipeline] synthesis failed ({type(exc).__name__}: {exc})")
        final_answer = f"(synthesis failed: {type(exc).__name__}: {exc})"
        citations = {}
        lineage_dot = ""

    if router is not None:
        await router.teardown()

    write_answer_files(output_dir, final_answer)
    if citations:
        (output_dir / "citations.json").write_text(
            json.dumps(citations, indent=2), encoding="utf-8"
        )
    if lineage_dot:
        (output_dir / "lineage.dot").write_text(lineage_dot, encoding="utf-8")

    # KB save (best effort)
    kb_diff: dict = {"new": 0, "matched": 0, "archived": 0}
    if not ignore_kb:
        try:
            from core.projection import build_projection
            final_projection = build_projection(store, has_validators=True,
                                                 task_type=task_type)
            diff = kb.save(final_projection, store, run_meta_dict, output_dir=output_dir)
            kb_diff = {
                "new": len(diff.get("new_entries", [])),
                "matched": len(diff.get("matched_entries", [])),
                "archived": len(diff.get("decayed_to_archive", [])),
            }
        except Exception as exc:
            print(f"[pipeline] kb save failed ({type(exc).__name__}: {exc})")

    # Audit flags
    try:
        audit_data = json.loads((output_dir / "renderer_audit.json").read_text(encoding="utf-8"))
        audit_flags = audit_data.get("total_flags", 0)
    except Exception:
        audit_flags = 0

    # summary.json — reuse the projection already computed for KB save when possible
    try:
        from core.projection import build_projection as _bp
        # Prefer the final projection already computed for KB save (stored as
        # `final_projection`). Fall back to a fresh one if KB was skipped or
        # the save failed before defining final_projection.
        try:
            _final_proj = final_projection  # type: ignore[name-defined]
        except NameError:
            _final_proj = _bp(store, has_validators=True, task_type=task_type)

        ver_scores = [cp.verification_score for cp in _final_proj.surviving + _final_proj.contested]
        support_depth_max = max(
            (cp.support_depth for cp in (
                _final_proj.surviving + _final_proj.contested
                + _final_proj.weakly_supported + _final_proj.rejected_by_field
                + _final_proj.unverified
            )),
            default=0,
        )

        # Genome quality metrics (P2.2 / R2 supplement)
        genome_cps = [
            cp for cp in (
                _final_proj.surviving + _final_proj.contested
                + _final_proj.weakly_supported
            )
            if cp.genome is not None
        ]
        genome_stats: dict = {}
        if genome_cps:
            fitness_vals = [cp.genome.composite_fitness for cp in genome_cps]
            grounding_vals = [
                cp.genome.fitness_breakdown.get("grounding", 0.0)
                for cp in genome_cps
            ]
            n_atoms_total = sum(len(cp.genome.atoms) for cp in genome_cps)
            genome_stats = {
                "n_genomes": len(genome_cps),
                "avg_composite_fitness": round(
                    sum(fitness_vals) / len(fitness_vals), 4
                ),
                "max_composite_fitness": round(max(fitness_vals), 4),
                "avg_grounding": round(sum(grounding_vals) / len(grounding_vals), 4),
                "total_atoms": n_atoms_total,
            }

        # Output diversity (P2.2 / R2 — continuous pipeline path)
        output_diversity: dict = {}
        try:
            from core.output_diversity import centroid_cosine_distance, self_bleu as _self_bleu
            all_deposit_texts = [
                s.content for s in store.all()
                if s.type in ("INITIAL", "SUPPORT") and len(s.content) > 20
            ]
            if len(all_deposit_texts) >= 2:
                output_diversity = {
                    "centroid_cosine_dist": centroid_cosine_distance(all_deposit_texts),
                    "self_bleu": _self_bleu(all_deposit_texts),
                    "n_texts": len(all_deposit_texts),
                }
        except Exception:
            pass

        # Silent-role check: a role whose backend made 0 calls all run produced
        # nothing (e.g. the hater that 404'd in lastgroqrun.txt). That is a
        # degraded run, so report it loudly and don't claim quality was met.
        silent_roles = []
        if router is not None and hasattr(router, "silent_roles"):
            try:
                silent_roles = list(router.silent_roles())
            except Exception:
                silent_roles = []
        if silent_roles:
            print(f"\n[pipeline] *** WARNING: roles produced 0 calls "
                  f"(degraded run): {silent_roles} ***", file=sys.stderr)

        # Embedder + clustering health. groqrun1.txt merged ZERO clusters all run
        # (every claim a singleton). Two distinct causes look identical in the
        # per-cluster audit, so disambiguate here and print it loudly + persist
        # it (survives the streamed-log truncation that hid the early NEW/JOIN
        # lines): (a) embedder unavailable -> semantic clustering is OFF; or
        # (b) embedder active but everything is still a singleton -> the join
        # threshold is too high for this embedding space.
        embedder = store.embedder_status()
        embedder_ok = embedder in ("all-MiniLM-L6-v2", "active")
        _all_cps = (_final_proj.surviving + _final_proj.contested
                    + _final_proj.weakly_supported + _final_proj.rejected_by_field)
        n_clusters_total = len(_all_cps)
        largest_cluster = max((len(cp.member_ids) for cp in _all_cps), default=0)
        multi_member = sum(1 for cp in _all_cps if len(cp.member_ids) > 1)
        n_initial_signals = sum(len(cp.member_ids) for cp in _all_cps)
        clustering_all_singletons = n_clusters_total >= 4 and largest_cluster <= 1
        clustering = {
            "embedder": embedder,
            "n_initial_signals": n_initial_signals,
            "n_clusters": n_clusters_total,
            "largest_cluster": largest_cluster,
            "multi_member_clusters": multi_member,
            # Calibration telemetry: set CLUSTER_JOIN_THRESHOLD at the antimode
            # of this distribution (see core/cluster_registry.join_sim_stats).
            "join_sim_stats": store.join_sim_stats(),
        }
        if not embedder_ok:
            print(f"\n[pipeline] *** WARNING: embedder {embedder} — semantic "
                  f"clustering + dedup OFF; every claim is its own cluster "
                  f"({n_clusters_total} singletons). Install sentence-transformers "
                  f"/ allow model download. ***", file=sys.stderr)
        elif clustering_all_singletons:
            print(f"\n[pipeline] *** WARNING: embedder active but ALL "
                  f"{n_clusters_total} clusters are singletons — join threshold "
                  f"(CLUSTER_JOIN_THRESHOLD) is too high for this embedding space; "
                  f"lower it via SWARM_CLUSTER_JOIN_THRESHOLD. ***", file=sys.stderr)

        degraded = bool(silent_roles) or not embedder_ok

        # Tier-0 timing breakdown: where did wall-clock go? `*_cumulative_s` sum
        # across concurrency (can exceed wall_clock_s); the ratio uses wall-clock
        # and is the actionable latency signal. Retrieval was the historical sink
        # (blocking HTTP); searches now run off the event loop (asyncio.to_thread).
        from core.search_tool import search_stats_snapshot as _search_stats_snap
        _search_stats = _search_stats_snap()
        # LLM timing is only populated when the progress wrapper is active
        # (legacy run_pipeline path). In the continuous-pool path the router
        # dispatches per-role engines directly, so report null rather than a
        # misleading 0 when uninstrumented.
        _llm_cum = (
            round(_LLM_INFLIGHT_METRICS.get("total_generate_s", 0.0), 1)
            if _LLM_INFLIGHT_METRICS.get("total_calls", 0) > 0 else None
        )
        timing = {
            "wall_clock_s": round(final_elapsed, 1),
            "search": _search_stats,
            "llm_generate_cumulative_s": _llm_cum,
            "search_cumulative_s": _search_stats.get("total_s", 0.0),
            "page_fetch_cumulative_s": _search_stats.get("fetch_total_s", 0.0),
            "search_fraction_of_wallclock": (
                round(_search_stats.get("total_s", 0.0) / final_elapsed, 3)
                if final_elapsed > 0 else 0.0
            ),
        }

        summary = {
            "task_type": task_type,
            "user_prompt": user_prompt,
            "execution_mode": "continuous_pool",
            "bundle": (router.bundle_name if router is not None else None),
            "engines": (router.engines_summary() if router is not None else None),
            "speculative_enabled": bool(router.speculative_enabled) if router is not None else False,
            "prefix_caching_enabled": True,
            "total_iterations": int(total_iterations),
            "wall_clock_s": round(final_elapsed, 1),
            "quality_met": bool(detector.state.quality_met) and not degraded,
            "silent_roles": silent_roles,
            "degraded": degraded,
            "embedder": embedder,
            "clustering": clustering,
            "convergence_reason": detector.state.reason or "unknown",
            "scout_gate_skips": int(ps.scout_gate_skips) if ps else 0,
            "action_shares": {k: round(v, 4) for k, v in final_shares.items()},
            "n_clusters": {
                "surviving": len(_final_proj.surviving),
                "contested": len(_final_proj.contested),
                "weakly_supported": len(_final_proj.weakly_supported),
                "rejected_by_field": len(_final_proj.rejected_by_field),
            },
            "max_verification_score": round(max(ver_scores), 4) if ver_scores else 0.0,
            "avg_verification_score": round(sum(ver_scores) / max(1, len(ver_scores)), 4) if ver_scores else 0.0,
            "support_depth_max": int(support_depth_max),
            "audit_flags": audit_flags,
            "kb_diff": kb_diff,
            "genome": genome_stats,
            "output_diversity": output_diversity,
            "timing": timing,
        }
    except Exception as exc:
        summary = {"error": str(exc)}

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[pipeline] summary.json:")
    print(json.dumps(summary, indent=2))
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(final_answer)
    print("=" * 60)
    return {"answer": final_answer, "iterations": total_iterations}


def _peek_pool_state(pool_task):
    """Return the PoolState attached to a run_pool task, if available.

    run_pool sets task._pool_state by assignment immediately after creating
    PoolState. We can't reach it via the task object directly without
    that, so check the task's frame for the local.
    """
    if pool_task.done():
        try:
            return pool_task.result()
        except Exception:
            return None
    # Fish out the running coroutine's locals; cheap and read-only.
    coro = pool_task.get_coro()
    frame = getattr(coro, "cr_frame", None)
    if frame is not None:
        return frame.f_locals.get("pool_state")
    return None


def _log_progress(iter_n: int, elapsed: float, detector,
                  pool_state, store, task_type: Optional[str] = None) -> None:
    """Per-tick progress line. Scaffold 6H format."""
    shares = pool_state.shares() if pool_state else {}
    shares_str = " ".join(
        f"{name[:4].upper()} {int(round(shares.get(name, 0.0) * 100))}%"
        for name in ("SCOUT", "DEVELOP", "CRITIQUE", "OBJECT", "VALIDATE",
                     "CHAIN", "REFINE")
    )
    proj = None
    try:
        from core.projection import build_projection
        proj = build_projection(store, has_validators=True, task_type=task_type)
    except Exception:
        pass
    surv = len(proj.surviving) if proj else 0
    weak = len(proj.weakly_supported) if proj else 0
    cold = (pool_state is not None and not pool_state.cold_start_done)
    # Live store counts — always read directly from the store so we can
    # distinguish "snapshot is stale" (n_init_live > 0, shares=SCOUT 100%)
    # from "deposits are silently failing" (n_init_live = 0 despite SCOUT share).
    _live_stats = store.stats().get("by_type", {})
    _n_init_live = _live_stats.get("INITIAL", 0)
    _n_supp_live = _live_stats.get("SUPPORT", 0)
    print(
        f"[swarm t={elapsed:.0f}s iter={iter_n}] "
        f"clusters: {surv} surviving, {weak} weakly\n"
        f"  shares: {shares_str}\n"
        f"  store: INITIAL={_n_init_live} SUPPORT={_n_supp_live}\n"
        f"  cold_start={cold} quality_met={detector.state.quality_met} "
        f"since_new_surv={detector.state.iterations_since_quality if detector.state.quality_met else detector.state.iterations_since_new_surviving} "
        f"novelty={detector.state.novelty_rate_last:.2f} "
        f"reason={detector.state.reason or '-'}"
    )


# ---------------------------------------------------------------------------
# Phase-isolated entry point
# ---------------------------------------------------------------------------

def _partitions_to_jsonable(partitions) -> list[dict]:
    out = []
    for p in partitions:
        out.append({
            "scout_index": p.scout_index,
            "chunks": [
                {"chunk_id": c.chunk_id, "text": c.text, "source_tag": c.source_tag}
                for c in p.chunks
            ],
        })
    return out


def _partitions_from_jsonable(data: list[dict]):
    from core.intake import CorpusChunk, ScoutPartition
    result = []
    for pd in data:
        chunks = [
            CorpusChunk(chunk_id=cd["chunk_id"], text=cd["text"],
                        source_tag=cd["source_tag"])
            for cd in pd["chunks"]
        ]
        result.append(ScoutPartition(scout_index=pd["scout_index"], chunks=chunks))
    return result


def _build_corpus_partitions(task_type: str, user_prompt: str, corpus_mode: str):
    """Same retrieval/partition logic as run_pipeline(), factored out."""
    if task_type == "coding":
        return [], partition_for_scouts([], NUM_SCOUTS)
    if corpus_mode == "placeholder":
        print(
            "[retrieval] WARNING: falling back to engineered placeholder corpus "
            "— partition diversity numbers from this run are not evidence"
        )
        corpus_text = trivial_corpus_from_thesis(user_prompt)
        chunks = chunk_corpus(corpus_text, source_tag="placeholder_corpus")
        return chunks, partition_for_scouts(chunks, NUM_SCOUTS)
    # real retrieval
    from core.retrieval import CachedRetriever, CompositeRetriever
    retrieved_chunks = CachedRetriever(CompositeRetriever()).retrieve(user_prompt)
    if retrieved_chunks:
        combined_text = "\n\n".join(c.text for c in retrieved_chunks)
        chunks = chunk_corpus(combined_text, source_tag="retrieved_corpus")
        for i, orig in enumerate(retrieved_chunks):
            for ch in chunks:
                if ch.source_tag == "retrieved_corpus":
                    ch.source_tag = orig.source_tag
                    break
    else:
        corpus_text = trivial_corpus_from_thesis(user_prompt)
        chunks = chunk_corpus(corpus_text, source_tag="placeholder_corpus")
    return chunks, partition_for_scouts(chunks, NUM_SCOUTS)


async def run_phase_isolated(
    task_type: str, user_prompt: str, run_dir: Path, phase: str, round_num: int,
    resume_dir: Optional[Path] = None,
    corpus_mode: str = "real",
    ignore_kb: bool = False,
    cloud_provider: str = "none",
) -> None:
    """Execute ONE phase of ONE round and persist a checkpoint to run_dir.

    The store_state.json + round_logs_partial.json + partitions.json files
    in run_dir together carry all state between subprocess invocations.
    """
    if phase not in VALID_PHASES:
        raise SystemExit(f"unknown phase {phase!r}; valid: {VALID_PHASES}")
    run_dir.mkdir(parents=True, exist_ok=True)
    task_prompt = build_task_prompt(task_type, user_prompt)

    active_roles = ROLES_FOR_TASK.get(task_type, {"critic", "hater", "validator"})
    n_critics = NUM_CRITICS if "critic" in active_roles else 0
    n_haters = NUM_HATERS if "hater" in active_roles else 0
    n_validators = NUM_VALIDATORS if "validator" in active_roles else 0

    # ------ load prior checkpoint state (signals, partitions, round_logs) -----
    store = SignalStore()
    state_path = (resume_dir or run_dir) / "store_state.json"
    if state_path.exists():
        store.load_state(state_path)
        print(f"[phase] resumed store from {state_path}: {store.stats()}")
    elif phase != "scouts" or round_num > 1:
        print(f"[phase] WARNING: no store_state.json at {state_path} but "
              f"phase={phase} round={round_num} expects prior state")

    partitions_path = (resume_dir or run_dir) / "partitions.json"
    chunks: list = []
    partitions: list = []
    if partitions_path.exists():
        pdata = json.loads(partitions_path.read_text(encoding="utf-8"))
        partitions = _partitions_from_jsonable(pdata.get("partitions", []))
        chunks = [c for p in partitions for c in p.chunks]
        print(f"[phase] loaded partitions: {[len(p.chunks) for p in partitions]}")
    else:
        chunks, partitions = _build_corpus_partitions(task_type, user_prompt, corpus_mode)
        (run_dir / "partitions.json").write_text(
            json.dumps({"partitions": _partitions_to_jsonable(partitions)}, indent=2),
            encoding="utf-8",
        )
        print(f"[phase] built and saved partitions: {[len(p.chunks) for p in partitions]}")

    round_logs_path = (resume_dir or run_dir) / "round_logs_partial.json"
    if round_logs_path.exists():
        round_logs = json.loads(round_logs_path.read_text(encoding="utf-8"))
    else:
        round_logs = []

    manifest_path = (resume_dir or run_dir) / "phase_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"task_type": task_type, "user_prompt": user_prompt,
                    "completed": []}

    # ------ knowledge base ----------------------------------------------------
    from core.knowledge_base import _topic_hash as _kb_topic_hash
    current_topic_hash = _kb_topic_hash(user_prompt)
    kb = KnowledgeBase()
    if ignore_kb:
        print("[kb] disabled (--use-kb to enable)")
    else:
        kb.load()
        n_priors = (len(kb.prior_consensus(current_topic_hash))
                    + len(kb.prior_rejections(current_topic_hash)))
        print(f"[kb] loaded {n_priors} priors — set --no-kb to disable")

    # ------ LLM (router for heterogeneous; single model otherwise) -----------
    if config.USE_HETEROGENEOUS:
        router = make_router(config._TIER)
        print(f"[phase] heterogeneous routing enabled "
              f"(router={type(router).__name__}, tier={config._TIER})")
        llm = None
    else:
        router = None
        llm = make_llm()
        print(f"[phase] llm backend: {llm.name}")

    role_cls = get_role_classes(task_type)
    ScoutClass = role_cls["scout"]
    DeveloperClass = role_cls["developer"]
    CriticClass = role_cls["critic"]
    HaterClass = role_cls["hater"]
    ValidatorClass = role_cls["validator"]

    # Ensure the partial round_logs entry for this round exists.
    def _round_entry(rn: int) -> dict:
        while len(round_logs) < rn:
            round_logs.append({
                "round": len(round_logs) + 1, "agents": [],
                "partition_deposits_this_round": {},
            })
        return round_logs[rn - 1]

    phase_start = time.time()

    # ------ dispatch ----------------------------------------------------------

    async def _run_agents(agents):
        """Run agents (router-grouped if heterogeneous, else direct gather)."""
        if router is None:
            return await asyncio.gather(
                *(a.run(store, ITERATIONS_PER_ROUND) for a in agents)
            )
        groups = router.group_agents(agents)
        stats_by_id: dict = {}
        for model_path, agent_subset in groups.items():
            async with router.acquire(model_path) as model_llm:
                for a in agent_subset:
                    a.llm = model_llm
                stats_subset = await asyncio.gather(
                    *(a.run(store, ITERATIONS_PER_ROUND) for a in agent_subset)
                )
                for a, s in zip(agent_subset, stats_subset):
                    stats_by_id[a.agent_id] = s
        return [stats_by_id[a.agent_id] for a in agents]

    if phase == "scouts":
        if task_type == "coding":
            scouts = [ScoutClass(agent_id=f"scout_R{round_num}_{i}", llm=llm,
                                 task_prompt=task_prompt)
                      for i in range(NUM_SCOUTS)]
        else:
            scouts = [ScoutClass(
                        agent_id=f"scout_R{round_num}_{i}", llm=llm,
                        config=ScoutConfig(
                        task_prompt=task_prompt,
                        partition=partitions[i],
                        # Phase 2C: agentic search by default; --corpus=placeholder
                        # disables it so the regression path still exercises the
                        # contiguous-partition diversity engine.
                        use_search=(corpus_mode != "placeholder"),
                        user_prompt=user_prompt,
                    ),
                      ) for i in range(NUM_SCOUTS)]
        stats = await _run_agents(scouts)
        entry = _round_entry(round_num)
        for a, s in zip(scouts, stats):
            entry["agents"].append({
                "agent_id": a.agent_id, "role": a.ROLE,
                "deposits": s.deposits, "rejected_dup": s.rejected_dup,
                "iterations": s.iterations, "novelty_per_iter": s.novelty_per_iter,
            })
            if s.deposits > 0:
                parts = a.agent_id.split("_")
                tag = f"partition_{parts[2]}" if len(parts) >= 3 else "partition_unknown"
                pd = entry.setdefault("partition_deposits_this_round", {})
                pd[tag] = pd.get(tag, 0) + s.deposits

    elif phase == "validators":
        validators = []
        for i in range(n_validators):
            name, strat = strategy_for_validator(i)
            validators.append(ValidatorClass(
                agent_id=f"validator_R{round_num}_{i}_{name}",
                llm=llm, strategy=strat, strategy_name=name,
                task_prompt=task_prompt, cloud_provider=cloud_provider,
            ))
        stats = await _run_agents(validators)
        entry = _round_entry(round_num)
        for a, s in zip(validators, stats):
            entry["agents"].append({
                "agent_id": a.agent_id, "role": a.ROLE,
                "deposits": s.deposits, "rejected_dup": s.rejected_dup,
                "iterations": s.iterations, "novelty_per_iter": s.novelty_per_iter,
            })

    elif phase == "foragers":
        foragers = []
        for i in range(NUM_FORAGERS):
            name, strat = strategy_for_forager(i)
            foragers.append(DeveloperClass(
                agent_id=f"forager_R{round_num}_{i}_{name}",
                llm=llm, strategy=strat, strategy_name=name, task_prompt=task_prompt,
            ))
        stats = await _run_agents(foragers)
        entry = _round_entry(round_num)
        for a, s in zip(foragers, stats):
            entry["agents"].append({
                "agent_id": a.agent_id, "role": a.ROLE,
                "deposits": s.deposits, "rejected_dup": s.rejected_dup,
                "iterations": s.iterations, "novelty_per_iter": s.novelty_per_iter,
            })

    elif phase == "critics":
        critics = []
        for i in range(n_critics):
            name, strat = strategy_for_critic(i)
            critics.append(CriticClass(
                agent_id=f"critic_R{round_num}_{i}_{name}",
                llm=llm, strategy=strat, strategy_name=name, task_prompt=task_prompt,
            ))
        stats = await _run_agents(critics)
        entry = _round_entry(round_num)
        for a, s in zip(critics, stats):
            entry["agents"].append({
                "agent_id": a.agent_id, "role": a.ROLE,
                "deposits": s.deposits, "rejected_dup": s.rejected_dup,
                "iterations": s.iterations, "novelty_per_iter": s.novelty_per_iter,
            })

    elif phase == "haters":
        haters = [HaterClass(agent_id=f"hater_R{round_num}_{i}", llm=llm,
                             task_prompt=task_prompt, task_type=task_type)
                  for i in range(n_haters)]
        stats = await _run_agents(haters)
        entry = _round_entry(round_num)
        for a, s in zip(haters, stats):
            entry["agents"].append({
                "agent_id": a.agent_id, "role": a.ROLE,
                "deposits": s.deposits, "rejected_dup": s.rejected_dup,
                "iterations": s.iterations, "novelty_per_iter": s.novelty_per_iter,
            })

    elif phase == "decay":
        store.decay_all()
        pruned = store.prune_weak()
        entry = _round_entry(round_num)
        entry["pruned"] = pruned
        # Finalize this round's diversity metrics from the deposits we tracked.
        # In phase-isolated mode we don't have AgentContextRecord objects here;
        # use the deposits as approximated by the per-agent stats logged above.
        # For richer metrics, the synth phase recomputes from signals.json.
        entry["stats"] = store.stats()
        entry["elapsed_s_round"] = round(time.time() - phase_start, 2)

    elif phase == "synth":
        SynthesizerClass = get_role_classes(task_type).get("synthesizer", Synthesizer)
        # Persist signals.json first so a synth crash doesn't lose state.
        signal_dump = [
            {"id": s.id, "type": s.type, "strength": round(s.strength, 4),
             "depositor": s.depositor, "parent_id": s.parent_id,
             "visits": s.visits, "metadata": s.metadata, "content": s.content}
            for s in store.all()
        ]
        (run_dir / "signals.json").write_text(
            json.dumps(signal_dump, indent=2), encoding="utf-8"
        )
        (run_dir / "round_log.json").write_text(
            json.dumps(round_logs, indent=2), encoding="utf-8"
        )
        try:
            if router is not None:
                async with router.acquire(router.model_for_role("synthesizer")) as synth_llm:
                    synth = SynthesizerClass(synth_llm, task_prompt)
                    final_answer, citations, lineage_dot = await synth.synthesize(
                        store, has_validators=(n_validators > 0),
                        prior_rejections=kb.prior_rejections(current_topic_hash) if not ignore_kb else None,
                        prior_consensus=kb.prior_consensus(current_topic_hash) if not ignore_kb else None,
                        output_dir=run_dir,
                    )
            else:
                synth = SynthesizerClass(llm, task_prompt)
                final_answer, citations, lineage_dot = await synth.synthesize(
                    store, has_validators=(n_validators > 0),
                    prior_rejections=kb.prior_rejections(current_topic_hash) if not ignore_kb else None,
                    prior_consensus=kb.prior_consensus(current_topic_hash) if not ignore_kb else None,
                    output_dir=run_dir,
                )
        except Exception as exc:
            print(f"[phase] synthesis failed ({type(exc).__name__}: {exc})")
            final_answer = f"(synthesis failed: {type(exc).__name__}: {exc})"
            citations = {}
            lineage_dot = ""
        if router is not None:
            await router.teardown()

        write_answer_files(run_dir, final_answer)
        if citations:
            (run_dir / "citations.json").write_text(
                json.dumps(citations, indent=2), encoding="utf-8"
            )
        if lineage_dot:
            (run_dir / "lineage.dot").write_text(lineage_dot, encoding="utf-8")
        backend_label = "heterogeneous" if router is not None else (llm.name if llm else "unknown")
        model_assignment = router.manifest() if router is not None else {"all": llm.name if llm else "unknown"}
        (run_dir / "run_meta.json").write_text(json.dumps({
            "task_type": task_type, "user_prompt": user_prompt,
            "llm_backend": backend_label,
            "is_mock": USE_MOCK_LLM,
            "timestamp": os.environ.get("SWARM_RUN_TIMESTAMP", datetime.now().isoformat()),
            "colab_tier": config._TIER,
            "vllm_concurrency": int(config.LLM_CONCURRENCY),
            "vllm_dtype": str(config.VLLM_DTYPE),
            "population_scaled_for": config._TIER if config._TIER else "laptop",
            "model_loaded_once": (router is None and not config.USE_HETEROGENEOUS),
            "model_name": config.MODEL_NAME,
            "active_roles": sorted(active_roles),
            "population": {"scouts": NUM_SCOUTS, "foragers": NUM_FORAGERS,
                            "critics": n_critics, "haters": n_haters,
                            "validators": n_validators},
            "model_assignment": model_assignment,
            "execution_mode": "phase_isolated",
        }, indent=2), encoding="utf-8")
        # Best-effort KB save
        if not ignore_kb:
            try:
                from core.projection import build_projection
                final_projection = build_projection(store, has_validators=(n_validators > 0))
                kb.save(final_projection, store, {
                    "task_type": task_type, "user_prompt": user_prompt,
                    "timestamp": datetime.now().isoformat(),
                }, output_dir=run_dir)
            except Exception as exc:
                print(f"[phase] kb save failed ({type(exc).__name__}: {exc})")
        print("=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(final_answer)
        print("=" * 60)

    # ------ persist checkpoint ----------------------------------------------
    store.save_state(run_dir / "store_state.json")
    (run_dir / "round_logs_partial.json").write_text(
        json.dumps(round_logs, indent=2), encoding="utf-8"
    )
    manifest.setdefault("completed", []).append({
        "phase": phase, "round": (round_num if phase != "synth" else None),
        "completed_at": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - phase_start, 2),
        "store_size_after": len(store.all()),
    })
    (run_dir / "phase_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[phase] {phase} R{round_num if phase != 'synth' else '-'} "
          f"completed in {time.time() - phase_start:.1f}s; "
          f"store={len(store.all())} signals")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    # Default flipped: KB is OFF unless --use-kb is passed. --ignore-kb is
    # preserved as a no-op alias for backwards compat with old scripts.
    use_kb = "--use-kb" in args
    ignore_kb = not use_kb
    reset_kb  = "--reset-kb"  in args
    prune_kb_before: Optional[str] = None
    _prune_flags = [a for a in args if a.startswith("--prune-kb-before=")]
    if _prune_flags:
        prune_kb_before = _prune_flags[-1].split("=", 1)[1]
    args = [a for a in args if not a.startswith("--prune-kb-before=")]
    show_partition_overlap = "--show-partition-overlap" in args
    heterogeneous = "--heterogeneous" in args
    # Continuous worker pool is the default execution mode (Phase 1 of the
    # emergent-swarm rewrite). Opt back to the round/phase scheduler with
    # --legacy-rounds for A/B comparison or to repro old artifacts.
    legacy_rounds = "--legacy-rounds" in args
    small = "--small" in args
    # --synth-verbose: restore the old combined answer.txt (Section 1 + all field
    # telemetry). Default is the clean reader answer + a separate diagnostics.md.
    # Sets the env var the write helper (core/clean_answer.py) reads, so it works
    # with or without the explicit flag.
    if "--synth-verbose" in args:
        os.environ["SWARM_SYNTH_VERBOSE"] = "1"
    args = [a for a in args if a not in (
        "--use-kb", "--ignore-kb", "--reset-kb",
        "--show-partition-overlap", "--heterogeneous", "--legacy-rounds",
        "--small", "--synth-verbose",
    )]
    # --workers=N for the continuous pool size
    workers = _CONTINUOUS_DEFAULT_WORKERS
    worker_flags = [a for a in args if a.startswith("--workers=")]
    if worker_flags:
        try:
            workers = int(worker_flags[-1].split("=", 1)[1])
        except ValueError:
            print(f"[pipeline] --workers expects an integer; got {worker_flags[-1]!r}")
            sys.exit(1)
    args = [a for a in args if not a.startswith("--workers=")]

    # --depth / --depth=N was already consumed into SWARM_RENDER_K at module
    # top (before config import). Strip both forms here so the value token in
    # the space form isn't mistaken for the task/prompt positional.
    _depth_clean: list = []
    _skip_next = False
    for _a in args:
        if _skip_next:
            _skip_next = False
            continue
        if _a == "--depth":
            _skip_next = True
            continue
        if _a.startswith("--depth="):
            continue
        _depth_clean.append(_a)
    args = _depth_clean

    # --bundle=NAME: pick a MODEL_BUNDLES entry, overriding TASK_TO_BUNDLE.
    # Only meaningful with the default continuous_pool execution mode.
    bundle: Optional[str] = None
    bundle_flags = [a for a in args if a.startswith("--bundle=")]
    if bundle_flags:
        bundle = bundle_flags[-1].split("=", 1)[1]
        from core.config import MODEL_BUNDLES as _BUNDLES
        if bundle not in _BUNDLES:
            print(f"[pipeline] --bundle={bundle!r} is not in MODEL_BUNDLES "
                  f"({sorted(_BUNDLES)})")
            sys.exit(1)
        # Setting the flag turns the bundle path on even if env was unset.
        config.USE_MODEL_BUNDLES = True
    args = [a for a in args if not a.startswith("--bundle=")]

    if heterogeneous:
        # Tier-gated. T4 cannot run heterogeneous: VRAM (16 GB) only fits one
        # 7B AWQ with usable KV cache. Per-role GGUF specialists were never the
        # plan on Colab anyway — on L4/A100 the path is vLLM multi-LoRA (see
        # core/llm_router.py:LoRAHeterogeneousRouter), which loads the base
        # once and selects adapters per-request. Laptop still uses the legacy
        # GGUF HeterogeneousRouter.
        if config._TIER == "t4":
            print("[pipeline] WARNING: --heterogeneous ignored on T4 "
                  "(VRAM-bound to a single 7B AWQ). vLLM single-model batching "
                  "is the intended path here.")
        else:
            # Mutate module attribute so run_pipeline's `config.USE_HETEROGENEOUS`
            # read picks up the flag (legacy-rounds path).
            config.USE_HETEROGENEOUS = True
            # Continuous pool (default): --heterogeneous maps to the bundle
            # path. MultiEngineRouter loads all engines resident; USE_HETEROGENEOUS
            # is only read by the legacy run_pipeline() scheduler. Without this,
            # run_continuous_pipeline() falls through to make_llm() and loads
            # a single model, silently ignoring the flag.
            if not legacy_rounds:
                config.USE_MODEL_BUNDLES = True
                print(
                    "[pipeline] --heterogeneous: continuous pool → bundle path activated "
                    "(auto-selects from TASK_TO_BUNDLE; override with --bundle=NAME)"
                )

    # --corpus={real,placeholder}
    corpus_mode = "real"
    corpus_flags = [a for a in args if a.startswith("--corpus=")]
    if corpus_flags:
        val = corpus_flags[-1].split("=", 1)[1]
        if val not in ("real", "placeholder"):
            print(f"[pipeline] unknown --corpus value {val!r}; use 'real' or 'placeholder'")
            sys.exit(1)
        corpus_mode = val
    args = [a for a in args if not a.startswith("--corpus=")]

    # --mode={stigmergic,baseline}
    run_mode = "stigmergic"
    mode_flags = [a for a in args if a.startswith("--mode=")]
    if mode_flags:
        val = mode_flags[-1].split("=", 1)[1]
        if val not in ("stigmergic", "baseline"):
            print(f"[pipeline] unknown --mode value {val!r}; use 'stigmergic' or 'baseline'")
            sys.exit(1)
        run_mode = val
    args = [a for a in args if not a.startswith("--mode=")]

    # --cloud-validator={none,anthropic,gemini}
    cloud_provider = "none"
    cv_flags = [a for a in args if a.startswith("--cloud-validator=")]
    if cv_flags:
        val = cv_flags[-1].split("=", 1)[1]
        if val not in ("none", "anthropic", "gemini"):
            print(f"[pipeline] unknown --cloud-validator value {val!r}; use 'none', 'anthropic', or 'gemini'")
            sys.exit(1)
        cloud_provider = val
    args = [a for a in args if not a.startswith("--cloud-validator=")]

    # --isolated: marker flag declaring this invocation is part of a
    # phase-isolated execution sequence (split away from in-process
    # parallelism). When set alongside --phase, the pipeline runs ONE
    # phase and exits; the orchestrator (tools/run_isolated.py) chains
    # invocations. When set without --phase, behaves like --phase=all
    # but records execution_mode=phase_isolated in run_meta.json.
    isolated = "--isolated" in args
    args = [a for a in args if a != "--isolated"]

    # --phase={scouts,validators,foragers,critics,haters,decay,synth,all}
    phase = "all"
    phase_flags = [a for a in args if a.startswith("--phase=")]
    if phase_flags:
        val = phase_flags[-1].split("=", 1)[1]
        if val not in (*VALID_PHASES, "all"):
            print(f"[pipeline] unknown --phase value {val!r}; use one of "
                  f"{(*VALID_PHASES, 'all')}")
            sys.exit(1)
        phase = val
        isolated = True  # any explicit --phase implies isolation
    args = [a for a in args if not a.startswith("--phase=")]

    # --round=N
    round_num = 1
    round_flags = [a for a in args if a.startswith("--round=")]
    if round_flags:
        try:
            round_num = int(round_flags[-1].split("=", 1)[1])
        except ValueError:
            print(f"[pipeline] --round expects an integer; got {round_flags[-1]!r}")
            sys.exit(1)
    args = [a for a in args if not a.startswith("--round=")]

    # --run-id=NAME: override default output dir name (used by orchestrator).
    run_id: Optional[str] = None
    run_id_flags = [a for a in args if a.startswith("--run-id=")]
    if run_id_flags:
        run_id = run_id_flags[-1].split("=", 1)[1]
    args = [a for a in args if not a.startswith("--run-id=")]

    # --resume=PATH: load store_state.json + partitions.json from this dir
    # before running the requested phase.
    resume_path: Optional[Path] = None
    resume_flags = [a for a in args if a.startswith("--resume=")]
    if resume_flags:
        resume_path = Path(resume_flags[-1].split("=", 1)[1])
    args = [a for a in args if not a.startswith("--resume=")]

    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    task_type = args[0]
    user_prompt = " ".join(args[1:])

    if small:
        # Dispatch on the EXECUTION PATH, not on --heterogeneous.
        # --heterogeneous is only meaningful on the legacy-rounds + isolated
        # path (GGUF sequential per-phase loading). The continuous pool uses
        # the bundle system and --heterogeneous has no effect there.
        _in_continuous_pool = (
            not legacy_rounds
            and run_mode != "baseline"
            and not (isolated and phase != "all")
        )
        if _in_continuous_pool:
            # Continuous pool (default): select the task-matched small vLLM
            # bundle. --heterogeneous is silently ignored here — it only
            # applies to the GGUF sequential path.
            _small_bundle = config.TASK_TO_BUNDLE_SMALL.get(task_type, "small")
            if bundle is None:
                bundle = _small_bundle
            config.USE_MODEL_BUNDLES = True
            config.HETEROGENEOUS_CONFIG_FILE = "heterogeneous_small.json"
            print(f"[pipeline] --small: continuous pool using bundle {bundle!r} "
                  f"(Qwen2.5-3B-Instruct + Qwen2.5-1.5B-Instruct)")
            if heterogeneous:
                print("[pipeline] --small: note — --heterogeneous ignored in "
                      "continuous-pool mode (use --legacy-rounds for GGUF "
                      "per-role routing)")
        elif heterogeneous:
            # Legacy-rounds / isolated + --heterogeneous: GGUF sequential
            # per-phase loading with the small model assignment.
            config.HETEROGENEOUS_CONFIG_FILE = "heterogeneous_small.json"
            print("[pipeline] --small --heterogeneous: "
                  "using configs/heterogeneous_small.json "
                  "(Qwen2.5-3B + Qwen2.5-1.5B)")
        else:
            # Legacy-rounds single-model or baseline.
            config.MODEL_NAME = config.SMALL_MODEL
            print(f"[pipeline] --small: model overridden to {config.MODEL_NAME!r}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # P0.1: mock runs go in a separate directory so they cannot be mistaken
    # for empirical artifacts.
    outputs_root = "outputs_mock" if USE_MOCK_LLM else "outputs"
    mode_suffix = "_baseline" if run_mode == "baseline" else ""
    # SWARM_OUTPUTS_BASE_DIR redirects the outputs/ and outputs_mock/ trees
    # to a custom location (Google Drive, scratch dir, etc.). Mock/real
    # subdirectory split is preserved per P0.1.
    _base_env = os.environ.get("SWARM_OUTPUTS_BASE_DIR")
    if _base_env:
        output_base = Path(_base_env) / outputs_root
    else:
        output_base = Path(__file__).parent / outputs_root
    if run_id is not None:
        output_dir = output_base / run_id
    else:
        output_dir = output_base / f"{task_type}_{timestamp}{mode_suffix}"

    if ignore_kb:
        print("[pipeline] --ignore-kb: knowledge base disabled for this run")
    if reset_kb:
        print("[pipeline] --reset-kb: quarantining existing knowledge base before run")
    if corpus_mode == "placeholder":
        # Loud, multi-line so it's obvious in Colab logs that this run's
        # corpus is 4 engineered chunks and downstream diversity / cluster
        # numbers are NOT empirical evidence. The --corpus flag is for
        # development only.
        print("=" * 72)
        print("[pipeline] --corpus=placeholder is DEVELOPMENT-ONLY.")
        print("[pipeline]   The engineered corpus produces ~4 chunks regardless of prompt;")
        print("[pipeline]   scouts at NUM_SCOUTS>=6 will get EMPTY partitions, foragers")
        print("[pipeline]   will heavily dedup-reject, and clusters will collapse to")
        print("[pipeline]   weakly_supported. Drop --corpus=placeholder for a real run.")
        print("=" * 72)
    if run_mode == "baseline":
        print("[pipeline] --mode=baseline: running non-stigmergic independent-agent baseline")
    if isolated:
        print(f"[pipeline] --isolated: phase-isolated execution mode "
              f"(phase={phase}, round={round_num}, run_dir={output_dir.name})")

    # Phase-isolated execution: one phase, then exit.
    if isolated and phase != "all":
        if run_mode == "baseline":
            print("[pipeline] --isolated + --phase is incompatible with --mode=baseline")
            sys.exit(1)
        asyncio.run(run_phase_isolated(
            task_type, user_prompt, output_dir, phase, round_num,
            resume_dir=resume_path,
            corpus_mode=corpus_mode,
            ignore_kb=ignore_kb,
            cloud_provider=cloud_provider,
        ))
        return

    if run_mode == "baseline":
        _run_baseline(task_type, user_prompt, output_dir, corpus_mode=corpus_mode)
    elif legacy_rounds:
        print("[pipeline] --legacy-rounds: using round/phase scheduler "
              "(continuous pool is the default).")
        asyncio.run(run_pipeline(
            task_type, user_prompt, output_dir,
            ignore_kb=ignore_kb,
            reset_kb=reset_kb,
            corpus_mode=corpus_mode,
            show_partition_overlap=show_partition_overlap,
            cloud_provider=cloud_provider,
            prune_kb_before=prune_kb_before,
        ))
    else:
        asyncio.run(run_continuous_pipeline(
            task_type, user_prompt, output_dir,
            ignore_kb=ignore_kb,
            reset_kb=reset_kb,
            n_workers=workers,
            cloud_provider=cloud_provider,
            bundle=bundle,
            prune_kb_before=prune_kb_before,
        ))


def _run_baseline(
    task_type: str,
    user_prompt: str,
    output_dir: Path,
    corpus_mode: str = "real",
) -> None:
    """Entry point for --mode=baseline: runs BaselineCoordinator and writes outputs."""
    from core.baseline import BaselineCoordinator
    from core.config import NUM_SCOUTS, NUM_FORAGERS, USE_MOCK_LLM

    task_prompt = build_task_prompt(task_type, user_prompt)
    llm = make_llm()

    # Corpus — same logic as stigmergic path
    corpus_text = ""
    if task_type != "coding":
        if corpus_mode == "placeholder":
            corpus_text = trivial_corpus_from_thesis(user_prompt)
        else:
            try:
                from core.retrieval import CachedRetriever, CompositeRetriever
                chunks = CachedRetriever(CompositeRetriever()).retrieve(user_prompt)
                corpus_text = "\n\n".join(c.text for c in chunks)
            except Exception as exc:
                print(f"[baseline] retrieval failed ({exc}); using placeholder corpus")
                corpus_text = trivial_corpus_from_thesis(user_prompt)

    coordinator = BaselineCoordinator(n_agents=NUM_SCOUTS + NUM_FORAGERS)
    result = asyncio.run(coordinator.run(
        task_type=task_type,
        user_prompt=user_prompt,
        task_prompt=task_prompt,
        llm=llm,
        corpus_text=corpus_text,
        output_dir=output_dir,
        is_mock=USE_MOCK_LLM,
    ))

    print("\n[baseline] summary.json:")
    print(json.dumps(result.summary_dict(), indent=2))
    print(f"\n[baseline] outputs: {output_dir}")
    print("=" * 60)
    print("BASELINE FINAL ANSWER")
    print("=" * 60)
    print(result.final_answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
