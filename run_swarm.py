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

sys.path.insert(0, str(Path(__file__).parent))

from core import config
from core.config import (
    NUM_SCOUTS, NUM_FORAGERS, NUM_CRITICS, NUM_HATERS, NUM_VALIDATORS,
    NUM_ROUNDS, ITERATIONS_PER_ROUND, USE_MOCK_LLM,
)
from core.signal_store import SignalStore
from core.intake import (
    chunk_corpus, partition_for_scouts, trivial_corpus_from_thesis,
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
from core.llm_router import HeterogeneousRouter
from core.knowledge_base import KnowledgeBase

from agents.scout import Scout, ScoutConfig
from agents.developer import Developer
from agents.forager import Forager  # backward-compat alias for Developer
from agents.critic import Critic
from agents.hater import Hater
from agents.validator import Validator
from agents.synthesizer import Synthesizer
from core.role_registry import get_role_classes


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
    "creative":        {"critic"},                       # no hater, no validator
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


def _wrap_llm_with_progress(llm, progress):
    """Wrap llm.generate so each call ticks the progress bar.

    Idempotent: re-wrapping the same llm (e.g. when the router reuses a
    cached model across phases) returns it untouched so the progress bar
    doesn't double-count.
    """
    if getattr(llm, "_progress_wrapped", False):
        return llm
    original = llm.generate

    async def generate(prompt, role="agent", max_tokens=120, temperature=0.7):
        result = await original(prompt, role=role, max_tokens=max_tokens, temperature=temperature)
        try:
            progress.update(1)
        except Exception:
            pass
        return result

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
                       cloud_provider: str = "none") -> dict:
    task_prompt = build_task_prompt(task_type, user_prompt)
    if config.USE_HETEROGENEOUS:
        router = HeterogeneousRouter()
        print(f"[pipeline] heterogeneous routing enabled.")
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

    # Knowledge base (optional — skip with --ignore-kb / reset with --reset-kb)
    from core.knowledge_base import _topic_hash as _kb_topic_hash
    current_topic_hash = _kb_topic_hash(user_prompt)
    kb = KnowledgeBase()
    if reset_kb:
        _reset_kb(kb.kb_dir)
    if not ignore_kb:
        kb.load()
        if not kb.is_empty():
            print(f"[pipeline] knowledge base loaded: "
                  f"{len(kb.prior_consensus(current_topic_hash))} prior survivors (topic-filtered), "
                  f"{len(kb.prior_rejections(current_topic_hash))} prior rejections (topic-filtered)")

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
        # Real retrieval: Wikipedia → Web → placeholder fallback
        from core.retrieval import CachedRetriever, CompositeRetriever
        retriever = CachedRetriever(CompositeRetriever())
        retrieved_chunks = retriever.retrieve(user_prompt)
        if retrieved_chunks:
            # Reconstruct full text and re-chunk with standard CHUNK_WORDS
            combined_text = "\n\n".join(c.text for c in retrieved_chunks)
            # Preserve source tags by re-tagging chunks from the first source
            chunks = chunk_corpus(combined_text, source_tag="retrieved_corpus")
            # Overlay source tags from the original retrieval chunks
            for i, orig in enumerate(retrieved_chunks):
                # Replace source_tag on chunks that came from this retrieved piece
                for ch in chunks:
                    if ch.source_tag == "retrieved_corpus":
                        ch.source_tag = orig.source_tag
                        break
        else:
            # CompositeRetriever already printed the warning
            corpus_text = trivial_corpus_from_thesis(user_prompt)
            chunks = chunk_corpus(corpus_text, source_tag="placeholder_corpus")
        partitions = partition_for_scouts(chunks, NUM_SCOUTS)

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
            scouts = [
                ScoutClass(
                    agent_id=f"scout_R{round_num}_{i}",
                    llm=llm,
                    config=ScoutConfig(task_prompt=task_prompt, partition=partitions[i]),
                )
                for i in range(NUM_SCOUTS)
            ]
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
            HaterClass(agent_id=f"hater_R{round_num}_{i}", llm=llm, task_prompt=task_prompt)
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
    (output_dir / "run_meta.json").write_text(json.dumps({
        "task_type": task_type,
        "user_prompt": user_prompt,
        "llm_backend": backend_label,
        "is_mock": USE_MOCK_LLM,
        "timestamp": datetime.now().isoformat(),
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
                )
        else:
            synth = SynthesizerClass(llm, task_prompt)
            final_answer, citations, lineage_dot = await synth.synthesize(
                store,
                has_validators=(n_validators > 0),
                prior_rejections=kb.prior_rejections(current_topic_hash) if not ignore_kb else None,
                prior_consensus=kb.prior_consensus(current_topic_hash) if not ignore_kb else None,
                output_dir=output_dir,
            )
    except Exception as exc:
        print(f"[pipeline] synthesis failed ({type(exc).__name__}: {exc})")
        print(f"[pipeline] signals.json is still on disk — run synthesize.py against it.")
        final_answer = f"(synthesis failed: {type(exc).__name__}: {exc})"
        citations = {}
        lineage_dot = ""

    if router is not None:
        await router.teardown()

    (output_dir / "answer.txt").write_text(final_answer, encoding="utf-8")
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
            final_projection = build_projection(store, has_validators=(n_validators > 0))
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

    # Grab audit flag count from the last written audit (best-effort).
    try:
        audit_data = json.loads((output_dir / "renderer_audit.json").read_text(encoding="utf-8"))
        audit_flags = audit_data.get("total_flags", 0)
    except Exception:
        audit_flags = -1

    # Final projection for cluster counts.
    try:
        from core.projection import build_projection as _bp
        _final_proj = _bp(store, has_validators=(n_validators > 0))
        ver_scores = [cp.verification_score for cp in _final_proj.surviving + _final_proj.contested]
        summary = {
            "task_type": task_type,
            "user_prompt": user_prompt,
            "llm_backend": backend_label,
            "total_llm_calls": total_llm_calls,
            "total_elapsed_s": round(total_elapsed, 1),
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
            "audit_flags": audit_flags,
            "kb_diff": kb_diff,
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
    if not ignore_kb:
        kb.load()

    # ------ LLM (router for heterogeneous; single model otherwise) -----------
    if config.USE_HETEROGENEOUS:
        router = HeterogeneousRouter()
        print(f"[phase] heterogeneous routing enabled (one model will be loaded)")
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
                        config=ScoutConfig(task_prompt=task_prompt, partition=partitions[i]),
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
                             task_prompt=task_prompt)
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

        (run_dir / "answer.txt").write_text(final_answer, encoding="utf-8")
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
            "timestamp": datetime.now().isoformat(),
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
    ignore_kb = "--ignore-kb" in args
    reset_kb  = "--reset-kb"  in args
    show_partition_overlap = "--show-partition-overlap" in args
    heterogeneous = "--heterogeneous" in args
    args = [a for a in args if a not in (
        "--ignore-kb", "--reset-kb", "--show-partition-overlap", "--heterogeneous",
    )]
    if heterogeneous:
        # Mutate module attribute so run_pipeline's `config.USE_HETEROGENEOUS`
        # read picks up the flag.
        config.USE_HETEROGENEOUS = True

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # P0.1: mock runs go in a separate directory so they cannot be mistaken
    # for empirical artifacts.
    outputs_root = "outputs_mock" if USE_MOCK_LLM else "outputs"
    mode_suffix = "_baseline" if run_mode == "baseline" else ""
    if run_id is not None:
        output_dir = Path(__file__).parent / outputs_root / run_id
    else:
        output_dir = Path(__file__).parent / outputs_root / f"{task_type}_{timestamp}{mode_suffix}"

    if ignore_kb:
        print("[pipeline] --ignore-kb: knowledge base disabled for this run")
    if reset_kb:
        print("[pipeline] --reset-kb: quarantining existing knowledge base before run")
    if corpus_mode == "placeholder":
        print("[pipeline] --corpus=placeholder: using engineered corpus (diversity numbers invalid)")
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
    else:
        asyncio.run(run_pipeline(
            task_type, user_prompt, output_dir,
            ignore_kb=ignore_kb,
            reset_kb=reset_kb,
            corpus_mode=corpus_mode,
            show_partition_overlap=show_partition_overlap,
            cloud_provider=cloud_provider,
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
