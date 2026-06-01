"""Synthesizer — two-layer read-out from the surviving signal DAG.

Layer 1 (core/projection.py): pure-Python DAG projection. No LLM.
  Classifies clusters as surviving / contested / weakly_supported / rejected_by_field.
  surviving clusters carry an `unverified` flag when no validator reached them.
  Computes support_diversity, dissent_pressure, verification_score.

Layer 2 (this file): structured multi-call LLM renderer.
  One LLM call per surviving cluster (Section 1) and per contested/dissented
  cluster (Section 2). Deterministic assembly. No single pooled call that lets
  the model hallucinate cross-cluster content.

Output structure
----------------
  Section 1 — Position synthesis      (1 paragraph per surviving cluster)
  Section 2 — Open questions/dissent  (1 paragraph per contested or challenged)
  Section 3 — Considered and filtered (brief, deterministic — no LLM)
  Section 4 — Citations               (deterministic stamp, no LLM)

Why per-cluster calls?
  - Faithfulness: each call only sees one cluster's evidence, so cross-cluster
    content pooling is structurally impossible.
  - Failure isolation: one bad cluster render doesn't corrupt the others.
  - Parallelism: when LLM_CONCURRENCY rises, cluster calls will parallelize.

External grounding (optional)
------------------------------
  At synthesis time, a Wikipedia lookup is performed for each surviving cluster's
  representative claim. The snippet is injected as "[External context]" in the
  renderer prompt and flagged in the output so it's distinguishable from
  agent-deposited content. Never raises; degrades gracefully offline.

Faithfulness audit
------------------
  After rendering, each paragraph is checked: for every cited cluster ID [INITIAL_XXXXX],
  does the surrounding text have at least a 4-gram overlap with that cluster's
  representative content? Failures are collected and written to renderer_audit.json
  when output_dir is provided.

Returns
-------
  answer_text:   str — prose with inline citation tags
  citations:     dict — maps each signal ID to provenance metadata
  lineage_dot:   str — Graphviz DOT of the surviving DAG
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Optional

from core.signal_store import SignalStore
from core.signal_types import (
    INITIAL, SUPPORT, CRITIQUE, CRITIQUE_POSITIVE, CRITIQUE_NEGATIVE,
    OBJECTION, VERIFICATION,
)
from core.config import (
    MAX_TOKENS_SYNTHESIZER,
    RENDER_POSITION_MAX_WORDS,
    RENDER_DISSENT_MAX_WORDS,
    SAMPLING_PER_ENGINE,
    LLM_CONCURRENCY,
)
from core.projection import (
    SynthesisProjection,
    ClusterProjection,
    SynthesisPlan,
    build_projection,
    build_plan,
)
from agents.base import strip_reasoning

# Fix R — guard: primary engine must use repetition_penalty=1.15 to prevent
# the synthesizer from emitting repetitive boilerplate across cluster paragraphs.
assert SAMPLING_PER_ENGINE["primary"]["repetition_penalty"] == 1.15, (
    f"[FIX R] primary engine repetition_penalty must be 1.15; "
    f"got {SAMPLING_PER_ENGINE['primary']['repetition_penalty']}"
)

_RENDERER_TEMPERATURE = 0.3
# Content budgets for renderer prompts (chars, not tokens)
_REPRESENTATIVE_CHARS = 800
_SUPPORT_CHARS = 400
_DISSENT_CHARS = 400
_SUMMARY_CHARS = 150   # for Section 3 "filtered" entries
_EXTERNAL_CHARS = 400  # Wikipedia snippet injected into Section 1 calls

# Fallback cap when the LLM planner returns nothing usable. The planner
# normally chooses the render set itself based on a structural digest that
# carries no signal content — see _plan_synthesis below. The cap only fires
# when the planner call fails or yields zero valid cluster IDs.
_SECTION_1_FALLBACK_CAP = 6
# Hard cap on Section 3 entries (after merging tail-of-surviving, unverified,
# rejected_by_field, and weakly_supported). Was 5; raised because we now
# route more buckets through this section.
_SECTION_3_RENDER_CAP = 10
# Threshold above which the run-end summary prints a faithfulness warning.
_AUDIT_WARNING_THRESHOLD = 20

# Revision loop parameters (improvement 5.2). K=1 round of critic + revise
# after sectioned rendering. Set to 0 to disable (useful when synthesis time
# is the bottleneck; each round adds 2 LLM calls). Self-Refine literature
# shows diminishing returns past K=2.
_SYNTHESIZER_REVISION_ROUNDS = 1
_REVISION_CRITIC_MAX_TOKENS = 600
_REVISION_REVISE_MAX_TOKENS = 3000
_REVISION_TEMPERATURE = 0.2

# Decomposed edge-graph composition (improvement 5.4). Stage B composes
# the independently-rendered Section-1 paragraphs using typed inter-cluster
# edges as transition scaffolding. Adds one LLM call per sectioned synthesis
# when edges are present. Set to False to skip and use bare paragraph join.
_SYNTHESIZER_USE_EDGE_COMPOSITION = True
_EDGE_COMPOSE_MAX_TOKENS = 2000

# Best-of-N cohesive composition (improvement 5.3). Run the integration call
# N times with diverse cluster orderings and (for exploration) diverse
# temperatures; score each candidate by cluster coverage + faithfulness
# overlap − audit flags; take the argmax. Set to 0 or 1 to disable (single
# candidate). Applies only to cohesive_exploration and cohesive_optimization
# strategies, not to sectioned. Each additional candidate costs one full
# LLM call; N=3 is the budget-safe default on a 6 GB laptop GPU.
# Reduce best-of-N on laptop path (LLM_CONCURRENCY=1). With genome-enhanced
# scoring, 2 candidates is sufficient — the genome coverage signal is more
# discriminating than word overlap alone, so fewer candidates are needed to
# find the best one. On vLLM/A100 (LLM_CONCURRENCY > 1) keep the full 3.
_BEST_OF_N_COHESIVE = 2 if LLM_CONCURRENCY <= 1 else 3
# Three-element lists so the modulo cycle `_temps[_i % len(_temps)]` in the
# best-of-N loop produces distinct temperatures for all N ≤ 3 candidates.
# The first two are used when N=2 (laptop); all three when N=3 (vLLM/A100).
_BEST_OF_N_EXPLORATION_TEMPS = [0.3, 0.5, 0.7]
_BEST_OF_N_OPTIMIZATION_TEMPS = [0.15, 0.25, 0.35]

# Debate frame for `alternatives` cluster sets (improvement 5.5).
# When ≥2 surviving clusters are connected by an `alternatives` edge and
# are within _DEBATE_PRIORITY_RATIO of each other's priority score, replace
# their Section-1 rendering with a 3-round debate: Round 1 each generates a
# position; Round 2 each responds to the strongest sibling; Round 3 a judge
# identifies the unresolved empirical question. Gate on _SYNTHESIZER_USE_DEBATE
# (False by default — adds 3+ LLM calls per alternatives set; enable only
# when compute budget permits).
_SYNTHESIZER_USE_DEBATE = True   # debate frame for alternatives cluster sets
_DEBATE_PRIORITY_RATIO = 0.8     # alternatives within 20% of each other's priority
_DEBATE_ROUND_MAX_TOKENS = 800
_DEBATE_JUDGE_MAX_TOKENS = 600
_DEBATE_TEMPERATURE = 0.4

# Alternative-of-the-best artifact (improvement 5.9). For exploration tasks,
# after the primary synthesis emit a second artifact from the highest-priority
# cluster that was NOT selected by the planner (section3_only or MMR-filtered).
# Framed as "the strongest alternative direction the swarm explored but did
# not select as primary." Gated on SYNTHESIZER_EMIT_ALTERNATIVE (False by
# default — adds one full LLM call; enable when compute budget allows).
# Only fires for cohesive_exploration strategy.
SYNTHESIZER_EMIT_ALTERNATIVE = True
_ALTERNATIVE_MAX_TOKENS = 1500
_ALTERNATIVE_TEMPERATURE = 0.5

# Feature flag for robust synthesizer logic: retry prompt interpretation,
# sanitize planner output, and trim excessive render sets.
_SYNTHESIZER_USE_ROBUST_PLAN_FALLBACK = True

# Calibrated abstention gate (improvement 5.8). Fires when surviving clusters
# exist but none clear a credibility bar: max verification_score is too low,
# max support_diversity is too low, AND at least one cluster is heavily
# contested. Conservative defaults — only fire on obviously broken field states.
# Set _SYNTHESIZER_USE_CALIBRATED_ABSTENTION = False to disable while tuning.
_SYNTHESIZER_USE_CALIBRATED_ABSTENTION = True
_ABSTAIN_VER_THRESHOLD = 0.15    # max verification_score must exceed this
_ABSTAIN_DIVERSITY_THRESHOLD = 2  # max support_diversity must exceed this
_ABSTAIN_DISSENT_THRESHOLD = 1.2  # max dissent_pressure trigger

# Genome-level abstention gate (Gap #3, design doc §11).
# Fires when every surviving cluster's genome fails all three tests:
#   composite_fitness < _ABSTAIN_GENOME_FITNESS_TAU
#   knowledge_base.grounding_score == 0 (purely parametric, no retrieval)
#   sensitivity.load_bearing_atoms is empty (no atom structure)
# Only fires when genomes are populated (i.e., after build_projection with
# the genome pipeline active). Falls back to passing when genomes are absent.
_SYNTHESIZER_USE_GENOME_ABSTENTION = True
_ABSTAIN_GENOME_FITNESS_TAU = 0.20   # composite_fitness floor; below this = suspect

# Planner is a single LLM call ahead of per-cluster rendering. Its prompt
# carries STRUCTURAL metadata only (cluster IDs, support / dissent / ver
# counts, scores) — never Signal.content — so the synthesizer can decide
# what to surface without ingesting the full DAG. Per-cluster renderers
# then read the specific signals the plan selected. This satisfies the
# no-leak rule: the planner sees structure, the renderer sees content,
# neither sees other agents' reasoning chains.
_PLAN_MAX_TOKENS = 1500
_PLAN_TEMPERATURE = 0.2

# Prompt-interpreter call. Reads the user's task prompt and extracts the
# structural contract (form, regime, constraints) BEFORE any rendering, so
# downstream stages know whether they're producing a haiku, a function,
# or an argument. Decouples task category (coarse) from output shape (fine).
_INTERPRET_MAX_TOKENS = 600
_INTERPRET_TEMPERATURE = 0.1

# Final integration call for cohesive output strategies. Sees the surviving
# cluster representatives + supports as raw materials and the prompt
# contract as the shape directive. Higher temp than per-cluster (which
# rendered prose paragraphs from a fixed template) because the integration
# call is producing the final user-facing artifact and benefits from a
# little freedom — but capped so coding outputs stay deterministic enough.
_INTEGRATE_MAX_TOKENS = 2500
_INTEGRATE_TEMPERATURE_EXPLORATION = 0.7
_INTEGRATE_TEMPERATURE_OPTIMIZATION = 0.3

# Lattice resolution: governs which level of the multi-resolution lattice
# the synthesizer primarily traverses when composing a cluster's paragraph.
# "atom" = atom-level citations from VERIFICATION metadata (analysis/coding
# tasks where exact sourcing matters); "cluster" = cluster-rep only (debate/
# problem_solving where positions matter more than citations); "frame" =
# aggregate frame-level framing (creative tasks with broad scope).
# "atom+cluster" = atom references annotated per cluster (coding).
_LATTICE_RESOLUTION_BY_TASK = {
    "creative":        "frame",
    "problem_solving": "cluster",
    "debate":          "cluster",
    "analysis":        "atom",
    "coding":          "atom+cluster",
}

# Map task_type to output strategy. Tasks not listed fall back to
# "sectioned" (the original multi-section format). Per the design memo:
# debate/analysis -> sectioned (positions presented separately is the
# whole point); creative/problem_solving -> cohesive_exploration (one
# unified user-facing artifact); coding -> cohesive_optimization (one
# working implementation).
_OUTPUT_STRATEGY_BY_TASK = {
    "debate":          "sectioned",
    "analysis":        "sectioned",
    "problem_solving": "cohesive_exploration",
    "creative":        "cohesive_exploration",
    "coding":          "cohesive_optimization",
}

# Composite cluster priority score used for rendering order.
# Higher support_diversity and verification_score = more credible.
# Higher dissent_pressure = more contested = lower priority for Section 1.
def _cluster_priority(cp) -> float:
    return (cp.support_diversity * max(0.01, cp.verification_score)
            / max(0.01, cp.dissent_pressure))


def _rank_clusters(clusters: list) -> list:
    """Sort clusters by composite priority score descending."""
    return sorted(clusters, key=_cluster_priority, reverse=True)


# Minimum embedding-cosine distance between two clusters picked into the
# Section-1 render set. Pure top-N by priority routinely picked 5
# near-duplicates from the same conceptual cluster family — the post-mortem
# showed all 5 paragraphs restating "the debate over free will...". MMR-style
# diversity-aware selection: each new pick must be at least DIVERSITY_MIN_DIST
# from every already-picked cluster. 0.35 ≈ "topically distinct".
_SECTION_1_DIVERSITY_MIN_DIST = 0.35

# Hard ceiling on the number of clusters the synthesizer renders in full.
# If the planner returns more than this, the set is trimmed to the most
# diverse, highest-priority clusters.
_SECTION_1_MAX_RENDER_FULL = 6

# Semaphore size for parallel per-cluster renders (Fix S1 from §9b).
# The per-cluster renders are mutually independent under the no-leak rule, so
# they can run concurrently. Capped at min(LLM_CONCURRENCY, 4) because the
# laptop GPU (RTX 3060, 6 GB VRAM) can safely decode ≤4 streams in parallel.
# With LLM_CONCURRENCY=1 (laptop/HF path) this collapses to serial — same
# behaviour as before, zero overhead. On vLLM / A100 paths it parallelizes.
_RENDER_SEM_SIZE = max(1, min(LLM_CONCURRENCY, 4))

# Retries for the prompt interpreter and planner when the LLM returns
# malformed or non-JSON output.
_CONTRACT_PROMPT_RETRIES = 1
_PLAN_PROMPT_RETRIES = 1


def _select_diverse_clusters(
    clusters: list, store, k: int,
    min_dist: float = _SECTION_1_DIVERSITY_MIN_DIST,
) -> tuple[list, list]:
    """Pick up to `k` clusters ranked by priority but spread by embedding distance.

    Greedy MMR: take the highest-priority cluster; for each subsequent slot
    pick the next-highest-priority cluster whose representative embedding is
    >= `min_dist` from EVERY already-picked cluster's representative.
    Clusters skipped for distance reasons are returned in the second list
    (Section-3 tail).

    Falls back to a pure top-N when embeddings aren't available — same
    behavior as before, but the tail list still gets populated so Section 3
    is correct.
    """
    ranked = _rank_clusters(clusters)
    if not ranked:
        return [], []
    picked: list = []
    picked_embs: list = []
    tail: list = []
    for cp in ranked:
        if len(picked) >= k:
            tail.append(cp)
            continue
        emb = store.get_embedding(cp.representative_id)
        if emb is None:
            # No embedding — take it unconditionally if there's a slot,
            # otherwise drop to tail.
            picked.append(cp)
            picked_embs.append(None)
            continue
        too_close = False
        for prev_emb in picked_embs:
            if prev_emb is None:
                continue
            sim = sum(a * b for a, b in zip(emb, prev_emb))
            if (1.0 - sim) < min_dist:
                too_close = True
                break
        if too_close:
            tail.append(cp)
            continue
        picked.append(cp)
        picked_embs.append(emb)
    return picked, tail


def _is_valid_contract(contract: dict) -> bool:
    if not isinstance(contract, dict):
        return False
    if contract.get("regime") not in ("exploration", "optimization"):
        return False
    form = contract.get("form")
    if not isinstance(form, str) or not form.strip():
        return False
    if contract.get("length_hint", "") not in ("short", "medium", "long"):
        return False
    return True


def _trim_render_full(plan: dict, candidates: list, store: SignalStore) -> dict:
    if len(plan["render_full"]) <= _SECTION_1_MAX_RENDER_FULL:
        return plan

    id_to_cp = {cp.representative_id: cp for cp in candidates}
    selected_candidates = [id_to_cp[cid] for cid in plan["render_full"]
                           if cid in id_to_cp]
    if not selected_candidates:
        return plan

    picked, tail = _select_diverse_clusters(
        selected_candidates, store, _SECTION_1_MAX_RENDER_FULL,
    )
    picked_ids = [cp.representative_id for cp in picked]
    tail_ids = [cp.representative_id for cp in tail]

    section3 = list(dict.fromkeys(plan.get("section3_only", []) + tail_ids))
    notes = str(plan.get("notes", "")).strip()
    if notes:
        notes += " "
    notes += (
        f"(render set trimmed to {_SECTION_1_MAX_RENDER_FULL} highest-priority, "
        f"diverse clusters)"
    )

    return {
        "render_full": picked_ids,
        "section3_only": section3,
        "merge_groups": plan.get("merge_groups", []),
        "notes": notes,
    }


def _split_ids(ids: list[str], valid_ids: set[str]) -> list[str]:
    return [cid for cid in ids if isinstance(cid, str) and cid in valid_ids]


def _parse_merge_groups(raw_groups, valid_ids: set[str]) -> list[list[str]]:
    merged: list[list[str]] = []
    if not isinstance(raw_groups, list):
        return merged
    for grp in raw_groups:
        if not isinstance(grp, list):
            continue
        cleaned = [cid for cid in grp if isinstance(cid, str) and cid in valid_ids]
        if len(cleaned) >= 2:
            merged.append(cleaned)
    return merged


def _sanitize_plan(raw_plan: dict, candidates: list, store: SignalStore) -> dict:
    valid_ids = {cp.representative_id for cp in candidates}
    if not isinstance(raw_plan, dict):
        return {"render_full": [], "section3_only": [], "merge_groups": [], "notes": ""}

    render_full = _split_ids(raw_plan.get("render_full", []), valid_ids)
    section3_only = _split_ids(raw_plan.get("section3_only", []), valid_ids)
    merge_groups = _parse_merge_groups(raw_plan.get("merge_groups", []), valid_ids)
    notes = str(raw_plan.get("notes", "")).strip()

    # Implicit demotion: any candidate not in render_full and not in section3
    # is routed to section 3 so no cluster disappears silently.
    used = set(render_full) | set(section3_only)
    for cp in candidates:
        if cp.representative_id not in used:
            section3_only.append(cp.representative_id)

    sanitized = {
        "render_full": list(dict.fromkeys(render_full)),
        "section3_only": list(dict.fromkeys(section3_only)),
        "merge_groups": merge_groups,
        "notes": notes,
    }
    if len(sanitized["render_full"]) > _SECTION_1_MAX_RENDER_FULL:
        return _trim_render_full(sanitized, candidates, store)
    return sanitized


def _detect_inter_cluster_contradictions(
    clusters: list, store: SignalStore
) -> list[tuple]:
    """Find pairs of surviving clusters that appear contradictory.

    A pair is flagged as contradictory when:
      - Both have dissent_pressure > 0.3 (both have field opposition)
      - Their representative content shares at least 3 content words
        (same topic) but the dissent signals overlap (same challenger claims)

    Returns list of (cp_a, cp_b) pairs. Small-n heuristic — fast enough.
    """
    contradictions = []
    import re as _re
    _stop = frozenset({"the","a","an","is","are","to","of","in","and","or","it","be"})

    def content_words(text: str) -> set[str]:
        return {w.lower() for w in _re.findall(r"[a-z]+", text.lower())
                if w.lower() not in _stop and len(w) > 3}

    for i, ca in enumerate(clusters):
        if ca.dissent_pressure < 0.3:
            continue
        rep_a = store.get(ca.representative_id)
        if rep_a is None:
            continue
        words_a = content_words(rep_a.content)
        for cb in clusters[i+1:]:
            if cb.dissent_pressure < 0.3:
                continue
            rep_b = store.get(cb.representative_id)
            if rep_b is None:
                continue
            words_b = content_words(rep_b.content)
            shared_topic = words_a & words_b
            shared_dissent = set(ca.dissent_set) & set(cb.dissent_set)
            if len(shared_topic) >= 3 and shared_dissent:
                contradictions.append((ca, cb))
    return contradictions


def _contradictions_from_projection(
    projection: "SynthesisProjection",
    store: "SignalStore",
) -> list[tuple]:
    """Return (ca, cb) contradiction pairs for Section 2.

    Prefers typed 'tension' edges from the inter-cluster graph when
    available — these are principled (embedding-based) rather than
    heuristic. Falls back to _detect_inter_cluster_contradictions when
    the edge graph has no tension edges (e.g. embeddings unavailable).
    """
    id_to_cp = {cp.representative_id: cp
                for cp in projection.surviving + projection.contested}
    tension_pairs = [
        (id_to_cp[e.source], id_to_cp[e.target])
        for e in getattr(projection, "inter_cluster_edges", [])
        if e.relation == "tension"
        and e.source in id_to_cp
        and e.target in id_to_cp
    ]
    if tension_pairs:
        return tension_pairs
    return _detect_inter_cluster_contradictions(projection.surviving, store)


class Synthesizer:
    ROLE = "synthesizer"

    def __init__(self, llm, task_prompt: str):
        self.llm = llm
        self.task_prompt = task_prompt

    async def synthesize(
        self,
        store: SignalStore,
        has_validators: bool = True,
        prior_rejections: Optional[list] = None,
        prior_consensus: Optional[list] = None,
        output_dir: Optional[Path] = None,
        task_type: Optional[str] = None,
    ) -> tuple[str, dict, str]:
        """Run Layer 1 then Layer 2.

        output_dir: when provided, renderer_audit.json is written there.
        task_type: drives the survival profile + position-taking variants
        in the per-cluster renderer prompts.
        Returns (answer_text, citations_dict, lineage_dot_str).
        """
        self._task_type = task_type
        # Layer 1: pure-Python DAG projection
        projection = build_projection(
            store,
            has_validators=has_validators,
            prior_rejections=prior_rejections,
            prior_consensus=prior_consensus,
            task_type=task_type,
        )

        # Layer 2: structured multi-call renderer
        answer_text = await self._render(projection, store, output_dir=output_dir)

        # Build citation and lineage artefacts (pure Python, no LLM)
        citations = _build_citations(projection, store)
        lineage_dot = _build_lineage_dot(projection, store)

        return answer_text, citations, lineage_dot

    # -----------------------------------------------------------------------
    # Layer 2: renderer
    # -----------------------------------------------------------------------

    async def _render(
        self,
        projection: SynthesisProjection,
        store: SignalStore,
        output_dir: Optional[Path] = None,
    ) -> str:
        if projection.no_consensus:
            surv = len(projection.surviving)
            cont = len(projection.contested)
            weak = len(projection.weakly_supported)
            rej  = len(projection.rejected_by_field)
            # Bug 3: write a minimal renderer_audit.json on the no-consensus
            # short-circuit so summary.json reads audit_flags=0 (ran clean,
            # zero flags) rather than the -1 sentinel that meant "file
            # missing". Skip the actual faithfulness audit — there's no
            # rendered answer to audit.
            if output_dir is not None:
                _write_no_consensus_audit(output_dir)
            return (
                f"The swarm did not converge on a stable answer for this task.\n\n"
                f"Projection summary: {surv} surviving, {cont} contested, "
                f"{weak} weakly_supported, {rej} rejected_by_field.\n\n"
                f"Consult signals.json and lineage.dot for the underlying signal "
                f"field. You can re-synthesize with different filter thresholds "
                f"via `python synthesize.py <run_dir>`."
            )

        # ------------------------------------------------------------------
        # Partial-convergence abstention gate (improvement 5.8).
        # Fires when surviving clusters exist but none clear the credibility
        # bar: low verification, low support diversity, and heavy dissent.
        # Only applies when validators actually ran (proxied by whether any
        # surviving cluster has non-empty verification_set).
        # ------------------------------------------------------------------
        if (
            _SYNTHESIZER_USE_CALIBRATED_ABSTENTION
            and projection.surviving
            and any(cp.verification_set for cp in projection.surviving)
        ):
            max_ver = max(cp.verification_score for cp in projection.surviving)
            max_div = max(cp.support_diversity for cp in projection.surviving)
            max_dis = max(cp.dissent_pressure for cp in projection.surviving)
            if (max_ver < _ABSTAIN_VER_THRESHOLD
                    and max_div < _ABSTAIN_DIVERSITY_THRESHOLD
                    and max_dis > _ABSTAIN_DISSENT_THRESHOLD):
                ranked = _rank_clusters(projection.surviving)
                fragments: list[str] = []
                for cp in ranked[:3]:
                    rep = store.get(cp.representative_id)
                    if rep:
                        fragments.append(
                            f"[{cp.representative_id}] "
                            f"(ver={cp.verification_score:.2f}, "
                            f"div={cp.support_diversity}, "
                            f"dissent={cp.dissent_pressure:.2f}): "
                            f"{_truncate(rep.content, 200)}"
                        )
                if output_dir is not None:
                    _write_no_consensus_audit(output_dir)
                frag_block = "\n\n".join(fragments) if fragments else "(none)"
                print(
                    f"[synthesizer] ABSTAINING: max_ver={max_ver:.2f} < "
                    f"{_ABSTAIN_VER_THRESHOLD}, max_div={max_div} < "
                    f"{_ABSTAIN_DIVERSITY_THRESHOLD}, max_dis={max_dis:.2f} > "
                    f"{_ABSTAIN_DISSENT_THRESHOLD}"
                )
                return (
                    f"The signal field did not reach credible convergence. "
                    f"Abstaining from synthesis.\n\n"
                    f"Abstention criteria: "
                    f"max_verification_score={max_ver:.2f} < {_ABSTAIN_VER_THRESHOLD}, "
                    f"max_support_diversity={max_div} < {_ABSTAIN_DIVERSITY_THRESHOLD}, "
                    f"max_dissent_pressure={max_dis:.2f} > {_ABSTAIN_DISSENT_THRESHOLD}.\n\n"
                    f"Strongest surviving fragments (unrendered):\n\n{frag_block}\n\n"
                    f"Consult signals.json for the full field state. Consider re-running "
                    f"with a larger corpus or more iterations."
                )

        # Genome-level abstention gate (Gap #3, design doc §11).
        # Only fires when all surviving clusters have populated genomes
        # AND all of them fail the composite_fitness+grounding+load_bearing triple.
        # Complement to the structural gate above: catches the case where
        # clusters exist and are structurally reasonable but have zero external
        # grounding, no load-bearing atoms, and poor composite fitness.
        if (
            _SYNTHESIZER_USE_GENOME_ABSTENTION
            and projection.surviving
            and all(cp.genome is not None for cp in projection.surviving)
        ):
            def _genome_passes(cp) -> bool:
                g = cp.genome
                if g is None:
                    return True   # no genome → pass (don't penalise legacy path)
                fitness_ok = g.composite_fitness >= _ABSTAIN_GENOME_FITNESS_TAU
                grounding_ok = g.knowledge_base.source_count > 0
                atoms_ok = len(g.sensitivity.load_bearing_atoms) > 0
                return fitness_ok or grounding_ok or atoms_ok

            if not any(_genome_passes(cp) for cp in projection.surviving):
                best = max(projection.surviving, key=lambda c: c.genome.composite_fitness)
                print(
                    f"[synthesizer] GENOME ABSTAIN: no surviving cluster clears "
                    f"fitness≥{_ABSTAIN_GENOME_FITNESS_TAU} OR grounding>0 OR load_bearing_atoms≥1. "
                    f"Best composite_fitness={best.genome.composite_fitness:.3f}"
                )
                if output_dir is not None:
                    _write_no_consensus_audit(output_dir)
                return (
                    f"The surviving signal clusters lack sufficient external grounding "
                    f"and atom-level support structure for confident synthesis. "
                    f"Abstaining from synthesis.\n\n"
                    f"Best cluster composite_fitness={best.genome.composite_fitness:.3f} "
                    f"(threshold={_ABSTAIN_GENOME_FITNESS_TAU}), "
                    f"grounding={best.genome.knowledge_base.source_count} source(s), "
                    f"load_bearing_atoms={len(best.genome.sensitivity.load_bearing_atoms)}.\n\n"
                    f"Consider re-running with more iterations or a larger retrieval corpus."
                )

        # ------------------------------------------------------------------
        # Stage 1: interpret the user's prompt into a structural contract.
        # Decouples task category (coarse) from output shape (fine). The
        # contract drives Stage 3's cohesive integration call.
        # ------------------------------------------------------------------
        contract = await self._interpret_prompt(self.task_prompt)
        self._prompt_contract = contract  # stashed for downstream + summary.json
        strategy = _OUTPUT_STRATEGY_BY_TASK.get(
            getattr(self, "_task_type", None), "sectioned",
        )
        print(f"[synthesizer] contract: form={contract.get('form')!r} "
              f"regime={contract.get('regime')!r} strategy={strategy!r}")

        sections: list[str] = []

        # ------------------------------------------------------------------
        # Synthesis plan FIRST — the executive summary then references it.
        # ------------------------------------------------------------------
        plan = await self._plan_synthesis(projection, store)
        plan_render_ids: set = set(plan.get("render_full", []))
        plan_section3_ids: set = set(plan.get("section3_only", []))
        merge_groups: list = list(plan.get("merge_groups", []))
        plan_notes: str = plan.get("notes", "")

        # ------------------------------------------------------------------
        # COHESIVE strategies short-circuit the multi-section render path.
        # The integration call produces the actual user-facing artifact;
        # process notes (exec summary, plan, citations) are appended
        # deterministically below.
        # ------------------------------------------------------------------
        if strategy in ("cohesive_exploration", "cohesive_optimization"):
            render_id_list = list(plan_render_ids) or [
                cp.representative_id for cp in
                (_rank_clusters(projection.surviving) or projection.contested)[:5]
            ]

            # Best-of-N (improvement 5.3): generate N candidates with diverse
            # cluster orderings + temperatures, score each deterministically
            # against the cluster lattice, and take the argmax. Falls back to
            # single-candidate when _BEST_OF_N_COHESIVE <= 1 or all fail.
            _n = _BEST_OF_N_COHESIVE if _BEST_OF_N_COHESIVE > 1 else 1
            _temps = (
                _BEST_OF_N_EXPLORATION_TEMPS if strategy == "cohesive_exploration"
                else _BEST_OF_N_OPTIMIZATION_TEMPS
            )
            candidates: list[tuple[float, str]] = []
            for _i in range(_n):
                _temp = _temps[_i % len(_temps)]
                # Shuffle render order deterministically per candidate
                import random as _random
                _rng = _random.Random(_i + 42)
                _shuffled = list(render_id_list)
                if _i > 0:
                    _rng.shuffle(_shuffled)
                try:
                    if strategy == "cohesive_exploration":
                        _cand = await self._render_cohesive_exploration(
                            contract, projection, store, _shuffled,
                            temperature=_temp,
                        )
                    else:
                        _cand = await self._render_cohesive_optimization(
                            contract, projection, store, _shuffled,
                            temperature=_temp,
                        )
                    if _cand:
                        _score = self._score_cohesive_candidate(
                            _cand, contract, projection, store,
                        )
                        candidates.append((_score, _cand))
                        print(f"[synthesizer] best-of-N candidate {_i}: "
                              f"temp={_temp:.2f} score={_score:.3f} "
                              f"len={len(_cand)}")
                except Exception as exc:
                    print(f"[synthesizer] best-of-N candidate {_i} failed: "
                          f"{type(exc).__name__}: {exc}")

            if candidates:
                best_score, artifact = max(candidates, key=lambda x: x[0])
                print(f"[synthesizer] best-of-N: selected candidate "
                      f"(score={best_score:.3f}) from {len(candidates)} total")
            else:
                print(f"[synthesizer] cohesive render failed (all {_n} candidates "
                      f"failed); falling back to sectioned output")
                strategy = "sectioned"
                artifact = None

            if strategy != "sectioned":
                # Assemble: artifact at the top, then deterministic process
                # notes (exec summary, plan), then citations stamp.
                exec_notes = await self._render_executive_summary(
                    projection, store, plan_notes=plan_notes,
                )
                parts = [artifact]

                # Alternative-of-the-best artifact (improvement 5.9).
                # For exploration tasks: find the highest-priority surviving
                # cluster NOT in render_full (section3_only or MMR-filtered)
                # and generate a second artifact from that cluster alone.
                if (
                    SYNTHESIZER_EMIT_ALTERNATIVE
                    and strategy == "cohesive_exploration"
                    and plan_section3_ids
                ):
                    alt_candidates = [
                        cp for cp in _rank_clusters(projection.surviving)
                        if cp.representative_id in plan_section3_ids
                    ]
                    if alt_candidates:
                        alt_cp = alt_candidates[0]
                        try:
                            alt_artifact = await self._render_alternative_artifact(
                                contract, projection, store, alt_cp,
                            )
                            if alt_artifact:
                                alt_why = (
                                    f"strongest non-selected cluster "
                                    f"[{alt_cp.representative_id}]: "
                                    f"support_diversity={alt_cp.support_diversity}, "
                                    f"verification_score={alt_cp.verification_score:.2f}"
                                )
                                parts.append(
                                    f"\n\n---\n\n## STRONGEST ALTERNATIVE\n\n"
                                    f"*{alt_why}*\n\n{alt_artifact}"
                                )
                        except Exception as exc:
                            print(f"[synthesizer] alternative artifact failed: "
                                  f"{type(exc).__name__}: {exc}")

                parts.append("\n\n---\n\n## PROCESS NOTES\n")
                parts.append(exec_notes)
                answer = "\n\n".join(p for p in parts if p)
                answer = _stamp_citations(answer, projection, store,
                                           merge_groups=[])  # Fix P: no post-hoc merge
                # Soft audit for cohesive outputs (looser than 4-gram).
                try:
                    audit_flags = _build_cohesive_audit(
                        artifact, contract, projection, store,
                    )
                    if output_dir is not None:
                        _write_faithfulness_audit(audit_flags, output_dir)
                except Exception as exc:
                    print(f"[synthesizer] cohesive audit crashed: "
                          f"{type(exc).__name__}: {exc}")
                    if output_dir is not None:
                        _write_crashed_audit(output_dir, exc)
                return answer

        # ------------------------------------------------------------------
        # SECTIONED strategy (debate/analysis fallback): keep the existing
        # multi-section render path. Executive summary appears first.
        # ------------------------------------------------------------------
        exec_summary = await self._render_executive_summary(
            projection, store, plan_notes=plan_notes,
        )
        if exec_summary:
            sections.append(exec_summary)

        # ------------------------------------------------------------------
        # Section 1: per-cluster rendering, driven by the synthesis plan
        # already produced above (no hard cap on Section-1 size).
        # ------------------------------------------------------------------
        if projection.surviving:
            rendered_surviving = [
                cp for cp in projection.surviving
                if cp.representative_id in plan_render_ids
            ]
            section1_tail = [
                cp for cp in projection.surviving
                if cp.representative_id not in plan_render_ids
            ]
            # Preserve priority order within the planned set.
            rendered_surviving = _rank_clusters(rendered_surviving)
        else:
            rendered_surviving = []
            section1_tail = []
        if rendered_surviving:
            # Identify debate-eligible alternatives groups (improvement 5.5).
            # Clusters in a debate group get _render_debate_frame instead of
            # per-cluster _render_cluster_position calls.
            debate_groups: list[list] = []
            debate_cluster_ids: set[str] = set()
            if _SYNTHESIZER_USE_DEBATE and getattr(projection, "inter_cluster_edges", []):
                debate_groups = self._identify_debate_clusters(
                    rendered_surviving, projection,
                )
                for grp in debate_groups:
                    debate_cluster_ids.update(
                        cp.representative_id for cp in grp
                    )

            fragments: list[str] = []
            # Semaphore-bounded parallel render (Fix S1, §9b).
            # Per-cluster renders are mutually independent under the no-leak
            # rule — each call reads only its own cluster's signals. Gather
            # runs them concurrently up to _RENDER_SEM_SIZE slots. On the
            # laptop (LLM_CONCURRENCY=1) the semaphore inside the LLM
            # serializes them anyway; on vLLM paths they genuinely parallelize.
            _render_sem = asyncio.Semaphore(_RENDER_SEM_SIZE)

            async def _guarded_debate(grp):
                async with _render_sem:
                    return await self._render_debate_frame(grp, store)

            async def _guarded_position(cp):
                async with _render_sem:
                    return await self._render_cluster_position(
                        cp, store, projection=projection,
                    )

            # Render debate groups and remaining clusters concurrently.
            # Debate groups first (preserves ordering in final output).
            debate_tasks = [_guarded_debate(grp) for grp in debate_groups]
            debate_results = await asyncio.gather(*debate_tasks, return_exceptions=True)

            for grp, result in zip(debate_groups, debate_results):
                if isinstance(result, BaseException):
                    # Debate frame raised: fall back to per-cluster rendering.
                    fallback_tasks = [_guarded_position(cp) for cp in grp]
                    fallback_results = await asyncio.gather(*fallback_tasks, return_exceptions=True)
                    for frag in fallback_results:
                        if isinstance(frag, str) and frag:
                            if frag[-1] not in ".!?\")'":
                                frag += "."
                            fragments.append(frag)
                elif result:
                    fragments.append(result)

            # Render remaining (non-debate) clusters in parallel, then
            # append in priority order (gather preserves task submission order).
            solo_clusters = [
                cp for cp in rendered_surviving
                if cp.representative_id not in debate_cluster_ids
            ]
            solo_tasks = [_guarded_position(cp) for cp in solo_clusters]
            solo_results = await asyncio.gather(*solo_tasks, return_exceptions=True)
            for frag in solo_results:
                if isinstance(frag, str) and frag:
                    if frag[-1] not in ".!?\")'":
                        frag += "."
                    fragments.append(frag)
            if fragments:
                # Stage B (improvement 5.4): compose paragraphs along the
                # typed edge graph. Falls back to plain join when no edges
                # touch the rendered cluster set or the call fails.
                if (
                    _SYNTHESIZER_USE_EDGE_COMPOSITION
                    and len(fragments) > 1
                    and getattr(projection, "inter_cluster_edges", [])
                ):
                    try:
                        composed = await self._compose_with_edges(
                            fragments, rendered_surviving, projection, store,
                        )
                    except Exception as exc:
                        print(f"[synthesizer] _compose_with_edges crashed: "
                              f"{type(exc).__name__}: {exc}; plain join")
                        composed = "\n\n".join(fragments)
                else:
                    composed = "\n\n".join(fragments)
                sections.append("## 1. POSITION SYNTHESIS\n\n" + composed)

        # ------------------------------------------------------------------
        # Section 2: Open questions and dissent — per contested cluster AND
        # per surviving cluster that attracted any dissent.
        # Also flags inter-cluster contradictions / tension edges.
        # ------------------------------------------------------------------
        contradictions = _contradictions_from_projection(projection, store)
        dissent_candidates: list[ClusterProjection] = list(projection.contested)
        dissent_candidates += [
            cp for cp in projection.surviving if cp.dissent_set
        ]
        if dissent_candidates or contradictions:
            fragments = []
            # Parallel dissent renders — same independence guarantee as Section 1.
            _dissent_sem = asyncio.Semaphore(_RENDER_SEM_SIZE)

            async def _guarded_dissent(cp):
                async with _dissent_sem:
                    return await self._render_cluster_dissent(cp, store)

            dissent_tasks = [_guarded_dissent(cp) for cp in dissent_candidates]
            dissent_results = await asyncio.gather(*dissent_tasks, return_exceptions=True)
            for frag in dissent_results:
                if isinstance(frag, str) and frag:
                    if frag[-1] not in ".!?\")'":
                        frag += "."
                    fragments.append(frag)
            for ca, cb in contradictions:
                fragments.append(
                    f"[INTER-CLUSTER CONTRADICTION] Clusters "
                    f"[{ca.representative_id}] and [{cb.representative_id}] "
                    f"share topic content and share dissent signals "
                    f"({', '.join(set(ca.dissent_set) & set(cb.dissent_set))[:3]}). "
                    f"These may be two framings of the same contested claim."
                )
            if fragments:
                sections.append(
                    "## 2. OPEN QUESTIONS AND DISSENT\n\n" + "\n\n".join(fragments)
                )

        # ------------------------------------------------------------------
        # Section 3: Considered and filtered — deterministic, no LLM call.
        # Now aggregates four buckets in priority order:
        #   1. rejected_by_field      (most diagnostic — explicit field rejection)
        #   2. unverified             (would have survived but lacked credibility)
        #   3. tail-of-surviving      (clusters the planner demoted to section 3)
        #   4. weakly_supported       (insufficient distinct supporters)
        # Each sorted by descending verification_score within its bucket.
        # ------------------------------------------------------------------
        rej_sorted = sorted(
            projection.rejected_by_field,
            key=lambda c: c.verification_score, reverse=True,
        )
        unver_sorted = sorted(
            projection.unverified,
            key=lambda c: c.verification_score, reverse=True,
        )
        weak_sorted = sorted(
            projection.weakly_supported,
            key=lambda c: c.verification_score, reverse=True,
        )
        # section1_tail is already ranked by priority.
        filtered = (rej_sorted + unver_sorted + section1_tail + weak_sorted)[
            :_SECTION_3_RENDER_CAP
        ]
        if filtered:
            lines: list[str] = []
            for cp in filtered:
                rep = store.get(cp.representative_id)
                content = _truncate(rep.content, _SUMMARY_CHARS) if rep else cp.representative_id
                if cp.status == "rejected_by_field":
                    reason = f"rejected: dissent_pressure={cp.dissent_pressure:.2f} > 1.5"
                elif cp.status == "unverified":
                    reason = (f"held: no verification, no dissent, "
                              f"support_diversity={cp.support_diversity} < 4")
                elif cp.status == "surviving":
                    reason = (f"tail (planner demoted): "
                              f"support_diversity={cp.support_diversity}, "
                              f"verification_score={cp.verification_score:.2f}")
                else:
                    reason = f"filtered: support_diversity={cp.support_diversity} < 3"
                lines.append(f"- [{cp.representative_id}] {content}  ({reason})")
            sections.append(
                "## 3. CONSIDERED AND FILTERED\n\n"
                + "\n".join(lines)
            )

        # ------------------------------------------------------------------
        # Section 5: Topology coverage gaps — cells the scouts were assigned
        # to but no surviving signal covered. Pure Python; no LLM call.
        # Only emitted when topology is present and gaps exist.
        # ------------------------------------------------------------------
        topology = getattr(projection, "topology", None)
        if topology is not None:
            uncovered = list(getattr(projection, "uncovered_cells", []))
            if uncovered:
                gap_lines: list[str] = []
                for cell in uncovered[:8]:
                    desc = (
                        topology.cell_description(cell)
                        if hasattr(topology, "cell_description") else str(cell)
                    )
                    gap_lines.append(f"- {desc}")
                sections.append(
                    "## 5. UNEXPLORED ANSWER-SPACE REGIONS\n\n"
                    "The following topology cells were assigned to scouts but no "
                    "surviving signal was deposited in these regions — they "
                    "represent bounds of the answer space the swarm did not "
                    "adequately explore:\n\n"
                    + "\n".join(gap_lines)
                )

        # Section 6: Out-of-bounds clusters — surviving clusters whose topology
        # coordinates fall outside the declared axes. Flags possible category
        # errors or prompt drift. Only emitted when OOB clusters exist.
        if topology is not None:
            oob = list(getattr(projection, "out_of_bounds_clusters", []))
            if oob:
                sections.append(
                    "## 6. OUT-OF-BOUNDS CLUSTERS\n\n"
                    f"{len(oob)} surviving cluster(s) have topology coordinates "
                    "that fall outside the declared answer-space axes. This may "
                    "indicate off-topic exploration or axis under-specification:\n\n"
                    + "\n".join(f"- [{c}]" for c in oob[:5])
                )

        # ------------------------------------------------------------------
        # Assemble sections
        # ------------------------------------------------------------------
        answer = "\n\n".join(sections)

        # Revision loop (improvement 5.2): K rounds of critic → revise.
        # Runs after section assembly but before citation stamping so the
        # critic sees clean prose without the Section 3/4 deterministic
        # noise. Skipped when sections is empty (no-consensus fallback
        # already returned above).
        if _SYNTHESIZER_REVISION_ROUNDS > 0 and sections:
            try:
                answer = await self._revision_loop(
                    answer, projection, store,
                    max_rounds=_SYNTHESIZER_REVISION_ROUNDS,
                )
            except Exception as exc:
                print(f"[synthesizer] revision loop crashed: "
                      f"{type(exc).__name__}: {exc}; keeping original assembly")

        # Section 4 — Citations: deterministic stamp appended to the answer.
        # Pass merge_groups so the reader can see when the planner treated
        # two clusters as a single position.
        answer = _stamp_citations(answer, projection, store,
                                   merge_groups=[])  # Fix P: no post-hoc merge

        # ------------------------------------------------------------------
        # Post-hoc faithfulness audit
        # ------------------------------------------------------------------
        # Bug 3: wrap so an audit crash writes a -2 sentinel + error message
        # rather than leaving renderer_audit.json absent (which made summary
        # report audit_flags=-1 ambiguously). Sentinel semantics now:
        #     0  = audit ran clean
        #     N>=1 = audit ran, found N flags
        #     -2 = audit crashed (audit_error field carries the reason)
        try:
            audit_flags = _build_faithfulness_audit(answer, projection, store)
            if output_dir is not None:
                _write_faithfulness_audit(audit_flags, output_dir)
            elif audit_flags:
                print(
                    f"[synthesizer] faithfulness audit: {len(audit_flags)} flag(s) "
                    f"(pass output_dir to write renderer_audit.json)"
                )
            # Loud end-of-run warning when the audit is heavily flagged. 20+
            # is the empirical threshold past which a renderer pass is
            # producing more noise than signal and the synthesis prose
            # should not be trusted without manual review.
            if len(audit_flags) >= _AUDIT_WARNING_THRESHOLD:
                audit_path = (
                    str(output_dir / "renderer_audit.json")
                    if output_dir is not None else "(audit not written — no output_dir)"
                )
                print(
                    f"[synthesizer] WARNING: {len(audit_flags)} faithfulness "
                    f"flags (>= {_AUDIT_WARNING_THRESHOLD}). Synthesis prose "
                    f"may diverge from cited signals; review "
                    f"{audit_path} before citing this run."
                )
        except Exception as exc:
            print(f"[synthesizer] faithfulness audit crashed: "
                  f"{type(exc).__name__}: {exc}")
            if output_dir is not None:
                _write_crashed_audit(output_dir, exc)

        return answer

    # -----------------------------------------------------------------------
    # Stage 1: Prompt interpreter — extracts the structural contract
    # -----------------------------------------------------------------------

    async def _interpret_prompt(self, user_prompt: str) -> dict:
        """Extract a structured contract from the user's task prompt.

        Decouples *task category* (coarse: creative/coding/debate/...) from
        *output form* (fine: haiku vs short story vs slogan; binary-search
        function vs JSON parser; etc.). Same swarm behavior produces the
        surviving ideas in all cases; this contract is what tells the
        synthesizer what *shape* to render them in.

        Returns a dict with at least:
            regime:     "exploration" | "optimization"
            form:       short string naming the target form
            structural: list of hard constraints (line count, signature,
                        complexity bound, format markers)
            soft:       list of style/voice/theme cues
            length_hint: "short" | "medium" | "long"

        Falls back to a heuristic contract on parse failure so downstream
        rendering can still proceed.
        """
        prompt = (
            f"Read the following user prompt and extract its structural "
            f"contract as JSON. Two regimes exist:\n"
            f"  - exploration: no oracle answer; reader wants novel/"
            f"interesting ideas (haiku, story, essay, slogan, argument, "
            f"action plan, analysis).\n"
            f"  - optimization: a correct answer or class of correct "
            f"answers exists (code, math, formal proof).\n\n"
            f"Hard constraints are things you can VERIFY post-hoc (line "
            f"count, syllable structure, function signature, presence of "
            f"a section, language requirement). Soft cues are stylistic "
            f"directions (tone, voice, themes) that the renderer must "
            f"interpret. If the prompt is vague, infer the most likely "
            f"form rather than listing nothing.\n\n"
            f"EXAMPLE — prompt 'Write a haiku about AI':\n"
            f'{{"regime": "exploration", "form": "haiku",\n'
            f'  "structural": ["3 lines", "5-7-5 syllable pattern"],\n'
            f'  "soft": ["AI or technology theme", "evocative imagery"],\n'
            f'  "length_hint": "short", "audience": "general"}}\n\n'
            f"EXAMPLE — prompt 'Implement binary search in Python':\n"
            f'{{"regime": "optimization", "form": "function",\n'
            f'  "structural": ["Python function", "O(log n) complexity", '
            f'"returns index or -1"],\n'
            f'  "soft": ["clean idiomatic Python"],\n'
            f'  "length_hint": "medium", "audience": "developers"}}\n\n'
            f"Now extract the contract for this prompt. Write real values — "
            f"never output angle-bracket placeholders like <short label>.\n\n"
            f"---USER PROMPT---\n{user_prompt}\n---END USER PROMPT---\n\n"
            f"CONTRACT (JSON only):"
        )
        try:
            raw = await self.llm.generate(
                prompt, role=self.ROLE,
                max_tokens=_INTERPRET_MAX_TOKENS,
                temperature=_INTERPRET_TEMPERATURE,
            )
        except Exception as exc:
            print(f"[synthesizer] prompt-interpret call failed: "
                  f"{type(exc).__name__}: {exc}")
            return self._fallback_contract(user_prompt)

        block = _extract_json_block(raw or "")
        if block is None:
            # OLD:
            # if block is None:
            #     return self._fallback_contract(user_prompt)
            if _SYNTHESIZER_USE_ROBUST_PLAN_FALLBACK:
                print("[synthesizer] prompt interpreter parse failed; retrying with stricter JSON instructions")
                try:
                    raw = await self.llm.generate(
                        prompt + "\nRespond with a JSON object only, no surrounding text.\nCONTRACT:",
                        role=self.ROLE,
                        max_tokens=_INTERPRET_MAX_TOKENS,
                        temperature=_INTERPRET_TEMPERATURE,
                    )
                except Exception as exc:
                    print(f"[synthesizer] prompt-interpret retry failed: "
                          f"{type(exc).__name__}: {exc}")
                    return self._fallback_contract(user_prompt)
                block = _extract_json_block(raw or "")
            else:
                return self._fallback_contract(user_prompt)

        if block is None:
            print("[synthesizer] prompt interpreter failed to produce valid JSON; using heuristic fallback")
            return self._fallback_contract(user_prompt)
        try:
            contract = json.loads(_repair_json(block))
        except Exception as exc:
            print(f"[synthesizer] prompt interpreter JSON parse error: {exc}; using heuristic fallback")
            return self._fallback_contract(user_prompt)
        # Coerce / sanitize fields. Anything missing gets a default.
        out = {
            "regime":      str(contract.get("regime", "exploration")).strip().lower(),
            "form":        str(contract.get("form", "free_response")).strip(),
            "structural":  [str(x) for x in contract.get("structural", []) if x],
            "soft":        [str(x) for x in contract.get("soft", []) if x],
            "length_hint": str(contract.get("length_hint", "medium")).strip().lower(),
            "audience":    str(contract.get("audience", "general")).strip(),
        }
        if out["regime"] not in ("exploration", "optimization"):
            out["regime"] = "exploration"
        if out["length_hint"] not in ("short", "medium", "long"):
            out["length_hint"] = "medium"
        return out

    def _fallback_contract(self, user_prompt: str) -> dict:
        """Heuristic contract used when the interpreter call fails.

        Inferred entirely from the task_type recorded on self._task_type.
        Less precise than the LLM extraction but always available.
        """
        tt = getattr(self, "_task_type", None) or "analysis"
        regime = "optimization" if tt == "coding" else "exploration"
        form_by_task = {
            "creative":        "free_creative",
            "coding":          "function",
            "problem_solving": "action_plan",
            "debate":          "argument",
            "analysis":        "analysis",
        }
        return {
            "regime":      regime,
            "form":        form_by_task.get(tt, "free_response"),
            "structural":  [],
            "soft":        [],
            "length_hint": "medium",
            "audience":    "general",
        }

    # -----------------------------------------------------------------------
    # Planner: structural digest -> render plan (no content ingestion)
    # -----------------------------------------------------------------------

    async def _plan_synthesis(
        self, projection: SynthesisProjection, store: SignalStore,
    ) -> dict:
        """Ask the LLM to plan the synthesis from STRUCTURE alone.

        DUAL-PLANNER NOTE: There are two planners in this codebase.
        (1) This method — LLM-based. Sees a structural digest (IDs, counts,
            80-char previews). Returns JSON with render_full / section3_only /
            merge_groups. Lives in the Synthesizer agent because it still makes
            an LLM call, even though it reads no signal content.
        (2) core/projection.build_plan() — pure-Python, deterministic. Uses
            composite scoring + MMR over embeddings. No LLM. Called by the
            ConvergenceDetector and any path that needs a plan without a GPU.
        This method falls back to build_plan() if the LLM call fails or
        yields no valid cluster IDs. TODO: evaluate whether the LLM planner
        adds measurable ranking quality over build_plan() in ablation runs;
        if not, retire it and always use build_plan().

        The planner is shown a digest of the surviving + contested clusters:
        cluster IDs, type, support / dissent / verification *counts* and
        *scores*, support_depth, and a short preview (first 80 chars of the
        representative claim — just enough to let the planner topic-cluster
        without consuming the full DAG). It is NOT shown the support /
        dissent / verification signal contents.

        It returns JSON:
            {"render_full":   [<cluster_id>, ...],
             "section3_only": [<cluster_id>, ...],
             "merge_groups":  [[<cluster_id>, <cluster_id>], ...],
             "notes":         "<one-sentence overview>"}

        `render_full` clusters get a full Section-1 paragraph via the
        per-cluster renderer. `section3_only` clusters are demoted. Merge
        groups are recorded in the citation block so the reader can see
        the planner identified them as one position. Falls back to the
        legacy diversity-aware top-N selection if the call or parse fails.

        No leak rule: the digest carries only IDs and aggregate counts,
        not other agents' reasoning. The per-cluster renderer is the only
        layer that reads Signal.content.
        """
        candidates = list(projection.surviving) + list(projection.contested)
        if not candidates:
            return {"render_full": [], "section3_only": [], "merge_groups": [], "notes": ""}

        # Build the digest. ID + 200-char representative + structural metrics.
        # 200 chars is enough for content-based merge detection (two clusters
        # rephrasing the same claim in different surface language look similar
        # at 200 chars but are indistinguishable at 80). Still a tight budget —
        # full content lives in the per-cluster renderer, not here.
        lines: list[str] = []
        for cp in candidates:
            rep = store.get(cp.representative_id)
            preview = _truncate(rep.content, 200) if rep else "(no rep)"
            # Genome digest: include composite_fitness and grounding when available
            # so the LLM planner can weight render priority by non-symbolic fitness
            genome_str = ""
            if cp.genome is not None:
                genome_str = (
                    f"  composite_fitness={cp.genome.composite_fitness:.3f}"
                    f"  grounding={cp.genome.fitness_breakdown.get('grounding', 0):.2f}"
                    f"  n_atoms={len(cp.genome.atoms)}"
                    f"  load_bearing={len(cp.genome.sensitivity.load_bearing_atoms)}"
                )
            lines.append(
                f"- {cp.representative_id}  status={cp.status}  "
                f"n_supports={len(cp.support_set)}  "
                f"n_dissent={len(cp.dissent_set)}  "
                f"n_verifications={len(cp.verification_set)}  "
                f"support_diversity={cp.support_diversity}  "
                f"support_depth={cp.support_depth}  "
                f"verification_score={cp.verification_score:.2f}  "
                f"dissent_pressure={cp.dissent_pressure:.2f}"
                f"{genome_str}  "
                f"representative=\"{preview}\""
            )
        digest = "\n".join(lines)

        # Edge graph digest: summarise typed inter-cluster relations so the
        # planner can use them for merge and surface decisions.
        edge_lines: list[str] = []
        for e in getattr(projection, "inter_cluster_edges", []):
            edge_lines.append(
                f"  {e.source} --[{e.relation}]--> {e.target} (weight={e.weight:.2f})"
            )
        edge_block = (
            "INTER-CLUSTER EDGES:\n" + "\n".join(edge_lines)
            if edge_lines
            else "INTER-CLUSTER EDGES: (none detected)"
        )

        # Pull the first two real IDs for the schema example so the model
        # sees the exact ID format and cannot reproduce angle-bracket placeholders.
        _ex = [cp.representative_id for cp in candidates[:2]]
        ex0 = _ex[0] if len(_ex) > 0 else "INITIAL_00001"
        ex1 = _ex[1] if len(_ex) > 1 else "INITIAL_00002"

        prompt = (
            f"TASK TYPE: {getattr(self, '_task_type', 'unknown')}\n"
            f"TASK: {self.task_prompt}\n\n"
            f"You are planning the structure of a synthesis. You see a "
            f"structural digest of claim clusters: their IDs, counts of "
            f"supporting / dissenting / verifying signals, scores, a "
            f"200-character representative excerpt, and a typed inter-cluster "
            f"edge graph. You do NOT see the full underlying signals' content "
            f"— that gets rendered in a separate pass per cluster.\n\n"
            f"Your job: decide which clusters deserve a full paragraph in "
            f"Section 1 (POSITION SYNTHESIS) and which can be demoted to "
            f"Section 3 (CONSIDERED AND FILTERED). Pick clusters that are:\n"
            f"  1. Topically distinct from each other (avoid 5 paragraphs "
            f"     restating the same position).\n"
            f"  2. Well-supported (high support_diversity, support_depth >= 2).\n"
            f"  3. Verified where possible (verification_score > 0.3).\n"
            f"  4. Contested clusters are valuable — surface them.\n\n"
            f"Use the INTER-CLUSTER EDGES to inform your plan:\n"
            f"  - shared_evidence / co_contested pairs likely discuss the same "
            f"    topic — consider merging them.\n"
            f"  - alternatives pairs represent genuinely different approaches — "
            f"    surface both when distinct.\n"
            f"  - tension edges mean source's position is contested by target's "
            f"    dissent — worth flagging in section3_only notes.\n"
            f"  - supersedes: prefer the source cluster; demote the target.\n\n"
            f"If two clusters appear to make the same claim in different "
            f"surface language (look at their 200-char representatives) "
            f"or share shared_evidence edges, name them in a merge_group.\n\n"
            f"You can choose up to {len(candidates)} clusters for "
            f"render_full. The downstream synthesizer may trim this set to "
            f"the {_SECTION_1_MAX_RENDER_FULL} highest-priority, diverse "
            f"clusters if necessary. Prefer a slightly smaller, "
            f"sharper set over a larger noisy one.\n\n"
            f"---DIGEST---\n{digest}\n---END DIGEST---\n\n"
            f"---{edge_block}---\n\n"
            f"Reply with a JSON object in exactly this shape. The IDs in the "
            f"example below are from your digest — use the REAL IDs you see "
            f"above. Never output angle-bracket placeholders.\n\n"
            f"EXAMPLE OUTPUT (IDs are from YOUR digest — adjust the lists to "
            f"match your actual plan):\n"
            f'{{"render_full":   ["{ex0}", "{ex1}"],\n'
            f'  "section3_only": [],\n'
            f'  "merge_groups":  [],\n'
            f'  "notes": "Two distinct positions selected for full render."}}\n\n'
            f"Your plan (JSON only, no other text):\n\n"
            f"PLAN:"
        )

        try:
            raw = await self.llm.generate(
                prompt, role=self.ROLE,
                max_tokens=_PLAN_MAX_TOKENS,
                temperature=_PLAN_TEMPERATURE,
            )
        except Exception as exc:
            print(f"[synthesizer] plan call failed: {type(exc).__name__}: {exc}")
            return self._fallback_plan(candidates, store)

        plan = self._parse_plan(raw, candidates, store)
        if _SYNTHESIZER_USE_ROBUST_PLAN_FALLBACK and plan and len(plan.get("render_full", [])) > _SECTION_1_MAX_RENDER_FULL:
            plan = _trim_render_full(plan, candidates, store)
        return plan

    def _parse_plan(
        self, raw: str, candidates: list, store: SignalStore,
    ) -> dict:
        """Parse the planner JSON. Falls back to legacy selection on failure."""
        block = _extract_json_block(raw or "")
        if block is None:
            print(f"[synthesizer] plan parse: no JSON block; using fallback")
            return self._fallback_plan(candidates, store)
        try:
            raw_plan = json.loads(_repair_json(block))
        except Exception as exc:
            print(f"[synthesizer] plan JSON parse error: {exc}; using fallback")
            return self._fallback_plan(candidates, store)

        # OLD:
        # valid_ids = {cp.representative_id for cp in candidates}
        # render_full = [cid for cid in raw_plan.get("render_full", []) if cid in valid_ids]
        # section3_only = [cid for cid in raw_plan.get("section3_only", []) if cid in valid_ids]
        # merge_groups = []
        # for grp in raw_plan.get("merge_groups", []):
        #     if isinstance(grp, list):
        #         cleaned = [cid for cid in grp if cid in valid_ids]
        #         if len(cleaned) >= 2:
        #             merge_groups.append(cleaned)
        # notes = str(raw_plan.get("notes", "")).strip()
        # used = set(render_full)
        # for grp in merge_groups:
        #     used.update(grp)
        # section3_set = set(section3_only)
        # for cp in candidates:
        #     if cp.representative_id not in used and cp.representative_id not in section3_set:
        #         section3_set.add(cp.representative_id)
        # plan = {
        #     "render_full": render_full,
        #     "section3_only": sorted(section3_set),
        #     "merge_groups": merge_groups,
        #     "notes": notes,
        # }

        plan = _sanitize_plan(raw_plan, candidates, store)
        if _SYNTHESIZER_USE_ROBUST_PLAN_FALLBACK and not plan.get("render_full"):
            print(f"[synthesizer] plan empty after sanitization; using fallback")
            return self._fallback_plan(candidates, store)

        return plan

    def _fallback_plan(self, candidates: list, store: SignalStore) -> dict:
        """Fix P: deterministic build_plan() fallback (MMR cluster selection)."""
        surviving = [c for c in candidates if c.status == "surviving"]
        contested = [c for c in candidates if c.status != "surviving"]
        mini_proj = SynthesisProjection(surviving=surviving, contested=contested)
        sp = build_plan(mini_proj, store)
        # Section 3 = held (redundant-strong) + demoted (weak) + contested not in dissent
        contested_ids = {c.representative_id for c in contested}
        dissent_set = set(sp.dissent_clusters)
        s3_extra = [cid for cid in contested_ids if cid not in dissent_set]
        s3_ids = list(dict.fromkeys(sp.held_clusters + sp.demoted_clusters + s3_extra))
        return {
            "render_full":   sp.render_clusters,
            "section3_only": s3_ids,
            "merge_groups":  [],   # Fix P: no post-hoc merge
            "notes":         sp.planner_notes,
        }

    # -----------------------------------------------------------------------
    # Executive summary
    # -----------------------------------------------------------------------

    async def _render_executive_summary(
        self, projection: SynthesisProjection, store: SignalStore,
        plan_notes: str = "",
    ) -> str:
        """Deterministic executive summary built directly from the projection.

        Previously this was an LLM call that produced "Fourteen distinct
        clusters" when there were 41, and routinely inverted the polarity
        of contested clusters in prose. Counting and polarity belong in
        Python: the projection has the numbers, so we render them straight.

        Kept async to preserve the existing await at the call site, but no
        LLM call is made — runtime is microseconds.
        """
        n_surv = len(projection.surviving)
        n_cont = len(projection.contested)
        n_unver = len(projection.unverified)
        n_rej = len(projection.rejected_by_field)
        n_weak = len(projection.weakly_supported)
        total = n_surv + n_cont + n_unver + n_rej + n_weak

        breakdown_parts: list[str] = []
        breakdown_parts.append(f"{n_surv} survived field pressure")
        breakdown_parts.append(f"{n_cont} contested")
        if n_unver:
            breakdown_parts.append(f"{n_unver} held as unverified")
        if n_rej:
            breakdown_parts.append(f"{n_rej} rejected")
        if n_weak:
            breakdown_parts.append(f"{n_weak} weakly supported")
        breakdown = ", ".join(breakdown_parts)

        lines = [
            "## EXECUTIVE SUMMARY",
            "",
            f"Of {total} claim cluster(s) projected from the signal field: {breakdown}.",
        ]

        # Topology coverage preamble — shows which answer-space cells were
        # explored vs. left uncovered. Pure Python; no LLM call.
        topology = getattr(projection, "topology", None)
        if topology is not None:
            n_cells = len(topology.all_cells()) if hasattr(topology, "all_cells") else 0
            covered_cells = list(getattr(projection, "topology_coverage", {}).keys())
            uncovered = list(getattr(projection, "uncovered_cells", []))
            oob = list(getattr(projection, "out_of_bounds_clusters", []))
            lines.append(
                f"Answer-space topology: {len(topology.axes)} axes, "
                f"{n_cells} cells. "
                f"Coverage: {len(covered_cells)}/{n_cells} cells explored. "
                + (f"Uncovered: {len(uncovered)} cell(s). " if uncovered else "")
                + (f"Out-of-bounds clusters: {len(oob)}. " if oob else "")
            )
            if uncovered:
                uc_strs = [topology.cell_description(c) for c in uncovered[:4]
                           if hasattr(topology, "cell_description")]
                if uc_strs:
                    lines.append(f"Uncovered regions: {'; '.join(uc_strs)}")

        if projection.surviving:
            ranked = _rank_clusters(projection.surviving)
            top = ranked[0]
            rep = store.get(top.representative_id)
            if rep:
                excerpt = _truncate(rep.content, 160)
                # Genome fitness annotation when available
                genome_note = ""
                if top.genome is not None:
                    genome_note = (
                        f", composite_fitness={top.genome.composite_fitness:.3f}"
                        f", grounding={top.genome.fitness_breakdown.get('grounding', 0):.2f}"
                        f", n_atoms={len(top.genome.atoms)}"
                    )
                lines.append(
                    f"Strongest surviving cluster [{top.representative_id}] "
                    f"(support_diversity={top.support_diversity}, "
                    f"verification_score={top.verification_score:.2f}, "
                    f"dissent_pressure={top.dissent_pressure:.2f}"
                    f"{genome_note}): {excerpt}"
                )

            # Genome field summary across all surviving clusters
            genome_cps = [cp for cp in projection.surviving if cp.genome is not None]
            if genome_cps:
                avg_fitness = sum(cp.genome.composite_fitness for cp in genome_cps) / len(genome_cps)
                avg_grounding = sum(
                    cp.genome.fitness_breakdown.get("grounding", 0.0) for cp in genome_cps
                ) / len(genome_cps)
                n_with_atoms = sum(1 for cp in genome_cps if cp.genome.atoms)
                lines.append(
                    f"Genome field: avg_composite_fitness={avg_fitness:.3f}, "
                    f"avg_grounding={avg_grounding:.2f}, "
                    f"{n_with_atoms}/{len(genome_cps)} clusters have atom-level verification."
                )

        if projection.contested:
            top_cont = max(
                projection.contested,
                key=lambda c: c.dissent_pressure,
            )
            rep_c = store.get(top_cont.representative_id)
            if rep_c:
                excerpt = _truncate(rep_c.content, 160)
                lines.append(
                    f"Most contested cluster [{top_cont.representative_id}] "
                    f"(dissent_pressure={top_cont.dissent_pressure:.2f}): {excerpt}"
                )

        # Planner overview — surface the structural reasoning the planner
        # used, so the reader can see *why* this many paragraphs (vs. some
        # other set of clusters). Empty when the planner fell back.
        if plan_notes:
            lines.append("")
            lines.append(f"Synthesis plan: {plan_notes}")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Stage 3: cohesive output — produces the user-facing artifact
    # -----------------------------------------------------------------------

    def _gather_raw_materials(
        self, projection: SynthesisProjection, store: SignalStore,
        cluster_ids: list[str],
    ) -> list[dict]:
        """Pack cluster reps + their best supports as integration-call input.

        Returns a list of {id, rep_content, support_excerpts: [...]} dicts.
        Used by both cohesive_exploration and cohesive_optimization. Reads
        only Signal.content — no reasoning chains, no metadata that could
        leak deposit ordering.
        """
        materials: list[dict] = []
        for cid in cluster_ids:
            cp = next(
                (c for c in projection.surviving + projection.contested
                 if c.representative_id == cid),
                None,
            )
            if cp is None:
                continue
            rep = store.get(cp.representative_id)
            if rep is None:
                continue
            supports: list[str] = []
            for sid in cp.support_set[:4]:
                s = store.get(sid)
                if s and s.content:
                    supports.append(_truncate(s.content, _SUPPORT_CHARS))
                    store.mark_read(sid)
            store.mark_read(cp.representative_id)
            materials.append({
                "id": cp.representative_id,
                "rep_content": _truncate(rep.content, _REPRESENTATIVE_CHARS),
                "support_excerpts": supports,
                "verification_score": cp.verification_score,
                "support_diversity": cp.support_diversity,
            })
        return materials

    def _format_materials_block(self, materials: list[dict]) -> str:
        """Render raw materials into a labeled block the integration call reads."""
        if not materials:
            return "(no raw materials surfaced — synthesizer must fall back to the task prompt alone)"
        lines: list[str] = []
        for m in materials:
            lines.append(f"--- THREAD [{m['id']}] (support_diversity="
                          f"{m['support_diversity']}, "
                          f"verification_score={m['verification_score']:.2f}) ---")
            lines.append(m["rep_content"])
            for ex in m["support_excerpts"]:
                lines.append(f"  · {ex}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _gather_genomes(
        self, projection: SynthesisProjection, cluster_ids: list[str],
    ) -> list:
        """Return ClusterGenome objects for the given cluster IDs (Fix S2, §9b).

        Prefer genome-based intake over raw-materials when genomes are populated.
        Returns empty list when no cluster has a genome — caller should fall back
        to _gather_raw_materials.
        """
        result = []
        all_cps = projection.surviving + projection.contested
        for cid in cluster_ids:
            cp = next((c for c in all_cps if c.representative_id == cid), None)
            if cp is not None and cp.genome is not None:
                result.append((cp, cp.genome))
        return result

    def _format_genome_bundle(self, genome_pairs: list) -> str:
        """Render a structured genome bundle for the integration call (Fix S2, §9b).

        Replaces the flat text dump from _format_materials_block with a compact
        structured form: atoms (id, text, score, source), fitness summary, and
        domain provenance. The integration call reduces over this structure rather
        than re-deriving it from prose.

        No-leak: only AtomFact.text, atom_id, scalar fields, and source_tag
        (no foreign reasoning chains, no signal ancestry text).
        """
        if not genome_pairs:
            return "(no genome bundle available)"
        lines: list[str] = []
        for cp, g in genome_pairs:
            fitness_str = (
                f"composite_fitness={g.composite_fitness:.3f} "
                f"(grounding={g.fitness_breakdown.get('grounding', 0):.2f}, "
                f"stability={g.fitness_breakdown.get('centroid_stability', 0):.2f}, "
                f"llm_judged={g.fitness_breakdown.get('semantic_strength', 0):.2f})"
            )
            lines.append(
                f"--- CLUSTER [{cp.representative_id}] {fitness_str} "
                f"status={cp.status} ---"
            )
            if g.atoms:
                lines.append("ATOMS:")
                for a in g.atoms[:6]:
                    src = f" [src:{a.source_tag}]" if a.source_tag and not a.source_tag.startswith("(") else ""
                    lines.append(
                        f"  [{a.atom_id}] {a.text} "
                        f"(score={a.verification_score:.2f}, w={a.weight:.2f}){src}"
                    )
            kb = g.knowledge_base
            if kb.source_domains:
                lines.append(
                    f"SOURCES: {', '.join(kb.source_domains[:5])} "
                    f"(domain_diversity={kb.domain_diversity:.2f})"
                )
            if g.sensitivity.load_bearing_atoms:
                lines.append(
                    f"LOAD-BEARING ATOMS: {', '.join(g.sensitivity.load_bearing_atoms[:3])}"
                )
            lines.append("")
        return "\n".join(lines).rstrip()

    async def _render_cohesive_exploration(
        self,
        contract: dict,
        projection: SynthesisProjection,
        store: SignalStore,
        render_ids: list[str],
        temperature: float = _INTEGRATE_TEMPERATURE_EXPLORATION,
    ) -> str:
        """Produce ONE unified artifact (haiku/story/plan/etc) from cluster threads.

        The integration call sees:
          - the prompt contract (form, structural constraints, soft cues)
          - the raw materials block (surviving cluster reps + supports)
          - a sharp instruction to produce THE THING, not a description of it.

        Single LLM call. Output is the final user-facing artifact, with
        process notes appended deterministically below. Faithfulness for
        cohesive outputs is graded by the soft audit (see
        _build_cohesive_audit) — content-word overlap and structural
        plausibility, not 4-gram exact match.
        """
        # Fix S2 (§9b): use structured genome bundle when genomes are available.
        # Falls back to raw-materials text dump on legacy paths (no genome).
        genome_pairs = self._gather_genomes(projection, render_ids)
        if genome_pairs:
            materials_block = self._format_genome_bundle(genome_pairs)
        else:
            materials = self._gather_raw_materials(projection, store, render_ids)
            materials_block = self._format_materials_block(materials)

        # Edge-graph cues (improvement 5.4): tell the integration call how
        # threads relate so it can choose transitions deliberately.
        render_ids_set = set(render_ids)
        rendered_clusters = [
            cp for cp in projection.surviving + projection.contested
            if cp.representative_id in render_ids_set
        ]
        edge_block = self._build_edge_cue_block(rendered_clusters, projection)
        edge_section = (
            f"\n{edge_block}\n\nUse these relationships to guide transitions "
            f"between threads — contrast alternatives, weave complements, "
            f"name tensions explicitly.\n"
            if edge_block else ""
        )

        form = contract.get("form", "free_response")
        structural = contract.get("structural", []) or ["(none explicit)"]
        soft = contract.get("soft", []) or ["(none explicit)"]
        length_hint = contract.get("length_hint", "medium")

        struct_block = "\n".join(f"  · {x}" for x in structural)
        soft_block = "\n".join(f"  · {x}" for x in soft)

        prompt = (
            f"USER PROMPT: {self.task_prompt}\n\n"
            f"You are producing the FINAL user-facing artifact. Do not "
            f"describe what the artifact would contain — write the artifact "
            f"itself. The swarm has already explored the conceptual space "
            f"and surfaced the threads below; your job is to integrate them "
            f"into a single coherent work that matches the prompt's form.\n\n"
            f"FORM: {form}\n"
            f"LENGTH HINT: {length_hint}\n\n"
            f"HARD STRUCTURAL CONSTRAINTS (must satisfy):\n{struct_block}\n\n"
            f"SOFT CUES (style/voice/themes):\n{soft_block}\n\n"
            f"RAW MATERIALS — surviving threads from the swarm's exploration. "
            f"These are NOT meant to be quoted verbatim; they are the kernels "
            f"of ideas you should integrate, transform, or build on. The "
            f"final artifact should USE these ideas, not list them.\n\n"
            f"{materials_block}{edge_section}\n"
            f"RULES:\n"
            f"  1. Produce the artifact itself — no preface, no 'Here is...', "
            f"no meta-commentary, no markdown headers unless the form demands them.\n"
            f"  2. Honor every hard structural constraint exactly.\n"
            f"  3. Integrate at least two distinct threads — don't just paraphrase "
            f"the strongest one.\n"
            f"  4. Do NOT invent named entities, citations, or quotes that aren't "
            f"in the raw materials.\n"
            f"  5. No bracketed cluster IDs (e.g. [INITIAL_00023]) in the final "
            f"artifact — those are process metadata.\n\n"
            f"ARTIFACT:"
        )
        raw = await self.llm.generate(
            prompt, role=self.ROLE,
            max_tokens=_INTEGRATE_MAX_TOKENS,
            temperature=temperature,
        )
        return strip_reasoning(raw.strip())

    async def _render_cohesive_optimization(
        self,
        contract: dict,
        projection: SynthesisProjection,
        store: SignalStore,
        render_ids: list[str],
        temperature: float = _INTEGRATE_TEMPERATURE_OPTIMIZATION,
    ) -> str:
        """Produce ONE working implementation from candidate approaches.

        Optimization regime: a correct answer exists. The swarm's surviving
        clusters are candidate approaches; the integration call selects the
        best one (or merges them) and emits a working solution that satisfies
        the spec. Same single-call shape as the exploration path, but lower
        temperature and a spec-shaped instruction.
        """
        # Fix S2 (§9b): use structured genome bundle when genomes are available.
        genome_pairs_opt = self._gather_genomes(projection, render_ids)
        if genome_pairs_opt:
            materials_block = self._format_genome_bundle(genome_pairs_opt)
        else:
            materials = self._gather_raw_materials(projection, store, render_ids)
            materials_block = self._format_materials_block(materials)

        # Edge-graph cues (improvement 5.4): tell the integration call which
        # approaches are alternatives (pick the better), complements (merge),
        # or superseded (prefer the successor).
        render_ids_set_opt = set(render_ids)
        rendered_clusters_opt = [
            cp for cp in projection.surviving + projection.contested
            if cp.representative_id in render_ids_set_opt
        ]
        edge_block_opt = self._build_edge_cue_block(rendered_clusters_opt, projection)
        edge_section_opt = (
            f"\nAPPROACH RELATIONSHIPS:\n{edge_block_opt}\n\n"
            f"Use these to decide: prefer the superseding approach; merge "
            f"complements into one solution; pick the stronger alternative.\n"
            if edge_block_opt else ""
        )

        form = contract.get("form", "function")
        structural = contract.get("structural", []) or ["(none explicit)"]
        soft = contract.get("soft", []) or ["(none explicit)"]

        struct_block = "\n".join(f"  · {x}" for x in structural)
        soft_block = "\n".join(f"  · {x}" for x in soft)

        prompt = (
            f"USER PROMPT: {self.task_prompt}\n\n"
            f"You are producing the FINAL implementation. The swarm has "
            f"surfaced candidate approaches below — your job is to select "
            f"the best one (or merge compatible elements) and write a "
            f"working, correct solution. The output is the implementation "
            f"itself, not commentary about it.\n\n"
            f"FORM: {form}\n\n"
            f"HARD SPEC (signature, complexity bounds, language, behavior):\n"
            f"{struct_block}\n\n"
            f"SOFT GUIDANCE (idioms, style):\n{soft_block}\n\n"
            f"CANDIDATE APPROACHES — surviving threads from the swarm's "
            f"exploration. Pick the strongest, or merge compatible ideas. "
            f"Do NOT include every approach — produce ONE solution.\n\n"
            f"{materials_block}{edge_section_opt}\n"
            f"RULES:\n"
            f"  1. Output the code/solution itself — no preface, no 'Here is...', "
            f"no English commentary outside docstrings/comments.\n"
            f"  2. Match the hard spec exactly (signature, types, complexity).\n"
            f"  3. Include a brief docstring stating the algorithm + complexity.\n"
            f"  4. Handle edge cases the swarm surfaced.\n"
            f"  5. No bracketed cluster IDs in the final code.\n\n"
            f"IMPLEMENTATION:"
        )
        raw = await self.llm.generate(
            prompt, role=self.ROLE,
            max_tokens=_INTEGRATE_MAX_TOKENS,
            temperature=temperature,
        )
        return strip_reasoning(raw.strip())

    # -----------------------------------------------------------------------
    # Alternative-of-the-best artifact (improvement 5.9)
    # -----------------------------------------------------------------------

    async def _render_alternative_artifact(
        self,
        contract: dict,
        projection: SynthesisProjection,
        store: SignalStore,
        alt_cp,
    ) -> str:
        """Generate a second artifact from the strongest non-selected cluster.

        Uses the same integration-call style as cohesive_exploration but
        with a single cluster's thread and an explicit "alternative direction"
        framing. Framed as "the strongest alternative direction the swarm
        explored but did not select as primary."

        Returns empty string on failure.
        """
        rep = store.get(alt_cp.representative_id)
        if rep is None:
            return ""

        form = contract.get("form", "free_response")
        structural = contract.get("structural", []) or ["(none explicit)"]
        soft = contract.get("soft", []) or ["(none explicit)"]
        struct_block = "\n".join(f"  · {x}" for x in structural)
        soft_block = "\n".join(f"  · {x}" for x in soft)

        support_excerpts: list[str] = []
        for sid in alt_cp.support_set[:4]:
            s = store.get(sid)
            if s and s.content:
                support_excerpts.append(_truncate(s.content, _SUPPORT_CHARS))

        thread_block = _truncate(rep.content, _REPRESENTATIVE_CHARS)
        if support_excerpts:
            thread_block += "\n" + "\n".join(f"  · {x}" for x in support_excerpts)

        prompt = (
            f"USER PROMPT: {self.task_prompt}\n\n"
            f"You are producing an ALTERNATIVE artifact — the strongest "
            f"direction the swarm explored but did not select as the primary "
            f"answer. Produce the artifact in the same form as the primary, "
            f"using ONLY the thread below. This is not a critique of the "
            f"primary — it is an independent exploration of a different "
            f"direction that also survived field pressure.\n\n"
            f"FORM: {form}\n\n"
            f"HARD STRUCTURAL CONSTRAINTS:\n{struct_block}\n\n"
            f"SOFT CUES:\n{soft_block}\n\n"
            f"ALTERNATIVE THREAD [{alt_cp.representative_id}] "
            f"(support_diversity={alt_cp.support_diversity}, "
            f"verification_score={alt_cp.verification_score:.2f}):\n"
            f"{thread_block}\n\n"
            f"RULES:\n"
            f"  1. Produce the artifact itself — no preface, no 'Here is...'.\n"
            f"  2. Honor every hard structural constraint.\n"
            f"  3. Do NOT reference 'the primary' or 'the main answer'.\n"
            f"  4. No bracketed cluster IDs in the final artifact.\n\n"
            f"ALTERNATIVE ARTIFACT:"
        )
        try:
            raw = await self.llm.generate(
                prompt, role=self.ROLE,
                max_tokens=_ALTERNATIVE_MAX_TOKENS,
                temperature=_ALTERNATIVE_TEMPERATURE,
            )
            return strip_reasoning((raw or "").strip())
        except Exception as exc:
            print(f"[synthesizer] alternative artifact call failed: "
                  f"{type(exc).__name__}: {exc}")
            return ""

    # -----------------------------------------------------------------------
    # Best-of-N scoring (improvement 5.3)
    # -----------------------------------------------------------------------

    def _score_cohesive_candidate(
        self,
        artifact: str,
        contract: dict,
        projection: SynthesisProjection,
        store: SignalStore,
    ) -> float:
        """Score a candidate cohesive artifact for best-of-N argmax selection.

        score = weighted_coverage_rate − audit_flag_penalty

        Genome-enhanced (when genomes are available):
          weighted_coverage_rate uses composite_fitness as the weight per cluster,
          so high-fitness clusters contribute more to the score than low-fitness
          ones. Coverage checks both representative content AND atom texts, so
          an artifact that uses precise atom propositions scores higher than one
          that only echoes the representative summary.

        Legacy fallback (when no genomes): original uniform coverage rate.

        audit_flag_penalty: number of soft-audit flags * 0.1, capped at 0.5.
        Pure Python — no additional LLM calls.
        """
        _STOP = frozenset({
            "the", "a", "an", "is", "are", "to", "of", "in", "and", "or",
            "it", "be", "that", "this", "with", "for", "on", "as", "by",
            "at", "from", "but", "not",
        })
        artifact_lower = (artifact or "").lower()
        artifact_words = {
            w for w in re.findall(r"[a-z]{4,}", artifact_lower)
            if w not in _STOP
        }

        all_clusters = projection.surviving + projection.contested
        total_weight = 0.0
        covered_weight = 0.0
        has_genomes = any(cp.genome is not None for cp in all_clusters)

        for cp in all_clusters:
            # Genome-aware weight: use composite_fitness when available
            weight = cp.genome.composite_fitness if (cp.genome is not None and has_genomes) else 1.0
            weight = max(0.01, weight)  # floor so zero-fitness clusters still count
            total_weight += weight

            # Coverage: check representative content
            rep = store.get(cp.representative_id)
            rep_covered = False
            if rep:
                rep_words = {
                    w for w in re.findall(r"[a-z]{4,}", rep.content.lower())
                    if w not in _STOP
                }
                rep_covered = bool(rep_words & artifact_words)

            # Genome-aware coverage: also check atom texts (more precise than rep content)
            atom_covered = False
            if cp.genome is not None and cp.genome.atoms:
                for atom in cp.genome.atoms:
                    atom_words = {
                        w for w in re.findall(r"[a-z]{4,}", atom.text.lower())
                        if w not in _STOP
                    }
                    if atom_words & artifact_words:
                        atom_covered = True
                        break

            if rep_covered or atom_covered:
                covered_weight += weight

        coverage_rate = covered_weight / max(1e-9, total_weight)

        try:
            flags = _build_cohesive_audit(artifact, contract, projection, store)
            flag_penalty = min(0.5, len(flags) * 0.1)
        except Exception:
            flag_penalty = 0.0

        return coverage_rate - flag_penalty

    # -----------------------------------------------------------------------
    # Revision loop (improvement 5.2): Self-Refine critic → revise
    # -----------------------------------------------------------------------

    async def _revision_loop(
        self,
        answer: str,
        projection: SynthesisProjection,
        store: SignalStore,
        max_rounds: int = 1,
    ) -> str:
        """K rounds of critic→revise grounded in the surviving evidence.

        The critic is shown the rendered answer and a compact evidence digest
        (cluster rep content only — no reasoning chains) and asked to flag
        three specific pathologies: unsupported claims, vague hedges when the
        evidence leans one way, and parametric bleed (training knowledge not
        in the evidence). The reviser fixes only the flagged paragraphs.

        No-leak rule: critic and reviser see Signal.content only, never other
        agents' ancestry text. Falls back to the original on any failure.
        """
        evidence_lines: list[str] = []
        for cp in (_rank_clusters(projection.surviving) + projection.contested)[:6]:
            rep = store.get(cp.representative_id)
            if rep:
                evidence_lines.append(
                    f"[{cp.representative_id}] "
                    f"(diversity={cp.support_diversity}, "
                    f"ver={cp.verification_score:.2f}, "
                    f"dissent={cp.dissent_pressure:.2f}): "
                    f"{_truncate(rep.content, 300)}"
                )
        evidence_block = "\n".join(evidence_lines) or "(no surviving clusters)"

        current = answer
        for round_i in range(max_rounds):
            # Critic call: identify faithfulness issues grounded in evidence.
            critic_prompt = (
                f"TASK: {self.task_prompt}\n\n"
                f"You are a faithfulness critic. Below is a synthesis answer "
                f"and the surviving cluster evidence it was derived from. "
                f"Identify specific issues with individual paragraphs:\n\n"
                f"  1. UNSUPPORTED: a paragraph cites a cluster ID but says "
                f"     something the evidence does not support.\n"
                f"  2. VAGUE_HEDGE: the paragraph defaults to 'remains "
                f"     contentious' or 'there is debate' when the evidence "
                f"     actually leans one way.\n"
                f"  3. PARAMETRIC_BLEED: the paragraph introduces named "
                f"     entities, examples, or arguments not present in the "
                f"     evidence (model used training knowledge instead of "
                f"     supplied evidence).\n\n"
                f"For each issue: ISSUE TYPE | cited cluster ID | one-sentence "
                f"description. Only report issues you can ground in the "
                f"evidence block — do NOT flag based on your own prior "
                f"knowledge.\n\n"
                f"---EVIDENCE---\n{evidence_block}\n---END EVIDENCE---\n\n"
                f"---ANSWER TO REVIEW---\n{current[:3000]}\n---END ANSWER---\n\n"
                f"ISSUES (bullet list, or 'NO ISSUES' if none found):"
            )
            try:
                critic_raw = await self.llm.generate(
                    critic_prompt, role=self.ROLE,
                    max_tokens=_REVISION_CRITIC_MAX_TOKENS,
                    temperature=_REVISION_TEMPERATURE,
                )
            except Exception as exc:
                print(f"[synthesizer] revision critic failed (round {round_i}): "
                      f"{type(exc).__name__}: {exc}")
                break

            critic_text = strip_reasoning((critic_raw or "").strip())
            if not critic_text or "NO ISSUES" in critic_text.upper()[:40]:
                print(f"[synthesizer] revision round {round_i}: critic found no issues")
                break

            print(f"[synthesizer] revision round {round_i}: "
                  f"critic flagged issues; revising")

            # Revise call: fix only the flagged paragraphs.
            revise_prompt = (
                f"TASK: {self.task_prompt}\n\n"
                f"Fix ONLY the faithfulness issues below in the synthesis "
                f"answer. Do not change paragraphs the critic did not flag. "
                f"Do not introduce content not in the original answer or the "
                f"evidence block. Keep all section headers and citation IDs "
                f"exactly as-is.\n\n"
                f"ISSUES TO FIX:\n{critic_text}\n\n"
                f"EVIDENCE AVAILABLE:\n{evidence_block}\n\n"
                f"ORIGINAL ANSWER:\n{current[:3000]}\n\n"
                f"REVISED ANSWER:"
            )
            try:
                revised_raw = await self.llm.generate(
                    revise_prompt, role=self.ROLE,
                    max_tokens=_REVISION_REVISE_MAX_TOKENS,
                    temperature=_REVISION_TEMPERATURE,
                )
            except Exception as exc:
                print(f"[synthesizer] revision revise failed (round {round_i}): "
                      f"{type(exc).__name__}: {exc}")
                break

            revised = strip_reasoning((revised_raw or "").strip())
            if not revised or len(revised) < len(current) * 0.3:
                print(f"[synthesizer] revision round {round_i}: "
                      f"revised too short ({len(revised)} vs {len(current)}); "
                      f"keeping original")
                break

            current = revised

        return current

    # -----------------------------------------------------------------------
    # Stage B: edge-graph composition of Section 1 (improvement 5.4)
    # -----------------------------------------------------------------------

    def _build_edge_cue_block(
        self,
        clusters: list,
        projection: SynthesisProjection,
    ) -> str:
        """Build transition cues from the typed inter-cluster edge graph.

        Returns an empty string when no edges touch the rendered cluster set —
        callers gate on this to skip the composition call entirely.
        """
        rendered_ids = {cp.representative_id for cp in clusters}
        _RELATION_INSTRUCTION = {
            "alternatives": (
                "contrast these positions; acknowledge both as distinct "
                "directions the evidence supports"
            ),
            "complements": (
                "weave these together; show how each reinforces the other"
            ),
            "tension": (
                "name this tension explicitly: one position is directly "
                "challenged by evidence supporting the other"
            ),
            "supersedes": (
                "foreground the first cluster; mention the second as a prior "
                "or narrower position it builds on"
            ),
            "shared_evidence": (
                "transition smoothly — both draw on overlapping evidence"
            ),
            "co_contested": (
                "both face similar objections; acknowledge this when relevant"
            ),
        }
        lines: list[str] = []
        for e in getattr(projection, "inter_cluster_edges", []):
            if e.source not in rendered_ids or e.target not in rendered_ids:
                continue
            instruction = _RELATION_INSTRUCTION.get(
                e.relation, f"note the {e.relation} relationship"
            )
            lines.append(
                f"  [{e.source}] —[{e.relation}]→ [{e.target}]: {instruction}"
            )
        return (
            "INTER-CLUSTER RELATIONSHIPS (use to guide transitions):\n"
            + "\n".join(lines)
            if lines else ""
        )

    async def _compose_with_edges(
        self,
        paragraphs: list[str],
        clusters: list,
        projection: SynthesisProjection,
        store: SignalStore,
    ) -> str:
        """Stage B: compose rendered paragraphs using typed inter-cluster edges.

        Inserts ONE bridge sentence between pairs of paragraphs where a typed
        edge relationship exists. Paragraphs are preserved verbatim; only
        bridges are added (and optional reordering to put related clusters
        adjacent). Falls back to a plain join on any failure.

        No-leak rule: the composition call sees rendered paragraph text
        (already derived from Signal.content) and edge-type labels. No
        raw signal content or ancestry metadata is introduced here.
        """
        if len(paragraphs) <= 1:
            return "\n\n".join(paragraphs)

        edge_cue_block = self._build_edge_cue_block(clusters, projection)
        if not edge_cue_block:
            return "\n\n".join(paragraphs)

        labeled_paras = [
            f"PARAGRAPH [{cp.representative_id}]:\n{para}"
            for cp, para in zip(clusters, paragraphs)
        ]
        labeled_block = "\n\n".join(labeled_paras)

        prompt = (
            f"TASK: {self.task_prompt}\n\n"
            f"You are composing a POSITION SYNTHESIS section from the "
            f"cluster paragraphs below. Each paragraph was independently "
            f"rendered for one surviving cluster. Your job:\n"
            f"  1. Reorder paragraphs if edge relationships suggest a "
            f"     better reading order (alternatives side by side; "
            f"     superseded cluster after its successor).\n"
            f"  2. Insert ONE bridge sentence between paragraph pairs where "
            f"     a typed edge exists (see INTER-CLUSTER RELATIONSHIPS).\n"
            f"  3. Preserve all paragraphs verbatim — do NOT rewrite, "
            f"     summarize, or merge them. Only add bridges.\n\n"
            f"{edge_cue_block}\n\n"
            f"Bridge sentence vocabulary:\n"
            f"  · alternatives  → 'In contrast,...' / 'A competing direction...'\n"
            f"  · complements   → 'Complementing this,...' / 'Reinforcing this...'\n"
            f"  · tension       → 'This position is directly challenged by...'\n"
            f"  · supersedes    → 'Superseding the earlier position...' / "
            f"'Building on and extending...'\n"
            f"  · shared_evidence → 'Drawing on similar evidence,...'\n\n"
            f"PARAGRAPHS TO COMPOSE:\n\n{labeled_block}\n\n"
            f"RULES:\n"
            f"  1. Preserve all [SIGNAL_ID] citation tags exactly as written.\n"
            f"  2. Do not produce markdown headers — output ONLY the composed "
            f"paragraphs and bridge sentences as prose.\n"
            f"  3. Do not introduce content not present in the paragraphs.\n\n"
            f"COMPOSED OUTPUT:"
        )

        try:
            raw = await self.llm.generate(
                prompt, role=self.ROLE,
                max_tokens=_EDGE_COMPOSE_MAX_TOKENS,
                temperature=_RENDERER_TEMPERATURE,
            )
        except Exception as exc:
            print(f"[synthesizer] edge composition call failed: "
                  f"{type(exc).__name__}: {exc}; falling back to plain join")
            return "\n\n".join(paragraphs)

        result = strip_reasoning((raw or "").strip())
        plain = "\n\n".join(paragraphs)
        if not result or len(result) < len(plain) * 0.4:
            print(f"[synthesizer] edge composition output implausibly short; "
                  f"falling back to plain join")
            return plain

        return result

    # -----------------------------------------------------------------------
    # Debate frame for `alternatives` cluster sets (improvement 5.5)
    # -----------------------------------------------------------------------

    def _identify_debate_clusters(
        self,
        rendered_clusters: list,
        projection: SynthesisProjection,
    ) -> list[list]:
        """Find groups of `alternatives` clusters comparable enough to debate.

        Returns a list of groups. Each group is a list of ClusterProjection
        objects that are mutually connected by `alternatives` edges and within
        _DEBATE_PRIORITY_RATIO of each other's _cluster_priority score.
        Groups with < 2 members are excluded.
        """
        rendered_ids = {cp.representative_id for cp in rendered_clusters}
        id_to_cp = {cp.representative_id: cp for cp in rendered_clusters}

        # Build adjacency: only `alternatives` edges between rendered clusters.
        neighbours: dict[str, set[str]] = {
            cp.representative_id: set() for cp in rendered_clusters
        }
        for e in getattr(projection, "inter_cluster_edges", []):
            if (e.relation == "alternatives"
                    and e.source in rendered_ids
                    and e.target in rendered_ids):
                neighbours[e.source].add(e.target)
                neighbours[e.target].add(e.source)

        # Find connected components among `alternatives` neighbours.
        visited: set[str] = set()
        groups: list[list] = []
        for cid in rendered_ids:
            if cid in visited:
                continue
            component: list[str] = []
            queue = [cid]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                queue.extend(neighbours[node] - visited)
            if len(component) < 2:
                continue
            # Priority-filter: keep only clusters within _DEBATE_PRIORITY_RATIO
            # of the highest-priority member.
            cps = [id_to_cp[c] for c in component]
            max_prio = max(_cluster_priority(cp) for cp in cps)
            close = [
                cp for cp in cps
                if _cluster_priority(cp) >= max_prio * _DEBATE_PRIORITY_RATIO
            ]
            if len(close) >= 2:
                groups.append(close)
        return groups

    async def _render_debate_frame(
        self,
        debate_group: list,
        store: SignalStore,
    ) -> str:
        """Three-round debate between `alternatives` cluster positions.

        Round 1: each cluster generates a position paragraph.
        Round 2: each cluster responds to the strongest sibling Round-1 para.
        Round 3: a judge call reads all positions + responses and names the
                 unresolved empirical question that distinguishes the alternatives.

        Returns the Round-3 judge output (the debate-rendered Section-1 block).
        Falls back to empty string on any failure so the caller can fall back
        to standard `_render_cluster_position` rendering.
        """
        if not debate_group:
            return ""

        # Round 1: per-cluster position
        round1: list[tuple[str, str]] = []  # (cluster_id, position_paragraph)
        for cp in debate_group:
            rep = store.get(cp.representative_id)
            if rep is None:
                continue
            content = _truncate(rep.content, _REPRESENTATIVE_CHARS)
            support_lines = []
            for sid in cp.support_set[:3]:
                s = store.get(sid)
                if s:
                    support_lines.append(
                        f"  - [{sid}] {_truncate(s.content, _SUPPORT_CHARS)}"
                    )
            support_block = (
                "\nSupporting evidence:\n" + "\n".join(support_lines)
                if support_lines else ""
            )
            prompt = (
                f"TASK: {self.task_prompt}\n\n"
                f"You are arguing FOR the following position in a structured "
                f"debate between alternative approaches. Write ONE focused "
                f"paragraph that: (a) states the specific claim clearly, "
                f"(b) cites the strongest supporting evidence, (c) anticipates "
                f"the central objection to this position.\n\n"
                f"Position [{cp.representative_id}]: {content}\n"
                f"{support_block}\n\n"
                f"POSITION PARAGRAPH:"
            )
            try:
                raw = await self.llm.generate(
                    prompt, role=self.ROLE,
                    max_tokens=_DEBATE_ROUND_MAX_TOKENS,
                    temperature=_DEBATE_TEMPERATURE,
                )
                para = strip_reasoning((raw or "").strip())
                if para:
                    round1.append((cp.representative_id, para))
            except Exception as exc:
                print(f"[synthesizer] debate round1 failed for "
                      f"{cp.representative_id}: {exc}")

        if len(round1) < 2:
            return ""

        # Round 2: each cluster responds to the strongest sibling's Round-1 para.
        round2: list[tuple[str, str]] = []
        for idx, (cid, _own_para) in enumerate(round1):
            # "Strongest sibling" = highest priority among other round-1 participants.
            sibling_paras = [
                (sid, para) for (sid, para) in round1 if sid != cid
            ]
            if not sibling_paras:
                continue
            _, strongest_sibling_para = sibling_paras[0]  # already priority-ordered
            prompt = (
                f"TASK: {self.task_prompt}\n\n"
                f"In a structured debate, respond to the following competing "
                f"position. Address its strongest specific claim directly. "
                f"Do not restate your own position in full — focus the response "
                f"on what distinguishes your approach from theirs.\n\n"
                f"Your cluster ID: [{cid}]\n\n"
                f"Competing position:\n{strongest_sibling_para}\n\n"
                f"RESPONSE (one focused paragraph):"
            )
            try:
                raw = await self.llm.generate(
                    prompt, role=self.ROLE,
                    max_tokens=_DEBATE_ROUND_MAX_TOKENS,
                    temperature=_DEBATE_TEMPERATURE,
                )
                response = strip_reasoning((raw or "").strip())
                if response:
                    round2.append((cid, response))
            except Exception as exc:
                print(f"[synthesizer] debate round2 failed for {cid}: {exc}")

        # Round 3: judge reads all positions + responses and names the unresolved
        # empirical or structural question that would distinguish the alternatives.
        positions_block = "\n\n".join(
            f"Position [{cid}]:\n{para}" for cid, para in round1
        )
        responses_block = "\n\n".join(
            f"Response [{cid}]:\n{resp}" for cid, resp in round2
        ) if round2 else "(no responses generated)"

        judge_prompt = (
            f"TASK: {self.task_prompt}\n\n"
            f"You are a judge reading a structured debate between alternative "
            f"approaches. Read all positions and responses, then write ONE "
            f"paragraph that:\n"
            f"  1. Names the specific empirical or structural question that "
            f"     remains unresolved after the exchange — the question that, "
            f"     if answered, would decide between the approaches.\n"
            f"  2. Characterizes what evidence or test would resolve it.\n"
            f"  3. Does NOT collapse the alternatives into a single answer or "
            f"     declare a winner. The alternatives remain open.\n\n"
            f"Do not hedge with 'both have merit' without specifying what "
            f"the actual unresolved question is.\n\n"
            f"POSITIONS:\n\n{positions_block}\n\n"
            f"RESPONSES:\n\n{responses_block}\n\n"
            f"JUDGE PARAGRAPH (name the unresolved question precisely):"
        )
        try:
            raw = await self.llm.generate(
                judge_prompt, role=self.ROLE,
                max_tokens=_DEBATE_JUDGE_MAX_TOKENS,
                temperature=_DEBATE_TEMPERATURE,
            )
            judge_para = strip_reasoning((raw or "").strip())
        except Exception as exc:
            print(f"[synthesizer] debate round3 (judge) failed: {exc}")
            judge_para = ""

        if not judge_para:
            return ""

        # Assemble: present both positions, the exchange, and the judge's finding.
        parts = [
            "**[DEBATE FRAME — alternatives cluster set]**\n",
        ]
        for cid, para in round1:
            parts.append(f"*Position [{cid}]:*\n{para}")
        if round2:
            parts.append("\n*Exchange:*\n")
            for cid, resp in round2:
                parts.append(f"[{cid}] responds: {resp}")
        parts.append(f"\n*Judge (unresolved question):*\n{judge_para}")
        return "\n\n".join(parts)

    # -----------------------------------------------------------------------
    # Per-cluster render helpers
    # -----------------------------------------------------------------------

    async def _render_cluster_position(
        self,
        cp: ClusterProjection,
        store: SignalStore,
        projection: Optional[SynthesisProjection] = None,
    ) -> str:
        """Render a one-paragraph position statement for one surviving cluster.

        Prompt is anti-parametric: it forbids framing introductions (the
        "the debate over X..." pathology where every paragraph restates the
        same setup) and forces the paragraph to LEAD with the cluster's
        specific claim. On non-factual tasks (debate / analysis /
        problem_solving) it also explicitly asks the renderer to characterize
        which way the supplied evidence actually points rather than hedging
        with "remains contentious."
        """
        rep = store.get(cp.representative_id)
        if rep is None:
            return ""

        # Fix S3 (§9b): template-first render for high-confidence clusters.
        # When ALL atoms have verification_score ≥ 0.60 AND no meaningful dissent
        # AND task type is factual (not creative/debate where prose nuance matters),
        # render deterministically without an LLM call. Reserve llm.generate() for
        # clusters that genuinely need prose smoothing or that have dissent to address.
        task_type_s3 = getattr(self, "_task_type", None)
        _TEMPLATE_VER_FLOOR = 0.60
        _TEMPLATE_DISSENT_MAX = 0.30
        if (
            cp.genome is not None
            and cp.genome.atoms
            and task_type_s3 not in {"creative", "debate"}
            and cp.dissent_pressure < _TEMPLATE_DISSENT_MAX
            and all(a.verification_score >= _TEMPLATE_VER_FLOOR
                    for a in cp.genome.atoms)
        ):
            atom_stmts = []
            for a in cp.genome.atoms:
                src = f" [{a.source_tag}]" if a.source_tag and not a.source_tag.startswith("(") else ""
                atom_stmts.append(f"{a.text}{src}")
            unver_pfx = ""
            if cp.unverified and task_type_s3 not in {"debate", "analysis", "problem_solving", "creative"}:
                unver_pfx = " (not externally verified)"
            template_para = (
                f"[{cp.representative_id}] "
                + " ".join(atom_stmts)
                + unver_pfx
                + f" (support_diversity={cp.support_diversity},"
                f" verification={cp.verification_score:.2f})"
                + "."
            )
            print(
                f"[RENDER] template-first {cp.representative_id}: "
                f"n_atoms={len(cp.genome.atoms)} "
                f"min_score={min(a.verification_score for a in cp.genome.atoms):.2f}"
            )
            return template_para

        content = _truncate(rep.content, _REPRESENTATIVE_CHARS)
        # Only annotate "not externally verified" on factual tasks. For debate
        # / analysis / problem_solving the projection no longer flags unverified
        # for absent web confirmation (sources don't corroborate interpretive
        # claims) — appending the phrase would just be reflexive hedging.
        task_type = getattr(self, "_task_type", None)
        is_non_factual = task_type in {"debate", "analysis", "problem_solving", "creative"}
        unver_note = (
            " (not externally verified)"
            if (cp.unverified and not is_non_factual) else ""
        )

        # Gather support excerpts (up to 3)
        support_lines: list[str] = []
        for sid in cp.support_set[:3]:
            s = store.get(sid)
            if s:
                support_lines.append(
                    f"  - [{sid}] {_truncate(s.content, _SUPPORT_CHARS)}"
                )

        # Validator grounding: read VERIFICATION signal contents deposited
        # during exploration instead of re-fetching Wikipedia at synthesis
        # time. validate_parse deposits the validator's one-sentence reasoning
        # as VERIFICATION content (core/actions.py:537); this is richer than
        # a Wikipedia snippet because it reflects judgement about this specific
        # claim. Falls back to Wikipedia only when no validators ran (e.g.
        # creative tasks that suppress the Validator role).
        validator_notes: list[str] = []
        for vid in cp.verification_set[:3]:
            vsig = store.get(vid)
            if vsig and vsig.content:
                note = _truncate(vsig.content, _EXTERNAL_CHARS)
                validator_notes.append(
                    f"[validator score={vsig.strength:.2f}] {note}"
                )
                store.mark_read(vid)
        if validator_notes:
            ext_block = (
                "\n[Validator notes from exploration]:\n"
                + "\n".join(f"  · {n}" for n in validator_notes)
                + "\n"
            )
        elif cp.genome is not None and cp.genome.atoms:
            # Prefer genome atom verification scores over live Wikipedia lookup.
            # Removes the synchronous network round-trip from the serial chain.
            verified = [
                a for a in cp.genome.atoms
                if a.source_tag and not a.source_tag.startswith("(")
            ]
            if verified:
                atom_lines = [
                    f"  [{a.source_tag}] (score={a.verification_score:.2f}): {a.text}"
                    for a in verified[:3]
                ]
                ext_block = (
                    "\n[External grounding from atom verification]:\n"
                    + "\n".join(atom_lines) + "\n"
                )
            else:
                ext_block = ""
        else:
            ext_ctx = _get_external_context(rep.content)
            ext_block = (
                f"\n[External context, not agent-derived]: {ext_ctx}\n"
                if ext_ctx else ""
            )

        partition_note = (
            f"Partition origins: {', '.join(cp.partition_origins)}"
            if cp.partition_origins else ""
        )

        support_block = (
            "\nSupporting evidence:\n" + "\n".join(support_lines)
            if support_lines else ""
        )

        # Dissent injection: if surviving cluster has meaningful field pressure,
        # include the strongest dissent signal and ask for an acknowledgement.
        dissent_block = ""
        if cp.dissent_pressure > 0.5 and cp.dissent_set:
            strongest = _strongest_signal_content(cp.dissent_set, store)
            if strongest:
                dissent_block = (
                    f"\nCounter-position (dissent_pressure={cp.dissent_pressure:.2f}): "
                    f"{_truncate(strongest, _DISSENT_CHARS)}\n"
                    f"Briefly acknowledge this counter-position in one sentence "
                    f"at the end of the paragraph.\n"
                )

        # Trajectory context: if iter_at_deposit data is available, tell the
        # renderer whether this claim survived scrutiny or was never challenged.
        traj = getattr(cp, "trajectory", None)
        traj_block = ""
        if traj is not None and traj.has_trajectory:
            traj_parts = []
            if traj.iter_first_dissent > 0 and traj.dissent_response_lag > 0:
                traj_parts.append(
                    f"challenged at iteration {traj.iter_first_dissent}, "
                    f"field responded {traj.dissent_response_lag} iteration(s) later "
                    f"— render this as a position that survived scrutiny"
                )
            elif traj.iter_first_dissent == 0:
                traj_parts.append("accumulated support without direct challenge")
            if traj.objection_survival > 0:
                traj_parts.append(
                    f"{traj.objection_survival} unanswered objection(s) — "
                    f"acknowledge remaining uncertainty"
                )
            if traj_parts:
                traj_block = f"\nField trajectory: {'; '.join(traj_parts)}.\n"

        # Adjacency context (improvement 5.10): structural metrics of clusters
        # connected to this one via the typed edge graph. The renderer sees
        # ONLY metrics (support_diversity, verification_score, relation type),
        # never the neighboring cluster's content — enough to orient the prose
        # without importing cross-cluster material. Skipped when projection is
        # None (backward compat) or no adjacent clusters are in the render set.
        adj_block = ""
        if projection is not None:
            adj_lines: list[str] = []
            all_proj_cps = {
                c.representative_id: c
                for c in projection.surviving + projection.contested
            }
            for e in getattr(projection, "inter_cluster_edges", []):
                if e.relation not in ("complements", "alternatives",
                                      "tension", "supersedes"):
                    continue
                neighbour_id = None
                if e.source == cp.representative_id and e.target in all_proj_cps:
                    neighbour_id = e.target
                    direction = "this cluster →"
                elif e.target == cp.representative_id and e.source in all_proj_cps:
                    neighbour_id = e.source
                    direction = "→ this cluster"
                if neighbour_id is None:
                    continue
                neighbour_cp = all_proj_cps[neighbour_id]
                adj_lines.append(
                    f"  [{e.relation}] [{neighbour_id}] "
                    f"(support_diversity={neighbour_cp.support_diversity}, "
                    f"ver={neighbour_cp.verification_score:.2f}, "
                    f"dissent={neighbour_cp.dissent_pressure:.2f})"
                )
            if adj_lines:
                adj_block = (
                    "\nAdjacent cluster relationships (structural only — "
                    "use to orient your prose, not to import their content):\n"
                    + "\n".join(adj_lines) + "\n"
                )

        # Sensitivity annotation (topology-lattice overhaul): inject robustness
        # signal from _build_sensitivities() when available. Tells the renderer
        # whether this cluster's survival is load-bearing on a few key supports
        # (fragile) or spread across many (robust). Purely structural metadata —
        # no other cluster's content is exposed.
        sensitivity_block = ""
        if projection is not None:
            sens = getattr(projection, "cluster_sensitivities", {}).get(
                cp.representative_id
            )
            if sens is not None:
                robustness = getattr(sens, "support_removal_robustness", None)
                load_bearing = getattr(sens, "load_bearing_supports", [])
                competing = getattr(sens, "competing_takeover", None)
                topo_gap = getattr(sens, "topology_uncovered_on_removal", [])
                parts: list[str] = []
                if robustness is not None:
                    parts.append(f"robustness={robustness:.2f}")
                if load_bearing:
                    parts.append(
                        f"load_bearing=[{', '.join(load_bearing[:2])}]"
                    )
                if competing:
                    parts.append(f"competing_cluster=[{competing}]")
                if topo_gap:
                    parts.append(
                        f"topology_gap_on_removal={len(topo_gap)}_cell(s)"
                    )
                if parts:
                    sensitivity_block = (
                        f"\nSensitivity: {', '.join(parts)}.\n"
                    )

        # Position-taking instruction varies by task type. Non-factual tasks
        # need an actual stance on the supplied evidence; factual tasks need
        # to stick to what's verifiable. Both forbid the "the debate over X"
        # framing intro that produced 5 near-duplicate paragraphs.
        if is_non_factual:
            stance_instruction = (
                f"Open the paragraph with the specific position this cluster "
                f"advances — do NOT begin with framing prose like 'The debate "
                f"over...', 'The question of...', 'X is a contested issue...'. "
                f"State what the supplied evidence ACTUALLY argues, then trace "
                f"why the supporting signals back it up. If the evidence leans "
                f"toward one side, say so; do NOT default to 'remains contentious' "
                f"unless the supplied signals themselves are split."
            )
        else:
            stance_instruction = (
                f"Open the paragraph with the specific claim — do NOT begin "
                f"with framing prose ('The debate over...', 'X has long been...'). "
                f"State the claim directly and trace why the supporting "
                f"signals back it up."
            )

        prompt = (
            f"TASK: {self.task_prompt}\n\n"
            f"Synthesize the following surviving claim into one focused "
            f"paragraph. Cite [{cp.representative_id}] inline.\n\n"
            f"HARD RULES (failure to follow = the paragraph is unusable):\n"
            f"  1. Use ONLY the supplied claim, supporting evidence, and "
            f"counter-position below. Do NOT introduce concepts, examples, "
            f"theories, or arguments not present in the input (no parametric "
            f"knowledge bleed — no quantum-physics gestures, no thinkers not "
            f"already cited).\n"
            f"  2. {stance_instruction}\n"
            f"  3. Do not repeat the task framing across paragraphs.\n\n"
            f"Claim [{cp.representative_id}]: {content}{unver_note}\n"
            f"support_diversity={cp.support_diversity}  "
            f"verification_score={cp.verification_score:.2f}\n"
            f"{support_block}"
            f"{dissent_block}"
            f"{traj_block}"
            f"{adj_block}"
            f"{sensitivity_block}"
            f"{ext_block}"
            f"{partition_note}\n\n"
            f"PARAGRAPH:"
        )

        print(
            f"[RENDER] position cluster {cp.representative_id}: "
            f"support_diversity={cp.support_diversity}, "
            f"verification_score={cp.verification_score:.2f}, "
            f"dissent_pressure={cp.dissent_pressure:.2f}"
        )
        raw = await self.llm.generate(
            prompt, role=self.ROLE,
            max_tokens=MAX_TOKENS_SYNTHESIZER,
            temperature=_RENDERER_TEMPERATURE,
        )
        result = strip_reasoning(raw.strip())
        valid_ids = (
            {cp.representative_id}
            | set(cp.member_ids)
            | set(cp.support_set)
            | set(cp.dissent_set)
            | set(cp.verification_set)
        )
        result = _strip_hallucinated_citations(result, valid_ids, cp.representative_id)
        result = _apply_word_budget(
            result, RENDER_POSITION_MAX_WORDS, cp.representative_id, "position"
        )
        return result

    async def _render_cluster_dissent(
        self, cp: ClusterProjection, store: SignalStore
    ) -> str:
        """Render a one-paragraph dissent summary for one contested or challenged cluster."""
        rep = store.get(cp.representative_id)
        if rep is None:
            return ""

        content = _truncate(rep.content, _REPRESENTATIVE_CHARS)
        strongest_dissent = _strongest_signal_content(cp.dissent_set, store)
        counter = (
            _truncate(strongest_dissent, _DISSENT_CHARS)
            if strongest_dissent else "(no explicit counter-position deposited)"
        )

        prompt = (
            f"TASK: {self.task_prompt}\n\n"
            f"Write one paragraph describing the following {'contested' if cp.status == 'contested' else 'challenged'} "
            f"claim, the strongest counter-position, and what evidence would resolve the dispute. "
            f"Cite [{cp.representative_id}] inline. Do not take sides.\n\n"
            f"Claim [{cp.representative_id}] (status={cp.status}, "
            f"dissent_pressure={cp.dissent_pressure:.2f}): {content}\n\n"
            f"Strongest counter-position: {counter}\n\n"
            f"PARAGRAPH:"
        )

        print(
            f"[RENDER] dissent cluster {cp.representative_id}: "
            f"status={cp.status}, "
            f"dissent_pressure={cp.dissent_pressure:.2f}, "
            f"n_dissent={len(cp.dissent_set)}"
        )
        raw = await self.llm.generate(
            prompt, role=self.ROLE,
            max_tokens=MAX_TOKENS_SYNTHESIZER,
            temperature=_RENDERER_TEMPERATURE,
        )
        result = strip_reasoning(raw.strip())
        valid_ids = (
            {cp.representative_id}
            | set(cp.member_ids)
            | set(cp.support_set)
            | set(cp.dissent_set)
            | set(cp.verification_set)
        )
        result = _strip_hallucinated_citations(result, valid_ids, cp.representative_id)
        result = _apply_word_budget(
            result, RENDER_DISSENT_MAX_WORDS, cp.representative_id, "dissent"
        )
        return result


# ---------------------------------------------------------------------------
# Post-generation citation stamping (Section 4 — deterministic)
# ---------------------------------------------------------------------------

def _stamp_citations(
    answer: str,
    projection: SynthesisProjection,
    store: SignalStore,
    merge_groups: Optional[list] = None,
) -> str:
    """Append a deterministic CITATIONS block to the rendered answer.

    Each surviving/contested cluster gets one entry:
        CLAIM [INITIAL_00023]:
          Supports:      SUPPORT_00041, SUPPORT_00058
          Challenges:    CRITIQUE_00047
          Verifications: VERIFICATION_00062
          Partition:     partition_2
          Coverage:      2 partition(s) contributed to this cluster

    When the planner identified merge groups, those appear as a trailing
    block so the reader sees which clusters were treated as one position.
    """
    active = projection.surviving + projection.contested
    if not active:
        return answer

    lines = ["", "", "## 4. CITATIONS", "=" * 60]
    for cp in active:
        rep = store.get(cp.representative_id)
        excerpt = _truncate(rep.content, 80) if rep else cp.representative_id
        status_tag = f"[{cp.status.upper()}]" if cp.status != "surviving" else ""
        lines.append(f"CLAIM {status_tag} [{cp.representative_id}]: {excerpt}")
        if cp.support_set:
            lines.append(f"  Supports:      {', '.join(cp.support_set[:8])}")
        if cp.dissent_set:
            lines.append(f"  Challenges:    {', '.join(cp.dissent_set[:5])}")
        if cp.verification_set:
            lines.append(f"  Verifications: {', '.join(cp.verification_set[:5])}")
        if cp.partition_origins:
            lines.append(f"  Partition:     {', '.join(cp.partition_origins)}")
        lines.append(
            f"  support_diversity={cp.support_diversity}  "
            f"dissent_pressure={cp.dissent_pressure:.2f}  "
            f"verification_score={cp.verification_score:.2f}"
        )
        lines.append("")

    # Merge groups: planner-identified equivalent clusters.
    if merge_groups:
        lines.append("MERGED POSITIONS (planner identified as same claim):")
        for grp in merge_groups:
            lines.append(f"  - {' <=> '.join(grp)}")
        lines.append("")

    return answer.rstrip() + "\n".join(lines)


# ---------------------------------------------------------------------------
# Faithfulness audit
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[([A-Z]+_\d+)\]")


def _apply_word_budget(text: str, max_words: int, cluster_id: str, label: str) -> str:
    """Fix R: truncate a rendered paragraph to at most max_words words.

    Logs [RENDER GUARD] loudly so Colab output reveals every truncation event.
    Adds terminal punctuation when the cut lands mid-sentence.
    """
    words = text.split()
    if len(words) > max_words:
        print(
            f"[RENDER GUARD] {label} cluster {cluster_id}: "
            f"{len(words)} words > {max_words}; truncating to {max_words} words"
        )
        text = " ".join(words[:max_words])
        if text and text[-1] not in ".!?\")'":
            text += "."
    return text


def _strip_hallucinated_citations(
    text: str, valid_ids: set, cluster_id: str
) -> str:
    """Fix R: remove [SIGNAL_XXXXX] tags not belonging to this cluster.

    The per-cluster renderer is shown only this cluster's signals, so any
    [ID] tag for an ID outside the cluster's representative, members,
    supports, dissent set, and verifications is hallucinated. Strip it and
    log [RENDER GUARD] loudly.
    """
    def _replace(m):
        cid = m.group(1)
        if cid not in valid_ids:
            print(
                f"[RENDER GUARD] cluster {cluster_id}: stripping hallucinated "
                f"citation [{cid}] — not in cluster signal IDs"
            )
            return ""
        return m.group(0)
    return _CITATION_RE.sub(_replace, text)


def _build_cohesive_audit(
    artifact: str,
    contract: dict,
    projection: SynthesisProjection,
    store: SignalStore,
) -> list[dict]:
    """Soft audit for cohesive outputs (creative / coding / problem_solving).

    The hard 4-gram overlap audit doesn't fit a haiku — by design the
    artifact transforms cluster content rather than quoting it. Instead
    we check:

      1. content_word_overlap_low: fewer than K distinct content words from
         surviving cluster reps appear in the artifact. Flags total
         hallucination — the integration call ignored its source material.
      2. structural_violation: hard constraints from contract.structural
         that can be checked deterministically (line count, presence of
         a specific token, expected length range).
      3. fabricated_id_in_artifact: a bracketed [INITIAL_XXXXX]-style tag
         leaked into the artifact (process metadata in user-facing output).
      4. length_implausible: artifact is way shorter / longer than the
         length_hint suggests (e.g. a "haiku" that's 500 chars).

    Returns the same flag-list shape as _build_faithfulness_audit so the
    audit writer / summary.json reader doesn't need to branch.
    """
    flags: list[dict] = []
    _STOP = frozenset({
        "the","a","an","is","are","to","of","in","and","or","it","be",
        "that","this","with","for","on","as","by","at","from","but","not",
    })
    _CITATION_TAG = re.compile(r"\[[A-Z]+_\d+\]")

    rep_word_pool: set[str] = set()
    for cp in projection.surviving + projection.contested:
        rep = store.get(cp.representative_id)
        if not rep:
            continue
        for w in re.findall(r"[a-z]{4,}", rep.content.lower()):
            if w not in _STOP:
                rep_word_pool.add(w)

    artifact_words: set[str] = set()
    for w in re.findall(r"[a-z]{4,}", (artifact or "").lower()):
        if w not in _STOP:
            artifact_words.add(w)
    overlap = rep_word_pool & artifact_words
    # Threshold: at least 4 distinct content words from surviving threads
    # should appear in a non-trivial artifact. Haikus often dip below this;
    # we only flag if the artifact is longer than ~100 chars (i.e. not a
    # form where lexical brevity is the whole point).
    if len(artifact or "") > 100 and len(overlap) < 4:
        flags.append({
            "issue": "content_word_overlap_low",
            "overlap_count": len(overlap),
            "rep_pool_size": len(rep_word_pool),
            "artifact_length": len(artifact or ""),
        })

    tag_hits = _CITATION_TAG.findall(artifact or "")
    if tag_hits:
        flags.append({
            "issue": "fabricated_id_in_artifact",
            "tags": tag_hits[:10],
        })

    length_hint = contract.get("length_hint", "medium")
    artifact_len = len(artifact or "")
    # Loose per-hint bounds. Calibrated against form-typical lengths;
    # crosses-the-line cases (a "short" creative artifact 2000+ chars,
    # or a "long" one under 200) are the ones worth surfacing.
    if length_hint == "short" and artifact_len > 1500:
        flags.append({
            "issue": "length_implausible",
            "detail": f"length={artifact_len} exceeds 'short' bound (1500)",
        })
    elif length_hint == "long" and artifact_len < 300:
        flags.append({
            "issue": "length_implausible",
            "detail": f"length={artifact_len} below 'long' floor (300)",
        })

    # Form-specific structural checks. Conservative — only fire on signals
    # we can verify cheaply and unambiguously.
    form = contract.get("form", "").lower()
    if form == "haiku":
        # A haiku is three lines. We don't enforce syllable count (that
        # requires phonetic analysis), but line count we can check.
        non_empty_lines = [ln for ln in (artifact or "").splitlines() if ln.strip()]
        if non_empty_lines and len(non_empty_lines) != 3:
            flags.append({
                "issue": "structural_violation",
                "form": "haiku",
                "detail": f"expected 3 non-empty lines, got {len(non_empty_lines)}",
            })
    elif form in ("function", "code"):
        # For code: try ast.parse if it looks like Python.
        if any(tok in (artifact or "") for tok in ("def ", "class ", "import ")):
            try:
                import ast as _ast
                _ast.parse(artifact or "")
            except SyntaxError as exc:
                flags.append({
                    "issue": "structural_violation",
                    "form": form,
                    "detail": f"python syntax error: {exc}",
                })
    return flags


def _build_faithfulness_audit(
    answer: str,
    projection: SynthesisProjection,
    store: SignalStore,
) -> list[dict]:
    """Check each cited cluster ID in the prose for faithfulness.

    Two checks per cited ID:
      1. Existence: the ID must correspond to a real signal in the store.
         Fabricated IDs (e.g. [OPP_00145] that never existed) are flagged.
      2. 4-gram overlap: for surviving/contested cluster reps, the paragraph
         must share at least one 4-word sequence with that cluster's content.
         Catches wrong-cluster citation and hallucinated prose.

    Also raises post-hoc flags for decoder pathology (these don't reject —
    they make the audit file useful for cross-run comparison):
      3. orphan_think_tag: paragraph contains a literal </think> or <think>.
      4. scratchpad_in_prose: paragraph matches the scratchpad marker regex.
      5. truncated_mid_sentence: paragraph ends without terminal punctuation
         and is followed by a section break (double newline).
    """
    from agents.base import _SCRATCHPAD_RE

    cluster_content: dict[str, str] = {}
    # Genome atom texts: list of lowercased atom texts per cluster (for the
    # genome-enhanced overlap check that replaces the 4-gram rep-content check
    # when atom texts are available — more precise and fewer false positives).
    cluster_atom_texts: dict[str, list[str]] = {}
    for cp in projection.surviving + projection.contested:
        rep = store.get(cp.representative_id)
        if rep:
            cluster_content[cp.representative_id] = rep.content.lower()
        if cp.genome is not None and cp.genome.atoms:
            cluster_atom_texts[cp.representative_id] = [
                a.text.lower() for a in cp.genome.atoms if a.text
            ]

    # Build the complete set of real signal IDs from the store (for existence check).
    valid_signal_ids: set[str] = {s.id for s in store.all()}

    flags: list[dict] = []
    paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]

    # Track which section each paragraph belongs to so prose-only checks
    # (truncation, short-paragraph) don't fire on Section 3 list items or
    # Section 4 citation blocks — both are deterministic structured output
    # that lacks terminal punctuation by design and produced ~22 false
    # positives per run before this gate.
    paragraph_sections: list[str] = []
    current_section = "preamble"
    for para in paragraphs:
        # Detect section headers. Synthesizer emits them as a Markdown H2
        # with a digit prefix ("## 1. POSITION SYNTHESIS") or the literal
        # "## EXECUTIVE SUMMARY".
        stripped = para.lstrip()
        if stripped.startswith("## EXECUTIVE SUMMARY") or stripped.startswith("# EXECUTIVE SUMMARY"):
            current_section = "exec_summary"
        elif stripped.startswith("## 1.") or "POSITION SYNTHESIS" in stripped.split("\n", 1)[0]:
            current_section = "section_1"
        elif stripped.startswith("## 2.") or "OPEN QUESTIONS" in stripped.split("\n", 1)[0]:
            current_section = "section_2"
        elif stripped.startswith("## 3.") or "CONSIDERED AND FILTERED" in stripped.split("\n", 1)[0]:
            current_section = "section_3"
        elif stripped.startswith("## 4.") or "CITATIONS" in stripped.split("\n", 1)[0]:
            current_section = "section_4"
        paragraph_sections.append(current_section)
    # Prose-only sections: truncation + short-paragraph checks run only here.
    _PROSE_SECTIONS = {"preamble", "exec_summary", "section_1", "section_2"}

    for i, para in enumerate(paragraphs):
        cited_ids = _CITATION_RE.findall(para)
        section_name = paragraph_sections[i]
        is_prose = section_name in _PROSE_SECTIONS

        for cid in cited_ids:
            # Check 1: ID must exist in the signal store (fabrication check).
            if cid not in valid_signal_ids:
                # Grab up to 200 chars of surrounding context.
                excerpt_start = max(0, answer.find(para) - 50)
                surrounding = answer[excerpt_start: excerpt_start + 200]
                flags.append({
                    "issue": "fabricated citation",
                    "cited_id": cid,
                    "paragraph_excerpt": surrounding,
                })
                continue

            # Check 2: overlap check between paragraph and cited cluster content.
            # Primary: 4-gram overlap against representative content.
            # Fallback (genome-enhanced): 3-gram overlap against any atom text.
            # An atom-text match is sufficient — atoms are precise propositions;
            # if the paragraph correctly paraphrases an atom, no flag needed.
            if cid not in cluster_content:
                continue  # not a cluster rep — skip overlap check
            para_lower = para.lower()

            # Primary: 4-gram overlap against representative content
            cluster_words = cluster_content[cid].split()
            found_overlap = (len(cluster_words) >= 4 and any(
                " ".join(cluster_words[j: j + 4]) in para_lower
                for j in range(len(cluster_words) - 3)
            ))

            # Genome fallback: 3-gram overlap against any atom text
            if not found_overlap and cid in cluster_atom_texts:
                for atom_text in cluster_atom_texts[cid]:
                    atom_words = atom_text.split()
                    if len(atom_words) >= 3 and any(
                        " ".join(atom_words[j: j + 3]) in para_lower
                        for j in range(len(atom_words) - 2)
                    ):
                        found_overlap = True
                        break

            if not found_overlap and len(cluster_words) >= 4:
                flags.append({
                    "issue": "no 4-gram overlap between paragraph and cited cluster content",
                    "cited_id": cid,
                    "paragraph_excerpt": para[:300],
                    "cluster_content_excerpt": cluster_content[cid][:300],
                })

        # Check 3: orphan think tags in prose (post-hoc, no rejection).
        if "</think>" in para or "<think>" in para.lower():
            flags.append({
                "issue": "orphan_think_tag",
                "paragraph_excerpt": para[:300],
            })

        # Check 4: scratchpad marker survived into prose.
        if _SCRATCHPAD_RE.search(para):
            flags.append({
                "issue": "scratchpad_in_prose",
                "paragraph_excerpt": para[:300],
            })

        # Check 5: truncated mid-sentence (no terminal punctuation before section break).
        # Prose sections only. Section 3 (list) and Section 4 (citation block)
        # are deterministic structured output that lacks terminal punctuation
        # by design — running this check there produced ~22 FPs/run all
        # pointing at the trailing `support_diversity=...` line of each
        # citation entry.
        if is_prose and i < len(paragraphs) - 1:
            last_char = para.rstrip()[-1] if para.rstrip() else ""
            if last_char not in {".", "!", "?", '"', ")", "'"}:
                flags.append({
                    "issue": "truncated_mid_sentence",
                    "paragraph_excerpt": para[-200:],
                    "section": section_name,
                })

        # Check 6: suspiciously short paragraph (likely incomplete render).
        # Prose only — Section 3 list entries are legitimately short.
        if is_prose and cited_ids and len(para) < 50:
            flags.append({
                "issue": "suspiciously_short_paragraph",
                "paragraph_excerpt": para,
                "length": len(para),
                "section": section_name,
            })

    return flags


def _write_faithfulness_audit(flags: list[dict], output_dir: Path) -> None:
    audit = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_flags": len(flags),
        "flags": flags,
    }
    try:
        (output_dir / "renderer_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        if flags:
            print(
                f"[synthesizer] faithfulness audit: {len(flags)} flag(s) "
                f"written to renderer_audit.json"
            )
        else:
            print("[synthesizer] faithfulness audit: 0 flags (all citations faithful)")
    except Exception as exc:
        print(f"[synthesizer] could not write renderer_audit.json: {exc}")


def _write_no_consensus_audit(output_dir: Path) -> None:
    """No-consensus short-circuit: there's no rendered answer to audit, so
    write total_flags=0 (audit ran clean) rather than leaving the file
    absent — which downstream reads as the -1 'unknown' sentinel."""
    audit = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_flags": 0,
        "flags": [],
        "reason": "no_consensus",
    }
    try:
        (output_dir / "renderer_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        print(f"[synthesizer] could not write no-consensus renderer_audit.json: {exc}")


def _write_crashed_audit(output_dir: Path, exc: Exception) -> None:
    """Audit crashed mid-call. Write total_flags=-2 with an audit_error
    field so downstream summary.audit_flags is unambiguous (-2 = crashed,
    not -1 = file missing for unknown reason)."""
    audit = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_flags": -2,
        "flags": [],
        "audit_error": f"{type(exc).__name__}: {exc}",
    }
    try:
        (output_dir / "renderer_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
    except Exception as exc2:
        print(f"[synthesizer] could not write crashed-audit renderer_audit.json: {exc2}")


# ---------------------------------------------------------------------------
# External grounding at synthesis time
# ---------------------------------------------------------------------------

def _get_external_context(content: str) -> Optional[str]:
    """Best-effort Wikipedia snippet for a cluster's representative claim.

    Injected into the Section 1 renderer prompt as "[External context]" so it's
    distinguishable from agent-deposited content. Returns None on any failure.
    """
    try:
        keyphrase = _extract_keyphrase(content, max_words=4)
        snippet = _wiki_lookup(keyphrase)
        if not snippet or "(no Wikipedia" in snippet or "(wikipedia package" in snippet:
            return None
        return snippet[:_EXTERNAL_CHARS]
    except Exception:
        return None


def _extract_keyphrase(text: str, max_words: int = 5) -> str:
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
        "and", "or", "but", "for", "on", "at", "by", "with", "as",
        "that", "this", "these", "those", "it", "its",
    }
    words = [w.strip(".,;:?!\"'()[]") for w in text.split()]
    keep = [w for w in words if w and w.lower() not in stop]
    return " ".join(keep[:max_words]) or text[:50]


def _wiki_lookup(query: str) -> str:
    try:
        import wikipedia  # type: ignore
        try:
            return wikipedia.summary(query, sentences=2, auto_suggest=True, redirect=True)[:600]
        except Exception:
            try:
                hits = wikipedia.search(query, results=1)
                if hits:
                    return wikipedia.summary(hits[0], sentences=2)[:600]
            except Exception:
                pass
        return f"(no Wikipedia article found for query: {query!r})"
    except ImportError:
        return f"(wikipedia package not installed; query was: {query!r})"


# ---------------------------------------------------------------------------
# Citation helpers
# ---------------------------------------------------------------------------

def _citation_tag(cp: ClusterProjection) -> str:
    """Build inline citation tag like [INITIAL_00023 ← SUPPORT_00041 | CRITIQUE_00047]."""
    support_part = ", ".join(cp.support_set[:5])
    dissent_part = ", ".join(cp.dissent_set[:3])
    ver_part = ", ".join(cp.verification_set[:3])

    parts = []
    if support_part:
        parts.append(support_part)
    tag_inner = cp.representative_id
    if parts:
        tag_inner += " ← " + " | ".join(parts)
    if dissent_part:
        tag_inner += f" | {dissent_part}"
    if ver_part:
        tag_inner += f" | {ver_part}"
    return f"[{tag_inner}]"


def _build_citations(
    projection: SynthesisProjection,
    store: SignalStore,
) -> dict:
    """Build citations.json content mapping signal IDs to provenance metadata."""
    result: dict = {}

    all_clusters = (
        projection.surviving
        + projection.contested
        + projection.weakly_supported
        + projection.rejected_by_field
    )

    for cp in all_clusters:
        all_ids = (
            [cp.representative_id]
            + cp.member_ids
            + cp.support_set
            + cp.dissent_set
            + cp.verification_set
        )
        for sid in all_ids:
            if sid in result:
                continue
            sig = store.get(sid)
            if sig is None:
                continue
            result[sid] = {
                "depositor_role": sig.depositor,
                "depositor_agent_id": sig.metadata.get("depositor_agent_id", ""),
                "scout_agent_id": sig.metadata.get("scout_agent_id", ""),
                "partition_origin": _parse_partition_tag_from_sig(sig),
                "deposit_timestamp": sig.timestamp,
                "content_excerpt": sig.content[:200],
                "chunk_ids": sig.metadata.get("chunk_ids", []),
                "parent_id": sig.parent_id,
                "signal_type": sig.type,
                "strength": round(sig.strength, 4),
            }

    return result


def _parse_partition_tag_from_sig(sig) -> str:
    scout_id = sig.metadata.get("scout_agent_id", "")
    if not scout_id:
        return ""
    parts = scout_id.split("_")
    if len(parts) >= 3:
        return f"partition_{parts[2]}"
    return ""


# ---------------------------------------------------------------------------
# Lineage DOT graph
# ---------------------------------------------------------------------------

def _build_lineage_dot(
    projection: SynthesisProjection,
    store: SignalStore,
) -> str:
    """Build a Graphviz DOT string for the surviving + contested DAG."""
    lines = ["digraph lineage {", "  rankdir=TB;", "  node [fontsize=10];"]

    active_clusters = projection.surviving + projection.contested

    node_attrs = {
        INITIAL:           'shape=box style=filled fillcolor="#d0e8ff"',
        SUPPORT:           'shape=ellipse style=filled fillcolor="#d0f0d0"',
        CRITIQUE_POSITIVE: 'shape=diamond style=filled fillcolor="#d0ffd0"',
        CRITIQUE_NEGATIVE: 'shape=diamond style=filled fillcolor="#ffd0d0"',
        CRITIQUE:          'shape=diamond style=filled fillcolor="#ffd0d0"',  # legacy
        OBJECTION:         'shape=diamond style=filled fillcolor="#ffb0b0"',
        VERIFICATION:      'shape=hexagon style=filled fillcolor="#ffffd0"',
    }

    seen_nodes: set[str] = set()
    seen_edges: set[tuple] = set()

    def add_node(sig_id: str) -> None:
        if sig_id in seen_nodes:
            return
        seen_nodes.add(sig_id)
        sig = store.get(sig_id)
        if sig is None:
            return
        attr = node_attrs.get(sig.type, "")
        label = f"{sig_id}\\nstr={sig.strength:.2f}"
        lines.append(f'  "{sig_id}" [label="{label}" {attr}];')

    def add_edge(parent_id: str, child_id: str) -> None:
        if (parent_id, child_id) in seen_edges:
            return
        seen_edges.add((parent_id, child_id))
        lines.append(f'  "{child_id}" -> "{parent_id}";')

    for cp in active_clusters:
        all_ids = (
            cp.member_ids
            + cp.support_set
            + cp.dissent_set
            + cp.verification_set
        )
        for sid in all_ids:
            add_node(sid)
            sig = store.get(sid)
            if sig and sig.parent_id:
                add_node(sig.parent_id)
                add_edge(sig.parent_id, sid)

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> Optional[str]:
    """Return the first complete {...} block from text using brace counting.

    The greedy regex r"\\{[\\s\\S]*\\}" fails when the model appends a comment,
    a trailing explanation, or a second JSON object after the real one —
    it grabs from the first '{' to the LAST '}', producing an invalid span.
    This function stops at the matching closing brace of the FIRST '{'.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _repair_json(s: str) -> str:
    """Fix the most common LLM JSON formatting mistakes before parsing."""
    # Trailing commas before } or ] — the most frequent failure mode.
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _truncate(text: str, max_chars: int = _REPRESENTATIVE_CHARS) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _strongest_signal_content(signal_ids: list, store: SignalStore) -> str:
    best = None
    best_strength = -1.0
    for sid in signal_ids:
        s = store.get(sid)
        if s and s.strength > best_strength:
            best_strength = s.strength
            best = s.content
    return best or ""
