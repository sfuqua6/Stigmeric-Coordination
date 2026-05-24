"""Layer 1 projection: pure-Python DAG analysis, no LLM.

Takes the final signal store and computes a structured summary used
by the synthesizer's Layer 2 LLM renderer. The renderer is then given
a pre-digested projection — it cannot rank, judge, or invent.

Key concepts
------------
- ClusterProjection: one group of semantically related INITIAL signals plus
  all SUPPORT / CRITIQUE / OBJECTION / VERIFICATION descendants.
- SynthesisProjection: the full set of clusters, tagged by survival status.
- Survival filter — four buckets, evaluated in order:

    1. rejected_by_field  dissent_pressure > 1.5
    2. weakly_supported   support_diversity < 2  (not enough distinct forager
                          strategies; typically caused by cluster fragmentation
                          when the clustering threshold is too tight)
    3. contested          0.5 <= dissent_pressure <= 1.5
    4. surviving          everything else

  Verification is a *flag* on surviving clusters (cp.unverified=True), not a
  separate bucket. A well-supported, low-dissent claim that no validator
  happened to reach is still rendered — with an "(not externally verified)"
  note — rather than being silently suppressed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .signal_store import SignalStore
from .signal_types import (
    INITIAL, SUPPORT, CRITIQUE, CRITIQUE_POSITIVE, CRITIQUE_NEGATIVE,
    OBJECTION, VERIFICATION,
)
from .config import (
    BOOST_THRESHOLD, CLUSTER_SIM_THRESHOLD,
    SURVIVAL_MIN_SUPPORT_DIVERSITY, SURVIVAL_REJECT_DISSENT_PRESSURE,
    SURVIVAL_CONTEST_MIN, SURVIVAL_CONTEST_MAX,
    SURVIVAL_VERIFY_MIN, SURVIVAL_BROAD_SUPPORT,
    SURVIVAL_TASK_PROFILES, SURVIVAL_DEFAULT_PROFILE,
    RENDER_K, DISSENT_K, PLANNER_MMR_LAMBDA,
    VERIFICATION_WEIGHT, DISSENT_WEIGHT,
)

# Module-private alias preserved for back-compat with existing code paths
# and tests. Sourced from config.CLUSTER_SIM_THRESHOLD which is tier-aware:
# 0.55 on laptop (quantized-model diversity at 0.65 fragments into 17 narrow
# clusters; a looser threshold yields 4–5 broad ones), 0.72 on Colab where
# fp16 Qwen-Instruct produces sufficiently distinct claims.
_CLUSTER_SIM_THRESHOLD = CLUSTER_SIM_THRESHOLD
# Phase 5: tightened from 0.75 → 0.85. The old threshold matched semantically
# related but distinct clusters across runs, dragging dissent_pressure into
# topics the prior KB didn't actually address. At 0.85 only near-duplicate
# representative claims trigger KB carry-over.
_KB_MATCH_THRESHOLD = 0.85      # similarity above which a prior KB entry is applied
_KB_REJECTION_PENALTY = 0.5     # added to dissent_pressure for prior-rejected clusters
_EPS = 1e-9


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClusterProjection:
    representative_id: str
    member_ids: list[str]
    support_set: list[str]     # SUPPORT signal IDs on lineage
    dissent_set: list[str]     # CRITIQUE + OBJECTION signal IDs on lineage
    verification_set: list[str]
    support_diversity: int     # distinct strategy names among support depositors
    dissent_pressure: float    # sum(dissent_str) / max(eps, sum(support_str))
    verification_score: float  # mean VERIFICATION strength on lineage
    partition_origins: list[str]  # partition tags of originating scouts
    # Phase 3C: max SUPPORT->SUPPORT chain depth in this cluster. Captures
    # how deeply agents have iteratively developed a claim via PARENT
    # proposals (depth 1 = flat forest of direct SUPPORT children only).
    support_depth: int = 1
    status: str = "unclassified"  # set by _apply_survival_filter
    unverified: bool = False      # True when surviving but no validator reached it


@dataclass
class SynthesisProjection:
    surviving: list[ClusterProjection] = field(default_factory=list)
    contested: list[ClusterProjection] = field(default_factory=list)
    weakly_supported: list[ClusterProjection] = field(default_factory=list)
    rejected_by_field: list[ClusterProjection] = field(default_factory=list)
    # Passes structural checks (not rejected, not weakly_supported, not contested)
    # but does not clear the credibility gate (no verification, no field dissent,
    # and only moderate support_diversity). Distinguished from weakly_supported
    # so the renderer can describe why it was held back.
    unverified: list[ClusterProjection] = field(default_factory=list)
    partition_coverage: dict = field(default_factory=dict)  # partition_tag -> count
    no_consensus: bool = False


@dataclass
class SynthesisPlan:
    """Structured output of the Python-level cluster selection planner (Fix P).

    Built by build_plan() from a SynthesisProjection using composite scoring +
    MMR diversity selection. No LLM involved — pure Python DAG analysis.

    render_clusters:  representative IDs for Section 1 (full position render)
    dissent_clusters: representative IDs for Section 2 (dissent / open questions)
    held_clusters:    high-scoring but too similar to a render cluster → Section 3
    demoted_clusters: low composite score or weakly supported → Section 3
    planner_notes:    one-line description of the selection rationale
    """
    render_clusters: list[str]
    dissent_clusters: list[str]
    held_clusters: list[str]
    demoted_clusters: list[str]
    planner_notes: str = ""


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_projection(
    store: SignalStore,
    has_validators: bool = True,
    prior_rejections: Optional[list[dict]] = None,
    prior_consensus: Optional[list[dict]] = None,
    task_type: Optional[str] = None,
) -> SynthesisProjection:
    """Build the Layer 1 projection from the live signal store.

    prior_rejections: list of KB entries from prior runs that were rejected.
        Matching new clusters have dissent_pressure raised by _KB_REJECTION_PENALTY.
    prior_consensus: list of KB entries from prior surviving clusters.
        Matching new clusters get their support_diversity boosted.
    """
    initials = store.by_type(INITIAL)
    if not initials:
        return SynthesisProjection(no_consensus=True)

    # Build per-initial metrics
    initial_metrics = {}
    for sig in initials:
        initial_metrics[sig.id] = _compute_initial_metrics(sig, store)

    # Cluster by content similarity using cached embeddings
    clusters = _cluster_initials(initials, store)

    # Aggregate metrics per cluster
    cluster_projections = []
    for rep_id, member_ids in clusters:
        cp = _aggregate_cluster(rep_id, member_ids, initial_metrics, store)
        cluster_projections.append(cp)

    # Apply prior KB penalties / boosts
    if prior_rejections or prior_consensus:
        _apply_kb(cluster_projections, store, prior_rejections, prior_consensus)

    # Apply survival filter. The task profile decides whether external
    # verification is required for credibility: factual tasks demand it,
    # non-factual tasks (debate / analysis / etc.) accept chain depth +
    # broad support as internal-coherence credibility instead.
    #
    # When task_type is None, fall through to the legacy factual gate so
    # existing callers (tests, KB import paths) don't change behavior.
    if task_type is None:
        task_profile = None  # preserves the pre-task-profile gate
    else:
        task_profile = SURVIVAL_TASK_PROFILES.get(
            task_type, SURVIVAL_DEFAULT_PROFILE,
        )
    for cp in cluster_projections:
        _apply_survival_filter(cp, has_validators=has_validators,
                                task_profile=task_profile)

    proj = SynthesisProjection()
    for cp in cluster_projections:
        if cp.status == "surviving":
            proj.surviving.append(cp)
        elif cp.status == "contested":
            proj.contested.append(cp)
        elif cp.status == "weakly_supported":
            proj.weakly_supported.append(cp)
        elif cp.status == "rejected_by_field":
            proj.rejected_by_field.append(cp)
        elif cp.status == "unverified":
            proj.unverified.append(cp)
        else:
            proj.weakly_supported.append(cp)  # fallback for unclassified

    # Partition coverage: count surviving + contested signals per partition
    for cp in (proj.surviving + proj.contested):
        for pt in cp.partition_origins:
            proj.partition_coverage[pt] = proj.partition_coverage.get(pt, 0) + 1

    proj.no_consensus = len(proj.surviving) + len(proj.contested) == 0
    return proj


def build_plan(
    projection: SynthesisProjection,
    store: SignalStore,
    render_k: int = RENDER_K,
    dissent_k: int = DISSENT_K,
    lam: float = PLANNER_MMR_LAMBDA,
    verification_weight: float = VERIFICATION_WEIGHT,
    dissent_weight: float = DISSENT_WEIGHT,
) -> SynthesisPlan:
    """Deterministic Python planner: composite score + MMR cluster selection (Fix P).

    Composite score per cluster:
        score = support_diversity
                + verification_weight * verification_score
                - dissent_weight * dissent_pressure

    render_clusters: top render_k surviving clusters by MMR (diversity-aware).
    dissent_clusters: top dissent_k from contested + surviving-with-dissent,
                      sorted by dissent_pressure descending.
    held_clusters: survived composite score floor but were within MMR distance
                   of a selected render cluster (redundant, not weak).
    demoted_clusters: remaining surviving + all other non-render clusters.

    Falls back to top-k by composite score when embeddings are unavailable.
    Logs [PLAN] for diagnostics; always returns a valid SynthesisPlan.
    """
    surviving = list(projection.surviving)
    contested = list(projection.contested)

    # --- Score all surviving clusters ---
    def _score(cp: ClusterProjection) -> float:
        return (cp.support_diversity
                + verification_weight * cp.verification_score
                - dissent_weight * cp.dissent_pressure)

    scored = sorted(surviving, key=_score, reverse=True)

    # --- MMR selection for render_clusters ---
    picked_ids: list[str] = []
    picked_embs: list = []
    held: list[ClusterProjection] = []

    for cp in scored:
        if len(picked_ids) >= render_k:
            held.append(cp)
            continue
        emb = store.get_embedding(cp.representative_id)
        if emb is None:
            # No embedding: take if slot available, else hold
            picked_ids.append(cp.representative_id)
            picked_embs.append(None)
            continue
        # MMR: skip if too similar to any already-picked cluster
        if picked_embs:
            max_sim = max(
                (_cosine_sim(emb, pe) for pe in picked_embs if pe is not None),
                default=0.0,
            )
            mmr_score = lam * _score(cp) - (1.0 - lam) * max_sim
            # Also compute what the next-best would score with no diversity penalty
            pure_score = lam * _score(cp)
            # If MMR score is very negative (near-duplicate of a picked cluster),
            # route to held rather than demote — it's a strong signal, just redundant
            if max_sim >= (1.0 - _SECTION_1_DIVERSITY_MIN_DIST):
                held.append(cp)
                continue
        picked_ids.append(cp.representative_id)
        picked_embs.append(emb)

    # --- Dissent cluster selection ---
    dissent_candidates: list[ClusterProjection] = list(contested)
    dissent_candidates += [
        cp for cp in surviving if cp.dissent_set and cp.representative_id not in set(picked_ids)
    ]
    dissent_candidates.sort(key=lambda c: c.dissent_pressure, reverse=True)
    dissent_ids = [c.representative_id for c in dissent_candidates[:dissent_k]]

    # --- Demoted: surviving clusters not in render or dissent ---
    rendered_or_dissent = set(picked_ids) | set(dissent_ids)
    held_ids = [cp.representative_id for cp in held if cp.representative_id not in rendered_or_dissent]
    demoted_ids = [
        cp.representative_id for cp in surviving
        if cp.representative_id not in rendered_or_dissent
        and cp.representative_id not in set(held_ids)
    ]

    notes = (
        f"render={len(picked_ids)} (of {len(surviving)} surviving, "
        f"k={render_k}), dissent={len(dissent_ids)}, "
        f"held={len(held_ids)}, demoted={len(demoted_ids)}"
    )
    print(
        f"[PLAN] {notes} | "
        f"contested={len(contested)} render_ids={picked_ids[:3]}"
    )

    assert len(picked_ids) <= render_k, (
        f"[FIX P] render_clusters={len(picked_ids)} > render_k={render_k}"
    )

    return SynthesisPlan(
        render_clusters=picked_ids,
        dissent_clusters=dissent_ids,
        held_clusters=held_ids,
        demoted_clusters=demoted_ids,
        planner_notes=notes,
    )


# Used by build_plan to mirror the synthesizer's diversity distance constant.
_SECTION_1_DIVERSITY_MIN_DIST = 0.35


# ---------------------------------------------------------------------------
# Per-initial metrics
# ---------------------------------------------------------------------------

def _compute_initial_metrics(sig, store: SignalStore) -> dict:
    """Collect all descendants of an INITIAL signal and compute metrics.

    Phase 3C: also computes `support_depth` — the maximum SUPPORT-only chain
    length under this INITIAL. Depth 1 means at least one direct SUPPORT
    child; depth 2 means at least one SUPPORT under a SUPPORT; etc. The
    PARENT-proposal mechanism turns this from a flat forest into chains.
    """
    support_ids = []
    dissent_ids = []
    ver_ids = []
    # support_depth via BFS: track depth-of-each-node from the INITIAL.
    # Only follow SUPPORT links so chains through critiques don't inflate
    # the metric.
    support_depth_max = 0

    visited: set[str] = set()
    # (child_id, parent_depth) — parent_depth is the running SUPPORT-chain
    # length at the parent. Root parent is the INITIAL with depth 0.
    queue: list[tuple[str, int]] = [(cid, 0) for cid in store.by_parent(sig.id)]
    while queue:
        child_id, parent_depth = queue.pop()
        if child_id in visited:
            continue
        visited.add(child_id)
        child = store.get(child_id)
        if child is None:
            continue
        if child.type == SUPPORT or child.type == CRITIQUE_POSITIVE:
            support_ids.append(child_id)
            new_depth = parent_depth + 1 if child.type == SUPPORT else parent_depth
            support_depth_max = max(support_depth_max, new_depth)
        elif child.type in (CRITIQUE_NEGATIVE, CRITIQUE, OBJECTION):
            dissent_ids.append(child_id)
            new_depth = parent_depth
        elif child.type == VERIFICATION:
            ver_ids.append(child_id)
            new_depth = parent_depth
        else:
            new_depth = parent_depth
        for grandchild in store.by_parent(child_id):
            queue.append((grandchild, new_depth))

    # support_diversity: count distinct "perspectives" contributing support.
    # Legacy pipeline used strategy_name parsed from agent_id like
    # `forager_R1_2_under_supported_clusters`. Worker pool agent_ids are
    # `worker_001` (no strategy suffix), so the legacy parse returns "" for
    # every worker deposit and support_diversity is permanently 0 —
    # which traps every cluster at the weakly_supported gate. The new
    # diversity dimension reads metadata["action"] (SUPPORT can come from
    # DEVELOP / CHAIN / REFINE / CRITIQUE_POSITIVE), falling back to the
    # legacy strategy parse so legacy runs and the round/phase scheduler
    # still work unchanged.
    strategy_names: set[str] = set()
    for sid in support_ids:
        s = store.get(sid)
        if s is None:
            continue
        action = s.metadata.get("action", "")
        if action:
            strategy_names.add(action)
            continue
        strategy = _parse_strategy_name(s.metadata.get("depositor_agent_id", ""))
        if strategy:
            strategy_names.add(strategy)
    support_diversity = len(strategy_names)

    # dissent_pressure
    support_strengths = [store.get(i).strength for i in support_ids if store.get(i)]
    dissent_strengths = [store.get(i).strength for i in dissent_ids if store.get(i)]
    dissent_pressure = sum(dissent_strengths) / max(_EPS, sum(support_strengths))

    # verification_score
    ver_score = store.verification_strength(sig.id)

    # partition origin — prefer the explicit field (Fix A), fall back to
    # metadata key (pre-Fix-A deposits), then legacy agent_id parse.
    partition_tag = (
        sig.partition_id
        or sig.metadata.get("partition_id", "")
        or _parse_partition_tag(sig.metadata.get("scout_agent_id", ""))
    )

    return {
        "support_ids": support_ids,
        "dissent_ids": dissent_ids,
        "ver_ids": ver_ids,
        "support_diversity": support_diversity,
        "dissent_pressure": dissent_pressure,
        "ver_score": ver_score,
        "partition_tag": partition_tag,
        "support_depth": max(1, support_depth_max),
    }


def _parse_strategy_name(agent_id: str) -> str:
    """Extract strategy name from agent_id like 'forager_R1_2_stratified_extremes'."""
    parts = agent_id.split("_")
    # agent_ids: forager_R{r}_{i}_{strategy_name}
    # strategy portion starts after index 2 (0=role, 1=R{r}, 2={i})
    if len(parts) >= 4:
        return "_".join(parts[3:])
    return ""


def _parse_partition_tag(scout_agent_id: str) -> str:
    """Extract partition tag from scout_agent_id like 'scout_R1_2' -> 'partition_2'."""
    parts = scout_agent_id.split("_")
    # scout_R{r}_{i}: parts[0]=scout, parts[1]=R{r}, parts[2]={i}
    if len(parts) >= 3:
        return f"partition_{parts[2]}"
    return "partition_unknown"


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two already-normalized vectors."""
    return float(sum(x * y for x, y in zip(a, b)))


def _cluster_initials(
    initials: list,
    store: SignalStore,
) -> list[tuple[str, list[str]]]:
    """Return clusters for INITIAL signals.

    Prefers ClusterRegistry (Fix C') when populated — clusters are computed
    incrementally at deposit time and are always consistent with the store.
    Falls back to greedy single-linkage when the registry has no clusters
    (pre-Fix-C' stores, or when no embedder is available).

    Returns list of (representative_id, [member_ids]).
    """
    # Fix C': try the registry first
    live = store.get_live_clusters(INITIAL)
    if live:
        return live

    # Legacy fallback: greedy single-linkage at _CLUSTER_SIM_THRESHOLD.
    sorted_initials = sorted(initials, key=lambda s: s.strength, reverse=True)
    clusters: list[tuple[str, list[str], Optional[list[float]]]] = []

    for sig in sorted_initials:
        emb = store.get_embedding(sig.id)
        assigned = False
        if emb is not None:
            for i, (rep_id, members, rep_emb) in enumerate(clusters):
                if rep_emb is not None:
                    sim = _cosine_sim(emb, rep_emb)
                    if sim >= _CLUSTER_SIM_THRESHOLD:
                        members.append(sig.id)
                        assigned = True
                        break
        if not assigned:
            clusters.append((sig.id, [sig.id], emb))

    return [(rep_id, members) for rep_id, members, _ in clusters]


# ---------------------------------------------------------------------------
# Cluster aggregation
# ---------------------------------------------------------------------------

def _aggregate_cluster(
    rep_id: str,
    member_ids: list[str],
    initial_metrics: dict,
    store: SignalStore,
) -> ClusterProjection:
    """Merge per-initial metrics into a single ClusterProjection."""
    all_support: list[str] = []
    all_dissent: list[str] = []
    all_ver: list[str] = []
    all_partitions: list[str] = []
    seen_support: set[str] = set()
    seen_dissent: set[str] = set()
    seen_ver: set[str] = set()

    for mid in member_ids:
        m = initial_metrics.get(mid, {})
        for sid in m.get("support_ids", []):
            if sid not in seen_support:
                seen_support.add(sid)
                all_support.append(sid)
        for sid in m.get("dissent_ids", []):
            if sid not in seen_dissent:
                seen_dissent.add(sid)
                all_dissent.append(sid)
        for sid in m.get("ver_ids", []):
            if sid not in seen_ver:
                seen_ver.add(sid)
                all_ver.append(sid)
        pt = m.get("partition_tag", "")
        if pt and pt not in all_partitions:
            all_partitions.append(pt)

    # Recompute aggregate dissent_pressure and support_diversity over cluster.
    #
    # Interaction-density age weighting + youth grace.
    # ------------------------------------------------
    # Raw count of dissents produced a ranking artifact in m13: claims
    # deposited early accumulated objections purely because they had more
    # wall-clock TIME to be seen. The fix that landed first used a wall-
    # clock half-life — that helped, but baked in survivor bias against
    # young signals (they hadn't lived long enough to grow EITHER supports
    # or dissents). Under the interaction-density model we weight each
    # contrarian / supporter by how many pool iterations have ALREADY
    # elapsed since it was deposited. Signals deposited in busy windows
    # age fast (lots of opportunity to be reviewed); signals deposited in
    # quiet windows stay young. The "youth grace" floor exempts very-new
    # signals from contributing to dissent_pressure at all — they get a
    # window to mature before being judged.
    #
    # Fallback: when the SignalStore was never told about iterations
    # (legacy round-based pipeline, isolated tests) the iter_at_deposit
    # field is 0 on every signal and current_iter is 0, so iter_age is 0
    # for everyone. We detect that and drop back to the wall-clock half-
    # life calculation so nothing regresses.
    from .config import MIN_INTERACTIONS_BEFORE_DECAY, INTERACTION_AGE_HALFLIFE

    support_sigs = [store.get(i) for i in all_support if store.get(i)]
    dissent_sigs = [store.get(i) for i in all_dissent if store.get(i)]
    support_strengths = [s.strength for s in support_sigs]
    dissent_strengths = [s.strength for s in dissent_sigs]

    current_iter = store.get_iteration() if hasattr(store, "get_iteration") else 0
    use_iteration_age = current_iter > 0 and any(
        s.iter_at_deposit > 0 for s in (support_sigs + dissent_sigs)
    )

    if use_iteration_age:
        # Interaction-density path. Age = iterations elapsed since deposit.
        # Newborns (iter_age < MIN_INTERACTIONS_BEFORE_DECAY) are dropped
        # entirely from the dissent_pressure calculation so they can't
        # be judged before they've had a fair shot. They DO count toward
        # the support side once they exist — what we're protecting them
        # from is being penalized as a CHALLENGE before they've existed
        # long enough to be challenged.
        def _iter_weight(sig) -> float:
            age = max(0, current_iter - sig.iter_at_deposit)
            if age < MIN_INTERACTIONS_BEFORE_DECAY:
                # Youth grace on dissent attribution. For supporters we
                # still want them counted at full weight as soon as they
                # land — that's how a fresh cluster bootstraps.
                return 0.0 if sig.type in (
                    "CRITIQUE_NEGATIVE", "OBJECTION",
                ) else 1.0
            return 0.5 ** (age / INTERACTION_AGE_HALFLIFE)

        weighted_dissent = sum(s.strength * _iter_weight(s) for s in dissent_sigs)
        weighted_support = sum(s.strength * _iter_weight(s) for s in support_sigs)
    else:
        # Wall-clock fallback (legacy path). Reference time = median support
        # timestamp; same half-life math as the prior fix.
        import time as _time
        if support_sigs:
            ref_ts = sorted(s.timestamp for s in support_sigs)[len(support_sigs) // 2]
        elif dissent_sigs:
            ref_ts = sorted(s.timestamp for s in dissent_sigs)[len(dissent_sigs) // 2]
        else:
            ref_ts = _time.time()
        _AGE_HALFLIFE_S = 300.0

        def _ts_weight(sig) -> float:
            age = max(0.0, ref_ts - sig.timestamp)
            return 0.5 ** (age / _AGE_HALFLIFE_S)

        weighted_dissent = sum(s.strength * _ts_weight(s) for s in dissent_sigs)
        weighted_support = sum(s.strength * _ts_weight(s) for s in support_sigs)

    dissent_pressure = math.log1p(
        weighted_dissent / max(_EPS, weighted_support)
    )

    # Fix D: support_diversity = number of distinct (partition_id, author_role) pairs
    # among SUPPORT signals that carry an explicit partition_id. This replaces
    # the action-name parse (which was 0 for all worker-pool agents) with a
    # dimension that's always populated after Fix A (partition wiring).
    support_partitions: set[tuple[str, str]] = set()
    for sid in all_support:
        s = store.get(sid)
        if s is None or s.type != SUPPORT:
            continue
        pid = s.partition_id or s.metadata.get("partition_id", "")
        if pid:
            support_partitions.add((pid, s.depositor))

    # Fallback to action/strategy parse for SUPPORT signals without partition_id
    # (pre-Fix-A deposits) and for CRITIQUE_POSITIVE signals (which count toward
    # support but don't carry partition_id).
    strategy_names: set[str] = set()
    for sid in all_support:
        s = store.get(sid)
        if not s:
            continue
        if s.type == SUPPORT and (s.partition_id or s.metadata.get("partition_id", "")):
            continue   # already counted in support_partitions
        action = s.metadata.get("action", "")
        if action:
            strategy_names.add(action)
            continue
        strategy = _parse_strategy_name(s.metadata.get("depositor_agent_id", ""))
        if strategy:
            strategy_names.add(strategy)

    # Combined diversity score: (partition, role) pairs dominate; action names
    # are additive for any non-partitioned contributions.
    total_support_diversity = len(support_partitions) + len(strategy_names)

    # Critic endorsements: CRITIQUE_POSITIVE signals in the support set.
    critic_endorsements = sum(
        1 for sid in all_support
        for s in (store.get(sid),)
        if s is not None and s.type == CRITIQUE_POSITIVE
    )

    ver_scores = []
    for mid in member_ids:
        vs = initial_metrics.get(mid, {}).get("ver_score", 0.0)
        ver_scores.append(vs)
    verification_score = sum(ver_scores) / max(1, len(ver_scores))

    # Phase 3C: cluster-level support_depth = max chain depth across members.
    support_depth = max(
        (initial_metrics.get(mid, {}).get("support_depth", 1) for mid in member_ids),
        default=1,
    )

    # Citations audit block (Fix D): log diversity breakdown for diagnostics.
    print(
        f"[CLUSTER AUDIT] rep={rep_id} members={len(member_ids)} "
        f"support_diversity={total_support_diversity}  "
        f"critic_endorsements={critic_endorsements}  "
        f"members_in_cluster={len(member_ids)}"
    )

    return ClusterProjection(
        representative_id=rep_id,
        member_ids=member_ids,
        support_set=all_support,
        dissent_set=all_dissent,
        verification_set=all_ver,
        support_diversity=total_support_diversity,
        dissent_pressure=dissent_pressure,
        verification_score=verification_score,
        partition_origins=all_partitions,
        support_depth=support_depth,
    )


# ---------------------------------------------------------------------------
# KB penalties / boosts
# ---------------------------------------------------------------------------

def _apply_kb(
    clusters: list[ClusterProjection],
    store: SignalStore,
    prior_rejections: Optional[list[dict]],
    prior_consensus: Optional[list[dict]],
) -> None:
    """Mutate cluster metrics based on prior KB entries."""
    for cp in clusters:
        rep = store.get(cp.representative_id)
        if rep is None:
            continue
        rep_emb = store.get_embedding(cp.representative_id)

        if prior_rejections and rep_emb is not None:
            for entry in prior_rejections:
                entry_emb = entry.get("representative_embedding")
                if entry_emb is not None:
                    sim = _cosine_sim(rep_emb, entry_emb)
                    if sim >= _KB_MATCH_THRESHOLD:
                        cp.dissent_pressure += _KB_REJECTION_PENALTY
                        break  # one penalty per cluster

        if prior_consensus and rep_emb is not None:
            for entry in prior_consensus:
                entry_emb = entry.get("representative_embedding")
                if entry_emb is not None:
                    sim = _cosine_sim(rep_emb, entry_emb)
                    if sim >= _KB_MATCH_THRESHOLD:
                        # Bounded boost: a prior cluster contributes at most +2,
                        # using run_count as the proxy for accumulated evidence.
                        # Caps the rich-get-richer dynamic from unbounded support_diversity.
                        run_count = entry.get("run_count", 1)
                        cp.support_diversity += min(2, run_count)
                        break


# ---------------------------------------------------------------------------
# Survival filter
# ---------------------------------------------------------------------------

def _apply_survival_filter(cp: ClusterProjection, has_validators: bool,
                           task_profile: Optional[dict] = None) -> None:
    """Classify a cluster by mutating cp.status (and cp.unverified) in-place.

    Thresholds come from config (SURVIVAL_*); task_profile selects the
    credibility-gate variant. Evaluated in priority order:

    1. rejected_by_field  — dissent_pressure > SURVIVAL_REJECT_DISSENT_PRESSURE (1.5)
    2. weakly_supported   — support_diversity < SURVIVAL_MIN_SUPPORT_DIVERSITY (3)
    3. contested          — SURVIVAL_CONTEST_MIN (0.5) <= dissent_pressure
                            <= SURVIVAL_CONTEST_MAX (1.5)
    4. credibility gate   — passed (1)-(3) but must satisfy ANY of:
                              FACTUAL profile (requires_verification=True):
                                verification_score >= SURVIVAL_VERIFY_MIN (0.3)
                                len(dissent_set)   >= 1
                                support_diversity  >= SURVIVAL_BROAD_SUPPORT (4)
                              NON-FACTUAL profile (requires_verification=False):
                                support_depth     >= credibility_chain_depth (3)
                                len(dissent_set)  >= 1
                                support_diversity >= SURVIVAL_BROAD_SUPPORT (4)
                                verification_score>= SURVIVAL_VERIFY_MIN (still
                                  honoured if a source happened to corroborate)
                            otherwise tagged `unverified` (fourth bucket).
    5. surviving          — everything else; cp.unverified=True flag set when
                            factual profile + no verification.
    """
    if cp.dissent_pressure > SURVIVAL_REJECT_DISSENT_PRESSURE:
        cp.status = "rejected_by_field"
        return

    # task_profile may override the global minimum for tasks where the natural
    # action-type pool is smaller (e.g. coding only produces DEVELOP+CHAIN).
    min_diversity = (
        task_profile.get("support_diversity_min", SURVIVAL_MIN_SUPPORT_DIVERSITY)
        if task_profile else SURVIVAL_MIN_SUPPORT_DIVERSITY
    )
    if cp.support_diversity < min_diversity:
        cp.status = "weakly_supported"
        return

    if SURVIVAL_CONTEST_MIN <= cp.dissent_pressure <= SURVIVAL_CONTEST_MAX:
        cp.status = "contested"
        return

    # When task_profile is None we replicate the legacy factual gate
    # (verify OR dissent OR broad-support) and the legacy unverified flag.
    if task_profile is None:
        requires_ver = True
        chain_floor = 999
    else:
        requires_ver = task_profile.get("requires_verification", True)
        chain_floor = int(task_profile.get("credibility_chain_depth", 999))

    # Credibility gate
    has_verification = cp.verification_score >= SURVIVAL_VERIFY_MIN
    has_dissent = len(cp.dissent_set) >= 1
    has_broad_support = cp.support_diversity >= SURVIVAL_BROAD_SUPPORT
    has_chain_depth = cp.support_depth >= chain_floor

    if requires_ver:
        if not (has_verification or has_dissent or has_broad_support):
            cp.status = "unverified"
            return
    else:
        # Non-factual: chain depth counts as internal-coherence credibility.
        if not (has_verification or has_dissent or has_broad_support
                or has_chain_depth):
            cp.status = "unverified"
            return

    # Survives. The legacy unverified flag (Section 1 annotation) only fires
    # for factual tasks where verification was expected but didn't arrive —
    # for non-factual tasks the absence of validation is the point, not a flaw.
    cp.status = "surviving"
    cp.unverified = (requires_ver and has_validators
                     and not cp.verification_set)
