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


# ---------------------------------------------------------------------------
# Public entry point
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

    # partition origin
    scout_agent_id = sig.metadata.get("scout_agent_id", "")
    partition_tag = _parse_partition_tag(scout_agent_id)

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
    """Greedy single-linkage clustering at _CLUSTER_SIM_THRESHOLD.

    Iterates INITIALs in strength-descending order. Assigns each to the
    first cluster whose representative has cosine similarity >= threshold.
    Falls back to one cluster per INITIAL if no embeddings are available.

    Returns list of (representative_id, [member_ids]).
    """
    sorted_initials = sorted(initials, key=lambda s: s.strength, reverse=True)
    clusters: list[tuple[str, list[str], Optional[list[float]]]] = []
    # (rep_id, member_ids, rep_embedding)

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
    # Temporal normalization (post-mortem on m13): raw count produced a
    # ranking artifact where claims deposited early in the run accumulated
    # objections purely because they had more wall-clock time to be seen.
    # We now reference the cluster representative's age as a baseline and
    # apply an age-equalizing weight to each contrarian: each dissent's
    # contribution shrinks with its OWN age, normalized to the support
    # signals' median age. Net effect: a claim that has been in the field
    # 10× longer than the typical support no longer gets a proportional
    # dissent stack — it gets a comparable one. Newer dissent dominates.
    support_sigs = [store.get(i) for i in all_support if store.get(i)]
    dissent_sigs = [store.get(i) for i in all_dissent if store.get(i)]
    support_strengths = [s.strength for s in support_sigs]
    dissent_strengths = [s.strength for s in dissent_sigs]

    # Reference time: median support timestamp if we have supports, else
    # the median dissent timestamp, else "now". Supports are the natural
    # baseline because the cluster's existence as a CLUSTER is established
    # by its support pattern, not its detractors.
    import time as _time
    if support_sigs:
        ref_ts = sorted(s.timestamp for s in support_sigs)[len(support_sigs) // 2]
    elif dissent_sigs:
        ref_ts = sorted(s.timestamp for s in dissent_sigs)[len(dissent_sigs) // 2]
    else:
        ref_ts = _time.time()

    # Half-life for age-decay weighting. 5 minutes ≈ the wall-clock window
    # over which a single "round" of activity completes in continuous-pool
    # mode (see DECAY_ROUND_BASELINE_S in worker_pool). A dissent twice as
    # old as ref_ts contributes ~71% weight; ten minutes older contributes ~25%.
    _AGE_HALFLIFE_S = 300.0
    def _age_weight(sig) -> float:
        age = max(0.0, ref_ts - sig.timestamp)
        return 0.5 ** (age / _AGE_HALFLIFE_S)

    weighted_dissent = sum(s.strength * _age_weight(s) for s in dissent_sigs)
    # Apply the same weighting to supports so the ratio stays dimensionally
    # honest: both numerator and denominator are now "current-relevance".
    weighted_support = sum(s.strength * _age_weight(s) for s in support_sigs)
    dissent_pressure = weighted_dissent / max(_EPS, weighted_support)

    strategy_names: set[str] = set()
    for sid in all_support:
        s = store.get(sid)
        if not s:
            continue
        action = s.metadata.get("action", "")
        if action:
            strategy_names.add(action)
            continue
        strategy = _parse_strategy_name(s.metadata.get("depositor_agent_id", ""))
        if strategy:
            strategy_names.add(strategy)

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

    return ClusterProjection(
        representative_id=rep_id,
        member_ids=member_ids,
        support_set=all_support,
        dissent_set=all_dissent,
        verification_set=all_ver,
        support_diversity=len(strategy_names),
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

    if cp.support_diversity < SURVIVAL_MIN_SUPPORT_DIVERSITY:
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
