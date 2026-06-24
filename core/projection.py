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
import hashlib

from .signal_types import (
    INITIAL, SUPPORT, CRITIQUE, CRITIQUE_POSITIVE, CRITIQUE_NEGATIVE,
    OBJECTION, VERIFICATION, SEARCH,
)
from .config import (
    BOOST_THRESHOLD, CLUSTER_SIM_THRESHOLD,
    SURVIVAL_MIN_SUPPORT_DIVERSITY, SURVIVAL_REJECT_DISSENT_PRESSURE,
    SURVIVAL_CONTEST_MIN, SURVIVAL_CONTEST_MAX,
    SURVIVAL_VERIFY_MIN, SURVIVAL_BROAD_SUPPORT,
    SURVIVAL_TASK_PROFILES, SURVIVAL_DEFAULT_PROFILE,
    RENDER_K, DISSENT_K, PLANNER_MMR_LAMBDA,
    VERIFICATION_WEIGHT, DISSENT_WEIGHT,
    MAX_KB_DIVERSITY_BOOST,
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
class TrajectoryFeatures:
    """Temporal evolution of a cluster's signal field across pool iterations.

    All iter_* values are 0 and has_trajectory is False when
    Signal.iter_at_deposit was not set (legacy round-based path or tests
    that don't call store.set_iteration()). Always check has_trajectory
    before building trajectory-dependent prose in the renderer.
    """
    iter_first_support: int = 0
    iter_first_dissent: int = 0
    iter_first_verification: int = 0
    support_growth_rate: float = 0.0   # supports deposited per elapsed iteration
    dissent_response_lag: int = 0      # iters from first dissent to first response
    objection_survival: int = 0        # OBJECTIONs with no subsequent support
    has_trajectory: bool = False


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
    # SUPPORT-id -> [direct SUPPORT child IDs]. Preserves the refinement tree
    # that CHAIN/DEVELOP actions built; the scalar support_depth is the max
    # depth of this tree. Exposed to the synthesizer for trajectory features
    # and decomposed integration (see docs/SYNTHESIZER_OVERHAUL.md §5.4, §5.6).
    support_tree: dict = field(default_factory=dict)
    # Temporal trajectory across pool iterations. has_trajectory=False when
    # iter_at_deposit is 0 for all signals (legacy / test path).
    trajectory: TrajectoryFeatures = field(default_factory=TrajectoryFeatures)
    # Heritable genome. None until _build_genome() populates it (Stage 2).
    genome: Optional["ClusterGenome"] = None


@dataclass
class InterClusterEdge:
    """Typed directed edge between two clusters in the inter-cluster graph.

    source, target: representative_id strings.
    relation: one of —
        shared_evidence  non-empty support_set intersection (symmetric)
        co_contested     non-empty dissent_set intersection (symmetric)
        complements      repr-repr cosine in [0.50, 0.84]; related but distinct
        alternatives     repr-repr cosine in [0.25, 0.64] + comparable priority
        supersedes       repr-repr cosine >= 0.85; source subsumes target
        tension          support of source vs dissent of target cosine > τ
    weight: cosine similarity or overlap fraction, in [0, 1].
    """
    source: str
    target: str
    relation: str
    weight: float = 1.0


@dataclass
class AtomProjection:
    """One SAFE atomic fact extracted from a VERIFICATION signal's metadata."""
    atom_id: str                   # "{verification_signal_id}_atom_{i}"
    text: str                      # the atomic proposition
    weight: float                  # centrality from SAFE decomposition
    verification_score: float      # per-atom score
    source_tag: str                # snippet source or "(no result)"
    query: str                     # the DDG query issued for this atom
    parent_cluster_id: str         # representative_id of parent cluster
    parent_verification_id: str    # which VERIFICATION signal this came from


@dataclass
class ClusterSensitivity:
    """Robustness of a cluster's survival status to signal perturbation."""
    cluster_id: str
    support_removal_robustness: float       # min strength of load-bearing support
    dissent_amplification_tolerance: float  # dissent delta needed to flip status
    load_bearing_supports: list[str]        # support IDs whose removal flips status
    marginal_supports: list[str]            # support IDs whose removal does not
    competing_takeover: Optional[str]       # cluster_id that would advance if this fell
    topology_uncovered_on_removal: list     # topology cell tuples losing coverage


# ---------------------------------------------------------------------------
# Cluster Genome — heritable per-cluster typed organism
# ---------------------------------------------------------------------------

@dataclass
class AtomFact:
    """Persistent genome version of AtomProjection.

    AtomProjection (line 134) is ephemeral — computed at projection time and
    discarded. AtomFact is the heritable record that survives fission/merge.
    The two share the same atom_id key for direct correspondence.

    No-leak invariant: agents may only read .text (proposition text),
    .atom_id, .verification_score, and .source_tag. The .extracted_from
    list carries signal IDs only — never foreign reasoning text.
    """
    atom_id: str                    # "{verification_signal_id}_atom_{i}"
    text: str                       # the atomic proposition (~10-25 words)
    weight: float                   # centrality from SAFE decomposition
    verification_score: float       # per-atom score from _safe_score_atom
    source_tag: str                 # snippet source domain or "(no result)"
    query: str                      # the DDG query issued for this atom
    extracted_from: list[str]       # VERIFICATION signal IDs (IDs only, no text)
    parent_cluster_id: str          # representative_id of the parent cluster


@dataclass
class ClusterKnowledgeBase:
    """Aggregated SEARCH-signal lineage for a cluster.

    This is genuinely new — SEARCH signals are declared inert in
    signal_types.py ("Not counted in support_set or dissent_set") and are
    never currently aggregated per cluster. Populated by _build_genome()
    at projection time by walking SEARCH signals in each cluster's
    transitive ancestry.
    """
    queries_issued: list[str]           # all DDG queries from members' SEARCH lineage
    source_domains: list[str]           # distinct top-level domains across the lineage
    source_count: int                   # total distinct retrieved chunks
    domain_diversity: float             # normalized Shannon entropy over domain frequencies
    parametric_content_ratio: float     # fraction of members with no SEARCH lineage
    cross_cluster_source_overlap: float # fraction of sources shared with other clusters


@dataclass
class TopologyExpression:
    """Topology cell this cluster occupies in the AnswerSpaceTopology."""
    coords: Optional[tuple]             # cell tuple from metadata["topology_coords"]
    cell_label: Optional[str]           # human-readable cell description
    is_anchor: bool                     # True if cluster occupies an anchor corner
    cell_occupancy_rank: int            # 0 = sole occupant; N = N-th cluster in this cell


@dataclass
class Phenotype:
    """Geometric identity of the cluster in embedding space.

    centroid and centroid_at_formation are both stored as lists so the
    genome is JSON-serializable. centroid_stability and centroid_drift are
    Tier 2 (model-derived embeddings) — not model-independent Tier 3.
    """
    centroid: list[float]               # current L2-normalized centroid
    centroid_at_formation: list[float]  # snapshot from _Cluster.centroid_at_formation
    centroid_drift: float               # cosine distance between formation and current
    centroid_stability: float           # 1.0 = stable, 0.0 = drifting wildly
    novelty_density: float              # inverse mean distance to other clusters' centroids


@dataclass
class FitnessTrajectory:
    """Fitness history across the run.

    Composes with (does not subsume) TrajectoryFeatures: TrajectoryFeatures
    records signal-event milestones (iter_first_support, etc.); FitnessTrajectory
    records the composite fitness scalar over time, plus strength and member-count
    histories sampled at fixed intervals.
    """
    formation_iteration: int
    fitness_history: list               # list[tuple[int, float]] (iter, composite_fitness)
    strength_history: list              # list[tuple[int, float]] (iter, mean_strength)
    member_count_history: list          # list[tuple[int, int]] (iter, member_count)
    monotone_growth: bool               # True if fitness rose monotonically
    consolidation_iteration: Optional[int]  # iter when fitness plateaued, or None


@dataclass
class GenomeSensitivity:
    """Atom-level robustness annotation.

    Distinct from ClusterSensitivity (support-level, built by
    _build_sensitivities). This is the genome-level version operating at
    atom granularity: load_bearing_atoms are AtomFact.atom_ids whose
    removal would flip the cluster's survival status. Populated by
    _build_genome() as an extension of _build_sensitivities.
    """
    load_bearing_atoms: list[str]       # atom_ids whose removal flips survival status
    marginal_atoms: list[str]           # atom_ids whose removal does not
    support_removal_robustness: float   # min strength of single support whose removal flips
    competing_takeover: Optional[str]   # cluster_id that would advance if this fell
    topology_cells_at_risk: list        # topology cell tuples losing coverage on removal


@dataclass
class GenomeRelations:
    """Lineage and inter-cluster relationships for the genome."""
    parent_genomes: list[str]           # cluster_ids this descended from (fission/merge)
    descendant_genomes: list[str]       # cluster_ids that descended from this (fission)
    inter_cluster_edges: list           # list[InterClusterEdge] from _build_inter_cluster_edges


@dataclass
class ClusterGenome:
    """The heritable typed organism attached to each ClusterProjection.

    genome_hash reuses _cluster_hash from core/knowledge_base.py (stable
    SHA-1 prefix over sorted atom content). Two clusters with identical
    atom sets hash identically — useful for KB dedup and fission detection.

    All text surfaced to agents must pass the no-leak rule: only
    AtomFact.text (proposition text), identifiers, and scalar fitness
    fields may be rendered into prompts. extracted_from carries signal IDs
    only.
    """
    cluster_id: str
    genome_hash: str                    # stable hash over sorted atom content
    formation_iteration: int
    atoms: list                         # list[AtomFact]
    atom_graph: dict                    # atom_id -> list[atom_id] (dependency DAG)
    topology_expression: TopologyExpression
    phenotype: Phenotype
    knowledge_base: ClusterKnowledgeBase
    sensitivity: GenomeSensitivity
    trajectory: FitnessTrajectory
    relations: GenomeRelations
    composite_fitness: float            # computed by FitnessCompositor (Stage 4)
    fitness_breakdown: dict             # str -> float, per-term contributions


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
    # Typed inter-cluster edges. Built by _build_inter_cluster_edges after
    # the survival filter runs. Empty when no surviving/contested clusters
    # exist or when the lattice has only one cluster.
    inter_cluster_edges: list = field(default_factory=list)  # list[InterClusterEdge]

    # --- Topology axis ---
    # Set externally by the caller of build_projection() via the `topology` arg.
    topology: Optional[object] = None            # AnswerSpaceTopology or None
    topology_coverage: dict = field(default_factory=dict)   # cell_tuple -> [rep_ids]
    uncovered_cells: list = field(default_factory=list)     # cell tuples with no cluster
    out_of_bounds_clusters: list = field(default_factory=list)  # rep_ids w/ "out_of_bounds"

    # --- Atom resolution (feeds genome.atoms via _build_genomes) ---
    atoms: list = field(default_factory=list)           # list[AtomProjection]

    # --- Sensitivity axis ---
    cluster_sensitivities: dict = field(default_factory=dict)  # rep_id -> ClusterSensitivity


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
# Inter-cluster edge graph
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Trajectory features
# ---------------------------------------------------------------------------

def _compute_trajectory(
    support_sigs: list,
    dissent_sigs: list,
    ver_sigs: list,
) -> TrajectoryFeatures:
    """Compute temporal trajectory from cluster-level signal lists.

    Reads Signal.iter_at_deposit. Returns a zero-valued TrajectoryFeatures
    (has_trajectory=False) when no signal has iter_at_deposit > 0, so the
    synthesizer can skip trajectory-dependent prose without branching.
    """
    sup_iters = [s.iter_at_deposit for s in support_sigs if s.iter_at_deposit > 0]
    dis_iters = [s.iter_at_deposit for s in dissent_sigs if s.iter_at_deposit > 0]
    ver_iters = [s.iter_at_deposit for s in ver_sigs if s.iter_at_deposit > 0]

    if not (sup_iters or dis_iters or ver_iters):
        return TrajectoryFeatures()

    iter_first_support = min(sup_iters) if sup_iters else 0
    iter_first_dissent = min(dis_iters) if dis_iters else 0
    iter_first_verification = min(ver_iters) if ver_iters else 0

    if len(sup_iters) > 1:
        growth_rate = len(sup_iters) / max(1, max(sup_iters) - min(sup_iters) + 1)
    else:
        growth_rate = float(len(sup_iters))

    lag = 0
    if iter_first_dissent > 0 and sup_iters:
        after = [x for x in sup_iters if x > iter_first_dissent]
        if after:
            lag = min(after) - iter_first_dissent

    objection_survival = 0
    for s in dissent_sigs:
        if s.iter_at_deposit == 0 or s.type != OBJECTION:
            continue
        if not any(x > s.iter_at_deposit for x in sup_iters):
            objection_survival += 1

    return TrajectoryFeatures(
        iter_first_support=iter_first_support,
        iter_first_dissent=iter_first_dissent,
        iter_first_verification=iter_first_verification,
        support_growth_rate=round(growth_rate, 3),
        dissent_response_lag=lag,
        objection_survival=objection_survival,
        has_trajectory=True,
    )


# Thresholds for edge classification. Calibrated against all-MiniLM-L6-v2
# on the laptop (4-bit quantized) embedding distribution; re-tune once
# empirical runs are available.
_EDGE_TENSION_THRESHOLD = 0.65      # cosine(support_of_A, dissent_of_B)
_EDGE_SUPERSEDE_SIM_MIN = 0.85      # repr-repr cosine for supersedes
_EDGE_COMPLEMENT_SIM_MIN = 0.50     # repr-repr cosine floor for complements
_EDGE_ALTERNATIVE_SIM_MIN = 0.25    # repr-repr cosine floor for alternatives
_EDGE_ALTERNATIVE_SIM_MAX = 0.64    # repr-repr cosine ceiling for alternatives
_EDGE_ALTERNATIVE_PRIORITY_RATIO = 0.5  # min/max priority ratio for alternatives


def _cp_priority(cp: "ClusterProjection") -> float:
    """Priority score used for edge classification and plan ranking.

    When a genome is available (Stage 4+), the composite_fitness is the
    primary score — it is a capped, multi-term, partially non-symbolic signal.
    Before genomes exist (early iterations, legacy tests without genome field),
    falls back to the old three-term LLM-adjacent heuristic so existing code
    paths and tests continue to work unchanged.
    """
    if cp.genome is not None and cp.genome.composite_fitness > 0.0:
        return cp.genome.composite_fitness
    # Legacy fallback (Tier 1 only — purely LLM-adjacent)
    return (cp.support_diversity * max(0.01, cp.verification_score)
            / max(0.01, cp.dissent_pressure))


def _max_tension(
    sup_set: list, dis_set: list, store: "SignalStore",
) -> float:
    """Return max cosine between any (support, dissent) pair, limited to top-3 each."""
    best = 0.0
    for sid in sup_set[:3]:
        emb_s = store.get_embedding(sid)
        if emb_s is None:
            continue
        for did in dis_set[:3]:
            emb_d = store.get_embedding(did)
            if emb_d is None:
                continue
            best = max(best, _cosine_sim(emb_s, emb_d))
    return best


def _build_inter_cluster_edges(
    clusters: list,
    store: "SignalStore",
) -> list:
    """Compute typed inter-cluster edges from surviving/contested clusters.

    Pure Python; uses signal IDs and cached embeddings already in the store.
    No LLM calls. O(n²) in cluster count; fast for n <= 20.

    Set-based relations are always computed; embedding-based ones are skipped
    gracefully when embeddings are unavailable.
    """
    edges: list = []

    for i, ca in enumerate(clusters):
        for cb in clusters[i + 1:]:
            # --- Set-based (no embeddings) ---
            shared_sup = set(ca.support_set) & set(cb.support_set)
            if shared_sup:
                w = len(shared_sup) / max(1, min(len(ca.support_set), len(cb.support_set)))
                edges.append(InterClusterEdge(
                    ca.representative_id, cb.representative_id,
                    "shared_evidence", round(w, 3),
                ))

            shared_dis = set(ca.dissent_set) & set(cb.dissent_set)
            if shared_dis:
                w = len(shared_dis) / max(1, min(len(ca.dissent_set), len(cb.dissent_set)))
                edges.append(InterClusterEdge(
                    ca.representative_id, cb.representative_id,
                    "co_contested", round(w, 3),
                ))

            # --- Embedding-based ---
            emb_a = store.get_embedding(ca.representative_id)
            emb_b = store.get_embedding(cb.representative_id)
            if emb_a is not None and emb_b is not None:
                sim = _cosine_sim(emb_a, emb_b)
                pa, pb = _cp_priority(ca), _cp_priority(cb)

                if sim >= _EDGE_SUPERSEDE_SIM_MIN:
                    src, tgt = (ca, cb) if pa >= pb else (cb, ca)
                    edges.append(InterClusterEdge(
                        src.representative_id, tgt.representative_id,
                        "supersedes", round(sim, 3),
                    ))
                elif sim >= _EDGE_COMPLEMENT_SIM_MIN:
                    edges.append(InterClusterEdge(
                        ca.representative_id, cb.representative_id,
                        "complements", round(sim, 3),
                    ))
                elif sim >= _EDGE_ALTERNATIVE_SIM_MIN:
                    if (pa > 0 and pb > 0
                            and min(pa, pb) / max(pa, pb) >= _EDGE_ALTERNATIVE_PRIORITY_RATIO):
                        edges.append(InterClusterEdge(
                            ca.representative_id, cb.representative_id,
                            "alternatives", round(sim, 3),
                        ))

                # Tension (directional)
                t_ab = _max_tension(ca.support_set, cb.dissent_set, store)
                if t_ab >= _EDGE_TENSION_THRESHOLD:
                    edges.append(InterClusterEdge(
                        ca.representative_id, cb.representative_id,
                        "tension", round(t_ab, 3),
                    ))
                t_ba = _max_tension(cb.support_set, ca.dissent_set, store)
                if t_ba >= _EDGE_TENSION_THRESHOLD:
                    edges.append(InterClusterEdge(
                        cb.representative_id, ca.representative_id,
                        "tension", round(t_ba, 3),
                    ))

    return edges


# ---------------------------------------------------------------------------
# Lattice-axis builders (Phase 1 of topology overhaul)
# ---------------------------------------------------------------------------

_SENSITIVITY_MAX_CLUSTERS = 20   # skip sensitivity if field is very large


def _build_atoms(
    clusters: list,
    store: "SignalStore",
) -> list:
    """Extract AtomProjection objects from VERIFICATION signal metadata.

    Reads metadata["atoms"] deposited by the SAFE pipeline in
    core/worker_pool.py:748-766. Returns empty list when atoms are absent
    (legacy runs, SAFE fallback path, mock mode).
    """
    atoms: list = []
    for cp in clusters:
        for vid in cp.verification_set:
            vsig = store.get(vid)
            if vsig is None:
                continue
            raw_atoms = vsig.metadata.get("atoms")
            if not raw_atoms or not isinstance(raw_atoms, list):
                continue
            for i, a in enumerate(raw_atoms):
                if not isinstance(a, dict):
                    continue
                atom_text = str(a.get("text", "")).strip()
                if not atom_text:
                    continue
                atoms.append(AtomProjection(
                    atom_id=f"{vid}_atom_{i}",
                    text=atom_text,
                    weight=float(a.get("weight", 1.0)),
                    verification_score=float(a.get("score", 0.5)),
                    source_tag=str(a.get("snippet_tag", "(no result)")),
                    query=str(a.get("query", "")),
                    parent_cluster_id=cp.representative_id,
                    parent_verification_id=vid,
                ))
    return atoms


def _build_sensitivities(
    clusters: list,
    store: "SignalStore",
    topology_coverage: Optional[dict] = None,
) -> dict:
    """Compute ClusterSensitivity for each surviving/contested cluster.

    For each support signal, simulates its removal and re-applies the simplified
    survival thresholds to determine whether status would change.

    topology_coverage: cell_tuple -> [rep_ids]. Used to detect which topology
    cells lose coverage if the cluster is removed.
    """
    if len(clusters) > _SENSITIVITY_MAX_CLUSTERS:
        return {}

    sensitivities: dict = {}
    for cp in clusters:
        if not cp.support_set:
            sensitivities[cp.representative_id] = ClusterSensitivity(
                cluster_id=cp.representative_id,
                support_removal_robustness=0.0,
                dissent_amplification_tolerance=0.0,
                load_bearing_supports=[],
                marginal_supports=list(cp.support_set),
                competing_takeover=None,
                topology_uncovered_on_removal=[],
            )
            continue

        support_sigs = [store.get(sid) for sid in cp.support_set if store.get(sid)]
        dissent_sigs = [store.get(sid) for sid in cp.dissent_set if store.get(sid)]
        total_support = sum(s.strength for s in support_sigs) or _EPS
        total_dissent = sum(s.strength for s in dissent_sigs)
        base_dp = math.log1p(total_dissent / total_support)

        load_bearing: list[str] = []
        marginal: list[str] = []
        min_lb_strength = 1.0

        for sid in cp.support_set:
            sig = store.get(sid)
            if sig is None:
                continue
            wp_without = max(_EPS, total_support - sig.strength)
            dp_without = math.log1p(total_dissent / wp_without)
            # Would removal flip the status?
            would_flip = (
                dp_without > SURVIVAL_REJECT_DISSENT_PRESSURE
                or dp_without > SURVIVAL_CONTEST_MAX
                and base_dp <= SURVIVAL_CONTEST_MIN
            )
            if would_flip:
                load_bearing.append(sid)
                min_lb_strength = min(min_lb_strength, sig.strength)
            else:
                marginal.append(sid)

        # Dissent amplification tolerance: how much additional dissent δ flips status.
        # dp(δ) = log1p((total_dissent + δ) / total_support) > SURVIVAL_CONTEST_MIN
        # Solve: δ = total_support * (exp(SURVIVAL_CONTEST_MIN) - 1) - total_dissent
        contest_threshold = SURVIVAL_CONTEST_MIN
        delta_to_contest = max(
            0.0,
            total_support * (math.exp(contest_threshold) - 1) - total_dissent,
        )

        # Competing takeover: highest-priority other cluster
        others = [c for c in clusters if c.representative_id != cp.representative_id]
        competing = None
        if others:
            best = max(others, key=_cp_priority)
            competing = best.representative_id

        # Topology cells that would be uncovered if this cluster were removed
        oob_cells: list = []
        if topology_coverage:
            for cell, rep_ids in topology_coverage.items():
                if (cp.representative_id in rep_ids
                        and len(rep_ids) == 1):
                    oob_cells.append(cell)

        sensitivities[cp.representative_id] = ClusterSensitivity(
            cluster_id=cp.representative_id,
            support_removal_robustness=min_lb_strength if load_bearing else 1.0,
            dissent_amplification_tolerance=round(delta_to_contest, 3),
            load_bearing_supports=load_bearing,
            marginal_supports=marginal,
            competing_takeover=competing,
            topology_uncovered_on_removal=oob_cells,
        )

    return sensitivities


def _extract_domain(source_tag: str) -> str:
    """Extract the bare domain from a source_tag like 'ipcc.ch/report/ar6'.

    Returns "(parametric)" for empty/no-result tags so they are countable
    as a distinct domain bucket for parametric_content_ratio.
    """
    tag = source_tag.strip()
    if not tag or tag.startswith("("):
        return "(parametric)"
    # Strip leading protocol if present
    for prefix in ("https://", "http://", "//"):
        if tag.startswith(prefix):
            tag = tag[len(prefix):]
            break
    return tag.split("/")[0].split("?")[0].lower()


def _build_cluster_kb(
    cp: "ClusterProjection",
    store: "SignalStore",
    all_search_sigs: list,
    atom_facts: list,
) -> "ClusterKnowledgeBase":
    """Populate ClusterKnowledgeBase by aggregating SEARCH-signal lineage.

    Three source channels per cluster:
    1. Scout SEARCH: SEARCH signals whose metadata["scout_agent_id"] matches
       the scout_agent_id of any INITIAL member of this cluster.
    2. Developer SEARCH: SEARCH signals that are direct children (parent_id)
       of any INITIAL member (deposited by the DEVELOP sparse-cluster hook).
    3. Validator sources: already captured in AtomFact.source_tag from the
       SAFE pipeline — no additional SEARCH signals needed.
    """
    # Build index from scout_agent_id → SEARCH signal for channel 1
    scout_id_to_search: dict[str, list] = {}
    dev_parent_to_search: dict[str, list] = {}
    for ssig in all_search_sigs:
        sid = ssig.metadata.get("scout_agent_id") or ssig.metadata.get("depositor_agent_id", "")
        if sid:
            scout_id_to_search.setdefault(sid, []).append(ssig)
        if ssig.parent_id:
            dev_parent_to_search.setdefault(ssig.parent_id, []).append(ssig)

    queries: list[str] = []
    source_tags: list[str] = []
    member_has_search: set[str] = set()

    for mid in cp.member_ids:
        sig = store.get(mid)
        if sig is None:
            continue
        # Channel 1: scout SEARCH by matching scout_agent_id
        s_id = sig.metadata.get("scout_agent_id") or sig.metadata.get("depositor_agent_id", "")
        for ssig in scout_id_to_search.get(s_id, []):
            q = ssig.metadata.get("query", "")
            if q:
                queries.append(q)
            # Parse source lines from content "QUERY: ...\n  - <src>"
            for line in ssig.content.splitlines():
                line = line.strip().lstrip("- ")
                if line and not line.startswith("QUERY:") and not line.startswith("(no"):
                    source_tags.append(line[:200])
            member_has_search.add(mid)

        # Channel 2: developer SEARCH children of this INITIAL
        for ssig in dev_parent_to_search.get(mid, []):
            q = ssig.metadata.get("query", "")
            if q:
                queries.append(q)
            for line in ssig.content.splitlines():
                line = line.strip().lstrip("- ")
                if line and not line.startswith("QUERY:") and not line.startswith("(no"):
                    source_tags.append(line[:200])
            member_has_search.add(mid)

    # Channel 3: validator atom source_tags
    for af in atom_facts:
        q = af.query
        if q:
            queries.append(q)
        tag = af.source_tag
        if tag and not tag.startswith("("):
            source_tags.append(tag)

    # Deduplicate
    queries = list(dict.fromkeys(queries))  # stable dedup
    source_tags = list(dict.fromkeys(source_tags))

    # Extract domains and count frequencies
    domain_freq: dict[str, int] = {}
    for tag in source_tags:
        d = _extract_domain(tag)
        domain_freq[d] = domain_freq.get(d, 0) + 1
    # Remove the parametric bucket from domain list (it's not a real domain)
    real_domains = [d for d in domain_freq if d != "(parametric)"]

    # Shannon entropy over domain frequencies (normalized by log2(N))
    total_hits = sum(domain_freq.get(d, 0) for d in real_domains) or 1
    entropy = 0.0
    for d in real_domains:
        p = domain_freq[d] / total_hits
        if p > 0:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(max(len(real_domains), 1))
    domain_diversity = round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0

    # parametric_content_ratio: fraction of members with no SEARCH linkage
    n_members = max(1, len(cp.member_ids))
    parametric_ratio = round(1.0 - len(member_has_search) / n_members, 4)

    # cross_cluster_source_overlap: fraction of this cluster's domains that
    # appear in at least one other cluster (computed across all genomes).
    # At this point other genomes are not yet assembled, so we defer to a
    # post-pass. Set to 0.0 here; _finalize_kb_overlap() fills it after all
    # genomes are assembled.
    return ClusterKnowledgeBase(
        queries_issued=queries,
        source_domains=real_domains,
        source_count=len(source_tags),
        domain_diversity=domain_diversity,
        parametric_content_ratio=parametric_ratio,
        cross_cluster_source_overlap=0.0,  # filled by _finalize_kb_overlap
    )


def _finalize_kb_overlap(clusters: list) -> None:
    """Fill cross_cluster_source_overlap in-place after all genomes are assembled."""
    # Build union of domains per cluster
    all_domain_sets: list[tuple[str, set]] = []
    for cp in clusters:
        if cp.genome:
            all_domain_sets.append(
                (cp.representative_id, set(cp.genome.knowledge_base.source_domains))
            )

    for cp in clusters:
        if cp.genome is None or not cp.genome.knowledge_base.source_domains:
            continue
        my_domains = set(cp.genome.knowledge_base.source_domains)
        n = len(my_domains)
        if n == 0:
            continue
        overlap_count = 0
        for other_id, other_domains in all_domain_sets:
            if other_id == cp.representative_id:
                continue
            overlap_count += len(my_domains & other_domains)
        # Fraction of my domain occurrences that appear in any other cluster
        overlap_frac = round(min(1.0, overlap_count / n), 4)
        # ClusterKnowledgeBase is a plain mutable dataclass — set field directly
        cp.genome.knowledge_base.cross_cluster_source_overlap = overlap_frac


def _build_genomes(
    clusters: list,
    store: "SignalStore",
    proj_atoms: list,
    inter_cluster_edges: list,
    cluster_sensitivities: dict,
    topology=None,
) -> None:
    """Populate cp.genome in-place for each cluster.

    Assembles ClusterGenome from the typed pieces already computed:
    - AtomFacts from proj_atoms (convert from AtomProjection)
    - Phenotype from ClusterRegistry centroid + centroid_at_formation
    - GenomeSensitivity from ClusterSensitivity (support-level → atom-level mapping)
    - GenomeRelations from inter_cluster_edges
    - TopologyExpression from signal metadata + topology
    - ClusterKnowledgeBase: populated from SEARCH-signal lineage (scout/developer/atom channels)
    - FitnessTrajectory: initialized from TrajectoryFeatures; history filled by Stage 4
    """
    # Index atoms by cluster
    atoms_by_cluster: dict[str, list] = {}
    for a in proj_atoms:
        atoms_by_cluster.setdefault(a.parent_cluster_id, []).append(a)

    # Pre-scan all SEARCH signals once for kb population (Stage 3)
    all_search_sigs = store.by_type(SEARCH)

    # Index edges by cluster (both directions)
    edges_by_cluster: dict[str, list] = {}
    for edge in inter_cluster_edges:
        edges_by_cluster.setdefault(edge.source, []).append(edge)
        edges_by_cluster.setdefault(edge.target, []).append(edge)

    # Precompute centroids keyed by representative_id for novelty_density.
    # Stored as a dict so each cluster can exclude its own centroid
    # (identity comparison on list copies is always False, so a keyed
    # lookup is the only reliable way to exclude self).
    registry = store._cluster_registry
    centroid_by_rep: dict[str, list] = {}
    for _cp in clusters:
        _cid = registry.get_cluster_id(_cp.representative_id)
        _cl = registry.get_cluster(_cid) if _cid else None
        if _cl and _cl.centroid:
            centroid_by_rep[_cp.representative_id] = _cl.centroid

    # Anchor coords for topology
    anchor_coords: set = set()
    if topology is not None:
        try:
            anchor_coords = topology.anchor_coords_set()
        except Exception:
            pass

    for cp in clusters:
        cluster_atoms_proj = atoms_by_cluster.get(cp.representative_id, [])

        # Convert AtomProjection -> AtomFact
        atom_facts = [
            AtomFact(
                atom_id=a.atom_id,
                text=a.text,
                weight=a.weight,
                verification_score=a.verification_score,
                source_tag=a.source_tag,
                query=a.query,
                extracted_from=[a.parent_verification_id],
                parent_cluster_id=a.parent_cluster_id,
            )
            for a in cluster_atoms_proj
        ]

        # Rule 3: deduplicate atoms with high content overlap.
        # Collapses near-duplicate atoms (word Jaccard ≥ 0.70) by keeping the
        # one with the higher verification_score and merging their extracted_from
        # lists. Uses 0.70 word-Jaccard as a proxy for cosine ≥ 0.85 (design doc
        # specifies cosine; word-Jaccard is used here since per-atom embeddings
        # are not available without an embedder call). Same 0.85 constant used
        # in knowledge_base.py for cluster-level dedup.
        _ATOM_DEDUP_JACCARD = 0.70

        def _word_jaccard_atoms(ta: str, tb: str) -> float:
            wa = set(ta.lower().split())
            wb = set(tb.lower().split())
            if not wa or not wb:
                return 0.0
            return len(wa & wb) / len(wa | wb)

        deduplicated: list = []
        for af in atom_facts:
            merged = False
            for existing in deduplicated:
                if _word_jaccard_atoms(af.text, existing.text) >= _ATOM_DEDUP_JACCARD:
                    # Merge: keep higher-score atom, union extracted_from
                    if af.verification_score > existing.verification_score:
                        # Replace existing with af's content/score but union extracted_from
                        new_extracted = list(dict.fromkeys(
                            af.extracted_from + existing.extracted_from
                        ))
                        existing.__class__  # confirm it's AtomFact
                        idx = deduplicated.index(existing)
                        deduplicated[idx] = AtomFact(
                            atom_id=af.atom_id,
                            text=af.text,
                            weight=max(af.weight, existing.weight),
                            verification_score=af.verification_score,
                            source_tag=af.source_tag or existing.source_tag,
                            query=af.query or existing.query,
                            extracted_from=new_extracted,
                            parent_cluster_id=af.parent_cluster_id,
                        )
                    else:
                        # Merge extracted_from into existing; existing text/score wins
                        new_extracted = list(dict.fromkeys(
                            existing.extracted_from + af.extracted_from
                        ))
                        idx = deduplicated.index(existing)
                        deduplicated[idx] = AtomFact(
                            atom_id=existing.atom_id,
                            text=existing.text,
                            weight=max(existing.weight, af.weight),
                            verification_score=existing.verification_score,
                            source_tag=existing.source_tag or af.source_tag,
                            query=existing.query or af.query,
                            extracted_from=new_extracted,
                            parent_cluster_id=existing.parent_cluster_id,
                        )
                    merged = True
                    break
            if not merged:
                deduplicated.append(af)

        atom_facts = deduplicated

        # atom_graph: dependency DAG over atoms in this cluster.
        # Rules (from §5 of the design doc):
        #   1. SUPPORT-derived atoms depend on INITIAL-derived atoms in this cluster
        #   2. Atoms sharing extracted_from siblings (same VERIFICATION) have no edge
        #   3. Atoms with content cosine ≥ 0.85 collapsed above (Rule 3)
        #   4. REFINE-targeted atoms depend on the atoms they address (via targets_atom)
        atom_graph: dict[str, list] = {a.atom_id: [] for a in atom_facts}

        # Build extracted_from lookup: verification_sig_id -> list[atom_id]
        ver_to_atoms: dict[str, list[str]] = {}
        for af in atom_facts:
            for vid in af.extracted_from:
                ver_to_atoms.setdefault(vid, []).append(af.atom_id)

        # Rule 1: if a VERIFICATION signal's parent is a SUPPORT (not INITIAL),
        # the atoms it produced depend on atoms from the INITIAL's VERIFICATION.
        initial_atom_ids: set[str] = set()
        support_atom_ids: set[str] = set()
        for af in atom_facts:
            for vid in af.extracted_from:
                vsig = store.get(vid)
                if vsig is None:
                    continue
                # Walk up one level: vsig.parent_id is the target signal
                parent_sig = store.get(vsig.parent_id) if vsig.parent_id else None
                if parent_sig and parent_sig.type == INITIAL:
                    initial_atom_ids.add(af.atom_id)
                elif parent_sig and parent_sig.type == SUPPORT:
                    support_atom_ids.add(af.atom_id)

        # Atoms extracted from SUPPORT-level verifications depend on INITIAL-level atoms
        for sa_id in support_atom_ids:
            for ia_id in initial_atom_ids:
                if sa_id != ia_id:
                    atom_graph[sa_id].append(ia_id)

        # Rule 4: REFINE-targeted atoms — check deposit metadata for targets_atom
        # links from SUPPORT signals in this cluster's support_set
        for sid in cp.support_set:
            sig = store.get(sid)
            if sig is None:
                continue
            targets = sig.metadata.get("targets_atom")
            if targets and targets in atom_graph:
                # Atoms in this SUPPORT signal's VERIFICATION depend on the targeted atom
                for af in atom_facts:
                    for vid in af.extracted_from:
                        vsig = store.get(vid)
                        if vsig and vsig.parent_id == sid:
                            if targets not in atom_graph[af.atom_id]:
                                atom_graph[af.atom_id].append(targets)

        # Genome hash: SHA-1 over sorted atom texts (same approach as _cluster_hash)
        sorted_texts = sorted(a.text for a in atom_facts)
        hash_input = "\n".join(sorted_texts) if sorted_texts else cp.representative_id
        genome_hash = hashlib.sha1(hash_input.encode()).hexdigest()[:16]

        # TopologyExpression
        rep_sig = store.get(cp.representative_id)
        topo_coords = None
        cell_label = None
        is_anchor = False
        if rep_sig:
            raw = rep_sig.metadata.get("topology_coords")
            if raw and raw != "out_of_bounds":
                topo_coords = tuple(raw) if isinstance(raw, list) else raw
        if topology is not None and topo_coords is not None:
            try:
                cell_label = topology.cell_description(topo_coords)
                is_anchor = topo_coords in anchor_coords
            except Exception:
                pass

        cell_occupancy_rank = 0
        if topo_coords is not None:
            for other_cp in clusters:
                if other_cp.representative_id == cp.representative_id:
                    continue
                other_sig = store.get(other_cp.representative_id)
                if other_sig:
                    other_raw = other_sig.metadata.get("topology_coords")
                    if other_raw and other_raw != "out_of_bounds":
                        other_coords = tuple(other_raw) if isinstance(other_raw, list) else other_raw
                        if other_coords == topo_coords:
                            cell_occupancy_rank += 1

        topo_expr = TopologyExpression(
            coords=topo_coords,
            cell_label=cell_label,
            is_anchor=is_anchor,
            cell_occupancy_rank=cell_occupancy_rank,
        )

        # Phenotype from ClusterRegistry
        cid = registry.get_cluster_id(cp.representative_id)
        cl = registry.get_cluster(cid) if cid else None
        centroid = cl.centroid[:] if cl and cl.centroid else []
        centroid_at_formation = (
            cl.centroid_at_formation[:] if cl and cl.centroid_at_formation else centroid[:]
        )

        if centroid and centroid_at_formation and len(centroid) == len(centroid_at_formation):
            dot = sum(a * b for a, b in zip(centroid, centroid_at_formation))
            centroid_drift = round(max(0.0, 1.0 - dot), 4)
        else:
            centroid_drift = 0.0

        if centroid_drift <= 0.1:
            centroid_stability = 1.0
        elif centroid_drift >= 0.5:
            centroid_stability = 0.0
        else:
            centroid_stability = round(1.0 - (centroid_drift - 0.1) / 0.4, 4)

        # novelty_density: mean cosine distance to all OTHER clusters' centroids.
        # centroid_by_rep excludes self via key lookup (identity comparison on
        # list copies is unreliable — this is the correct approach).
        other_centroids = [
            c for rid, c in centroid_by_rep.items()
            if rid != cp.representative_id
        ]
        if centroid and other_centroids:
            dists = [
                max(0.0, 1.0 - sum(a * b for a, b in zip(centroid, c)))
                for c in other_centroids
            ]
            novelty_density = round(sum(dists) / len(dists), 4)
        else:
            novelty_density = 0.0

        phenotype = Phenotype(
            centroid=centroid,
            centroid_at_formation=centroid_at_formation,
            centroid_drift=centroid_drift,
            centroid_stability=centroid_stability,
            novelty_density=novelty_density,
        )

        # ClusterKnowledgeBase: populated from SEARCH-signal lineage
        kb = _build_cluster_kb(
            cp, store, all_search_sigs, atom_facts
        )

        # GenomeSensitivity: derived from ClusterSensitivity (support-level)
        # load_bearing_atoms = atoms extracted from a load_bearing_support signal
        cs = cluster_sensitivities.get(cp.representative_id)
        if cs:
            lb_ids = set(cs.load_bearing_supports)
            lb_atoms = [
                a.atom_id for a in atom_facts
                if any(vid in lb_ids for vid in a.extracted_from)
            ]
            mg_atoms = [a.atom_id for a in atom_facts if a.atom_id not in set(lb_atoms)]
            genome_sens = GenomeSensitivity(
                load_bearing_atoms=lb_atoms,
                marginal_atoms=mg_atoms,
                support_removal_robustness=cs.support_removal_robustness,
                competing_takeover=cs.competing_takeover,
                topology_cells_at_risk=cs.topology_uncovered_on_removal,
            )
        else:
            genome_sens = GenomeSensitivity(
                load_bearing_atoms=[],
                marginal_atoms=[a.atom_id for a in atom_facts],
                support_removal_robustness=1.0,
                competing_takeover=None,
                topology_cells_at_risk=[],
            )

        # FitnessTrajectory: seeded from TrajectoryFeatures (Stage 4 fills history)
        fitness_traj = FitnessTrajectory(
            formation_iteration=cp.trajectory.iter_first_support,
            fitness_history=[],
            strength_history=[],
            member_count_history=[],
            monotone_growth=False,
            consolidation_iteration=None,
        )

        # GenomeRelations: edges involving this cluster
        genome_relations = GenomeRelations(
            parent_genomes=[],
            descendant_genomes=[],
            inter_cluster_edges=edges_by_cluster.get(cp.representative_id, []),
        )

        cp.genome = ClusterGenome(
            cluster_id=cp.representative_id,
            genome_hash=genome_hash,
            formation_iteration=cp.trajectory.iter_first_support,
            atoms=atom_facts,
            atom_graph=atom_graph,
            topology_expression=topo_expr,
            phenotype=phenotype,
            knowledge_base=kb,
            sensitivity=genome_sens,
            trajectory=fitness_traj,
            relations=genome_relations,
            composite_fitness=0.0,
            fitness_breakdown={},
        )

    # Post-pass: fill cross_cluster_source_overlap now that all genomes exist
    _finalize_kb_overlap(clusters)


def _build_topology_coverage(
    clusters: list,
    store: "SignalStore",
    topology,  # Optional[AnswerSpaceTopology]
) -> tuple[dict, list, list]:
    """Build topology_coverage, uncovered_cells, and out_of_bounds_clusters.

    Returns (topology_coverage, uncovered_cells, out_of_bounds_clusters).
    topology_coverage: cell_tuple -> [representative_ids of clusters in that cell]
    uncovered_cells: cell tuples with no covering cluster
    out_of_bounds_clusters: representative_ids whose INITIAL has coords="out_of_bounds"
    """
    if topology is None:
        return {}, [], []

    coverage: dict = {}
    oob: list = []

    for cp in clusters:
        rep_sig = store.get(cp.representative_id)
        if rep_sig is None:
            continue
        coords = rep_sig.metadata.get("topology_coords")
        if coords is None:
            # Also check member signals
            for mid in cp.member_ids:
                msig = store.get(mid)
                if msig is not None:
                    coords = msig.metadata.get("topology_coords")
                    if coords is not None:
                        break
        if coords is None:
            continue
        if coords == "out_of_bounds":
            oob.append(cp.representative_id)
            continue
        if isinstance(coords, list):
            coords = tuple(coords)
        if not isinstance(coords, tuple):
            continue
        coverage.setdefault(coords, [])
        if cp.representative_id not in coverage[coords]:
            coverage[coords].append(cp.representative_id)

    all_cells = topology.all_cells()
    uncovered = [c for c in all_cells if c not in coverage]

    return coverage, uncovered, oob


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_projection(
    store: SignalStore,
    has_validators: bool = True,
    prior_rejections: Optional[list[dict]] = None,
    prior_consensus: Optional[list[dict]] = None,
    task_type: Optional[str] = None,
    topology=None,  # Optional[AnswerSpaceTopology] from core/topology.py
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

    # Build typed inter-cluster edge graph from surviving + contested clusters.
    active = proj.surviving + proj.contested
    if len(active) >= 2:
        proj.inter_cluster_edges = _build_inter_cluster_edges(active, store)

    # --- Topology axis ---
    proj.topology = topology
    if topology is not None:
        proj.topology_coverage, proj.uncovered_cells, proj.out_of_bounds_clusters = (
            _build_topology_coverage(active, store, topology)
        )

    # --- Atom resolution ---
    # Build atoms across all clusters (not just active) so weakly_supported
    # and rejected clusters also carry atoms in their genomes.
    proj.atoms = _build_atoms(cluster_projections, store)

    # --- Sensitivity axis (gated on field size) ---
    if active:
        proj.cluster_sensitivities = _build_sensitivities(
            active, store, topology_coverage=proj.topology_coverage,
        )

    # --- Genome assembly ---
    # Runs last so it can read all already-computed typed pieces (atoms,
    # inter_cluster_edges, cluster_sensitivities, topology_coverage).
    # Built for ALL clusters so weakly_supported/rejected clusters also carry
    # a genome for KB persistence and fission tracking.
    _build_genomes(
        cluster_projections, store,
        proj_atoms=proj.atoms,
        inter_cluster_edges=proj.inter_cluster_edges,
        cluster_sensitivities=proj.cluster_sensitivities,
        topology=topology,
    )

    # --- Composite fitness (Stage 4) ---
    # Populate composite_fitness and fitness_breakdown on each genome.
    # Requires all genomes to exist so novelty_density and topology
    # cross-cluster terms are well-defined.
    from .fitness import compute_composite_fitness as _compute_fitness
    all_genomes = [cp.genome for cp in cluster_projections if cp.genome is not None]
    for cp in cluster_projections:
        if cp.genome is None:
            continue
        cf, breakdown = _compute_fitness(cp.genome, task_type, all_genomes)
        cp.genome.composite_fitness = cf
        cp.genome.fitness_breakdown.update(breakdown)

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
    # When a genome is available, composite_fitness is the primary score
    # (capped, multi-term, partially non-symbolic).  The old three-term
    # heuristic is retained as a fallback so tests without genomes pass.
    def _score(cp: ClusterProjection) -> float:
        if cp.genome is not None and cp.genome.composite_fitness > 0.0:
            return cp.genome.composite_fitness
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
        f"contested={len(contested)} render_ids={picked_ids}"
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
    # support_tree: SUPPORT-id -> [direct SUPPORT child IDs]. Records the
    # actual refinement tree so the synthesizer can reason about chain
    # structure, not just the scalar depth.
    support_tree: dict[str, list[str]] = {}

    visited: set[str] = set()
    # (child_id, parent_id, parent_depth) — parent_id is the direct DAG
    # parent of child_id; used to build support_tree edges.
    queue: list[tuple[str, str, int]] = [
        (cid, sig.id, 0) for cid in store.by_parent(sig.id)
    ]
    while queue:
        child_id, parent_id, parent_depth = queue.pop()
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
            # Build tree: record SUPPORT→SUPPORT parent-child edges only.
            if child.type == SUPPORT:
                parent_sig = store.get(parent_id)
                if parent_sig is not None and parent_sig.type == SUPPORT:
                    support_tree.setdefault(parent_id, []).append(child_id)
        elif child.type in (CRITIQUE_NEGATIVE, CRITIQUE, OBJECTION):
            dissent_ids.append(child_id)
            new_depth = parent_depth
        elif child.type == VERIFICATION:
            ver_ids.append(child_id)
            new_depth = parent_depth
        else:
            new_depth = parent_depth
        for grandchild in store.by_parent(child_id):
            queue.append((grandchild, child_id, new_depth))

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
        "support_tree": support_tree,
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
    """Cosine similarity. signal_store normalizes at encode time so live
    signals are unit-length; KB embeddings from older runs may not be.
    Always computes full cosine to handle both cases correctly."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    denom = norm_a * norm_b
    return float(dot / denom) if denom > _EPS else 0.0


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

    # Trajectory features from cluster-level signal lists.
    ver_sigs_for_traj = [store.get(i) for i in all_ver if store.get(i)]
    trajectory = _compute_trajectory(support_sigs, dissent_sigs, ver_sigs_for_traj)

    # Merge per-member support trees into one cluster-level tree.
    merged_support_tree: dict[str, list[str]] = {}
    for mid in member_ids:
        for parent_id, children in initial_metrics.get(mid, {}).get("support_tree", {}).items():
            existing = merged_support_tree.setdefault(parent_id, [])
            for cid in children:
                if cid not in existing:
                    existing.append(cid)

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
        support_tree=merged_support_tree,
        trajectory=trajectory,
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
                        run_count = entry.get("run_count", 1)
                        cp.support_diversity += min(MAX_KB_DIVERSITY_BOOST, run_count)
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
