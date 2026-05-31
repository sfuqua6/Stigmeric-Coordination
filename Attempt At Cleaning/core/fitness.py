"""FitnessCompositor — composite, non-symbolic fitness for ClusterGenome.

Computes a weighted sum of seven terms, with the LLM-judged term hard-capped
at cap_llm (default 0.35) so the closed-loop pathology cannot dominate selection.

Tier classification (honesty caveat from the design doc §6):
  Tier 1 — self-consistency: semantic_strength (cap-limited LLM-judged term)
  Tier 2 — model-derived but not LLM-judged: centroid_stability, novelty_density
            (computed over the base model's own embeddings)
  Tier 3 — model-independent: grounding (domain diversity / SEARCH lineage),
            topology (cell occupancy geometry), trajectory (monotone growth test),
            entity_resolution (Wikidata — gated off on laptop by default)

capping semantic_strength alone does not fully sever the closed loop because
centroid_stability and novelty_density are Tier 2.  They are not LLM-judged
but they are computed over model embeddings.  Only entity_resolution is fully
independent of the model family.

Per-task weight tables encode the prior over which terms are informative:
  coding        — grounding + entity_resolution heavy (tests are authoritative)
  factual tasks — same direction
  debate/analysis/problem_solving — topology + novelty_density + stability
  creative      — novelty_density dominant; grounding low (no external facts needed)
"""

from __future__ import annotations

import math
from typing import Optional

# ---------------------------------------------------------------------------
# Public constants (overridable via config.py gating in run_swarm.py)
# ---------------------------------------------------------------------------

CAP_LLM: float = 0.35      # max contribution from LLM-judged semantic_strength
USE_ENTITY_RESOLUTION: bool = False  # Wikidata gating; set True on Colab/A100

# Per-task weight tables.  All rows sum to ~1.0 before normalization inside
# compute_composite_fitness.  Weights that are 0.0 skip their term computation.
_WEIGHTS: dict[str, dict[str, float]] = {
    "coding": {
        "semantic_strength": 0.10,
        "grounding":         0.30,
        "topology":          0.10,
        "centroid_stability":0.10,
        "novelty_density":   0.10,
        "trajectory":        0.10,
        "entity_resolution": 0.20,
    },
    "analysis": {
        "semantic_strength": 0.20,
        "grounding":         0.20,
        "topology":          0.15,
        "centroid_stability":0.20,
        "novelty_density":   0.15,
        "trajectory":        0.10,
        "entity_resolution": 0.00,
    },
    "debate": {
        "semantic_strength": 0.15,
        "grounding":         0.15,
        "topology":          0.20,
        "centroid_stability":0.15,
        "novelty_density":   0.20,
        "trajectory":        0.15,
        "entity_resolution": 0.00,
    },
    "problem_solving": {
        "semantic_strength": 0.15,
        "grounding":         0.25,
        "topology":          0.15,
        "centroid_stability":0.15,
        "novelty_density":   0.15,
        "trajectory":        0.15,
        "entity_resolution": 0.00,
    },
    "creative": {
        "semantic_strength": 0.10,
        "grounding":         0.05,
        "topology":          0.15,
        "centroid_stability":0.15,
        "novelty_density":   0.30,
        "trajectory":        0.25,
        "entity_resolution": 0.00,
    },
}
_WEIGHTS_DEFAULT = _WEIGHTS["analysis"]


# ---------------------------------------------------------------------------
# Per-term helpers
# ---------------------------------------------------------------------------

def _grounding_score(kb) -> float:
    """Weighted combination of domain_diversity and retrieval coverage.

    Two coverage channels:
    1. SEARCH-deposit coverage: 1 - parametric_content_ratio. Measures whether
       scout/developer agents filed SEARCH signal deposits (stigmergic traces).
    2. Atom-source coverage: source_count > 0, i.e., whether the SAFE pipeline
       or SEARCH signals left external source_tag traces in atoms. This catches
       the case where the validator found external sources via SAFE internal DDG
       calls without filing a SEARCH deposit (the source is real but wasn't
       stigmergically announced).

    Takes the MAX of the two channels so clusters with SAFE-found atom sources
    are not penalized for not having explicit SEARCH deposits.
    """
    search_coverage = 1.0 - kb.parametric_content_ratio  # [0, 1]
    # Atom-level coverage: source_count/5 capped at 0.6 (discounted vs. direct search)
    atom_coverage = min(0.6, kb.source_count / 5.0) if kb.source_count > 0 else 0.0
    retrieval_coverage = max(search_coverage, atom_coverage)
    if retrieval_coverage == 0.0:
        return 0.0
    # Blend diversity (quality of sources) with coverage (fact that searches happened)
    score = 0.6 * kb.domain_diversity + 0.4 * retrieval_coverage
    return round(min(1.0, score), 4)


def _topology_contribution(topo_expr, all_genomes: list) -> float:
    """Cell occupancy geometry term.

    1.0 if cluster is the sole occupant of its cell.
    Lower when multiple clusters share the cell.
    Anchor cells get a 1.5× multiplier (capped at 1.0 after normalization).
    Returns 0.5 when topology coords are unknown (no penalty, no bonus).
    """
    if topo_expr.coords is None:
        return 0.5
    # Count co-occupants
    n_co = topo_expr.cell_occupancy_rank  # 0 = sole occupant
    base = 1.0 / max(1, n_co + 1)        # 1.0 sole, 0.5 second, 0.33 third...
    if topo_expr.is_anchor:
        base = min(1.0, base * 1.5)
    return round(base, 4)


def _trajectory_score(traj) -> float:
    """Temporal growth quality.

    1.0 monotone growth with consolidation.
    0.5 monotone without consolidation.
    0.2 oscillating (fitness history exists but not monotone).
    0.0 no history yet (early run).
    """
    if not traj.fitness_history:
        return 0.0
    if traj.monotone_growth:
        return 1.0 if traj.consolidation_iteration is not None else 0.5
    return 0.2


def _entity_resolution_score(atoms: list) -> float:
    """Fraction of atom-text entities resolving in Wikidata.

    Requires USE_ENTITY_RESOLUTION=True and a live Wikidata API connection.
    Returns 0.5 (neutral) when gated off — does not penalize or reward.
    On the laptop path (RTX 3060, 6 GB VRAM) this is always off by default.
    """
    if not USE_ENTITY_RESOLUTION or not atoms:
        return 0.5
    try:
        return _wikidata_resolution_fraction(atoms)
    except Exception:
        return 0.5


def _wikidata_resolution_fraction(atoms: list) -> float:
    """Placeholder for Wikidata entity resolution (Stage 6 / Colab path).

    Not called on the laptop path (USE_ENTITY_RESOLUTION=False).
    """
    return 0.5


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_composite_fitness(
    genome,           # ClusterGenome
    task_type: Optional[str],
    all_genomes: list,
    cap_llm: float = CAP_LLM,
) -> tuple[float, dict]:
    """Compute composite fitness and per-term breakdown for one genome.

    Returns (composite_fitness, fitness_breakdown).

    fitness_breakdown maps term name -> contribution value (before weighting),
    so the synthesizer can render "this cluster's fitness comes mostly from
    grounding (0.72) and centroid stability (0.88); LLM-judged at cap (0.35)."

    semantic_strength is read from genome.fitness_breakdown["llm_judged"] if
    already present (set by an LLM quality call); otherwise falls back to
    verification_score as the best available proxy.
    """
    weights = _WEIGHTS.get(task_type or "", _WEIGHTS_DEFAULT)

    # semantic_strength: LLM-judged quality proxy.
    # If an explicit llm_judged value was set (e.g., by a quality scorer),
    # use it; otherwise fall back to verification_score from the genome's
    # atoms (Tier 2 — not fully independent, but better than 0).
    if "llm_judged" in genome.fitness_breakdown:
        llm_raw = float(genome.fitness_breakdown["llm_judged"])
    elif genome.atoms:
        total_w = sum(a.weight for a in genome.atoms) or 1.0
        llm_raw = sum(a.verification_score * a.weight for a in genome.atoms) / total_w
    else:
        llm_raw = 0.0

    terms: dict[str, float] = {
        "semantic_strength":  min(llm_raw, cap_llm),
        "grounding":          _grounding_score(genome.knowledge_base),
        "topology":           _topology_contribution(genome.topology_expression, all_genomes),
        "centroid_stability": genome.phenotype.centroid_stability,
        "novelty_density":    genome.phenotype.novelty_density,
        "trajectory":         _trajectory_score(genome.trajectory),
        "entity_resolution":  _entity_resolution_score(genome.atoms),
    }

    # Weighted sum — skip terms with weight=0 to avoid useless computation
    total_w = sum(weights.get(k, 0.0) for k in terms)
    if total_w == 0.0:
        composite = 0.0
    else:
        composite = sum(weights.get(k, 0.0) * v for k, v in terms.items()) / total_w

    breakdown = {k: round(v, 4) for k, v in terms.items()}
    breakdown["cap_llm"] = cap_llm
    breakdown["task_type"] = task_type or "default"

    return round(composite, 4), breakdown
