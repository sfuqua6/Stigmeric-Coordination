"""B2: FitnessCompositor unit tests (design doc benchmark plan §13).

Run with:
    pytest tests/test_fitness_compositor.py -v

No GPU or LLM required.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fitness import (
    compute_composite_fitness,
    CAP_LLM,
    _WEIGHTS,
    _grounding_score,
    _topology_contribution,
    _trajectory_score,
)
from core.projection import (
    AtomFact, ClusterGenome, ClusterKnowledgeBase, TopologyExpression,
    Phenotype, FitnessTrajectory, GenomeSensitivity, GenomeRelations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_genome(
    atoms=None, kb=None, topo=None, phenotype=None, trajectory=None,
    sensitivity=None, fitness_breakdown=None,
) -> ClusterGenome:
    """Build a ClusterGenome with sensible defaults for testing."""
    if atoms is None:
        atoms = []
    if kb is None:
        kb = ClusterKnowledgeBase(
            queries_issued=[], source_domains=[], source_count=0,
            domain_diversity=0.0, parametric_content_ratio=1.0,
            cross_cluster_source_overlap=0.0,
        )
    if topo is None:
        topo = TopologyExpression(coords=None, cell_label=None,
                                  is_anchor=False, cell_occupancy_rank=0)
    if phenotype is None:
        phenotype = Phenotype(
            centroid=[], centroid_at_formation=[],
            centroid_drift=0.0, centroid_stability=1.0, novelty_density=0.5,
        )
    if trajectory is None:
        trajectory = FitnessTrajectory(
            formation_iteration=0, fitness_history=[], strength_history=[],
            member_count_history=[], monotone_growth=False,
            consolidation_iteration=None,
        )
    if sensitivity is None:
        sensitivity = GenomeSensitivity(
            load_bearing_atoms=[], marginal_atoms=[],
            support_removal_robustness=1.0, competing_takeover=None,
            topology_cells_at_risk=[],
        )
    return ClusterGenome(
        cluster_id="test_cluster",
        genome_hash="deadbeef00000000",
        formation_iteration=0,
        atoms=atoms,
        atom_graph={},
        topology_expression=topo,
        phenotype=phenotype,
        knowledge_base=kb,
        sensitivity=sensitivity,
        trajectory=trajectory,
        relations=GenomeRelations(parent_genomes=[], descendant_genomes=[],
                                  inter_cluster_edges=[]),
        composite_fitness=0.0,
        fitness_breakdown=fitness_breakdown or {},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFitnessCompositorBasics(unittest.TestCase):

    def test_output_in_unit_interval(self):
        """composite_fitness must be in [0, 1]."""
        genome = _make_genome()
        cf, _ = compute_composite_fitness(genome, task_type="debate", all_genomes=[genome])
        self.assertGreaterEqual(cf, 0.0)
        self.assertLessEqual(cf, 1.0)

    def test_llm_judged_capped(self):
        """semantic_strength contribution must not exceed CAP_LLM."""
        atoms = [
            AtomFact("a1", "Claim one.", weight=1.0, verification_score=0.99,
                     source_tag="source.org", query="q", extracted_from=[],
                     parent_cluster_id="test_cluster"),
        ]
        genome = _make_genome(atoms=atoms)
        _, breakdown = compute_composite_fitness(genome, task_type="debate",
                                                 all_genomes=[genome])
        self.assertLessEqual(breakdown["semantic_strength"], CAP_LLM + 1e-9)

    def test_weights_sum_to_one_per_task(self):
        """Each per-task weight table should sum to 1.0."""
        for task, w in _WEIGHTS.items():
            total = sum(w.values())
            self.assertAlmostEqual(total, 1.0, places=9,
                                   msg=f"weights for task={task!r} sum to {total}")

    def test_purely_parametric_cluster_has_zero_grounding(self):
        """ClusterKnowledgeBase with no sources → grounding=0.0."""
        kb = ClusterKnowledgeBase(
            queries_issued=[], source_domains=[], source_count=0,
            domain_diversity=0.0, parametric_content_ratio=1.0,
            cross_cluster_source_overlap=0.0,
        )
        self.assertEqual(_grounding_score(kb), 0.0)

    def test_grounded_cluster_has_positive_grounding(self):
        """ClusterKnowledgeBase with sources → grounding > 0."""
        kb = ClusterKnowledgeBase(
            queries_issued=["CO2 climate"], source_domains=["ipcc.ch", "nature.com"],
            source_count=3, domain_diversity=1.0, parametric_content_ratio=0.0,
            cross_cluster_source_overlap=0.0,
        )
        self.assertGreater(_grounding_score(kb), 0.0)
        self.assertLessEqual(_grounding_score(kb), 1.0)

    def test_atom_source_coverage_gives_nonzero_grounding(self):
        """source_count > 0 with parametric_content_ratio=1 still gives grounding > 0.

        This covers SAFE-pipeline sources found without a SEARCH deposit.
        """
        kb = ClusterKnowledgeBase(
            queries_issued=["q"], source_domains=["ipcc.ch"],
            source_count=2, domain_diversity=0.0, parametric_content_ratio=1.0,
            cross_cluster_source_overlap=0.0,
        )
        self.assertGreater(_grounding_score(kb), 0.0)

    def test_sole_cell_occupant_topology_contribution(self):
        """Cluster that is sole occupant of its topology cell → contribution 1.0."""
        topo = TopologyExpression(
            coords=("pos_a", "empirical"), cell_label="pos_a, empirical",
            is_anchor=False, cell_occupancy_rank=0,
        )
        self.assertAlmostEqual(_topology_contribution(topo, []), 1.0)

    def test_anchor_cell_topology_bonus(self):
        """Anchor-cell cluster gets 1.5× contribution (capped at 1.0)."""
        topo = TopologyExpression(
            coords=("pos_a", "empirical"), cell_label="pos_a, empirical",
            is_anchor=True, cell_occupancy_rank=0,
        )
        contribution = _topology_contribution(topo, [])
        self.assertAlmostEqual(contribution, 1.0)   # 1.5 * 1.0, capped

    def test_trajectory_score_empty(self):
        traj = FitnessTrajectory(0, [], [], [], False, None)
        self.assertEqual(_trajectory_score(traj), 0.0)

    def test_trajectory_score_monotone(self):
        traj = FitnessTrajectory(
            0, [(10, 0.3), (35, 0.45), (60, 0.55)], [], [], True, None
        )
        self.assertEqual(_trajectory_score(traj), 0.5)  # monotone, no consolidation

    def test_trajectory_score_consolidated(self):
        traj = FitnessTrajectory(
            0, [(10, 0.3), (35, 0.5), (60, 0.51)], [], [], True, 35
        )
        self.assertEqual(_trajectory_score(traj), 1.0)

    def test_trajectory_score_oscillating(self):
        traj = FitnessTrajectory(
            0, [(10, 0.5), (35, 0.3), (60, 0.48)], [], [], False, None
        )
        self.assertEqual(_trajectory_score(traj), 0.2)

    def test_fitness_breakdown_contains_all_terms(self):
        """fitness_breakdown must contain all 7 term keys."""
        genome = _make_genome()
        _, breakdown = compute_composite_fitness(genome, task_type="analysis",
                                                 all_genomes=[genome])
        expected_keys = {
            "semantic_strength", "grounding", "topology",
            "centroid_stability", "novelty_density",
            "trajectory", "entity_resolution",
        }
        self.assertTrue(expected_keys.issubset(set(breakdown.keys())))

    def test_higher_grounding_raises_fitness(self):
        """A well-grounded cluster should have higher fitness than a parametric one."""
        # Parametric genome
        g_param = _make_genome()
        cf_param, _ = compute_composite_fitness(g_param, "analysis", [g_param])

        # Grounded genome
        kb_grounded = ClusterKnowledgeBase(
            queries_issued=["q1", "q2"],
            source_domains=["ipcc.ch", "nature.com", "science.org"],
            source_count=5, domain_diversity=1.0, parametric_content_ratio=0.0,
            cross_cluster_source_overlap=0.0,
        )
        g_grounded = _make_genome(kb=kb_grounded)
        cf_grounded, _ = compute_composite_fitness(g_grounded, "analysis", [g_grounded])

        self.assertGreater(cf_grounded, cf_param,
                           "Grounded cluster should have higher fitness")


class TestFitnessCompositorPerTask(unittest.TestCase):
    """Verify per-task weight tables produce sensible ordering."""

    def test_coding_weights_grounding_heavy(self):
        """For coding, grounding weight >= semantic_strength weight."""
        w = _WEIGHTS["coding"]
        self.assertGreaterEqual(w["grounding"], w["semantic_strength"])

    def test_creative_weights_novelty_heavy(self):
        """For creative, novelty_density is the dominant term."""
        w = _WEIGHTS["creative"]
        self.assertEqual(w["novelty_density"], max(w.values()))

    def test_debate_topology_nonzero(self):
        """Debate weights topology substantially."""
        w = _WEIGHTS["debate"]
        self.assertGreater(w["topology"], 0.10)


if __name__ == "__main__":
    unittest.main()
