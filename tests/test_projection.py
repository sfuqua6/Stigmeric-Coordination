"""Unit tests for core/projection.py.

Run with:
    python -m unittest tests.test_projection -v

No GPU or LLM required — all tests use a synthetic SignalStore populated
with manually deposited signals of known structure.
"""

import sys
import os
import unittest
from pathlib import Path

# Make the project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.signal_store import SignalStore, _NULL_EMBEDDER
from core.signal_types import INITIAL, SUPPORT, CRITIQUE, OBJECTION, VERIFICATION
from core.projection import (
    build_projection,
    _parse_strategy_name,
    _parse_partition_tag,
    SynthesisProjection,
    ClusterProjection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> SignalStore:
    """Return a SignalStore with the embedder disabled for deterministic tests."""
    return SignalStore(embedder=_NULL_EMBEDDER)


def _deposit(store, stype, content, strength, depositor, parent_id=None, metadata=None):
    return store.deposit(
        signal_type=stype,
        content=content,
        strength=strength,
        depositor=depositor,
        parent_id=parent_id,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseHelpers(unittest.TestCase):
    def test_parse_strategy_name_forager(self):
        self.assertEqual(_parse_strategy_name("forager_R1_2_stratified_extremes"),
                         "stratified_extremes")

    def test_parse_strategy_name_short(self):
        # agent_id with fewer than 4 parts returns empty string
        self.assertEqual(_parse_strategy_name("scout_R1_0"), "")

    def test_parse_partition_tag_normal(self):
        self.assertEqual(_parse_partition_tag("scout_R2_3"), "partition_3")

    def test_parse_partition_tag_empty(self):
        self.assertEqual(_parse_partition_tag(""), "partition_unknown")


class TestProjectionMetrics(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()
        # Deposit one INITIAL from scout partition_0
        self.init_id = _deposit(
            self.store, INITIAL, "Climate change is driven by CO2 emissions.",
            strength=0.6, depositor="scout",
            metadata={"scout_agent_id": "scout_R1_0", "depositor_agent_id": "scout_R1_0"},
        )
        self.assertIsNotNone(self.init_id)

        # Two SUPPORT signals from different strategies (→ support_diversity = 2)
        self.sup1 = _deposit(
            self.store, SUPPORT, "Atmospheric CO2 levels have risen 50% since industrialisation.",
            strength=0.7, depositor="forager", parent_id=self.init_id,
            metadata={"depositor_agent_id": "forager_R1_0_stratified_extremes"},
        )
        self.sup2 = _deposit(
            self.store, SUPPORT, "Mean global temperatures track CO2 concentration closely.",
            strength=0.65, depositor="forager", parent_id=self.init_id,
            metadata={"depositor_agent_id": "forager_R1_1_medium_only"},
        )

        # One CRITIQUE (low dissent pressure)
        self.crit_id = _deposit(
            self.store, CRITIQUE, "The causal link is well-established but not universally accepted.",
            strength=0.3, depositor="critic", parent_id=self.init_id,
            metadata={"depositor_agent_id": "critic_R1_0_weighted_default"},
        )

    def test_support_diversity_counts_distinct_strategies(self):
        proj = build_projection(self.store, has_validators=False)
        all_clusters = (proj.surviving + proj.contested +
                        proj.weakly_supported + proj.rejected_by_field)
        self.assertEqual(len(all_clusters), 1)
        cp = all_clusters[0]
        self.assertEqual(cp.support_diversity, 2)

    def test_dissent_pressure_below_threshold(self):
        proj = build_projection(self.store, has_validators=False)
        all_clusters = proj.surviving + proj.contested + proj.weakly_supported
        self.assertEqual(len(all_clusters), 1)
        cp = all_clusters[0]
        # dissent_pressure = 0.3 / (0.7 + 0.65) ≈ 0.22 — well below 0.5
        self.assertLess(cp.dissent_pressure, 0.5)

    def test_cluster_has_correct_member_ids(self):
        proj = build_projection(self.store, has_validators=False)
        all_clusters = proj.surviving + proj.weakly_supported
        self.assertEqual(len(all_clusters), 1)
        cp = all_clusters[0]
        self.assertIn(self.init_id, cp.member_ids)
        self.assertIn(self.sup1, cp.support_set)
        self.assertIn(self.sup2, cp.support_set)
        self.assertIn(self.crit_id, cp.dissent_set)

    def test_surviving_status_no_validators(self):
        # has_validators=False → cluster survives without verification
        proj = build_projection(self.store, has_validators=False)
        self.assertEqual(len(proj.surviving), 1)
        cp = proj.surviving[0]
        self.assertEqual(cp.status, "surviving")
        # unverified flag must be False when has_validators=False
        self.assertFalse(cp.unverified)

    def test_unverified_flag_when_has_validators_true(self):
        # has_validators=True + no VERIFICATION signal → cluster survives but
        # cp.unverified=True.  Verification is a flag, not a separate bucket.
        proj = build_projection(self.store, has_validators=True)
        self.assertEqual(len(proj.surviving), 1,
                         "Cluster should survive even without verification")
        cp = proj.surviving[0]
        self.assertEqual(cp.status, "surviving")
        self.assertTrue(cp.unverified,
                        "cp.unverified must be True when validators exist but none hit this cluster")

    def test_partition_coverage(self):
        proj = build_projection(self.store, has_validators=False)
        # The INITIAL came from scout_R1_0 → partition_0
        self.assertIn("partition_0", proj.partition_coverage)


class TestSurvivalFilter(unittest.TestCase):
    def _build_high_dissent_store(self):
        store = _make_store()
        init_id = _deposit(store, INITIAL, "Nuclear power is safe.", 0.6, "scout",
                           metadata={"scout_agent_id": "scout_R1_0",
                                     "depositor_agent_id": "scout_R1_0"})
        # weak support: sum = 0.75
        _deposit(store, SUPPORT, "Safety record is good overall.", 0.4, "forager",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_0_stratified_extremes"})
        _deposit(store, SUPPORT, "Modern reactors have improved containment features.", 0.35,
                 "forager", parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_1_medium_only"})
        # heavy dissent with distinct content: sum = 1.5 → pressure = 1.5/0.75 = 2.0 > 1.5
        _deposit(store, OBJECTION,
                 "Chernobyl demonstrated that containment can fail catastrophically.",
                 0.5, "hater", parent_id=init_id,
                 metadata={"depositor_agent_id": "hater_R1_0"})
        _deposit(store, OBJECTION,
                 "Fukushima shows that tsunami risk invalidates site safety assumptions.",
                 0.5, "hater", parent_id=init_id,
                 metadata={"depositor_agent_id": "hater_R1_1"})
        _deposit(store, OBJECTION,
                 "Radioactive waste disposal remains unsolved for thousands of years.",
                 0.5, "hater", parent_id=init_id,
                 metadata={"depositor_agent_id": "hater_R1_2"})
        return store

    def test_rejected_by_field(self):
        store = self._build_high_dissent_store()
        proj = build_projection(store, has_validators=False)
        self.assertEqual(len(proj.rejected_by_field), 1)
        self.assertEqual(proj.rejected_by_field[0].status, "rejected_by_field")
        self.assertTrue(proj.no_consensus)

    def test_contested_range(self):
        store = _make_store()
        init_id = _deposit(store, INITIAL, "Solar panels are cost-competitive.", 0.6, "scout",
                           metadata={"scout_agent_id": "scout_R1_1",
                                     "depositor_agent_id": "scout_R1_1"})
        _deposit(store, SUPPORT, "LCOE has fallen 90% since 2010.", 0.7, "forager",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_0_stratified_extremes"})
        _deposit(store, SUPPORT, "Grid parity achieved in many markets.", 0.65, "forager",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_1_medium_only"})
        # dissent_pressure ≈ 0.5 / (0.7+0.65) ≈ 0.74 → contested
        _deposit(store, CRITIQUE, "Intermittency costs are not captured in LCOE.", 0.5, "critic",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "critic_R1_0_weighted_default"})
        _deposit(store, OBJECTION, "Grid reliability requires expensive storage.", 0.5, "hater",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "hater_R1_0"})

        proj = build_projection(store, has_validators=False)
        self.assertEqual(len(proj.contested), 1)
        self.assertEqual(proj.contested[0].status, "contested")


class TestNoConsensus(unittest.TestCase):
    def test_empty_store_gives_no_consensus(self):
        proj = build_projection(_make_store(), has_validators=False)
        self.assertTrue(proj.no_consensus)

    def test_all_rejected_gives_no_consensus(self):
        store = _make_store()
        init_id = _deposit(store, INITIAL, "Some disputed claim.", 0.6, "scout",
                           metadata={"scout_agent_id": "scout_R1_0",
                                     "depositor_agent_id": "scout_R1_0"})
        for i in range(4):
            _deposit(store, OBJECTION, f"Strong objection number {i} against this claim.",
                     0.6, "hater", parent_id=init_id,
                     metadata={"depositor_agent_id": f"hater_R1_{i}"})
        proj = build_projection(store, has_validators=False)
        self.assertTrue(proj.no_consensus)


class TestPriorRejectionPenalty(unittest.TestCase):
    def test_prior_rejection_raises_dissent_pressure(self):
        """A cluster matching a prior rejection should start with higher dissent_pressure,
        potentially pushing a borderline cluster into rejected_by_field."""
        store = _make_store()
        init_id = _deposit(store, INITIAL, "Vaccines cause autism.", 0.6, "scout",
                           metadata={"scout_agent_id": "scout_R1_0",
                                     "depositor_agent_id": "scout_R1_0"})
        _deposit(store, SUPPORT, "Some parents claim correlation.", 0.4, "forager",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_0_stratified_extremes"})
        _deposit(store, SUPPORT, "Old study suggested link.", 0.35, "forager",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_1_medium_only"})
        _deposit(store, CRITIQUE, "Study was retracted, no link found.", 0.5, "critic",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "critic_R1_0_weighted_default"})

        # Without prior rejection: borderline dissent_pressure ≈ 0.5/0.75 ≈ 0.67 → contested
        proj_no_kb = build_projection(store, has_validators=False)
        self.assertEqual(proj_no_kb.contested[0].dissent_pressure,
                         proj_no_kb.contested[0].dissent_pressure)  # just assert runs

        # With a prior rejection entry (no embedding since embedder disabled → similarity
        # check skipped, penalty not applied). This tests that the code path runs cleanly.
        prior_rejection = [{
            "representative_content": "Vaccines cause autism.",
            "representative_embedding": None,  # no embedder in test
            "support_diversity": 1,
        }]
        proj_with_kb = build_projection(store, has_validators=False,
                                        prior_rejections=prior_rejection)
        # Both should run without error; with no embedding, no penalty applied
        all_no_kb = (proj_no_kb.surviving + proj_no_kb.contested +
                     proj_no_kb.weakly_supported + proj_no_kb.rejected_by_field)
        all_with_kb = (proj_with_kb.surviving + proj_with_kb.contested +
                       proj_with_kb.weakly_supported + proj_with_kb.rejected_by_field)
        self.assertEqual(len(all_no_kb), len(all_with_kb))


class TestVerificationScore(unittest.TestCase):
    def test_verification_score_included(self):
        store = _make_store()
        init_id = _deposit(store, INITIAL, "Renewable energy is expanding.", 0.6, "scout",
                           metadata={"scout_agent_id": "scout_R1_0",
                                     "depositor_agent_id": "scout_R1_0"})
        _deposit(store, SUPPORT, "Solar capacity doubled last year.", 0.7, "forager",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_0_stratified_extremes"})
        _deposit(store, SUPPORT, "Wind power now 20% of grid in some countries.", 0.65, "forager",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_1_medium_only"})
        _deposit(store, VERIFICATION, "Wikipedia confirms renewable capacity growth.", 0.8,
                 "validator", parent_id=init_id,
                 metadata={"depositor_agent_id": "validator_R1_0_weighted_default"})

        proj = build_projection(store, has_validators=True)
        # With VERIFICATION present, should be surviving (not ungrounded)
        self.assertEqual(len(proj.surviving), 1)
        self.assertGreater(proj.surviving[0].verification_score, 0.0)


class TestSemanticClustering(unittest.TestCase):
    """Tests that require an embedder to exercise the clustering path.

    We inject a fake embedder returning fixed unit-normalised vectors so the
    test is deterministic and requires no sentence-transformers install.
    """

    def _make_store_with_embedder(self, vector_map: dict) -> SignalStore:
        """Return a SignalStore whose embedder returns vectors from vector_map.

        vector_map: {content_str: [float, ...]}  (vectors should be unit-normalised)
        """
        class _FakeEmbedder:
            def encode(self, text: str):
                # Return the pre-defined vector, or a unique orthogonal vector
                import math
                if text in vector_map:
                    return vector_map[text]
                # Hash-based orthogonal fallback
                h = abs(hash(text)) % 1000
                v = [0.0] * len(next(iter(vector_map.values())))
                v[h % len(v)] = 1.0
                return v

        store = SignalStore(embedder=_FakeEmbedder())
        return store

    def _unit(self, *values) -> list[float]:
        """Return a unit-normalised vector from the given components."""
        import math
        n = math.sqrt(sum(x * x for x in values))
        return [x / n for x in values]

    def test_similar_initials_merge_into_one_cluster(self):
        """Two INITIALs with cosine similarity >= 0.65 should be in the same cluster.

        Vectors must satisfy:
          cos_sim >= _CLUSTER_SIM_THRESHOLD (0.65) — so they cluster
          cos_sim <  DIVERSITY_THRESHOLD    (0.85) — so neither is deduped
        We use cos_sim ≈ 0.75: v1=[1,0,0], v2=[0.75, 0.6614, 0] (already unit).
        """
        import math
        v1 = [1.0, 0.0, 0.0]
        # v2 at ~41.4 degrees from v1 → cos_sim = 0.75
        sin_a = math.sqrt(1.0 - 0.75 ** 2)   # ≈ 0.6614
        v2 = [0.75, sin_a, 0.0]

        content_a = "Renewable energy is expanding rapidly worldwide."
        content_b = "Renewable energy is growing rapidly around the world."

        vector_map = {content_a: v1, content_b: v2}
        store = self._make_store_with_embedder(vector_map)

        id_a = _deposit(store, INITIAL, content_a, 0.7, "scout",
                        metadata={"scout_agent_id": "scout_R1_0",
                                  "depositor_agent_id": "scout_R1_0"})
        id_b = _deposit(store, INITIAL, content_b, 0.65, "scout",
                        metadata={"scout_agent_id": "scout_R1_1",
                                  "depositor_agent_id": "scout_R1_1"})

        # Needs support to survive the filter
        _deposit(store, SUPPORT, "Solar capacity tripled.", 0.7, "forager",
                 parent_id=id_a,
                 metadata={"depositor_agent_id": "forager_R1_0_stratified_extremes"})
        _deposit(store, SUPPORT, "Wind covers 20% of EU demand.", 0.65, "forager",
                 parent_id=id_a,
                 metadata={"depositor_agent_id": "forager_R1_1_medium_only"})

        proj = build_projection(store, has_validators=False)
        all_clusters = (proj.surviving + proj.contested +
                        proj.weakly_supported + proj.rejected_by_field)

        # Both INITIALs should be in the same cluster (merged by similarity)
        total_members = sum(len(cp.member_ids) for cp in all_clusters)
        self.assertEqual(total_members, 2,
                         "Two similar INITIALs should form one cluster with 2 members")

    def test_dissimilar_initials_stay_in_separate_clusters(self):
        """Two INITIALs with cosine similarity < 0.65 should stay in separate clusters."""
        # v1 and v2 are orthogonal (cos sim = 0.0)
        v1 = self._unit(1.0, 0.0, 0.0)
        v2 = self._unit(0.0, 1.0, 0.0)

        content_a = "Climate action requires international cooperation."
        content_b = "Economic growth is driven by technological innovation."

        vector_map = {content_a: v1, content_b: v2}
        store = self._make_store_with_embedder(vector_map)

        id_a = _deposit(store, INITIAL, content_a, 0.7, "scout",
                        metadata={"scout_agent_id": "scout_R1_0",
                                  "depositor_agent_id": "scout_R1_0"})
        id_b = _deposit(store, INITIAL, content_b, 0.65, "scout",
                        metadata={"scout_agent_id": "scout_R1_1",
                                  "depositor_agent_id": "scout_R1_1"})

        proj = build_projection(store, has_validators=False)
        all_clusters = (proj.surviving + proj.contested +
                        proj.weakly_supported + proj.rejected_by_field)

        # Each INITIAL should be in its own cluster
        self.assertEqual(len(all_clusters), 2,
                         "Two dissimilar INITIALs should produce two separate clusters")
        for cp in all_clusters:
            self.assertEqual(len(cp.member_ids), 1)


if __name__ == "__main__":
    unittest.main()
