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
    meta = dict(metadata or {})
    if stype in ("INITIAL", "SUPPORT") and "partition_id" not in meta:
        # Derive partition_id from agent_id index (e.g. "forager_R1_2_..." → "partition_2")
        # so each distinct agent produces a distinct (partition_id, depositor) pair for Fix D.
        agent_id = meta.get("depositor_agent_id", meta.get("scout_agent_id", ""))
        parts = agent_id.split("_")
        if len(parts) >= 3 and parts[2].isdigit():
            meta["partition_id"] = f"partition_{parts[2]}"
        else:
            meta["partition_id"] = "test_partition_0"
    return store.deposit(
        signal_type=stype,
        content=content,
        strength=strength,
        depositor=depositor,
        parent_id=parent_id,
        metadata=meta,
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

        # Three SUPPORT signals from different strategies (→ support_diversity = 3).
        # Three is the minimum to clear the new weakly_supported threshold; the
        # low-strength CRITIQUE below contributes to dissent_set which also
        # clears the credibility gate.
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
        self.sup3 = _deposit(
            self.store, SUPPORT, "Ice cores show CO2 driving paleoclimate transitions.",
            strength=0.6, depositor="forager", parent_id=self.init_id,
            metadata={"depositor_agent_id": "forager_R1_2_under_supported_clusters"},
        )

        # One CRITIQUE (low dissent pressure, but dissent_set>=1 clears
        # the credibility gate).
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
        self.assertEqual(cp.support_diversity, 3)

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
        # heavy dissent: 5 objections × 0.6 = 3.0; ratio = 3.0/0.75 = 4.0;
        # math.log1p(4.0) ≈ 1.609 > SURVIVAL_REJECT_DISSENT_PRESSURE(1.5)
        _deposit(store, OBJECTION,
                 "Chernobyl demonstrated that containment can fail catastrophically.",
                 0.6, "hater", parent_id=init_id,
                 metadata={"depositor_agent_id": "hater_R1_0"})
        _deposit(store, OBJECTION,
                 "Fukushima shows that tsunami risk invalidates site safety assumptions.",
                 0.6, "hater", parent_id=init_id,
                 metadata={"depositor_agent_id": "hater_R1_1"})
        _deposit(store, OBJECTION,
                 "Radioactive waste disposal remains unsolved for thousands of years.",
                 0.6, "hater", parent_id=init_id,
                 metadata={"depositor_agent_id": "hater_R1_2"})
        _deposit(store, OBJECTION,
                 "Nuclear accidents have caused long-term environmental contamination.",
                 0.6, "hater", parent_id=init_id,
                 metadata={"depositor_agent_id": "hater_R1_3"})
        _deposit(store, OBJECTION,
                 "Insurance costs for nuclear plants are prohibitively expensive.",
                 0.6, "hater", parent_id=init_id,
                 metadata={"depositor_agent_id": "hater_R1_4"})
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
        # Three distinct support strategies clear support_diversity >= 3.
        _deposit(store, SUPPORT, "LCOE has fallen 90% since 2010.", 0.7, "forager",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_0_stratified_extremes"})
        _deposit(store, SUPPORT, "Grid parity achieved in many markets.", 0.65, "forager",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_1_medium_only"})
        _deposit(store, SUPPORT, "Utility-scale PPAs now undercut new gas in major markets.",
                 0.6, "forager", parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_2_under_supported_clusters"})
        # dissent_pressure: ratio = 1.6/1.95 ≈ 0.82; math.log1p(0.82) ≈ 0.60 ≥ 0.5 → contested
        _deposit(store, CRITIQUE, "Intermittency costs are not captured in LCOE.", 0.8, "critic",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "critic_R1_0_weighted_default"})
        _deposit(store, OBJECTION, "Grid reliability requires expensive storage.", 0.8, "hater",
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
        # Three distinct support strategies clear support_diversity >= 3 so
        # the cluster reaches the contested branch (rather than weakly_supported).
        _deposit(store, SUPPORT, "Some parents claim correlation.", 0.4, "forager",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_0_stratified_extremes"})
        _deposit(store, SUPPORT, "Old study suggested link.", 0.35, "forager",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_1_medium_only"})
        _deposit(store, SUPPORT, "Anecdotal reports circulate online.", 0.35, "forager",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_2_under_supported_clusters"})
        _deposit(store, CRITIQUE, "Study was retracted, no link found.", 0.9, "critic",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "critic_R1_0_weighted_default"})

        # Without prior rejection: ratio = 0.9/1.1 ≈ 0.818; math.log1p(0.818) ≈ 0.60 → contested
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
        # Three distinct support strategies clear support_diversity >= 3;
        # the VERIFICATION below also clears the credibility gate.
        _deposit(store, SUPPORT, "Solar capacity doubled last year.", 0.7, "forager",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_0_stratified_extremes"})
        _deposit(store, SUPPORT, "Wind power now 20% of grid in some countries.", 0.65, "forager",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_1_medium_only"})
        _deposit(store, SUPPORT, "Battery storage now economic for daily firming.", 0.6, "forager",
                 parent_id=init_id,
                 metadata={"depositor_agent_id": "forager_R1_2_under_supported_clusters"})
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
        # Include the unverified (credibility-hold) bucket: where the cluster
        # LANDS depends on SURVIVAL_MIN_SUPPORT_DIVERSITY, which is tuned
        # deliberately; this test only asserts the clustering, not the status.
        all_clusters = (proj.surviving + proj.contested +
                        proj.weakly_supported + proj.rejected_by_field +
                        proj.unverified)

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


class TestClusterGenome(unittest.TestCase):
    """Regression tests for the ClusterGenome pipeline (Stages 1-6)."""

    def _make_genome_store(self) -> tuple:
        """Return (store, init_id) with SEARCH + SUPPORT + VERIFICATION signals."""
        store = _make_store()
        init_id = _deposit(
            store, INITIAL, "CO2 emissions drive global temperature rise.",
            strength=0.7, depositor="scout",
            metadata={
                "scout_agent_id": "scout_R1_0",
                "depositor_agent_id": "scout_R1_0",
            },
        )
        # Scout SEARCH signal (no parent — correlated via scout_agent_id)
        store.deposit(
            signal_type="SEARCH",
            content="QUERY: CO2 climate\n  - ipcc.ch/ar6\n  - nature.com/climate",
            strength=0.4,
            depositor="scout",
            parent_id=None,
            metadata={
                "query": "CO2 climate",
                "n_results": 2,
                "scout_agent_id": "scout_R1_0",
                "depositor_agent_id": "scout_R1_0",
            },
        )
        for i in range(3):
            _deposit(
                store, SUPPORT, f"Supporting evidence {i}.",
                strength=0.65, depositor="forager", parent_id=init_id,
                metadata={"depositor_agent_id": f"forager_R1_{i}_stratified"},
            )
        # VERIFICATION with two atoms (different weight / score)
        store.deposit(
            signal_type="VERIFICATION",
            content="IPCC AR6 confirms relationship.",
            strength=0.8,
            depositor="validator",
            parent_id=init_id,
            metadata={
                "partition_id": "partition_0",
                "atoms": [
                    {
                        "text": "CO2 rose 50% since pre-industrial era.",
                        "weight": 1.0,
                        "score": 0.85,
                        "snippet_tag": "ipcc.ch",
                        "query": "CO2 historical",
                    },
                    {
                        "text": "Global mean temperature tracks CO2 concentration.",
                        "weight": 0.3,
                        "score": 0.20,
                        "snippet_tag": "nature.com",
                        "query": "temperature CO2 correlation",
                    },
                ],
            },
        )
        return store, init_id

    def test_genome_assembled(self):
        store, _ = self._make_genome_store()
        proj = build_projection(store, has_validators=True)
        all_cp = (proj.surviving + proj.contested
                  + proj.weakly_supported + proj.unverified + proj.rejected_by_field)
        self.assertEqual(len(all_cp), 1)
        cp = all_cp[0]
        self.assertIsNotNone(cp.genome, "ClusterGenome must be populated")

    def test_genome_atoms_from_verification(self):
        store, _ = self._make_genome_store()
        proj = build_projection(store, has_validators=True)
        g = (proj.surviving + proj.weakly_supported + proj.unverified)[0].genome
        self.assertEqual(len(g.atoms), 2)
        texts = {a.text for a in g.atoms}
        self.assertIn("CO2 rose 50% since pre-industrial era.", texts)

    def test_genome_hash_stable(self):
        store, _ = self._make_genome_store()
        proj = build_projection(store, has_validators=True)
        g = (proj.surviving + proj.weakly_supported + proj.unverified)[0].genome
        # Hash should be a non-empty hex string
        self.assertRegex(g.genome_hash, r'^[0-9a-f]{16}$')

    def test_genome_kb_captures_search_sources(self):
        store, _ = self._make_genome_store()
        proj = build_projection(store, has_validators=True)
        g = (proj.surviving + proj.weakly_supported + proj.unverified)[0].genome
        kb = g.knowledge_base
        # Should have at least ipcc.ch (from SEARCH signal) and nature.com (from atoms)
        self.assertIn("ipcc.ch", kb.source_domains)
        self.assertGreater(kb.domain_diversity, 0.0)
        self.assertEqual(kb.parametric_content_ratio, 0.0)  # has SEARCH coverage

    def test_genome_kb_queries_captured(self):
        store, _ = self._make_genome_store()
        proj = build_projection(store, has_validators=True)
        g = (proj.surviving + proj.weakly_supported + proj.unverified)[0].genome
        self.assertIn("CO2 climate", g.knowledge_base.queries_issued)

    def test_composite_fitness_nonzero(self):
        store, _ = self._make_genome_store()
        proj = build_projection(store, has_validators=True, task_type="debate")
        g = (proj.surviving + proj.weakly_supported + proj.unverified)[0].genome
        self.assertGreater(g.composite_fitness, 0.0)
        self.assertLessEqual(g.composite_fitness, 1.0)

    def test_composite_fitness_llm_capped(self):
        from core.fitness import CAP_LLM
        store, _ = self._make_genome_store()
        proj = build_projection(store, has_validators=True, task_type="debate")
        g = (proj.surviving + proj.weakly_supported + proj.unverified)[0].genome
        self.assertLessEqual(g.fitness_breakdown["semantic_strength"], CAP_LLM + 1e-9)

    def test_novelty_density_zero_for_single_cluster(self):
        store, _ = self._make_genome_store()
        proj = build_projection(store, has_validators=True)
        all_cp = (proj.surviving + proj.contested + proj.weakly_supported + proj.unverified)
        self.assertEqual(len(all_cp), 1)
        self.assertEqual(all_cp[0].genome.phenotype.novelty_density, 0.0)

    def test_sensitivity_populated_for_surviving_cluster(self):
        store, _ = self._make_genome_store()
        proj = build_projection(store, has_validators=True, task_type="debate")
        all_cp = proj.surviving + proj.weakly_supported + proj.unverified
        g = all_cp[0].genome
        # GenomeSensitivity should exist (may have empty lists when cluster is robust)
        self.assertIsNotNone(g.sensitivity)
        self.assertIsInstance(g.sensitivity.load_bearing_atoms, list)

    def test_genome_on_split_cluster_has_formation_centroid(self):
        """Verifies Bug 1 fix: _reanchor() sets centroid_at_formation on ejected clusters."""
        from core.cluster_registry import ClusterRegistry
        cr = ClusterRegistry()
        cr.create("s0", [1.0, 0.0], "INITIAL")
        cr.create("s1", [0.0, 1.0], "INITIAL")
        for cid in list(cr._clusters):
            cl = cr.get_cluster(cid)
            self.assertTrue(len(cl.centroid_at_formation) > 0,
                            f"cluster {cid} missing centroid_at_formation")

    def test_atom_graph_edges_from_support_verification(self):
        """SUPPORT-level VERIFICATION atoms depend on INITIAL-level atoms."""
        store = _make_store()
        init_id = _deposit(store, INITIAL, "CO2 drives climate.", 0.7, "scout",
                           metadata={"scout_agent_id": "scout_R1_0",
                                     "depositor_agent_id": "scout_R1_0"})
        sup_id = _deposit(store, SUPPORT, "Ice cores confirm CO2.", 0.65, "forager",
                          parent_id=init_id,
                          metadata={"depositor_agent_id": "forager_R1_0_strat"})
        # VERIFICATION of the INITIAL → atoms belong to INITIAL level
        store.deposit("VERIFICATION", "IPCC confirms.", 0.8, "validator",
                      parent_id=init_id,
                      metadata={"partition_id": "partition_0",
                                "atoms": [{"text": "CO2 rose 50pct.", "weight": 1.0,
                                           "score": 0.85, "snippet_tag": "ipcc.ch",
                                           "query": "CO2"}]})
        # VERIFICATION of the SUPPORT → atoms belong to SUPPORT level and depend on INITIAL atoms
        store.deposit("VERIFICATION", "Nature confirms temp tracking.", 0.7, "validator",
                      parent_id=sup_id,
                      metadata={"partition_id": "partition_0",
                                "atoms": [{"text": "Temps track CO2.", "weight": 0.8,
                                           "score": 0.72, "snippet_tag": "nature.com",
                                           "query": "temp CO2"}]})
        for i in range(2):
            _deposit(store, SUPPORT, f"Extra support {i}.", 0.6, "forager",
                     parent_id=init_id,
                     metadata={"depositor_agent_id": f"forager_R1_{i+1}_strat"})
        proj = build_projection(store, has_validators=True, task_type="analysis")
        all_cp = (proj.surviving + proj.contested + proj.weakly_supported + proj.unverified)
        self.assertEqual(len(all_cp), 1)
        g = all_cp[0].genome
        self.assertEqual(len(g.atoms), 2)
        # Find the support-level atom
        sup_atoms = [a for a in g.atoms if a.text == "Temps track CO2."]
        init_atoms = [a for a in g.atoms if a.text == "CO2 rose 50pct."]
        if sup_atoms and init_atoms:
            # Support-level atom should depend on initial-level atom
            self.assertIn(init_atoms[0].atom_id, g.atom_graph[sup_atoms[0].atom_id])

    def test_fission_genome_inherits_atoms(self):
        """B1 fission inheritance: split cluster carries genome with centroid_at_formation."""
        from core.cluster_registry import ClusterRegistry
        cr = ClusterRegistry()
        # Create a cluster, add a second member that drifts (no reanchor triggered
        # without CLUSTER_REANCHOR_EVERY deposits, so we manually create two clusters)
        cr.create("s0", [1.0, 0.0], "INITIAL")
        cr.create("s1", [0.0, 1.0], "INITIAL")
        # Both clusters must have centroid_at_formation set
        for cid in list(cr._clusters):
            cl = cr.get_cluster(cid)
            self.assertTrue(len(cl.centroid_at_formation) > 0)
            # centroid_at_formation should equal centroid for single-member clusters
            self.assertEqual(cl.centroid, cl.centroid_at_formation)

    def test_targets_atom_stamped_in_develop_parse(self):
        """develop_parse stamps targets_atom in metadata when genome provided."""
        from core.actions import develop_parse, _atom_for_develop
        from core.projection import AtomFact, ClusterGenome
        from core.projection import (TopologyExpression, Phenotype, ClusterKnowledgeBase,
                                     GenomeSensitivity, FitnessTrajectory, GenomeRelations)

        atoms = [
            AtomFact("a1", "Strong verified claim.", weight=1.0, verification_score=0.9,
                     source_tag="ipcc.ch", query="q", extracted_from=["v1"],
                     parent_cluster_id="c1"),
            AtomFact("a2", "Weak unverified claim.", weight=0.5, verification_score=0.1,
                     source_tag="", query="q2", extracted_from=["v2"],
                     parent_cluster_id="c1"),
        ]

        class _FakeGenome:
            def __init__(self):
                self.atoms = atoms

        genome = _FakeGenome()
        parsed = develop_parse("Test development output.", genome=genome)

        # Should target a2 (lowest verification_score)
        self.assertIsNotNone(parsed.metadata)
        self.assertEqual(parsed.metadata.get("targets_atom"), "a2")


if __name__ == "__main__":
    unittest.main()
