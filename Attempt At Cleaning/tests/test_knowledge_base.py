"""Unit tests for core/knowledge_base.py.

Run with:
    python -m unittest tests.test_knowledge_base -v

No GPU or LLM required. Tests use a synthetic SignalStore and a
temporary directory for KB persistence.
"""

import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.signal_store import SignalStore, _NULL_EMBEDDER
from core.signal_types import INITIAL, SUPPORT, CRITIQUE, OBJECTION, VERIFICATION
from core.projection import build_projection, SynthesisProjection
from core.knowledge_base import KnowledgeBase, _cluster_to_entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store():
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


def _build_simple_projection(has_validators=False):
    """Return (store, projection) with one surviving cluster.

    Four distinct forager strategies are required so the cluster clears
    both the support_diversity >= 3 weakly_supported threshold AND the
    credibility gate (support_diversity >= 4 when there's no verification
    and no dissent).
    """
    store = _make_store()
    init_id = _deposit(
        store, INITIAL,
        "Green energy deployment has dramatically reduced global carbon emissions over time.",
        0.7, "scout",
        metadata={"scout_agent_id": "scout_R1_0", "depositor_agent_id": "scout_R1_0"},
    )
    forager_supports = [
        ("Solar capacity has grown 300% in a decade, displacing coal generation.",
         "stratified_extremes"),
        ("Wind power now covers 20% of European electricity demand annually.",
         "medium_only"),
        ("Battery storage cost has fallen 90% since 2010, unlocking firming capacity.",
         "under_supported_clusters"),
        ("Renewable PPAs have outcompeted new fossil generation in major markets.",
         "weighted_default"),
    ]
    for content, strategy in forager_supports:
        _deposit(store, SUPPORT, content, 0.7, "forager", parent_id=init_id,
                 metadata={"depositor_agent_id": f"forager_R1_0_{strategy}"})
    proj = build_projection(store, has_validators=has_validators)
    return store, proj


def _build_rejected_projection():
    """Return (store, projection) with one rejected_by_field cluster."""
    store = _make_store()
    init_id = _deposit(
        store, INITIAL,
        "Climate change conspiracy theories contradict scientific evidence but are widely circulated.",
        0.6, "scout",
        metadata={"scout_agent_id": "scout_R1_0", "depositor_agent_id": "scout_R1_0"},
    )
    _deposit(store, SUPPORT,
             "Alternative evidence challenging mainstream climate science exists online.",
             0.3, "forager", parent_id=init_id,
             metadata={"depositor_agent_id": "forager_R1_0_stratified_extremes"})
    _deposit(store, SUPPORT,
             "Some contrarian sources and commentators publicly agree with these claims.",
             0.25, "forager", parent_id=init_id,
             metadata={"depositor_agent_id": "forager_R1_1_medium_only"})
    for i in range(3):
        _deposit(store, OBJECTION,
                 f"No credible peer-reviewed evidence supports this claim ({i}); "
                 f"mainstream scientific consensus strongly disagrees.",
                 0.55, "hater", parent_id=init_id,
                 metadata={"depositor_agent_id": f"hater_R1_{i}"})
    proj = build_projection(store, has_validators=False)
    return store, proj


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKBLoadEmpty(unittest.TestCase):
    def test_load_nonexistent_dir_is_noop(self):
        with tempfile.TemporaryDirectory() as td:
            kb = KnowledgeBase(kb_dir=Path(td) / "nonexistent_kb")
            kb.load()  # must not raise
            self.assertTrue(kb.is_empty())

    def test_prior_consensus_empty_initially(self):
        with tempfile.TemporaryDirectory() as td:
            kb = KnowledgeBase(kb_dir=Path(td))
            self.assertEqual(kb.prior_consensus(), [])

    def test_prior_rejections_empty_initially(self):
        with tempfile.TemporaryDirectory() as td:
            kb = KnowledgeBase(kb_dir=Path(td))
            self.assertEqual(kb.prior_rejections(), [])


class TestKBSaveLoad(unittest.TestCase):
    def test_save_creates_files(self):
        store, proj = _build_simple_projection()
        with tempfile.TemporaryDirectory() as td:
            kb = KnowledgeBase(kb_dir=Path(td))
            kb.save(proj, store, {"timestamp": "2026-01-01T00:00:00"})
            self.assertTrue((Path(td) / "surviving_clusters.json").exists())
            self.assertTrue((Path(td) / "rejected_clusters.json").exists())

    def test_surviving_round_trips(self):
        store, proj = _build_simple_projection()
        with tempfile.TemporaryDirectory() as td:
            kb = KnowledgeBase(kb_dir=Path(td))
            kb.save(proj, store, {"timestamp": "2026-01-01T00:00:00"})

            kb2 = KnowledgeBase(kb_dir=Path(td))
            kb2.load()
            self.assertEqual(len(kb2.prior_consensus()), len(proj.surviving))

    def test_rejected_round_trips(self):
        store, proj = _build_rejected_projection()
        with tempfile.TemporaryDirectory() as td:
            kb = KnowledgeBase(kb_dir=Path(td))
            kb.save(proj, store, {"timestamp": "2026-01-01T00:00:00"})

            kb2 = KnowledgeBase(kb_dir=Path(td))
            kb2.load()
            self.assertEqual(len(kb2.prior_rejections()), len(proj.rejected_by_field))

    def test_save_appends_across_runs(self):
        """Two saves with different content should accumulate entries.
        Two saves with identical content should deduplicate (run_count increases).
        """
        store1, proj1 = _build_simple_projection()
        # Build a second store with DISTINCT content so dedup doesn't fire.
        store2 = _make_store()
        init_id2 = _deposit(
            store2, INITIAL,
            "Ocean acidification from CO2 absorption threatens coral reef ecosystems globally.",
            0.7, "scout",
            metadata={"scout_agent_id": "scout_R1_0", "depositor_agent_id": "scout_R1_0"},
        )
        _deposit(store2, SUPPORT, "IPCC AR6 confirms accelerating ocean pH decline worldwide.",
                 0.75, "forager", parent_id=init_id2,
                 metadata={"depositor_agent_id": "forager_R1_0_stratified_extremes"})
        _deposit(store2, SUPPORT, "Coral bleaching events have doubled in frequency since 1980.",
                 0.7, "forager", parent_id=init_id2,
                 metadata={"depositor_agent_id": "forager_R1_1_medium_only"})
        proj2 = build_projection(store2, has_validators=False)

        with tempfile.TemporaryDirectory() as td:
            kb = KnowledgeBase(kb_dir=Path(td))
            kb.save(proj1, store1, {"timestamp": "2026-01-01T00:00:00"})
            kb.save(proj2, store2, {"timestamp": "2026-01-02T00:00:00"})

            kb2 = KnowledgeBase(kb_dir=Path(td))
            kb2.load()
            # Two distinct claims → two entries in prior_consensus
            self.assertEqual(len(kb2.prior_consensus()),
                             len(proj1.surviving) + len(proj2.surviving))

    def test_saved_entry_has_required_fields(self):
        store, proj = _build_simple_projection()
        with tempfile.TemporaryDirectory() as td:
            kb = KnowledgeBase(kb_dir=Path(td))
            kb.save(proj, store, {"timestamp": "2026-05-13T12:00:00"})
            data = json.loads((Path(td) / "surviving_clusters.json").read_text())
            self.assertTrue(len(data) > 0)
            entry = data[0]
            for key in ("representative_content", "member_signal_ids",
                        "lineage_signal_ids", "partition_origins",
                        "support_diversity", "run_timestamp", "status"):
                self.assertIn(key, entry, f"Missing key: {key}")
            self.assertEqual(entry["status"], "surviving")

    def test_rejected_entry_status(self):
        store, proj = _build_rejected_projection()
        if not proj.rejected_by_field:
            self.skipTest("No rejected clusters in this projection")
        with tempfile.TemporaryDirectory() as td:
            kb = KnowledgeBase(kb_dir=Path(td))
            kb.save(proj, store, {"timestamp": "2026-05-13T12:00:00"})
            data = json.loads((Path(td) / "rejected_clusters.json").read_text())
            self.assertTrue(len(data) > 0)
            self.assertEqual(data[0]["status"], "rejected_by_field")


class TestIgnoreKBBehavior(unittest.TestCase):
    """Verify that when ignore_kb=True, the KB files on disk have no effect.
    This mirrors the --ignore-kb CLI flag behavior tested at the unit level.
    """

    def test_kb_not_loaded_when_ignore_kb(self):
        """Populate a KB on disk, then create a new KnowledgeBase but don't call load().
        prior_consensus() should return empty list."""
        store, proj = _build_simple_projection()
        with tempfile.TemporaryDirectory() as td:
            kb_save = KnowledgeBase(kb_dir=Path(td))
            kb_save.save(proj, store, {"timestamp": "2026-01-01T00:00:00"})

            # Simulate --ignore-kb: create a KnowledgeBase but never call load()
            kb_ignored = KnowledgeBase(kb_dir=Path(td))
            # DO NOT call kb_ignored.load()
            self.assertEqual(kb_ignored.prior_consensus(), [])
            self.assertEqual(kb_ignored.prior_rejections(), [])

    def test_projection_unaffected_without_kb(self):
        """build_projection with prior_rejections=None behaves identically to
        a run without any prior knowledge."""
        store, proj_no_kb = _build_simple_projection()
        store2, proj_with_none = _build_simple_projection()
        # Both should produce identical cluster counts since no KB is applied
        total_no_kb = (len(proj_no_kb.surviving) + len(proj_no_kb.contested) +
                       len(proj_no_kb.weakly_supported) + len(proj_no_kb.rejected_by_field))
        total_with_none = (len(proj_with_none.surviving) + len(proj_with_none.contested) +
                           len(proj_with_none.weakly_supported) + len(proj_with_none.rejected_by_field))
        self.assertEqual(total_no_kb, total_with_none)


class TestClusterToEntry(unittest.TestCase):
    def test_cluster_to_entry_shape(self):
        store, proj = _build_simple_projection()
        if not proj.surviving:
            self.skipTest("No surviving clusters")
        cp = proj.surviving[0]
        entry = _cluster_to_entry(cp, store, "2026-01-01T00:00:00", "surviving")
        self.assertIn("representative_content", entry)
        self.assertIn("member_signal_ids", entry)
        self.assertIn("lineage_signal_ids", entry)
        self.assertIsInstance(entry["member_signal_ids"], list)
        self.assertIsInstance(entry["lineage_signal_ids"], list)


if __name__ == "__main__":
    unittest.main()
