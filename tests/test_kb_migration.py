"""Regression tests for KB migration and junk-filter enforcement at KB save time.

Tests that:
  - Junk entries are excluded from KB save (is_junk_output at _build_entries).
  - Clean entries survive save and load round-trip.
  - Dedup merge works: second save of same content updates match_count, not appends.

Run with:
    python -m unittest tests.test_kb_migration -v

No GPU or LLM required — uses SignalStore with embedder disabled.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.signal_store import SignalStore, _NULL_EMBEDDER
from core.signal_types import INITIAL, SUPPORT, CRITIQUE
from core.projection import build_projection
from core.knowledge_base import KnowledgeBase, _SURVIVING_FILE, _REJECTED_FILE


def _make_store() -> SignalStore:
    return SignalStore(embedder=_NULL_EMBEDDER)


def _deposit(store, stype, content, strength=0.7, depositor="scout",
             parent_id=None, metadata=None):
    meta = dict(metadata or {})
    if stype in ("INITIAL", "SUPPORT") and "partition_id" not in meta:
        # Derive partition from agent index so distinct foragers get distinct
        # (partition_id, depositor) pairs — required for support_diversity > 1.
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


class TestKBJunkFilter(unittest.TestCase):
    """Junk entries must be excluded at KB save, even if they made it past deposit."""

    def _build_surviving_store(self, claim: str) -> tuple[SignalStore, object]:
        store = _make_store()
        iid = _deposit(store, INITIAL, claim,
                       metadata={"scout_agent_id": "scout_R1_0"})
        if iid is None:
            self.skipTest("deposit rejected (store dedup)")
        # Four distinct forager supports with distinct agent indices so each
        # gets a unique (partition_id, depositor) pair — satisfies diversity >= 4.
        for i, strategy in enumerate((
            "stratified_extremes", "medium_only",
            "under_supported_clusters", "weighted_default",
        )):
            _deposit(store, SUPPORT, f"Evidence supports the claim ({strategy}).",
                     depositor="forager", parent_id=iid,
                     metadata={"depositor_agent_id": f"forager_R1_{i}_{strategy}"})
        proj = build_projection(store, has_validators=False)
        return store, proj

    def test_junk_claim_excluded_from_kb(self):
        """A claim that passes is_junk_output=True must not appear in saved KB."""
        junk_claim = "[No claim yet] Hmm... Frogs have"
        store, proj = self._build_surviving_store(junk_claim)

        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(kb_dir=Path(tmpdir))
            kb.save(proj, store, {"task_type": "debate", "user_prompt": "test"})

            surviving_path = Path(tmpdir) / _SURVIVING_FILE
            if surviving_path.exists():
                data = json.loads(surviving_path.read_text())
                contents = [e["representative_content"] for e in data]
                self.assertNotIn(junk_claim, contents,
                                 "Junk claim must not be persisted to KB")

    def test_clean_claim_survives_kb(self):
        """A clean claim that crosses support_diversity must appear in saved KB."""
        clean_claim = (
            "Carbon dioxide concentration has exceeded 420 ppm, "
            "the highest level in 800,000 years of ice core records."
        )
        store, proj = self._build_surviving_store(clean_claim)

        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(kb_dir=Path(tmpdir))
            kb.save(proj, store, {"task_type": "debate", "user_prompt": "test"})

            surviving_path = Path(tmpdir) / _SURVIVING_FILE
            self.assertTrue(surviving_path.exists(), "surviving_clusters.json must be written")
            data = json.loads(surviving_path.read_text())
            self.assertGreater(len(data), 0, "At least one surviving cluster must be saved")


class TestKBRoundTrip(unittest.TestCase):
    """Save then load produces the same entries (no silent loss)."""

    def test_save_load_roundtrip(self):
        store = _make_store()
        iid = _deposit(store, INITIAL,
                       "Global temperatures have risen 1.1°C since the industrial revolution.",
                       metadata={"scout_agent_id": "scout_R1_0"})
        if iid is None:
            self.skipTest("deposit rejected")
        # Four distinct forager strategies with distinct agent indices so each
        # gets a unique (partition_id, depositor) pair — satisfies diversity >= 4.
        for i, strategy in enumerate((
            "stratified_extremes", "medium_only",
            "under_supported_clusters", "weighted_default",
        )):
            _deposit(store, SUPPORT, f"Confirming evidence ({strategy}).",
                     depositor="forager", parent_id=iid,
                     metadata={"depositor_agent_id": f"forager_R1_{i}_{strategy}"})
        proj = build_projection(store, has_validators=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(kb_dir=Path(tmpdir))
            kb.save(proj, store, {"task_type": "debate", "user_prompt": "test"})

            kb2 = KnowledgeBase(kb_dir=Path(tmpdir))
            kb2.load()
            self.assertFalse(kb2.is_empty(), "KB must not be empty after load")
            self.assertGreater(len(kb2.prior_consensus()), 0)


class TestKBDedup(unittest.TestCase):
    """Saving the same cluster twice must update match_count, not create a duplicate."""

    def test_dedup_on_second_save(self):
        """Without embeddings, dedup falls back to string-sim. Two identical
        representative_content strings must deduplicate."""
        claim = (
            "Sea level rise is accelerating due to polar ice melt, "
            "threatening coastal populations worldwide."
        )
        store = _make_store()
        iid = _deposit(store, INITIAL, claim,
                       metadata={"scout_agent_id": "scout_R1_0"})
        if iid is None:
            self.skipTest("deposit rejected")
        _deposit(store, SUPPORT, "IPCC AR6 confirms accelerating sea level rise.",
                 depositor="forager", parent_id=iid,
                 metadata={"depositor_agent_id": "forager_R1_0_stratified_extremes"})
        _deposit(store, SUPPORT, "Tide gauge data shows 3.6 mm/year acceleration.",
                 depositor="forager", parent_id=iid,
                 metadata={"depositor_agent_id": "forager_R1_1_medium_only"})

        proj = build_projection(store, has_validators=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(kb_dir=Path(tmpdir))
            kb.save(proj, store, {"task_type": "debate", "user_prompt": "test"})
            count_after_first = len(json.loads(
                (Path(tmpdir) / _SURVIVING_FILE).read_text()
            ))

            # Second save of the same projection
            kb2 = KnowledgeBase(kb_dir=Path(tmpdir))
            kb2.load()
            kb2.save(proj, store, {"task_type": "debate", "user_prompt": "test"})
            count_after_second = len(json.loads(
                (Path(tmpdir) / _SURVIVING_FILE).read_text()
            ))

            self.assertEqual(
                count_after_first, count_after_second,
                "Duplicate save must not grow KB entry count"
            )


if __name__ == "__main__":
    unittest.main()
