"""Tests for citation-tag aliasing and KB-aware scout novelty.

Aliasing: composers replace [TYPE_NNNNN] provenance tags with short [S#]
aliases (mid-size models drop long tags in rewrites — 3 consecutive real
runs at 0/6 retention) and re-map deterministically afterward.

KB novelty: prior-consensus claims registered as novelty references steer
scout claim selection away from what the knowledge base already holds.

Run with:
    pytest tests/test_alias_and_kb_novelty.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.signal_store import SignalStore, _NULL_EMBEDDER
from core.signal_types import INITIAL
from core.actions import select_novel_claim
from agents.synthesizer import Synthesizer


def _make_store() -> SignalStore:
    return SignalStore(embedder=_NULL_EMBEDDER)


# ---------------------------------------------------------------------------
# Alias round-trip
# ---------------------------------------------------------------------------

class TestAliasRoundTrip(unittest.TestCase):
    def test_alias_and_unalias(self):
        texts = [
            "Claim one [INITIAL_00005] with support [SUPPORT_00012].",
            "Claim two [INITIAL_00007]; echoes claim one [INITIAL_00005].",
        ]
        aliased, mapping = Synthesizer._alias_citation_tags(texts)
        self.assertEqual(aliased[0],
                         "Claim one [S1] with support [S2].")
        # Repeated real tag reuses its alias.
        self.assertEqual(aliased[1],
                         "Claim two [S3]; echoes claim one [S1].")
        self.assertEqual(mapping["[S1]"], "[INITIAL_00005]")
        round_tripped = Synthesizer._unalias_citation_tags(aliased[0], mapping)
        self.assertEqual(round_tripped, texts[0])

    def test_double_digit_aliases_unambiguous(self):
        # [S12] must not be corrupted by replacing [S1] first.
        texts = [f"Claim [INITIAL_{i:05d}]." for i in range(12)]
        aliased, mapping = Synthesizer._alias_citation_tags(texts)
        self.assertIn("[S12]", aliased[11])
        out = Synthesizer._unalias_citation_tags(aliased[11], mapping)
        self.assertEqual(out, texts[11])

    def test_no_tags_passthrough(self):
        aliased, mapping = Synthesizer._alias_citation_tags(["No tags here."])
        self.assertEqual(aliased, ["No tags here."])
        self.assertEqual(mapping, {})


# ---------------------------------------------------------------------------
# KB novelty references
# ---------------------------------------------------------------------------

class TestKBNoveltyReferences(unittest.TestCase):
    def test_references_raise_similarity(self):
        store = _make_store()
        n = store.set_novelty_references(
            ["Banning private cars reduces carbon emissions in cities."])
        self.assertEqual(n, 1)
        sim = store.max_similarity_to_recent(
            "Banning private cars reduces carbon emissions in cities.",
            INITIAL)
        self.assertGreater(sim, 0.95)

    def test_selection_avoids_kb_known_claim(self):
        """A candidate matching KB prior consensus loses to a fresh one
        even when the store itself is empty."""
        store = _make_store()
        store.set_novelty_references(
            ["Banning private cars reduces carbon emissions in cities."])
        candidates = [
            "Banning private cars reduces carbon emissions in urban areas.",
            "Freight delivery costs rise when vans lose curbside access.",
        ]
        self.assertEqual(select_novel_claim(candidates, store),
                         candidates[1])

    def test_empty_and_blank_references(self):
        store = _make_store()
        self.assertEqual(store.set_novelty_references(["", "  ", None or ""]), 0)
        self.assertEqual(
            store.max_similarity_to_recent("anything", INITIAL), 0.0)

    def test_reference_cap(self):
        store = _make_store()
        texts = [f"Distinct prior claim number {i} about topic {i}."
                 for i in range(80)]
        self.assertEqual(store.set_novelty_references(texts, cap=50), 50)


if __name__ == "__main__":
    unittest.main()
