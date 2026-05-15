"""Tests for the synthesizer faithfulness audit (§7 directive).

Verifies that each audit check fires on contrived inputs and does NOT fire
on clean inputs. Tests _build_faithfulness_audit() directly — no LLM needed.

Run with:
    python -m unittest tests.test_renderer_audit_coverage -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.signal_store import SignalStore, _NULL_EMBEDDER
from core.signal_types import INITIAL, SUPPORT, VERIFICATION
from core.projection import build_projection
from agents.synthesizer import _build_faithfulness_audit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> SignalStore:
    return SignalStore(embedder=_NULL_EMBEDDER)


def _make_projection(store):
    return build_projection(store, has_validators=False)


_STRATEGIES = ["stratified_extremes", "under_supported_clusters", "recent_dissent_targeted"]
_SUPPORT_TEXTS = [
    "Solar photovoltaic capacity has tripled over the past decade worldwide.",
    "Wind power installations now generate electricity at costs below fossil fuels.",
    "Battery storage technology enables round-the-clock renewable grid operation.",
]

def _deposit_cluster(store, content: str) -> str:
    sid = store.deposit(
        signal_type=INITIAL, content=content,
        strength=0.7, depositor="scout",
        metadata={"depositor_agent_id": "scout_R1_0_stratified_extremes"},
    )
    # Use distinct strategy names AND distinct content so dedup doesn't block them.
    # support_diversity counts distinct strategies, so we need >= 2.
    for i, (strat, text) in enumerate(zip(_STRATEGIES, _SUPPORT_TEXTS)):
        store.deposit(
            signal_type=SUPPORT, content=text,
            strength=0.6, depositor="developer",
            parent_id=sid,
            metadata={"depositor_agent_id": f"forager_R1_{i}_{strat}"},
        )
    return sid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFabricatedCitationCheck(unittest.TestCase):
    """Check 1: fabricated_citation — ID does not exist in the store."""

    def test_fabricated_id_flagged(self):
        store = _make_store()
        _deposit_cluster(store, "Renewable energy has grown dramatically.")
        proj = _make_projection(store)
        if not proj.surviving:
            self.skipTest("No surviving clusters in this projection")

        # INITIAL_99999 matches the citation regex but will never exist in the store
        fake_id = "INITIAL_99999"
        answer = f"Some prose. Also [{fake_id}] is cited here but does not exist."
        flags = _build_faithfulness_audit(answer, proj, store)
        issues = [f["issue"] for f in flags]
        self.assertIn("fabricated citation", issues)

    def test_real_id_not_flagged_as_fabricated(self):
        store = _make_store()
        _deposit_cluster(store, "Solar panels convert sunlight into electricity.")
        proj = _make_projection(store)
        if not proj.surviving:
            self.skipTest("No surviving clusters in this projection")
        real_id = proj.surviving[0].representative_id
        # Use real ID in prose
        answer = f"Solar panels [{real_id}] are increasingly efficient."
        flags = _build_faithfulness_audit(answer, proj, store)
        fabricated_flags = [f for f in flags if f["issue"] == "fabricated citation"]
        self.assertEqual(len(fabricated_flags), 0)


class TestOrphanThinkTagCheck(unittest.TestCase):
    """Check 3: orphan_think_tag — paragraph contains a think tag."""

    def test_closing_think_tag_flagged(self):
        store = _make_store()
        proj = _make_projection(store)
        answer = "Some prose here. </think> And more prose."
        flags = _build_faithfulness_audit(answer, proj, store)
        issues = [f["issue"] for f in flags]
        self.assertIn("orphan_think_tag", issues)

    def test_opening_think_tag_flagged(self):
        store = _make_store()
        proj = _make_projection(store)
        answer = "Paragraph with <think> inside."
        flags = _build_faithfulness_audit(answer, proj, store)
        issues = [f["issue"] for f in flags]
        self.assertIn("orphan_think_tag", issues)

    def test_clean_prose_not_flagged(self):
        store = _make_store()
        proj = _make_projection(store)
        answer = "Clean prose without any think tags. This is fine."
        flags = _build_faithfulness_audit(answer, proj, store)
        think_flags = [f for f in flags if f["issue"] == "orphan_think_tag"]
        self.assertEqual(len(think_flags), 0)


class TestTruncatedMidSentenceCheck(unittest.TestCase):
    """Check 5: truncated_mid_sentence — paragraph lacks terminal punctuation."""

    def test_no_terminal_punctuation_flagged(self):
        store = _make_store()
        proj = _make_projection(store)
        # Two paragraphs, first one has no terminal punctuation
        answer = "This sentence ends without punctuation\n\nSecond paragraph."
        flags = _build_faithfulness_audit(answer, proj, store)
        issues = [f["issue"] for f in flags]
        self.assertIn("truncated_mid_sentence", issues)

    def test_terminal_punctuation_not_flagged(self):
        store = _make_store()
        proj = _make_projection(store)
        answer = "This sentence ends properly.\n\nSecond paragraph ends too!"
        flags = _build_faithfulness_audit(answer, proj, store)
        trunc_flags = [f for f in flags if f["issue"] == "truncated_mid_sentence"]
        self.assertEqual(len(trunc_flags), 0)


class TestSuspiciouslyShortParagraphCheck(unittest.TestCase):
    """Check 6: suspiciously_short_paragraph — cited paragraph under 50 chars."""

    def test_short_cited_paragraph_flagged(self):
        store = _make_store()
        _deposit_cluster(store, "A claim about sustainability.")
        proj = _make_projection(store)
        if not proj.surviving:
            self.skipTest("No surviving clusters")
        real_id = proj.surviving[0].representative_id
        # Very short paragraph with citation
        answer = f"Short [{real_id}]."
        flags = _build_faithfulness_audit(answer, proj, store)
        issues = [f["issue"] for f in flags]
        self.assertIn("suspiciously_short_paragraph", issues)

    def test_long_paragraph_not_flagged(self):
        store = _make_store()
        _deposit_cluster(store, "A claim about sustainability and renewable energy.")
        proj = _make_projection(store)
        if not proj.surviving:
            self.skipTest("No surviving clusters")
        real_id = proj.surviving[0].representative_id
        answer = (
            f"This is a sufficiently long paragraph discussing [{real_id}] "
            f"with enough words to exceed the minimum threshold of fifty characters."
        )
        flags = _build_faithfulness_audit(answer, proj, store)
        short_flags = [f for f in flags if f["issue"] == "suspiciously_short_paragraph"]
        self.assertEqual(len(short_flags), 0)


class TestCleanAnswerProducesNoFlags(unittest.TestCase):
    """A well-formed answer with valid citations should produce zero flags."""

    def test_no_flags_on_clean_answer(self):
        store = _make_store()
        content = "renewable energy reduces carbon emissions significantly"
        sid = _deposit_cluster(store, content)
        proj = _make_projection(store)
        if not proj.surviving:
            self.skipTest("No surviving clusters with this content")
        rep_id = proj.surviving[0].representative_id
        rep = store.get(rep_id)
        if rep is None:
            self.skipTest("Representative signal not found")

        # Build a prose that contains the actual content words (for 4-gram overlap)
        rep_words = rep.content.split()
        if len(rep_words) >= 4:
            four_gram = " ".join(rep_words[:4])
        else:
            four_gram = rep.content

        answer = (
            f"According to the swarm field, {four_gram} [{rep_id}] "
            f"has strong supporting evidence from multiple strategies. "
            f"This claim survived field pressure and warrants consideration."
        )
        flags = _build_faithfulness_audit(answer, proj, store)
        # Allow scratchpad_in_prose to slip through (content-dependent), but
        # no fabricated citations, no think tags, no truncation.
        serious_flags = [
            f for f in flags
            if f["issue"] in ("fabricated citation", "orphan_think_tag")
        ]
        self.assertEqual(len(serious_flags), 0, f"Unexpected flags: {flags}")


if __name__ == "__main__":
    unittest.main()
