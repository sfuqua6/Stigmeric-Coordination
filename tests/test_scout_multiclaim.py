"""Tests for multi-claim scout sampling.

Covers:
  - split_scout_claims: numbered-portfolio parsing (the prompt pre-seeds
    "1." so the first claim arrives headless), single-claim passthrough,
    junk filtering.
  - SignalStore.max_similarity_to_recent: scalar similarity vs recent
    same-type signals (string fallback path).
  - select_novel_claim: picks the candidate least similar to the existing
    INITIAL field; ties resolve to the model's first (highest-confidence)
    claim.
  - scout_prompt: asks for the K-claim portfolio.

No GPU or LLM required.

Run with:
    pytest tests/test_scout_multiclaim.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.signal_store import SignalStore, _NULL_EMBEDDER
from core.signal_types import INITIAL, SUPPORT
from core.actions import split_scout_claims, select_novel_claim, scout_prompt
from core.config import SCOUT_CLAIMS_PER_CALL


def _make_store() -> SignalStore:
    return SignalStore(embedder=_NULL_EMBEDDER)


def _deposit_initial(store, content, pid="partition_0"):
    return store.deposit(
        signal_type=INITIAL, content=content, strength=0.6,
        depositor="scout", parent_id=None,
        metadata={"partition_id": pid, "scout_agent_id": "scout_R1_0"},
    )


# ---------------------------------------------------------------------------
# split_scout_claims
# ---------------------------------------------------------------------------

class TestSplitScoutClaims(unittest.TestCase):
    def test_headless_first_claim_from_preseeded_prompt(self):
        # Prompt ends "CLAIMS:\n1." — the model continues without a marker.
        text = (
            "Cities lose parking-fee revenue when downtown bans take effect.\n"
            "2. Freight delivery costs rise sharply under blanket car bans.\n"
            "3. Electric buses cut more emissions per dollar than car bans."
        )
        claims = split_scout_claims(text)
        self.assertEqual(len(claims), 3)
        self.assertTrue(claims[0].startswith("Cities lose parking-fee"))
        self.assertTrue(claims[1].startswith("Freight delivery"))

    def test_fully_numbered(self):
        text = (
            "1. Madrid's restriction zone lowered nitrogen dioxide levels.\n"
            "2) Tradespeople carrying tools cannot switch to bicycles.\n"
        )
        claims = split_scout_claims(text)
        self.assertEqual(len(claims), 2)

    def test_single_claim_passthrough(self):
        text = "Car bans reduce emissions in dense urban cores."
        self.assertEqual(split_scout_claims(text), [text])

    def test_short_fragments_filtered(self):
        text = (
            "Transit investment must precede restrictions in sparse regions.\n"
            "2. Yes.\n"
            "3. Night-shift workers travel when transit barely operates."
        )
        claims = split_scout_claims(text)
        self.assertEqual(len(claims), 2)
        self.assertNotIn("Yes.", claims)

    def test_empty_returns_empty(self):
        self.assertEqual(split_scout_claims(""), [])

    def test_multiline_claim_flattened(self):
        text = (
            "1. A claim that spans\nmultiple lines in the response body.\n"
            "2. Another distinct claim about freight costs and timing."
        )
        claims = split_scout_claims(text)
        self.assertEqual(len(claims), 2)
        self.assertNotIn("\n", claims[0])


# ---------------------------------------------------------------------------
# SignalStore.max_similarity_to_recent
# ---------------------------------------------------------------------------

class TestMaxSimilarityToRecent(unittest.TestCase):
    def test_empty_store_returns_zero(self):
        store = _make_store()
        self.assertEqual(
            store.max_similarity_to_recent("anything", INITIAL), 0.0)

    def test_identical_text_scores_high(self):
        store = _make_store()
        _deposit_initial(store, "Car bans reduce urban emissions sharply.")
        sim = store.max_similarity_to_recent(
            "Car bans reduce urban emissions sharply.", INITIAL)
        self.assertGreater(sim, 0.95)

    def test_unrelated_text_scores_low(self):
        store = _make_store()
        _deposit_initial(store, "Car bans reduce urban emissions sharply.")
        sim = store.max_similarity_to_recent(
            "Quantum entanglement enables novel cryptographic protocols.",
            INITIAL)
        self.assertLess(sim, 0.5)

    def test_type_filtered(self):
        store = _make_store()
        init_id = _deposit_initial(
            store, "Car bans reduce urban emissions sharply.")
        store.deposit(
            signal_type=SUPPORT, content="Totally different support text here.",
            strength=0.5, depositor="forager", parent_id=init_id,
            metadata={"partition_id": "partition_1"},
        )
        # Comparing against SUPPORT: the INITIAL must not be considered.
        sim = store.max_similarity_to_recent(
            "Car bans reduce urban emissions sharply.", SUPPORT)
        self.assertLess(sim, 0.95)


# ---------------------------------------------------------------------------
# select_novel_claim
# ---------------------------------------------------------------------------

class TestSelectNovelClaim(unittest.TestCase):
    def test_picks_candidate_far_from_existing_field(self):
        store = _make_store()
        _deposit_initial(
            store, "Banning private cars reduces carbon emissions in cities.")
        candidates = [
            "Banning private cars reduces carbon emissions in urban areas.",
            "Freight logistics costs rise when delivery vans lose access.",
        ]
        chosen = select_novel_claim(candidates, store)
        self.assertEqual(chosen, candidates[1])

    def test_empty_store_ties_resolve_to_first(self):
        store = _make_store()
        candidates = [
            "First claim about urban transit and equity outcomes.",
            "Second claim about freight costs and delivery windows.",
        ]
        self.assertEqual(select_novel_claim(candidates, store), candidates[0])

    def test_single_candidate_passthrough(self):
        store = _make_store()
        self.assertEqual(
            select_novel_claim(["Only claim available."], store),
            "Only claim available.")

    def test_empty_candidates(self):
        store = _make_store()
        self.assertEqual(select_novel_claim([], store), "")


# ---------------------------------------------------------------------------
# scout_prompt portfolio ask
# ---------------------------------------------------------------------------

class TestScoutPrompt(unittest.TestCase):
    def test_prompt_asks_for_k_distinct_claims(self):
        prompt = scout_prompt("Should cities ban cars?", [])
        if SCOUT_CLAIMS_PER_CALL > 1:
            self.assertIn(f"{SCOUT_CLAIMS_PER_CALL} DISTINCT", prompt)
            self.assertTrue(prompt.endswith("CLAIMS:\n1."))
        else:
            self.assertIn("ONE concise initial claim", prompt)

    def test_reseed_excerpt_included(self):
        prompt = scout_prompt(
            "Should cities ban cars?", [],
            prior_own_content="Cars cause emissions downtown.")
        self.assertIn("Cars cause emissions downtown.", prompt)
        self.assertIn("genuinely different", prompt)


if __name__ == "__main__":
    unittest.main()
