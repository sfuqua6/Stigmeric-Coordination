"""Regression tests for core/filters.py::is_junk_output (exposed via core/junk_filter.py).

Covers every _JUNK_PATTERNS entry from the task spec against a known-bad
example AND 8–10 legitimate claims that must NOT match.

Run with:
    python -m unittest tests.test_junk_filter -v

No GPU or LLM required.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.junk_filter import is_junk
from core.filters import is_junk_output


class TestJunkFilterAlias(unittest.TestCase):
    """core/junk_filter.is_junk must be the same function as core/filters.is_junk_output."""

    def test_alias(self):
        self.assertIs(is_junk, is_junk_output)


class TestJunkFilterBadInputs(unittest.TestCase):
    """Every bad pattern from the spec must be caught."""

    # Template-leakage / placeholder patterns
    def test_no_claim_yet(self):
        self.assertTrue(is_junk("[No claim yet] Hmm... Frogs have"))

    def test_insert_placeholder(self):
        self.assertTrue(is_junk("[Insert your claim here about the topic]"))

    def test_placeholder_bracket(self):
        self.assertTrue(is_junk("[placeholder] This is where the claim goes"))

    # Mid-stream interjections now added to filters.py
    def test_but_wait_colon(self):
        self.assertTrue(is_junk(
            "Renewable energy is growing. But wait: I need to reconsider this."
        ))

    def test_but_wait_space(self):
        self.assertTrue(is_junk(
            "Climate change is real. But wait, there are other factors."
        ))

    def test_now_its_your_turn(self):
        self.assertTrue(is_junk(
            "I've presented the evidence. Now it's your turn to respond."
        ))

    def test_now_its_your_turn_no_apostrophe(self):
        self.assertTrue(is_junk("Now its your turn to evaluate this claim."))

    def test_now_i_need_to(self):
        self.assertTrue(is_junk(
            "The evidence supports this. Now I need to write a claim."
        ))

    def test_thats_my_claim(self):
        self.assertTrue(is_junk("That's my claim about the evidence."))

    def test_thats_my_first_order_claim(self):
        self.assertTrue(is_junk("That's my first-order claim based on partition data."))

    def test_i_remember_hearing(self):
        self.assertTrue(is_junk(
            "I remember hearing that renewable energy is the future."
        ))

    def test_im_supposed_to(self):
        self.assertTrue(is_junk("I'm supposed to produce a claim from this evidence."))

    def test_this_initial_claim(self):
        self.assertTrue(is_junk("This initial claim is about climate change."))

    def test_this_is_my_initial_claim(self):
        self.assertTrue(is_junk("This is my initial claim about the topic."))

    def test_based_on_the_evidence_assigned(self):
        self.assertTrue(is_junk("Based on the evidence assigned to me, I conclude..."))

    # Signal delimiter leakage
    def test_signal_delimiter(self):
        self.assertTrue(is_junk("Here is my claim.\n---SIGNAL\nContent follows."))

    def test_artifact_delimiter(self):
        self.assertTrue(is_junk("---ARTIFACT\nSome content here."))

    def test_evidence_delimiter(self):
        self.assertTrue(is_junk("Some text.\n---EVIDENCE\nMore text."))

    def test_end_delimiter(self):
        self.assertTrue(is_junk("Some content.\n---END\nTrailing."))

    # Chat-template leakage
    def test_assistant_line_end(self):
        self.assertTrue(is_junk(
            "The response is complete.\nAssistant"
        ))

    # Family 1 openers (already in filters.py)
    def test_so_i_have_to(self):
        self.assertTrue(is_junk("So I have to produce a relevant claim here."))

    def test_alright_opener(self):
        self.assertTrue(is_junk("Alright, let me think about this carefully."))

    def test_let_me_opener(self):
        self.assertTrue(is_junk("Let me consider the evidence before responding."))

    def test_i_need_to_opener(self):
        self.assertTrue(is_junk("I need to produce a clear and concise claim."))

    def test_my_claim_is(self):
        self.assertTrue(is_junk("My claim is that climate action is necessary."))

    # Too short
    def test_too_short(self):
        self.assertTrue(is_junk("Short."))

    # Empty
    def test_empty(self):
        self.assertTrue(is_junk(""))

    def test_whitespace_only(self):
        self.assertTrue(is_junk("   \n  "))

    # Decoder repetition
    def test_decoder_repetition(self):
        sentence = "Climate action is critically important for future generations."
        self.assertTrue(is_junk((sentence + " ") * 5))


class TestJunkFilterGoodInputs(unittest.TestCase):
    """Legitimate claims must NOT be flagged as junk."""

    GOOD_CLAIMS = [
        "Renewable energy costs have declined 89% since 2010, making solar competitive with coal.",
        "Greenhouse gas concentrations have reached 421 ppm CO2 as of 2023, the highest in 800,000 years.",
        "Carbon pricing mechanisms have been adopted by 46 countries, covering 23% of global emissions.",
        "The Paris Agreement commits signatories to limiting warming to 1.5°C above pre-industrial levels.",
        "Ocean acidification from CO2 absorption threatens coral reef ecosystems globally.",
        "Deforestation accounts for approximately 10% of annual global carbon emissions.",
        "Battery storage costs have fallen 97% since 1991, enabling viable grid-scale renewables.",
        "Methane has 84× the warming potential of CO2 over 20 years but a shorter atmospheric lifetime.",
        "Electric vehicle adoption doubled in 2022, reaching 10% of new car sales worldwide.",
        "Climate adaptation costs in developing nations are projected to reach $300 billion annually by 2030.",
    ]

    def test_all_good_claims_pass(self):
        for claim in self.GOOD_CLAIMS:
            with self.subTest(claim=claim[:60]):
                self.assertFalse(
                    is_junk(claim),
                    msg=f"Legitimate claim incorrectly flagged as junk: {claim[:60]!r}",
                )


if __name__ == "__main__":
    unittest.main()
