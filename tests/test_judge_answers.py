"""Tests for the pairwise LLM answer judge (tools/judge_answers.py).

The LLM is faked so these run without a model. They pin: JSON extraction,
position-label -> A/B mapping, position-bias detection (winner flips with
order -> tie), genuine order-invariant wins, and graceful fallback on garbage.
"""

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.judge_answers import judge_pair, _extract_json, _winner_for


def _run(coro):
    return asyncio.run(coro)


class _FixedLLM:
    """Returns a fixed string regardless of prompt (models position bias)."""
    def __init__(self, out):
        self.out = out
    async def generate(self, prompt, **kw):
        return self.out


class _AwareLLM:
    """Order-invariant judge: always prefers the answer containing `marker`."""
    def __init__(self, marker):
        self.marker = marker
    async def generate(self, prompt, **kw):
        i1, i2 = prompt.find("ANSWER 1:"), prompt.find("ANSWER 2:")
        marker_in_1 = self.marker in prompt[i1:i2]
        w = "1" if marker_in_1 else "2"
        return '{"winner":"%s","scores":{"1":{},"2":{}},"rationale":"r"}' % w


class TestJsonExtraction(unittest.TestCase):
    def test_extracts_from_prose(self):
        self.assertEqual(_extract_json('prefix {"winner": "1"} suffix')["winner"], "1")

    def test_none_on_no_json(self):
        self.assertIsNone(_extract_json("no braces here"))

    def test_none_on_malformed(self):
        self.assertIsNone(_extract_json('{"winner": '))


class TestWinnerMapping(unittest.TestCase):
    def test_position_one_maps_to_first_shown(self):
        self.assertEqual(_winner_for(True, {"winner": "1"}), "A")   # A shown first
        self.assertEqual(_winner_for(False, {"winner": "1"}), "B")  # B shown first
        self.assertEqual(_winner_for(True, {"winner": "2"}), "B")
        self.assertEqual(_winner_for(False, {"winner": "2"}), "A")

    def test_tie_and_unknown(self):
        self.assertEqual(_winner_for(True, {"winner": "tie"}), "tie")
        self.assertIsNone(_winner_for(True, {"winner": "banana"}))


class TestJudgePair(unittest.TestCase):
    def test_position_bias_detected_as_tie(self):
        # LLM always picks shown-position-1 -> winner flips with order -> tie.
        v = _run(judge_pair("q", "AAA", "BBB",
                            _FixedLLM('{"winner":"1","scores":{"1":{},"2":{}},"rationale":"r"}')))
        self.assertEqual(v["winner"], "tie")
        self.assertFalse(v["agreement"])

    def test_genuine_order_invariant_win(self):
        v = _run(judge_pair("q", "AAA", "BBB", _AwareLLM("AAA")))
        self.assertEqual(v["winner"], "A")
        self.assertTrue(v["agreement"])

    def test_genuine_win_for_b(self):
        v = _run(judge_pair("q", "AAA", "BBB", _AwareLLM("BBB")))
        self.assertEqual(v["winner"], "B")
        self.assertTrue(v["agreement"])

    def test_garbage_output_falls_back_to_tie(self):
        v = _run(judge_pair("q", "AAA", "BBB", _FixedLLM("I cannot comply")))
        self.assertEqual(v["winner"], "tie")
        self.assertFalse(v["rounds"][0]["parsed"])

    def test_consensus_tie_when_both_orders_tie(self):
        v = _run(judge_pair("q", "AAA", "BBB",
                            _FixedLLM('{"winner":"tie","scores":{"1":{},"2":{}},"rationale":"close"}')))
        self.assertEqual(v["winner"], "tie")
        self.assertTrue(v["agreement"])  # both orders agree it's a tie


if __name__ == "__main__":
    unittest.main()
