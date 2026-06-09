"""Regression tests for agents/base.py::strip_reasoning.

Run with:
    python -m unittest tests.test_strip_reasoning -v

No GPU or LLM required.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import strip_reasoning


class TestStripReasoning(unittest.TestCase):

    # ------------------------------------------------------------------
    # Core acceptance criterion from the task spec
    # ------------------------------------------------------------------

    def test_orphan_close_think(self):
        """Everything before a standalone </think> is reasoning; take what follows."""
        inp = "some reasoning\n</think>\n\nactual answer"
        out = strip_reasoning(inp)
        self.assertEqual(out, "actual answer")

    def test_orphan_close_think_with_content_before(self):
        """Full chain-of-thought before </think> is stripped."""
        inp = (
            "Let me think step by step.\n"
            "First: renewable energy costs are falling.\n"
            "Second: fossil fuels dominate.\n"
            "</think>\n\n"
            "Renewable energy is critical for climate action because costs have dropped."
        )
        out = strip_reasoning(inp)
        self.assertNotIn("</think>", out)
        self.assertIn("Renewable energy is critical", out)

    def test_paired_think_tags(self):
        """Paired <think>…</think> block is removed entirely."""
        inp = "<think>I need to reason carefully here.</think>\n\nThe answer is 42."
        out = strip_reasoning(inp)
        self.assertNotIn("<think>", out)
        self.assertNotIn("</think>", out)
        self.assertIn("The answer is 42", out)

    def test_orphan_open_think(self):
        """Orphan <think> with no closing tag — strip from <think> onward."""
        inp = "This is valid prose.\n<think>Hmm, let me reconsider this entirely."
        out = strip_reasoning(inp)
        self.assertNotIn("<think>", out)
        self.assertIn("This is valid prose", out)

    def test_scratchpad_only_output(self):
        """Output that is ONLY scratchpad should return empty string (falsy)."""
        inp = "Alright, so I need to produce a claim about climate action."
        out = strip_reasoning(inp)
        # After stripping the scratchpad opener, nothing should remain
        self.assertFalse(out.strip())

    def test_valid_prose_untouched(self):
        """Clean substantive prose is not altered."""
        inp = (
            "Climate action is necessary because greenhouse gas emissions are driving "
            "unprecedented changes to global temperature patterns."
        )
        out = strip_reasoning(inp)
        self.assertEqual(out, inp)

    def test_multiple_close_think_takes_last(self):
        """If multiple </think> tags exist, content after the LAST one is returned."""
        inp = "reasoning1\n</think>\nmore reasoning\n</think>\n\nfinal answer"
        out = strip_reasoning(inp)
        self.assertNotIn("</think>", out)
        self.assertIn("final answer", out)
        self.assertNotIn("reasoning1", out)
        self.assertNotIn("more reasoning", out)

    def test_none_input(self):
        """None input returns None (not a crash)."""
        self.assertIsNone(strip_reasoning(None))

    def test_empty_string(self):
        """Empty string returns empty string."""
        self.assertEqual(strip_reasoning(""), "")

    def test_chinese_chars_after_think(self):
        """Chinese characters embedded in reasoning block are stripped with it."""
        inp = "some thinking 潜水 鳃\n</think>\n\nFrogs breathe through their skin."
        out = strip_reasoning(inp)
        self.assertNotIn("潜水", out)
        self.assertNotIn("鳃", out)
        self.assertIn("Frogs breathe", out)

    def test_wait_scratchpad_stripped(self):
        """'Wait,' mid-generation self-correction is stripped."""
        inp = "Wait, I should reconsider this."
        out = strip_reasoning(inp)
        self.assertFalse(out.strip())

    def test_hmm_scratchpad_stripped(self):
        """'Hmm,' opener is stripped."""
        inp = "Hmm, this is tricky."
        out = strip_reasoning(inp)
        self.assertFalse(out.strip())


if __name__ == "__main__":
    unittest.main()
