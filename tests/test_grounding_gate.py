"""Tests for the deposit-time number-grounding gate and planner retirement.

The fabrication failure mode (real runs: '$54M Oslo', '90K tons Copenhagen'
with verification ~0.04): the particulars push made workers produce figures
no retrieved chunk contained. STORM-style fix: a figure must appear in
evidence the worker was actually shown, or the deposit is rejected (evidence
present) / tagged numbers_grounded=false (no evidence this iteration).

Run with:
    pytest tests/test_grounding_gate.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.actions import ungrounded_numbers


_CHUNK = ("Oslo allocated $54 million to cycling infrastructure between "
          "2015 and 2019, and bicycle commuting rose 23 percent.")


class TestUngroundedNumbers(unittest.TestCase):
    def test_grounded_figures_pass(self):
        content = ("Oslo invested $54 million over the 2015-2019 period, "
                   "lifting bike commuting 23 percent.")
        self.assertEqual(ungrounded_numbers(content, [_CHUNK]), [])

    def test_fabricated_figure_caught(self):
        content = ("Copenhagen avoided 90,000 tons of CO2 annually after "
                    "its restrictions.")
        bad = ungrounded_numbers(content, [_CHUNK])
        self.assertTrue(any("90,000" in t or "90000" in t for t in bad))

    def test_comma_normalization(self):
        content = "The program cost $54,000,000 in total."
        # 54,000,000 normalizes to 54000000 — NOT the same number as 54.
        bad = ungrounded_numbers(content, [_CHUNK])
        self.assertTrue(bad)
        self.assertEqual(
            ungrounded_numbers("Spending hit 90,000 tons.",
                               ["emissions fell by 90000 tons"]),
            [])

    def test_single_digits_ignored(self):
        # 'step 2' / 'version 3' style structural numbers never trip the gate.
        self.assertEqual(
            ungrounded_numbers("In step 2 the function returns 3 values.",
                               ["no numbers here at all"]),
            [])

    def test_parent_content_grounds(self):
        parent = "The 2019 Oslo program cut emissions 11 percent."
        content = "Building on the 11 percent reduction recorded in 2019."
        self.assertEqual(ungrounded_numbers(content, [parent]), [])

    def test_no_figures_no_flags(self):
        self.assertEqual(
            ungrounded_numbers("Transit equity matters for access.", []),
            [])

    def test_no_sources_all_figures_ungrounded(self):
        bad = ungrounded_numbers("Emissions fell 32% by 2020.", [])
        self.assertEqual(len(bad), 2)


class TestPlannerRetired(unittest.TestCase):
    def test_llm_planner_default_off(self):
        from core.config import USE_LLM_PLANNER
        self.assertFalse(USE_LLM_PLANNER)


if __name__ == "__main__":
    unittest.main()
