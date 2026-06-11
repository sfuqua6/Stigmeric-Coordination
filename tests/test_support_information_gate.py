"""Tests for the paraphrase-support gate and planner digest cap.

support_adds_information() rejects DEVELOP/CHAIN deposits that merely
restate their parent — the "claims, not evidence" failure mode where
agreement-paraphrases inflate support_diversity without adding facts.

Run with:
    pytest tests/test_support_information_gate.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.actions import (
    support_adds_information,
    _new_particulars,
    scout_prompt,
    develop_prompt,
)


_PARENT = ("Banning private cars in cities could significantly reduce "
           "carbon emissions and combat climate change.")


class TestSupportAddsInformation(unittest.TestCase):
    def test_pure_paraphrase_rejected(self):
        support = ("Banning private cars in cities could significantly "
                   "reduce carbon emissions and fight climate change.")
        self.assertFalse(support_adds_information(support, _PARENT))

    def test_new_fact_accepted(self):
        support = ("Madrid's low-emission zone cut nitrogen dioxide levels "
                   "by 32 percent within the first year of enforcement.")
        self.assertTrue(support_adds_information(support, _PARENT))

    def test_similar_wording_with_new_particular_accepted(self):
        # High overlap with the parent, but carries a new number + name.
        support = ("Banning private cars in cities could significantly "
                   "reduce carbon emissions, as Oslo's 2019 program cut "
                   "them by 11 percent.")
        self.assertTrue(support_adds_information(support, _PARENT))

    def test_genuinely_different_content_accepted_without_particulars(self):
        support = ("Reduced traffic noise improves sleep quality and "
                   "lowers chronic stress for residents near arterials.")
        self.assertTrue(support_adds_information(support, _PARENT))

    def test_empty_inputs_pass_through(self):
        self.assertTrue(support_adds_information("", _PARENT))
        self.assertTrue(support_adds_information("anything", ""))


class TestNewParticulars(unittest.TestCase):
    def test_finds_numbers_and_names(self):
        out = _new_particulars(
            "Madrid cut emissions 32% by 2021.", _PARENT)
        self.assertIn("Madrid", out)
        self.assertTrue(any("32" in t for t in out))

    def test_parent_tokens_excluded(self):
        out = _new_particulars(
            "Banning cars helps cities.", _PARENT)
        # 'Banning' appears in the parent — not a new particular.
        self.assertNotIn("Banning", out)


class TestPromptsDemandParticulars(unittest.TestCase):
    def test_develop_prompt_demands_particular(self):
        from types import SimpleNamespace
        target = SimpleNamespace(id="INITIAL_00001", content=_PARENT)
        prompt = develop_prompt("task", target)
        self.assertIn("concrete particular", prompt)
        self.assertIn("rejected", prompt)


class TestPlannerDigestCap(unittest.TestCase):
    def test_cap_constant_sane(self):
        import agents.synthesizer as s
        self.assertGreaterEqual(s._PLANNER_DIGEST_MAX_CLUSTERS, 8)
        self.assertLessEqual(s._PLANNER_DIGEST_MAX_CLUSTERS, 24)


if __name__ == "__main__":
    unittest.main()
