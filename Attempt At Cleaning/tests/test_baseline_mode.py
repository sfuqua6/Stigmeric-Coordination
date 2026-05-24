"""Tests for core/baseline.py (§3 directive).

Run with:
    python -m unittest tests.test_baseline_mode -v
"""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.baseline import (
    BaselineCoordinator,
    BaselineResult,
    _deduplicate,
    _jaccard_sim,
    _normalise,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(responses=None):
    """Return a mock LLM whose generate() returns successive canned responses."""
    llm = MagicMock()
    llm.name = "mock"
    responses = responses or ["Response A", "Response B", "Response C"]
    it = iter(responses)

    async def generate(prompt, role="agent", max_tokens=120, temperature=0.7):
        try:
            return next(it)
        except StopIteration:
            return "fallback response"

    llm.generate = generate
    return llm


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

class TestJaccardSim(unittest.TestCase):
    def test_identical_strings(self):
        self.assertAlmostEqual(_jaccard_sim("hello world", "hello world"), 1.0)

    def test_completely_different(self):
        self.assertAlmostEqual(_jaccard_sim("alpha beta", "gamma delta"), 0.0)

    def test_partial_overlap(self):
        sim = _jaccard_sim("the quick brown fox", "the slow brown dog")
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)


class TestDeduplicate(unittest.TestCase):
    def test_removes_exact_duplicates(self):
        result = _deduplicate(["hello world", "hello world", "something else"])
        self.assertEqual(len(result), 2)

    def test_preserves_distinct_responses(self):
        result = _deduplicate(["alpha beta gamma", "completely different words here"])
        self.assertEqual(len(result), 2)

    def test_empty_strings_filtered(self):
        result = _deduplicate(["", "  ", "real content here"])
        self.assertEqual(len(result), 1)

    def test_near_duplicates_removed(self):
        # High Jaccard overlap → one should be kept
        a = "climate change is a global emergency requiring immediate action"
        b = "climate change is a global emergency requiring immediate response"
        result = _deduplicate([a, b], sim_threshold=0.7)
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# BaselineCoordinator
# ---------------------------------------------------------------------------

class TestBaselineCoordinator(unittest.TestCase):
    def test_returns_baseline_result(self):
        llm = _make_llm(["Answer one", "Answer two", "Answer three"] * 4)
        coord = BaselineCoordinator(n_agents=4, max_tokens=50)
        result = asyncio.get_event_loop().run_until_complete(coord.run(
            task_type="debate",
            user_prompt="Is AI good?",
            task_prompt="Argue both sides of: Is AI good?",
            llm=llm,
            corpus_text="Some context about AI.",
        ))
        self.assertIsInstance(result, BaselineResult)
        self.assertIsInstance(result.final_answer, str)
        self.assertGreater(len(result.final_answer), 0)

    def test_n_calls_matches_n_agents(self):
        llm = _make_llm(["response"] * 8)
        coord = BaselineCoordinator(n_agents=8, max_tokens=50)
        result = asyncio.get_event_loop().run_until_complete(coord.run(
            task_type="debate",
            user_prompt="test",
            task_prompt="test",
            llm=llm,
        ))
        self.assertEqual(result.n_calls, 8)

    def test_writes_output_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = _make_llm(["Unique response X", "Unique response Y"] * 3)
            coord = BaselineCoordinator(n_agents=4, max_tokens=50)
            asyncio.get_event_loop().run_until_complete(coord.run(
                task_type="debate",
                user_prompt="test prompt",
                task_prompt="test prompt",
                llm=llm,
                output_dir=Path(tmpdir),
            ))
            self.assertTrue((Path(tmpdir) / "answer.txt").exists())
            self.assertTrue((Path(tmpdir) / "summary.json").exists())

    def test_summary_dict_has_required_keys(self):
        llm = _make_llm(["A", "B", "C"] * 3)
        coord = BaselineCoordinator(n_agents=3, max_tokens=50)
        result = asyncio.get_event_loop().run_until_complete(coord.run(
            task_type="analysis",
            user_prompt="What is entropy?",
            task_prompt="Analyse: What is entropy?",
            llm=llm,
        ))
        d = result.summary_dict()
        for key in ("mode", "task_type", "total_llm_calls", "n_clusters", "total_elapsed_s"):
            self.assertIn(key, d, f"summary_dict missing key {key!r}")
        self.assertEqual(d["mode"], "baseline")
        self.assertIn("surviving", d["n_clusters"])

    def test_empty_corpus_handled(self):
        llm = _make_llm(["answer"] * 4)
        coord = BaselineCoordinator(n_agents=4, max_tokens=50)
        result = asyncio.get_event_loop().run_until_complete(coord.run(
            task_type="debate",
            user_prompt="test",
            task_prompt="test",
            llm=llm,
            corpus_text="",
        ))
        self.assertIsInstance(result.final_answer, str)

    def test_all_empty_responses_gives_fallback_message(self):
        llm = _make_llm(["", "  ", ""] * 3)
        coord = BaselineCoordinator(n_agents=3, max_tokens=50)
        result = asyncio.get_event_loop().run_until_complete(coord.run(
            task_type="debate",
            user_prompt="test",
            task_prompt="test",
            llm=llm,
        ))
        self.assertIn("no non-empty responses", result.final_answer)


# ---------------------------------------------------------------------------
# compare_runs CLI (import smoke-test)
# ---------------------------------------------------------------------------

class TestCompareRunsImport(unittest.TestCase):
    def test_compare_module_importable(self):
        import importlib
        mod = importlib.util.spec_from_file_location(
            "compare_runs",
            str(Path(__file__).parent.parent / "tools" / "compare_runs.py"),
        )
        self.assertIsNotNone(mod)

    def test_compare_two_summary_dicts(self):
        """compare() must not raise when both files are valid."""
        import io
        from contextlib import redirect_stdout
        from tools.compare_runs import compare
        import tempfile

        d_a = {
            "mode": "stigmergic",
            "task_type": "debate",
            "user_prompt": "test",
            "total_llm_calls": 100,
            "total_elapsed_s": 42.0,
            "n_clusters": {"surviving": 3, "contested": 1, "weakly_supported": 0, "rejected_by_field": 0},
        }
        d_b = {
            "mode": "baseline",
            "task_type": "debate",
            "user_prompt": "test",
            "total_llm_calls": 8,
            "total_elapsed_s": 5.0,
            "n_clusters": {"surviving": 4, "contested": 0, "weakly_supported": 0, "rejected_by_field": 0},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fa:
            json.dump(d_a, fa)
            path_a = fa.name
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fb:
            json.dump(d_b, fb)
            path_b = fb.name

        out = io.StringIO()
        with redirect_stdout(out):
            compare(path_a, path_b)
        output = out.getvalue()
        self.assertIn("stigmergic", output)
        self.assertIn("baseline", output)


if __name__ == "__main__":
    unittest.main()
