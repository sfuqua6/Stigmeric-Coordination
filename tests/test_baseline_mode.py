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
        result = asyncio.run(coord.run(
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
        result = asyncio.run(coord.run(
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
            asyncio.run(coord.run(
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
        result = asyncio.run(coord.run(
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
        result = asyncio.run(coord.run(
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
        result = asyncio.run(coord.run(
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


class TestCompareRunsTimingAndVerdict(unittest.TestCase):
    """A/B harness extensions: deep nested lookup, timing block, verdict."""

    def test_nested_get_arbitrary_depth(self):
        from tools.compare_runs import _nested_get, _MISSING
        d = {"timing": {"search": {"calls": 7}}}
        self.assertEqual(_nested_get(d, "timing.search.calls"), 7)
        self.assertIs(_nested_get(d, "timing.search.missing"), _MISSING)
        self.assertIs(_nested_get(d, "timing.absent.x"), _MISSING)
        self.assertIs(_nested_get({"a": 5}, "a.b"), _MISSING)  # non-dict traversal

    def test_verdict_flags_quality_regression(self):
        from tools.compare_runs import _verdict
        # higher_is_better dropped 50% -> regression; lower_is_better rose 50% -> regression
        scored = [
            ("max verification score", "higher_is_better", 0.4, 0.2),
            ("output diversity (self-BLEU)", "lower_is_better", 0.4, 0.6),
            ("avg verification score", "higher_is_better", 0.30, 0.30),  # unchanged: ok
        ]
        regressions = _verdict(scored)
        self.assertEqual(len(regressions), 2)
        self.assertTrue(any("max verification" in r for r in regressions))

    def test_verdict_quality_held_within_tolerance(self):
        from tools.compare_runs import _verdict
        scored = [("max verification score", "higher_is_better", 0.40, 0.39)]  # -2.5% < 5%
        self.assertEqual(_verdict(scored), [])

    def test_timing_block_surfaces_in_output(self):
        import io, json, tempfile
        from contextlib import redirect_stdout
        from tools.compare_runs import compare
        d_a = {"user_prompt": "p", "wall_clock_s": 100.0,
               "timing": {"wall_clock_s": 100.0, "search": {"calls": 50, "total_s": 80.0},
                          "search_fraction_of_wallclock": 0.8}}
        d_b = {"user_prompt": "p", "wall_clock_s": 40.0,
               "timing": {"wall_clock_s": 40.0, "search": {"calls": 50, "total_s": 80.0},
                          "search_fraction_of_wallclock": 0.2}}
        paths = []
        for d in (d_a, d_b):
            f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
            json.dump(d, f); f.close(); paths.append(f.name)
        out = io.StringIO()
        with redirect_stdout(out):
            compare(*paths)
        output = out.getvalue()
        self.assertIn("search frac of wallclock", output)
        self.assertIn("wall-clock:", output)        # verdict line
        self.assertIn("B is faster", output)


if __name__ == "__main__":
    unittest.main()
