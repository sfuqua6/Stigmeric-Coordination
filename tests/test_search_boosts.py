"""Tests for search quality boosts and the free backends.

Covers:
  - _fact_density: numeric/currency/percent token scoring.
  - _quality_rerank: fact-dense chunks rise; coding domain prior fires
    only for task_type='coding'.
  - _stackexchange_search: response parsing with mocked HTTP (no network).
  - _wikipedia_search: fail-soft without the package / with mocked module.
  - summarize_for_signal: SEARCH traces now carry a content excerpt.

Run with:
    pytest tests/test_search_boosts.py -v
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

import core.search_tool as st
from core.search_tool import (
    CorpusChunk,
    _fact_density,
    _quality_rerank,
    _stackexchange_search,
    _wikipedia_search,
    summarize_for_signal,
)


def _chunk(cid, text, tag=""):
    return CorpusChunk(chunk_id=cid, text=text, source_tag=tag)


# ---------------------------------------------------------------------------
# Fact density
# ---------------------------------------------------------------------------

class TestFactDensity(unittest.TestCase):
    def test_dense_text_scores_higher(self):
        dense = ("Oslo invested $54 million over 4 years; emissions fell "
                 "11% by 2019 and 90,000 tons of CO2 were avoided.")
        sparse = ("Many cities have considered various approaches to "
                  "reducing emissions through transportation policy changes.")
        self.assertGreater(_fact_density(dense), _fact_density(sparse))
        self.assertEqual(_fact_density(sparse), 0.0)

    def test_empty_text(self):
        self.assertEqual(_fact_density(""), 0.0)

    def test_clamped_to_one(self):
        self.assertLessEqual(_fact_density("1 2 3 4 5"), 1.0)


# ---------------------------------------------------------------------------
# Quality rerank
# ---------------------------------------------------------------------------

class TestQualityRerank(unittest.TestCase):
    def test_fact_dense_chunk_rises(self):
        vague = _chunk("a", "General discussion of urban policy approaches "
                            "and their many possible considerations overall.")
        dense = _chunk("b", "Madrid's LEZ cut NO2 32% in 2019; the city "
                            "spent $200 million on 480 monitoring stations.")
        out = _quality_rerank([vague, dense], task_type=None)
        self.assertEqual(out[0].chunk_id, "b")

    def test_coding_domain_prior_fires_for_coding_only(self):
        blog = _chunk("blog", "How to do binary search, a beginner's guide.",
                      "Guide | https://someblog.example.com/binary-search")
        so = _chunk("so", "Binary search implementation pitfalls explained.",
                    "Q&A | https://stackoverflow.com/questions/123")
        out_coding = _quality_rerank([blog, so], task_type="coding")
        self.assertEqual(out_coding[0].chunk_id, "so")
        out_debate = _quality_rerank([blog, so], task_type="debate")
        self.assertEqual(out_debate[0].chunk_id, "blog")  # incoming order kept

    def test_disabled_weights_passthrough(self):
        a, b = _chunk("a", "text 1 2 3"), _chunk("b", "text")
        with mock.patch.object(st, "_fact_density") as fd:
            import core.config as cfg
            with mock.patch.object(cfg, "SEARCH_FACT_DENSITY_WEIGHT", 0.0), \
                 mock.patch.object(cfg, "SEARCH_CODING_DOMAIN_BOOST", 0.0):
                out = _quality_rerank([a, b], task_type="coding")
        self.assertEqual([c.chunk_id for c in out], ["a", "b"])
        fd.assert_not_called()


# ---------------------------------------------------------------------------
# Stack Exchange backend (mocked HTTP)
# ---------------------------------------------------------------------------

class TestStackExchange(unittest.TestCase):
    def test_parses_items_and_strips_html(self):
        payload = {"items": [{
            "title": "How to implement binary search?",
            "body": "<p>Use <code>lo + (hi - lo) // 2</code> to avoid "
                    "overflow.</p>",
            "link": "https://stackoverflow.com/q/1",
        }]}
        fake_resp = SimpleNamespace(
            status_code=200, json=lambda: payload)
        with mock.patch("requests.get", return_value=fake_resp):
            out = _stackexchange_search("binary search", 3)
        self.assertEqual(len(out), 1)
        self.assertIn("binary search", out[0].text.lower())
        self.assertNotIn("<p>", out[0].text)
        self.assertIn("stackoverflow.com", out[0].source_tag)

    def test_http_error_fail_soft(self):
        fake_resp = SimpleNamespace(status_code=429, json=lambda: {})
        with mock.patch("requests.get", return_value=fake_resp):
            self.assertEqual(_stackexchange_search("q", 3), [])

    def test_network_exception_fail_soft(self):
        with mock.patch("requests.get", side_effect=OSError("down")):
            self.assertEqual(_stackexchange_search("q", 3), [])


# ---------------------------------------------------------------------------
# Wikipedia backend (mocked module)
# ---------------------------------------------------------------------------

class TestWikipediaBackend(unittest.TestCase):
    def test_parses_summaries(self):
        fake_wiki = SimpleNamespace(
            search=lambda q, results=4: ["Binary search algorithm"],
            summary=lambda t, sentences=5, auto_suggest=False: (
                "In computer science, binary search finds a target in a "
                "sorted array in O(log n) comparisons, described in 1946."),
        )
        with mock.patch.dict(sys.modules, {"wikipedia": fake_wiki}):
            out = _wikipedia_search("binary search", 3)
        self.assertEqual(len(out), 1)
        self.assertIn("en.wikipedia.org/wiki/Binary_search_algorithm",
                      out[0].source_tag)

    def test_search_failure_fail_soft(self):
        def _boom(q, results=4):
            raise RuntimeError("api down")
        fake_wiki = SimpleNamespace(search=_boom, summary=None)
        with mock.patch.dict(sys.modules, {"wikipedia": fake_wiki}):
            self.assertEqual(_wikipedia_search("q", 3), [])


# ---------------------------------------------------------------------------
# SEARCH trace excerpt
# ---------------------------------------------------------------------------

class TestSummarizeForSignal(unittest.TestCase):
    def test_includes_top_excerpt(self):
        chunks = [_chunk("a", "Oslo spent $54 million on cycling paths.",
                         "Oslo report | https://example.org/oslo")]
        out = summarize_for_signal("oslo cycling investment", chunks)
        self.assertIn("QUERY: oslo cycling investment", out)
        self.assertIn("TOP: Oslo spent $54 million", out)

    def test_no_results(self):
        out = summarize_for_signal("q", [])
        self.assertIn("(no results)", out)


if __name__ == "__main__":
    unittest.main()
