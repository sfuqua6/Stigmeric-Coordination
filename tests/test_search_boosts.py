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

import os
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


class TestAdaptiveRelevanceGate(unittest.TestCase):
    """Tier 'refine search': adaptive relevance gate, default-off."""

    class _FakeEmbedder:
        """encode(text) -> 1-D vector [sim]; query encodes to [1.0] so cs@q == sim."""
        def __init__(self, sim_by_text):
            self.sim_by_text = sim_by_text
        def encode(self, x, normalize_embeddings=True):
            import numpy as np
            if isinstance(x, str):
                return np.array([1.0]) if x == "QUERY" else np.array([self.sim_by_text[x]])
            return np.array([[self.sim_by_text[t]] for t in x])

    def _run_gate(self, sims, rel_margin, min_sim=0.15):
        from core import config as cfg
        chunks = [_chunk(f"c{i}", f"text{i}") for i in range(len(sims))]
        sim_by = {f"text{i}": s for i, s in enumerate(sims)}
        emb = self._FakeEmbedder(sim_by)
        with mock.patch.object(st, "_get_embedder", return_value=emb), \
             mock.patch.object(cfg, "SEARCH_RELEVANCE_MIN", min_sim), \
             mock.patch.object(cfg, "SEARCH_RELEVANCE_REL_MARGIN", rel_margin):
            return st._relevance_filter(chunks, "QUERY")

    def test_default_off_keeps_absolute_floor_behaviour(self):
        # margin 0.0 -> only the absolute floor (0.15) applies.
        kept = self._run_gate([0.9, 0.5, 0.2, 0.05], rel_margin=0.0)
        self.assertEqual(len(kept), 3)  # 0.05 dropped, rest >= 0.15

    def test_relative_margin_prunes_inferior(self):
        # top=0.9, margin 0.25 -> threshold max(0.15, 0.65) = 0.65; keep >=0.65.
        kept = self._run_gate([0.9, 0.7, 0.5, 0.2], rel_margin=0.25)
        self.assertEqual(len(kept), 2)  # 0.9, 0.7

    def test_never_drops_below_two(self):
        kept = self._run_gate([0.9, 0.1, 0.05], rel_margin=0.5)
        self.assertGreaterEqual(len(kept), 2)


class TestEnrichAfterDiversify(unittest.TestCase):
    """Tier 1.3: page enrichment runs on final survivors, not raw DDG top-K."""

    def test_diversify_enriches_only_when_flagged(self):
        chunks = [_chunk(f"c{i}", f"snippet {i} about innovation drivers", f"t{i}|u{i}")
                  for i in range(5)]
        with mock.patch.object(st, "_enrich_with_pages",
                               side_effect=lambda cs: cs) as m:
            st._diversify(list(chunks), "innovation", 3, enrich_pages=False)
            self.assertEqual(m.call_count, 0)
            st._diversify(list(chunks), "innovation", 3, enrich_pages=True)
            self.assertEqual(m.call_count, 1)
            # enrichment receives the FINAL (<= max_results) ranked set, not raw input
            enriched_arg = m.call_args[0][0]
            self.assertLessEqual(len(enriched_arg), 3)

    def test_ddg_search_no_longer_enriches_inline(self):
        # _ddg_search must NOT call enrichment now (deferred to _diversify).
        fake_ddgs = mock.MagicMock()
        ctx = fake_ddgs.return_value.__enter__.return_value
        ctx.text.return_value = [
            {"href": f"https://e{i}.org", "title": f"t{i}", "body": f"body {i} text"}
            for i in range(4)
        ]
        with mock.patch.dict("sys.modules", {"ddgs": mock.MagicMock(DDGS=fake_ddgs)}):
            with mock.patch.object(st, "_enrich_with_pages",
                                   side_effect=lambda cs: cs) as m:
                st._ddg_search("query", 4)
                self.assertEqual(m.call_count, 0)


class TestPageCache(unittest.TestCase):
    """Tier 1.2: in-run URL->text memo avoids re-fetching the same page."""

    def setUp(self):
        self._orig = st._fetch_page_text
        st.reset_search_stats()

    def tearDown(self):
        st._fetch_page_text = self._orig
        st.reset_search_stats()

    def test_repeat_url_served_from_cache(self):
        n = {"calls": 0}
        def fake(url, timeout):
            n["calls"] += 1
            return "content " * 100
        st._fetch_page_text = fake
        st._fetch_page_text_cached("https://a.org", 5)
        st._fetch_page_text_cached("https://a.org", 5)
        st._fetch_page_text_cached("https://b.org", 5)
        self.assertEqual(n["calls"], 2)  # a fetched once, b once
        self.assertEqual(st.search_stats_snapshot()["fetch_cache_hits"], 1)

    def test_reset_clears_cache(self):
        st._fetch_page_text = lambda u, t: "x" * 50
        st._fetch_page_text_cached("https://a.org", 5)
        st.reset_search_stats()
        n = {"calls": 0}
        def fake(url, timeout):
            n["calls"] += 1
            return "y" * 50
        st._fetch_page_text = fake
        st._fetch_page_text_cached("https://a.org", 5)  # must re-fetch after reset
        self.assertEqual(n["calls"], 1)


class TestSearchStats(unittest.TestCase):
    """Tier-0 latency instrumentation: search() records timing into stats."""

    def setUp(self):
        self._orig_impl = st._search_impl
        self._orig_mock = os.environ.pop("MOCK_LLM", None)
        st.reset_search_stats()

    def tearDown(self):
        st._search_impl = self._orig_impl
        if self._orig_mock is not None:
            os.environ["MOCK_LLM"] = self._orig_mock

    def test_counts_calls_and_empties(self):
        st._search_impl = lambda q, m=8, t=None: ([] if "empty" in q else [_chunk("c", "txt", "t|u")])
        st.search("topic one")
        st.search("topic two")
        st.search("empty topic")
        snap = st.search_stats_snapshot()
        self.assertEqual(snap["calls"], 3)
        self.assertEqual(snap["empty_results"], 1)
        self.assertGreaterEqual(snap["total_s"], 0.0)
        self.assertIn("avg_s", snap)

    def test_reset_zeroes(self):
        st._search_impl = lambda q, m=8, t=None: [_chunk("c", "txt", "t|u")]
        st.search("x")
        st.reset_search_stats()
        self.assertEqual(st.search_stats_snapshot()["calls"], 0)

    def test_mock_runs_are_not_timed(self):
        os.environ["MOCK_LLM"] = "1"
        st.reset_search_stats()
        st.search("anything")  # hits the MOCK short-circuit, must not count
        self.assertEqual(st.search_stats_snapshot()["calls"], 0)


if __name__ == "__main__":
    unittest.main()
