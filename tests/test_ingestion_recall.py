"""Data-ingestion recall improvements (PIPELINE_MAP #15 + evidence intake).

The swarm's only structural edge over a flagship direct call is evidence the
flagship doesn't have at answer time — retrieved particulars. These tests pin
the three recall mechanisms added for that:

  1. `relax_query` — deterministic keyword fallback for zero-hit queries
     (planned atom queries fail on specificity, not topic).
  2. `_snippet_from_hits` — the widened two-source atom-evidence window
     (one 300-char snippet clipped the sentence carrying the figure).
  3. `_semanticscholar_search` — key-free aux backend for analysis/debate
     (paper abstracts carry the quantified particulars DDG snippets don't).
  4. The relaxed-query rescue rung in `_search_impl` before
     ALL BACKENDS FAILED.

No network, no LLM: backends are faked.
"""

import types

from core.query_planner import relax_query
from core.worker_pool import _snippet_from_hits
from core import search_tool
from core.intake import CorpusChunk


# ---------------------------------------------------------------------------
# relax_query
# ---------------------------------------------------------------------------

def test_relax_query_drops_connective_tissue_keeps_particulars():
    q = ("whether the ban of private cars in Copenhagen has reduced "
         "emissions by 23 percent since 2008")
    relaxed = relax_query(q)
    assert relaxed
    for term in ("Copenhagen", "23", "2008", "emissions"):
        assert term in relaxed
    for stop in ("whether", "the", "of", "in", "has", "by"):
        assert f" {stop} " not in f" {relaxed} "


def test_relax_query_caps_terms():
    q = " ".join(f"keyword{i}" for i in range(20))
    relaxed = relax_query(q, keep=8)
    assert len(relaxed.split()) == 8


def test_relax_query_preserves_order_and_dedupes():
    relaxed = relax_query("solar Solar power power costs solar")
    assert relaxed.split() == ["solar", "power", "costs"]


def test_relax_query_noop_returns_empty():
    # Already-minimal keyword queries must return "" so callers skip an
    # identical (pointless) retry.
    assert relax_query("solar power costs") == ""
    assert relax_query("") == ""


# ---------------------------------------------------------------------------
# _snippet_from_hits (widened atom-evidence window)
# ---------------------------------------------------------------------------

def _hit(text, tag):
    return types.SimpleNamespace(text=text, source_tag=tag)


def test_snippet_from_hits_two_sources():
    hits = [_hit("body one " * 60, "Source A | http://a"),
            _hit("body two " * 60, "Source B | http://b"),
            _hit("body three", "Source C | http://c")]
    snippet, tag = _snippet_from_hits(hits)
    assert tag == "Source A | http://a"
    assert "[Source A | http://a]" in snippet
    assert "[Source B | http://b]" in snippet
    assert "Source C" not in snippet          # max_hits=2
    # per-hit cap applies to each source, not the joint snippet
    assert snippet.index("body two") > snippet.index("body one")


def test_snippet_from_hits_skips_empty_bodies():
    hits = [_hit("", "Empty | http://e"), _hit("real content", "Real | http://r")]
    snippet, tag = _snippet_from_hits(hits)
    assert tag == "Real | http://r"
    assert "real content" in snippet


def test_snippet_from_hits_no_usable_hits():
    snippet, tag = _snippet_from_hits([_hit("", "x"), _hit("   ", "y")])
    assert snippet == "(no result)"
    assert tag == "(no result)"


# ---------------------------------------------------------------------------
# Semantic Scholar aux backend
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def test_semanticscholar_parses_papers(monkeypatch):
    payload = {"data": [
        {"title": "Car-free zones and urban emissions",
         "abstract": "We find a 12% reduction across 14 cities.",
         "url": "https://s2/p1", "year": 2021, "citationCount": 40},
        {"title": "No abstract paper", "abstract": None,
         "url": "https://s2/p2", "year": 2020},
    ]}
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp(200, payload))
    out = search_tool._semanticscholar_search("car free zones emissions", 5)
    assert len(out) == 1                      # abstract-less paper skipped
    assert "12% reduction" in out[0].text
    assert "2021" in out[0].text
    assert out[0].source_tag.endswith("https://s2/p1")


def test_semanticscholar_fail_soft(monkeypatch):
    import requests

    def _boom(*a, **k):
        raise requests.exceptions.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", _boom)
    assert search_tool._semanticscholar_search("anything", 5) == []
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp(429, {}))
    assert search_tool._semanticscholar_search("anything", 5) == []


def test_scholar_toggle(monkeypatch):
    monkeypatch.delenv("SWARM_SEARCH_SCHOLAR", raising=False)
    assert search_tool._scholar_enabled()
    monkeypatch.setenv("SWARM_SEARCH_SCHOLAR", "0")
    assert not search_tool._scholar_enabled()


# ---------------------------------------------------------------------------
# Relaxed-query rescue rung in _search_impl
# ---------------------------------------------------------------------------

def test_search_impl_relaxed_rescue(monkeypatch):
    """When every backend returns nothing for the original query, the relaxed
    keyword form is retried before declaring ALL BACKENDS FAILED."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setattr(search_tool, "_load_cache", lambda *a, **k: None)
    monkeypatch.setattr(search_tool, "_save_cache", lambda *a, **k: None)
    monkeypatch.setattr(search_tool, "_tavily_search", lambda *a, **k: [])
    monkeypatch.setattr(search_tool, "_wikipedia_search", lambda *a, **k: [])
    monkeypatch.setattr(search_tool, "_cohere_search", lambda *a, **k: [])
    monkeypatch.setattr(search_tool, "_build_followup_query", lambda *a, **k: "")
    monkeypatch.setattr(search_tool, "_diversify",
                        lambda chunks, *a, **k: chunks)

    original = ("whether the ban of private cars in Copenhagen has "
                "reduced emissions since 2008")
    seen_queries = []

    def _fake_ddg(query, max_results):
        seen_queries.append(query)
        if query == search_tool._sanitize_query(original):
            return []                          # original phrasing: zero hits
        return [CorpusChunk(chunk_id="c1", text="rescued evidence",
                            source_tag="Rescue | http://r")]

    monkeypatch.setattr(search_tool, "_ddg_search", _fake_ddg)
    out = search_tool._search_impl(original, max_results=4)
    assert out and out[0].text == "rescued evidence"
    # second DDG call used the relaxed keyword form, not the original
    assert len(seen_queries) == 2
    assert seen_queries[1] == relax_query(search_tool._sanitize_query(original))


def test_search_impl_no_rescue_when_relax_is_noop(monkeypatch):
    """A query that is already minimal keywords must NOT be retried."""
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setattr(search_tool, "_load_cache", lambda *a, **k: None)
    monkeypatch.setattr(search_tool, "_save_cache", lambda *a, **k: None)
    monkeypatch.setattr(search_tool, "_tavily_search", lambda *a, **k: [])
    monkeypatch.setattr(search_tool, "_wikipedia_search", lambda *a, **k: [])
    monkeypatch.setattr(search_tool, "_cohere_search", lambda *a, **k: [])
    monkeypatch.setattr(search_tool, "_build_followup_query", lambda *a, **k: "")
    calls = []
    monkeypatch.setattr(search_tool, "_ddg_search",
                        lambda q, n: calls.append(q) or [])
    assert search_tool._search_impl("solar power costs", max_results=4) == []
    assert len(calls) == 1
