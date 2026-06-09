"""Fix S — Search diversity tests.

Verifies source-cap, BM25, RRF, MMR fallback, and the _diversify pipeline.
All tests run without GPU or live network (embedder mocked out where needed).

Run with:
    pytest tests/test_search_diversity.py -v
"""

import sys
import os
import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.intake import CorpusChunk
from core.search_tool import (
    apply_source_cap,
    apply_rrf,
    apply_mmr,
    _bm25_rank,
    _diversify,
    _source_domain,
)


def _chunk(cid: str, text: str, source: str) -> CorpusChunk:
    return CorpusChunk(chunk_id=cid, text=text, source_tag=source)


# ---------------------------------------------------------------------------
# _source_domain
# ---------------------------------------------------------------------------

def test_source_domain_extracts_hostname():
    ch = _chunk("c0", "text", "BBC News | https://www.bbc.co.uk/news/article-123")
    assert _source_domain(ch) == "bbc.co.uk"


def test_source_domain_no_pipe():
    ch = _chunk("c0", "text", "plain-tag")
    domain = _source_domain(ch)
    assert isinstance(domain, str)
    assert len(domain) >= 0  # at minimum empty string, not an error


# ---------------------------------------------------------------------------
# apply_source_cap
# ---------------------------------------------------------------------------

def test_source_cap_limits_to_max():
    chunks = [
        _chunk(f"a{i}", f"AAA {i}", f"Site A | https://sitea.com/{i}")
        for i in range(5)
    ]
    result = apply_source_cap(chunks, max_per_source=2)
    domains = [_source_domain(c) for c in result]
    assert domains.count("sitea.com") <= 2


def test_source_cap_preserves_other_sources():
    chunks = [
        _chunk("a0", "AAA 0", "Site A | https://sitea.com/0"),
        _chunk("a1", "AAA 1", "Site A | https://sitea.com/1"),
        _chunk("a2", "AAA 2", "Site A | https://sitea.com/2"),
        _chunk("b0", "BBB 0", "Site B | https://siteb.com/0"),
    ]
    result = apply_source_cap(chunks, max_per_source=2)
    ids = {c.chunk_id for c in result}
    assert "b0" in ids, "source B should survive regardless of cap"
    assert "a2" not in ids, "third chunk from site A should be capped"


def test_source_cap_preserves_order():
    chunks = [
        _chunk("a0", "first",         "Site A | https://sitea.com/0"),
        _chunk("b0", "second",        "Site B | https://siteb.com/0"),
        _chunk("a1", "third (capped)","Site A | https://sitea.com/1"),
        _chunk("c0", "fourth",        "Site C | https://sitec.com/0"),
    ]
    result = apply_source_cap(chunks, max_per_source=1)
    assert [c.chunk_id for c in result] == ["a0", "b0", "c0"]


def test_source_cap_empty():
    assert apply_source_cap([], max_per_source=2) == []


def test_source_cap_below_limit_passes_all():
    chunks = [
        _chunk("a0", "AAA", "Site A | https://sitea.com/0"),
        _chunk("b0", "BBB", "Site B | https://siteb.com/0"),
    ]
    assert len(apply_source_cap(chunks, max_per_source=5)) == 2


# ---------------------------------------------------------------------------
# _bm25_rank
# ---------------------------------------------------------------------------

def test_bm25_rank_match_scores_higher():
    chunks = [
        _chunk("match",   "climate change impact greenhouse emissions warming", "src"),
        _chunk("nomatch", "baseball stadium score winning team pitcher batter",  "src"),
    ]
    pairs = _bm25_rank(chunks, "climate change")
    scores = {c.chunk_id: s for s, c in pairs}
    assert scores["match"] > scores["nomatch"]


def test_bm25_rank_empty():
    assert _bm25_rank([], "query") == []


def test_bm25_rank_single():
    ch = _chunk("only", "some content here", "src")
    pairs = _bm25_rank([ch], "content")
    assert len(pairs) == 1
    assert pairs[0][1].chunk_id == "only"


def test_bm25_rank_returns_all():
    chunks = [_chunk(f"c{i}", f"text about topic {i}", "src") for i in range(6)]
    pairs = _bm25_rank(chunks, "topic")
    assert len(pairs) == 6


# ---------------------------------------------------------------------------
# apply_rrf
# ---------------------------------------------------------------------------

def test_rrf_returns_all_chunks():
    chunks = [_chunk(f"c{i}", f"text {i}", "src") for i in range(5)]
    result = apply_rrf([chunks, list(reversed(chunks))], k=60)
    assert {c.chunk_id for c in result} == {c.chunk_id for c in chunks}


def test_rrf_top_of_single_list_is_first():
    chunks = [_chunk(f"c{i}", f"text {i}", "src") for i in range(3)]
    result = apply_rrf([chunks], k=60)
    assert result[0].chunk_id == "c0"


def test_rrf_empty_inputs():
    assert apply_rrf([], k=60) == []
    assert apply_rrf([[]], k=60) == []


def test_rrf_symmetric_tie():
    """c0 rank-1 in list1 / rank-2 in list2; c1 rank-2 in list1 / rank-1 in list2.
    Both get same RRF score; both must appear in output."""
    c0 = _chunk("c0", "alpha", "src")
    c1 = _chunk("c1", "beta",  "src")
    result = apply_rrf([[c0, c1], [c1, c0]], k=60)
    ids = {c.chunk_id for c in result}
    assert ids == {"c0", "c1"}


def test_rrf_high_ranked_appears_first():
    """Chunk that is rank-1 in BOTH lists should beat one that is rank-1 in one
    and rank-last in the other."""
    top = _chunk("top",  "alpha", "src")
    mid = _chunk("mid",  "beta",  "src")
    bot = _chunk("bot",  "gamma", "src")
    list1 = [top, mid, bot]
    list2 = [top, bot, mid]   # top still first in both
    result = apply_rrf([list1, list2], k=60)
    assert result[0].chunk_id == "top"


# ---------------------------------------------------------------------------
# apply_mmr — fallback when embedder unavailable
# ---------------------------------------------------------------------------

def test_mmr_fallback_preserves_order(monkeypatch):
    """With no embedder, apply_mmr returns the original order unchanged."""
    import core.search_tool as st
    monkeypatch.setattr(st, "_get_embedder", lambda: None)
    chunks = [_chunk(f"c{i}", f"text {i}", "src") for i in range(4)]
    result = apply_mmr(chunks, "query", lam=0.7)
    assert [c.chunk_id for c in result] == [c.chunk_id for c in chunks]


def test_mmr_single_chunk():
    ch = _chunk("only", "some text", "src")
    result = apply_mmr([ch], "query")
    assert result[0].chunk_id == "only"


def test_mmr_empty():
    assert apply_mmr([], "query") == []


# ---------------------------------------------------------------------------
# _diversify — full pipeline smoke (no embedder)
# ---------------------------------------------------------------------------

def test_diversify_returns_at_most_max_results(monkeypatch):
    import core.search_tool as st
    monkeypatch.setattr(st, "_get_embedder", lambda: None)

    chunks = [
        _chunk(f"c{i}", f"content about topic {i}",
               f"Site {i % 3} | https://site{i % 3}.com/{i}")
        for i in range(12)
    ]
    result = _diversify(chunks, "topic query", max_results=5)
    assert len(result) <= 5


def test_diversify_all_corpus_chunk_instances(monkeypatch):
    import core.search_tool as st
    monkeypatch.setattr(st, "_get_embedder", lambda: None)

    chunks = [_chunk(f"c{i}", f"text {i}", f"src{i} | https://src{i}.com/") for i in range(6)]
    result = _diversify(chunks, "query", max_results=4)
    assert all(isinstance(c, CorpusChunk) for c in result)


def test_diversify_applies_source_cap(monkeypatch):
    """_diversify should cap over-represented domains."""
    import core.search_tool as st
    monkeypatch.setattr(st, "_get_embedder", lambda: None)

    same = [
        _chunk(f"s{i}", f"text {i}", f"Big Site | https://bigsite.com/{i}")
        for i in range(6)
    ]
    other = [_chunk("o0", "other content", "Other | https://other.com/1")]
    chunks = same + other

    from core.config import MAX_CHUNKS_PER_SOURCE
    result = _diversify(chunks, "query", max_results=8)
    domains = [_source_domain(c) for c in result]
    assert domains.count("bigsite.com") <= MAX_CHUNKS_PER_SOURCE
    assert "other.com" in domains


def test_diversify_empty_input(monkeypatch):
    import core.search_tool as st
    monkeypatch.setattr(st, "_get_embedder", lambda: None)
    assert _diversify([], "query", max_results=5) == []


def test_diversify_single_chunk(monkeypatch):
    import core.search_tool as st
    monkeypatch.setattr(st, "_get_embedder", lambda: None)
    ch = _chunk("only", "some text about the topic", "src | https://src.com/")
    result = _diversify([ch], "topic", max_results=5)
    assert len(result) == 1
    assert result[0].chunk_id == "only"
