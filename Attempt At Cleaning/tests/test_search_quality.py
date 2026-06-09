"""Search-quality fixes: page-content extraction + relevance gate (the GIGO fix).
Offline — `requests` is monkeypatched and the embedder is a fake.
"""

import pytest

import core.config as cfg
import core.search_tool as st
from core.intake import CorpusChunk

_HTML = """
<html><head><title>T</title><style>.x{}</style><script>bad()</script></head>
<body>
<nav>home about contact</nav><header>site header</header>
<article>
  <h1>Cars and Cities</h1>
  <p>Banning private cars in dense cities can cut transport emissions substantially according to several studies.</p>
  <p>Short.</p>
  <p>Congestion pricing in London reduced traffic volumes by about twenty percent in the first year of operation.</p>
</article>
<footer>copyright</footer><script>tracker()</script>
</body></html>
"""


class _Resp:
    def __init__(self, text, status=200, ctype="text/html; charset=utf-8"):
        self.text, self.status_code = text, status
        self.headers = {"content-type": ctype}


def _chunk(text, url):
    return CorpusChunk(chunk_id="c", text=text, source_tag=f"Title | {url}")


def test_url_of_parses_source_tag():
    assert st._url_of(_chunk("x", "https://example.com/a")) == "https://example.com/a"


# ---- page fetch / extraction --------------------------------------------

def test_fetch_extracts_main_text_and_strips_chrome(monkeypatch):
    pytest.importorskip("bs4")
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(_HTML))
    text = st._fetch_page_text("https://example.com/cars", timeout=5)
    assert "Banning private cars" in text and "twenty percent" in text
    assert "tracker" not in text and "site header" not in text and "home about" not in text
    assert "Short." not in text          # tiny <p> dropped


def test_fetch_non_html_returns_empty(monkeypatch):
    pytest.importorskip("bs4")
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp("{}", ctype="application/json"))
    assert st._fetch_page_text("https://x/api", timeout=5) == ""


def test_fetch_bad_status_and_errors_return_empty(monkeypatch):
    pytest.importorskip("bs4")
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp("<p>hi</p>", status=404))
    assert st._fetch_page_text("https://x", timeout=5) == ""
    def boom(*a, **k): raise RuntimeError("net")
    monkeypatch.setattr(requests, "get", boom)
    assert st._fetch_page_text("https://x", timeout=1) == ""


def test_enrich_replaces_snippet_with_page(monkeypatch):
    pytest.importorskip("bs4")
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(_HTML))
    monkeypatch.setattr(cfg, "SEARCH_FETCH_PAGES", True)
    monkeypatch.setattr(cfg, "SEARCH_FETCH_TOP_K", 1)
    monkeypatch.setattr(cfg, "SEARCH_FETCH_MIN_CHARS", 50)
    out = st._enrich_with_pages([_chunk("tiny snippet", "https://example.com/cars")])
    assert "Banning private cars" in out[0].text


def test_enrich_disabled_keeps_snippet(monkeypatch):
    monkeypatch.setattr(cfg, "SEARCH_FETCH_PAGES", False)
    assert st._enrich_with_pages([_chunk("tiny snippet", "u")])[0].text == "tiny snippet"


# ---- relevance gate ------------------------------------------------------

class _FakeEmbedder:
    def encode(self, x, normalize_embeddings=True):
        import numpy as np
        def vec(s): return np.array([1.0, 0.0]) if "car" in s.lower() else np.array([0.0, 1.0])
        return vec(x) if isinstance(x, str) else np.array([vec(s) for s in x])


def test_relevance_gate_drops_offtopic(monkeypatch):
    pytest.importorskip("numpy")
    monkeypatch.setattr(cfg, "SEARCH_RELEVANCE_MIN", 0.5)
    monkeypatch.setattr(st, "_get_embedder", lambda: _FakeEmbedder())
    chunks = [_chunk("car ban evidence", "u1"), _chunk("cars and congestion", "u2"),
              _chunk("plastic waste recycling", "u3")]
    out = st._relevance_filter(chunks, "should cities ban cars")
    assert "plastic waste recycling" not in [c.text for c in out]
    assert len(out) == 2


def test_relevance_gate_never_empties(monkeypatch):
    pytest.importorskip("numpy")
    monkeypatch.setattr(cfg, "SEARCH_RELEVANCE_MIN", 0.99)   # strict — nothing passes
    monkeypatch.setattr(st, "_get_embedder", lambda: _FakeEmbedder())
    chunks = [_chunk("plastic a", "u1"), _chunk("plastic b", "u2"), _chunk("plastic c", "u3")]
    assert len(st._relevance_filter(chunks, "cars")) == 2   # keeps the 2 best


def test_relevance_gate_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(cfg, "SEARCH_RELEVANCE_MIN", 0.0)
    chunks = [_chunk("a", "u1"), _chunk("b", "u2"), _chunk("c", "u3")]
    assert st._relevance_filter(chunks, "q") is chunks
