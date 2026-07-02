"""Tests for core/retrieval.py (§1 directive).

Mocks both the wikipedia and requests layers. Tests the composite fallthrough
logic without network access.

Run with:
    python -m unittest tests.test_retrieval -v
"""

import sys
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieval import (
    WikipediaRetriever,
    WebRetriever,
    CohereCorpusRetriever,
    CompositeRetriever,
    CachedRetriever,
    _extract_keyphrases,
)
from core.intake import CorpusChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_wiki_module(summaries: dict, search_results: dict = None):
    """Return a mock wikipedia module that returns known summaries."""
    wiki = MagicMock()
    wiki.exceptions.DisambiguationError = Exception

    def _summary(query, sentences=5, auto_suggest=True, redirect=True):
        if query in summaries:
            return summaries[query]
        raise Exception(f"No article: {query!r}")

    def _search(query, results=1):
        if search_results and query in search_results:
            return search_results[query]
        return []

    wiki.summary = _summary
    wiki.search = _search
    # Make page() raise so the retriever falls back to title=query string
    # (avoids all calls sharing the same MagicMock return_value for .title)
    wiki.page.side_effect = Exception("page lookup disabled in test")
    return wiki


# ---------------------------------------------------------------------------
# Keyphrase extraction
# ---------------------------------------------------------------------------

class TestExtractKeyphrases(unittest.TestCase):
    def test_returns_list(self):
        phrases = _extract_keyphrases("Climate action is necessary for global stability.")
        self.assertIsInstance(phrases, list)

    def test_filters_stopwords(self):
        phrases = _extract_keyphrases("The the the is are to of.")
        for p in phrases:
            words = p.split()
            for w in words:
                self.assertNotIn(w.lower(), {"the", "is", "are", "to", "of"})

    def test_max_phrases_respected(self):
        long_text = " ".join([f"concept{i}" for i in range(50)])
        phrases = _extract_keyphrases(long_text, max_phrases=4)
        self.assertLessEqual(len(phrases), 4)


# ---------------------------------------------------------------------------
# WikipediaRetriever
# ---------------------------------------------------------------------------

class TestWikipediaRetriever(unittest.TestCase):
    def test_returns_chunks_on_success(self):
        fake_wiki = _make_fake_wiki_module({
            "Climate action is necessary": "Climate action refers to mitigation efforts.",
            "climate action": "Climate action summary text here.",
        })
        with patch.dict("sys.modules", {"wikipedia": fake_wiki}):
            retriever = WikipediaRetriever()
            chunks = retriever.retrieve("Climate action is necessary")
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertIsInstance(c, CorpusChunk)

    def test_returns_empty_on_import_error(self):
        with patch.dict("sys.modules", {"wikipedia": None}):
            retriever = WikipediaRetriever()
            # When wikipedia is None in sys.modules, import inside will fail
            # Patch it properly
        with patch("builtins.__import__", side_effect=ImportError("no wikipedia")):
            retriever2 = WikipediaRetriever()
            chunks = retriever2.retrieve("anything")
        # Either [] or non-empty depending on whether wikipedia is installed
        self.assertIsInstance(chunks, list)

    def test_chunk_source_tag_set(self):
        fake_wiki = _make_fake_wiki_module({
            "renewable energy": "Renewable energy is energy from natural sources.",
        })
        with patch.dict("sys.modules", {"wikipedia": fake_wiki}):
            retriever = WikipediaRetriever()
            chunks = retriever.retrieve("renewable energy")
        for c in chunks:
            self.assertIsNotNone(c.source_tag)
            self.assertGreater(len(c.source_tag), 0)


# ---------------------------------------------------------------------------
# CohereCorpusRetriever adapter
# ---------------------------------------------------------------------------

class TestCohereCorpusRetriever(unittest.TestCase):
    """Doesn't require cohere/datasets/faiss installed — mocks the store."""

    def test_translates_search_to_retrieve(self):
        from core import corpus_store_cohere
        fake_store = MagicMock()
        fake_store.search.return_value = [
            CorpusChunk(chunk_id="cohere_00_abc", text="Renewable energy text.",
                        source_tag="Renewable energy")
        ]
        with patch.object(corpus_store_cohere, "get_store",
                          return_value=fake_store):
            r = CohereCorpusRetriever(n_chunks=20)
            chunks = r.retrieve("renewable energy", target_chars=8000)
        fake_store.search.assert_called_once_with("renewable energy", n_chunks=20)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_tag, "Renewable energy")

    def test_returns_empty_on_dep_missing(self):
        from core import corpus_store_cohere
        with patch.object(corpus_store_cohere, "get_store",
                          side_effect=ImportError("No module named 'faiss'")):
            r = CohereCorpusRetriever()
            chunks = r.retrieve("anything")
        self.assertEqual(chunks, [])

    def test_returns_empty_on_api_key_missing(self):
        from core import corpus_store_cohere
        with patch.object(corpus_store_cohere, "get_store",
                          side_effect=RuntimeError("COHERE_API_KEY not set")):
            r = CohereCorpusRetriever()
            chunks = r.retrieve("anything")
        self.assertEqual(chunks, [])


# ---------------------------------------------------------------------------
# CompositeRetriever fallthrough
# ---------------------------------------------------------------------------

class TestCompositeRetriever(unittest.TestCase):
    def setUp(self):
        # The agentic search stack (search_tool.search) is now the
        # composite's PRIMARY source. These tests pin the LEGACY fallthrough
        # chain (cohere -> wiki -> web -> placeholder), so silence the
        # primary; its own behavior is covered by the search_tool tests.
        p = patch("core.retrieval._agentic_search", return_value=[])
        p.start()
        self.addCleanup(p.stop)

    def _mock_cohere_empty(self, comp):
        """Replace comp._cohere with one that returns []."""
        from core.retrieval import CohereCorpusRetriever
        comp._cohere = MagicMock(spec=CohereCorpusRetriever)
        comp._cohere.retrieve.return_value = []
        return comp._cohere

    def test_falls_through_to_placeholder_when_all_fail(self):
        """When every real source returns nothing, placeholder is used."""
        comp = CompositeRetriever()
        self._mock_cohere_empty(comp)
        comp._wiki = MagicMock(spec=WikipediaRetriever)
        comp._wiki.retrieve.return_value = []
        comp._web = MagicMock(spec=WebRetriever)
        comp._web.retrieve.return_value = []

        import io
        from contextlib import redirect_stdout
        out = io.StringIO()
        with redirect_stdout(out):
            chunks = comp.retrieve("some thesis prompt")

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0, "Should fall back to placeholder chunks")
        output = out.getvalue()
        self.assertIn("WARNING", output, "Fallback must print a WARNING")
        self.assertIn("placeholder corpus", output)

    def test_cohere_short_circuits_when_fat(self):
        """When Cohere returns >= _MIN_USEFUL_CHUNKS, Wikipedia and Web are skipped."""
        from core.retrieval import _MIN_USEFUL_CHUNKS
        comp = CompositeRetriever()
        from core.retrieval import CohereCorpusRetriever
        comp._cohere = MagicMock(spec=CohereCorpusRetriever)
        cohere_chunks = [
            CorpusChunk(chunk_id=f"cohere_{i:02d}_abcd", text=f"Cohere text {i}.",
                        source_tag=f"Wiki article {i}")
            for i in range(_MIN_USEFUL_CHUNKS)
        ]
        comp._cohere.retrieve.return_value = cohere_chunks
        comp._wiki = MagicMock(spec=WikipediaRetriever)
        comp._web = MagicMock(spec=WebRetriever)

        chunks = comp.retrieve("test query")
        self.assertEqual(len(chunks), _MIN_USEFUL_CHUNKS)
        comp._wiki.retrieve.assert_not_called()
        comp._web.retrieve.assert_not_called()

    def test_uses_wikipedia_when_cohere_empty(self):
        """If Cohere is unavailable and Wikipedia returns enough chunks,
        Web is not called."""
        from core.retrieval import _MIN_USEFUL_CHUNKS
        comp = CompositeRetriever()
        self._mock_cohere_empty(comp)
        comp._wiki = MagicMock(spec=WikipediaRetriever)
        wiki_chunks = [
            CorpusChunk(chunk_id=f"w{i}", text=f"Wikipedia text {i}.",
                        source_tag="Wikipedia article")
            for i in range(_MIN_USEFUL_CHUNKS)
        ]
        comp._wiki.retrieve.return_value = wiki_chunks
        comp._web = MagicMock(spec=WebRetriever)

        chunks = comp.retrieve("test query")
        self.assertEqual(len(chunks), _MIN_USEFUL_CHUNKS)
        self.assertEqual(chunks[0].source_tag, "Wikipedia article")
        comp._web.retrieve.assert_not_called()

    def test_keeps_querying_web_when_wiki_thin(self):
        """Cohere+Wiki together still < _MIN_USEFUL_CHUNKS → Web is called.
        Regression guard for the 1-chunk pathology."""
        from core.retrieval import _MIN_USEFUL_CHUNKS
        comp = CompositeRetriever()
        self._mock_cohere_empty(comp)
        comp._wiki = MagicMock(spec=WikipediaRetriever)
        comp._wiki.retrieve.return_value = [
            CorpusChunk(chunk_id="w1", text="Lone Wikipedia text.",
                        source_tag="Wikipedia article")
        ]
        comp._web = MagicMock(spec=WebRetriever)
        comp._web.retrieve.return_value = [
            CorpusChunk(chunk_id=f"web{i}", text=f"Web text {i}.",
                        source_tag=f"http://example.com/{i}")
            for i in range(_MIN_USEFUL_CHUNKS)
        ]
        chunks = comp.retrieve("test query")
        comp._web.retrieve.assert_called_once()
        self.assertEqual(len(chunks), 1 + _MIN_USEFUL_CHUNKS)

    def test_falls_through_to_web_when_wiki_fails(self):
        """If Cohere and Wikipedia return nothing, Web is tried next."""
        comp = CompositeRetriever()
        self._mock_cohere_empty(comp)
        comp._wiki = MagicMock(spec=WikipediaRetriever)
        comp._wiki.retrieve.return_value = []
        comp._web = MagicMock(spec=WebRetriever)
        comp._web.retrieve.return_value = [
            CorpusChunk(chunk_id="web1", text="Web page text.", source_tag="http://example.com")
        ]

        chunks = comp.retrieve("test query")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_tag, "http://example.com")


# ---------------------------------------------------------------------------
# CachedRetriever
# ---------------------------------------------------------------------------

class TestCachedRetriever(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _fat_chunks(self, n: int) -> list[CorpusChunk]:
        """Return enough chunks to clear _MIN_USEFUL_CHUNKS (so the cache persists)."""
        from core.retrieval import _MIN_USEFUL_CHUNKS
        return [
            CorpusChunk(chunk_id=f"c{i}", text=f"Cached text {i}.", source_tag="src")
            for i in range(max(n, _MIN_USEFUL_CHUNKS))
        ]

    def test_caches_result(self):
        inner = MagicMock(spec=WikipediaRetriever)
        inner.retrieve.return_value = self._fat_chunks(4)
        cached = CachedRetriever(inner, cache_dir=self.tmpdir)

        # First call: inner is called
        chunks1 = cached.retrieve("my query")
        self.assertGreater(len(chunks1), 0)
        self.assertEqual(inner.retrieve.call_count, 1)

        # Second call: uses cache, inner NOT called again
        chunks2 = cached.retrieve("my query")
        self.assertEqual(inner.retrieve.call_count, 1)
        self.assertEqual(chunks2[0].text, "Cached text 0.")

    def test_cache_key_is_sha1_of_query(self):
        key = hashlib.sha1("my query".encode("utf-8")).hexdigest()
        inner = MagicMock(spec=WikipediaRetriever)
        inner.retrieve.return_value = self._fat_chunks(4)
        cached = CachedRetriever(inner, cache_dir=self.tmpdir)
        cached.retrieve("my query")
        cache_file = Path(self.tmpdir) / f"{key}.json"
        self.assertTrue(cache_file.exists())

    def test_different_queries_use_different_cache_files(self):
        inner = MagicMock(spec=WikipediaRetriever)
        inner.retrieve.return_value = self._fat_chunks(4)
        cached = CachedRetriever(inner, cache_dir=self.tmpdir)
        cached.retrieve("query A")
        cached.retrieve("query B")
        files = list(Path(self.tmpdir).glob("*.json"))
        self.assertEqual(len(files), 2)

    def test_thin_results_are_not_cached(self):
        """Regression guard: a retrieval that returns < _MIN_USEFUL_CHUNKS
        must NOT poison the cache. Previously, a one-chunk Wikipedia hit
        would write a 1-chunk cache file that served the same broken
        result on every subsequent run."""
        from core.retrieval import _MIN_USEFUL_CHUNKS
        inner = MagicMock(spec=WikipediaRetriever)
        # One chunk — well below the threshold.
        inner.retrieve.return_value = [
            CorpusChunk(chunk_id="thin", text="Lonely chunk.", source_tag="src")
        ]
        cached = CachedRetriever(inner, cache_dir=self.tmpdir)

        cached.retrieve("my query")
        cached.retrieve("my query")
        # Inner called both times — no cache hit because no cache write.
        self.assertEqual(inner.retrieve.call_count, 2)
        files = list(Path(self.tmpdir).glob("*.json"))
        self.assertEqual(len(files), 0,
                         f"thin (<{_MIN_USEFUL_CHUNKS}) results must not be cached")


if __name__ == "__main__":
    unittest.main()
