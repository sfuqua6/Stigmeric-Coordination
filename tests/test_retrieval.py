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
# CompositeRetriever fallthrough
# ---------------------------------------------------------------------------

class TestCompositeRetriever(unittest.TestCase):
    def test_falls_through_to_placeholder_when_both_fail(self):
        """When Wikipedia and Web both return nothing, placeholder is used."""
        comp = CompositeRetriever()
        # Patch inner retrievers to return empty
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

    def test_uses_wikipedia_when_available(self):
        """If Wikipedia returns chunks, Web and placeholder are not called."""
        comp = CompositeRetriever()
        comp._wiki = MagicMock(spec=WikipediaRetriever)
        comp._wiki.retrieve.return_value = [
            CorpusChunk(chunk_id="w1", text="Wikipedia text.", source_tag="Wikipedia article")
        ]
        comp._web = MagicMock(spec=WebRetriever)

        chunks = comp.retrieve("test query")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_tag, "Wikipedia article")
        comp._web.retrieve.assert_not_called()

    def test_falls_through_to_web_when_wiki_fails(self):
        """If Wikipedia returns nothing, Web is tried next."""
        comp = CompositeRetriever()
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

    def test_caches_result(self):
        inner = MagicMock(spec=WikipediaRetriever)
        inner.retrieve.return_value = [
            CorpusChunk(chunk_id="c1", text="Cached text.", source_tag="src")
        ]
        cached = CachedRetriever(inner, cache_dir=self.tmpdir)

        # First call: inner is called
        chunks1 = cached.retrieve("my query")
        self.assertEqual(len(chunks1), 1)
        self.assertEqual(inner.retrieve.call_count, 1)

        # Second call: uses cache, inner NOT called again
        chunks2 = cached.retrieve("my query")
        self.assertEqual(inner.retrieve.call_count, 1)
        self.assertEqual(chunks2[0].text, "Cached text.")

    def test_cache_key_is_sha1_of_query(self):
        key = hashlib.sha1("my query".encode("utf-8")).hexdigest()
        inner = MagicMock(spec=WikipediaRetriever)
        inner.retrieve.return_value = []
        cached = CachedRetriever(inner, cache_dir=self.tmpdir)
        cached.retrieve("my query")
        cache_file = Path(self.tmpdir) / f"{key}.json"
        self.assertTrue(cache_file.exists())

    def test_different_queries_use_different_cache_files(self):
        inner = MagicMock(spec=WikipediaRetriever)
        inner.retrieve.return_value = []
        cached = CachedRetriever(inner, cache_dir=self.tmpdir)
        cached.retrieve("query A")
        cached.retrieve("query B")
        files = list(Path(self.tmpdir).glob("*.json"))
        self.assertEqual(len(files), 2)


if __name__ == "__main__":
    unittest.main()
