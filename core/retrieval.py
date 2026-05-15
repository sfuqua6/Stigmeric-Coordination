"""Real retriever implementations (§1 directive).

Replaces trivial_corpus_from_thesis() in core/intake.py with actual evidence
retrieval. The CompositeRetriever tries Wikipedia → Web → placeholder, falling
back to the engineered corpus ONLY if both fail, and prints a loud warning
when it does so.

# FUTURE-CLAUDE NOTE: The trivial_corpus_from_thesis fallback in
# CompositeRetriever exists ONLY as a last-resort safety net so the pipeline
# can run offline. If you see the fallback warning in stdout, the run's
# diversity numbers are NOT valid evidence. Do not cite them. Fix the
# retriever connectivity before reporting results.

Classes
-------
    Retriever(ABC)          — abstract base
    WikipediaRetriever      — uses `wikipedia` package; noun-phrase keyphrases
    WebRetriever            — DuckDuckGo HTML search + beautifulsoup4
    CompositeRetriever      — Wikipedia → Web → placeholder fallback
    CachedRetriever         — on-disk SHA1-keyed cache wrapping any retriever

All retrievers return list[CorpusChunk] from core.intake.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .intake import CorpusChunk

# ---------------------------------------------------------------------------
# Timeouts and limits
# ---------------------------------------------------------------------------
_WEB_TOTAL_TIMEOUT_S = 30   # total budget for all web requests in one retrieve() call
_MAX_CONTENT_CHARS = 2000   # max chars per chunk from any source
_WIKI_MAX_SENTENCES = 5     # max sentences per Wikipedia summary
_WEB_MAX_RESULTS = 4        # max DuckDuckGo results to pull
_WIKI_QUERIES_MAX = 8       # max keyphrase queries to send to Wikipedia


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, target_chars: int = 8000) -> list[CorpusChunk]:
        """Retrieve evidence for the given query.

        Parameters
        ----------
        query:        The thesis / task prompt to retrieve evidence for.
        target_chars: Soft total character target (chunks may exceed this slightly).

        Returns
        -------
        List of CorpusChunk objects, each with a source_tag.
        """


# ---------------------------------------------------------------------------
# Noun-phrase keyphrase extraction (no NLP deps — re-based)
# ---------------------------------------------------------------------------

_STOP = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
    "and", "or", "but", "for", "on", "at", "by", "with", "as",
    "that", "this", "these", "those", "it", "its", "be", "been",
    "not", "no", "can", "will", "would", "should", "may", "might",
    "does", "do", "did", "has", "have", "had", "if", "then", "than",
    "we", "i", "you", "he", "she", "they", "our", "my", "your",
    "very", "more", "most", "also", "how", "what", "when", "why",
    "which", "who", "all", "any", "each", "from",
})

_WORD_RE = re.compile(r"\b([A-Za-z][a-z]*(?:\s+[A-Za-z][a-z]+){0,2})\b")


def _extract_keyphrases(text: str, max_phrases: int = 8) -> list[str]:
    """Extract up to max_phrases noun-phrase keyphrases from text.

    Uses a simple re-based approach: find capitalized word groups,
    filter stopwords, deduplicate, and return the longest unique phrases first.
    """
    # Sentences as units to extract from
    sentences = re.split(r"[.!?]+", text)
    phrases: list[str] = []

    for sent in sentences:
        # Find consecutive non-stop title-cased or content words
        words = sent.strip().split()
        i = 0
        buf: list[str] = []
        for w in words:
            clean = w.strip(".,;:?!\"'()[]{}").lower()
            if clean and clean not in _STOP and len(clean) > 2:
                buf.append(clean)
            else:
                if len(buf) >= 1:
                    phrases.append(" ".join(buf))
                buf = []
        if buf:
            phrases.append(" ".join(buf))

    # Deduplicate and prefer multi-word phrases
    seen: set[str] = set()
    unique: list[str] = []
    for p in sorted(phrases, key=len, reverse=True):
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            unique.append(p)

    return unique[:max_phrases]


# ---------------------------------------------------------------------------
# WikipediaRetriever
# ---------------------------------------------------------------------------

class WikipediaRetriever(Retriever):
    """Retrieve evidence from Wikipedia using the `wikipedia` package.

    Generates 4–8 keyphrase queries via noun-phrase extraction, then pulls
    article summaries (up to _WIKI_MAX_SENTENCES sentences each). Each
    article summary becomes one CorpusChunk.
    """

    def retrieve(self, query: str, target_chars: int = 8000) -> list[CorpusChunk]:
        try:
            import wikipedia  # type: ignore
        except ImportError:
            return []

        keyphrases = _extract_keyphrases(query, max_phrases=_WIKI_QUERIES_MAX)
        # Always include the full query itself as the first keyphrase attempt
        all_queries = [query] + keyphrases
        all_queries = all_queries[:_WIKI_QUERIES_MAX]

        chunks: list[CorpusChunk] = []
        seen_titles: set[str] = set()
        n = 0

        for q in all_queries:
            if n >= _WIKI_QUERIES_MAX:
                break
            n += 1
            try:
                # Try direct summary first
                try:
                    summary = wikipedia.summary(
                        q, sentences=_WIKI_MAX_SENTENCES,
                        auto_suggest=True, redirect=True,
                    )
                    title = q  # best-effort title
                    try:
                        page = wikipedia.page(q, auto_suggest=True, redirect=True)
                        title = page.title
                    except Exception:
                        pass
                except wikipedia.exceptions.DisambiguationError as e:
                    # Pick the first option
                    try:
                        option = e.options[0] if e.options else q
                        summary = wikipedia.summary(option, sentences=_WIKI_MAX_SENTENCES)
                        title = option
                    except Exception:
                        continue
                except Exception:
                    # Try search fallback
                    try:
                        hits = wikipedia.search(q, results=2)
                        if not hits:
                            continue
                        summary = wikipedia.summary(hits[0], sentences=_WIKI_MAX_SENTENCES)
                        title = hits[0]
                    except Exception:
                        continue

                if title in seen_titles:
                    continue
                seen_titles.add(title)

                text = summary[:_MAX_CONTENT_CHARS]
                chunks.append(CorpusChunk(
                    chunk_id=f"wiki_{hashlib.sha1(title.encode()).hexdigest()[:8]}",
                    text=text,
                    source_tag=title,
                ))
            except Exception:
                continue

        return chunks


# ---------------------------------------------------------------------------
# WebRetriever
# ---------------------------------------------------------------------------

class WebRetriever(Retriever):
    """Retrieve evidence via DuckDuckGo HTML search + page text scraping.

    Uses requests + beautifulsoup4. Respects a 30s total timeout budget.
    No API key required (uses DDG HTML endpoint).
    """

    def retrieve(self, query: str, target_chars: int = 8000) -> list[CorpusChunk]:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return []

        chunks: list[CorpusChunk] = []
        deadline = time.time() + _WEB_TOTAL_TIMEOUT_S

        # DuckDuckGo HTML search — no API key, but rate-limited
        ddg_url = "https://html.duckduckgo.com/html/"
        headers = {"User-Agent": "Mozilla/5.0 (academic research bot; not commercial)"}
        params = {"q": query, "t": "h_"}

        try:
            resp = requests.post(
                ddg_url, data=params, headers=headers,
                timeout=min(10, deadline - time.time()),
            )
            if resp.status_code != 200:
                return chunks
            soup = BeautifulSoup(resp.text, "html.parser")
            result_links = []
            for a in soup.select("a.result__url"):
                href = a.get("href", "")
                if href.startswith("http"):
                    result_links.append(href)
            result_links = result_links[:_WEB_MAX_RESULTS]
        except Exception:
            return chunks

        for i, url in enumerate(result_links):
            if time.time() >= deadline:
                break
            try:
                remaining = max(3.0, deadline - time.time())
                page_resp = requests.get(
                    url, headers=headers, timeout=min(8, remaining),
                )
                soup2 = BeautifulSoup(page_resp.text, "html.parser")
                # Remove scripts and styles
                for tag in soup2(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup2.get_text(separator=" ", strip=True)
                text = re.sub(r"\s+", " ", text).strip()[:_MAX_CONTENT_CHARS]
                if len(text) < 200:
                    continue

                chunks.append(CorpusChunk(
                    chunk_id=f"web_{i}_{hashlib.sha1(url.encode()).hexdigest()[:8]}",
                    text=text,
                    source_tag=url[:80],
                ))
            except Exception:
                continue

        return chunks


# ---------------------------------------------------------------------------
# CompositeRetriever
# ---------------------------------------------------------------------------

class CompositeRetriever(Retriever):
    """Wikipedia → Web → placeholder fallback.

    When the fallback fires, prints a loud warning so any diversity numbers
    produced by the run are clearly flagged as invalid.
    """

    def __init__(self) -> None:
        self._wiki = WikipediaRetriever()
        self._web = WebRetriever()

    def retrieve(self, query: str, target_chars: int = 8000) -> list[CorpusChunk]:
        # Try Wikipedia
        chunks = self._wiki.retrieve(query, target_chars)
        if chunks:
            return chunks

        # Try Web
        chunks = self._web.retrieve(query, target_chars)
        if chunks:
            return chunks

        # Last resort: engineered placeholder corpus
        print(
            "[retrieval] WARNING: falling back to engineered placeholder corpus "
            "— partition diversity numbers from this run are not evidence"
        )
        from .intake import trivial_corpus_from_thesis, chunk_corpus
        text = trivial_corpus_from_thesis(query)
        raw_chunks = chunk_corpus(text, source_tag="placeholder_corpus")
        # Return the text content packed into CorpusChunks
        return raw_chunks


# ---------------------------------------------------------------------------
# CachedRetriever
# ---------------------------------------------------------------------------

class CachedRetriever(Retriever):
    """On-disk cache wrapping another retriever.

    Cache is keyed by SHA1(query). Stored under `retrieval_cache/` in the
    current working directory. Each entry is a JSON file containing the list
    of CorpusChunk dicts.
    """

    def __init__(self, inner: Retriever, cache_dir: Optional[str] = None) -> None:
        self._inner = inner
        self._cache_dir = Path(cache_dir or "retrieval_cache")

    def retrieve(self, query: str, target_chars: int = 8000) -> list[CorpusChunk]:
        key = hashlib.sha1(query.encode("utf-8")).hexdigest()
        cache_path = self._cache_dir / f"{key}.json"

        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return [
                    CorpusChunk(
                        chunk_id=d["chunk_id"],
                        text=d["text"],
                        source_tag=d["source_tag"],
                    )
                    for d in data
                ]
            except Exception:
                pass  # corrupted cache; re-fetch

        chunks = self._inner.retrieve(query, target_chars)

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps([
                    {"chunk_id": c.chunk_id, "text": c.text, "source_tag": c.source_tag}
                    for c in chunks
                ], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass  # cache write failure is non-fatal

        return chunks
