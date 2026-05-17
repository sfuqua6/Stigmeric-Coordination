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
_WEB_MAX_RESULTS = 6        # max DuckDuckGo results to pull
_WIKI_QUERIES_MAX = 8       # max keyphrase queries to send to Wikipedia

# Below this chunk count, CompositeRetriever keeps trying additional sources
# instead of short-circuiting. Also the threshold below which CachedRetriever
# refuses to persist a result (thin caches are usually one-off scrape failures
# that the user will hit forever after, since the cache key is just sha1(query)).
_MIN_USEFUL_CHUNKS = 4


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
        print(f"[retrieval] attempting wikipedia for query: {query!r}")
        try:
            import wikipedia  # type: ignore
        except ImportError as exc:
            print(f"[retrieval] wikipedia: package not installed ({exc}); "
                  f"install with `pip install wikipedia` (also in requirements-colab.txt)")
            return []

        keyphrases = _extract_keyphrases(query, max_phrases=_WIKI_QUERIES_MAX)
        # Always include the full query itself as the first keyphrase attempt
        all_queries = [query] + keyphrases
        all_queries = all_queries[:_WIKI_QUERIES_MAX]

        chunks: list[CorpusChunk] = []
        seen_titles: set[str] = set()
        n = 0
        last_exc: Optional[Exception] = None

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
                    except Exception as exc:
                        last_exc = exc
                        continue
                except Exception as exc:
                    last_exc = exc
                    # Try search fallback
                    try:
                        hits = wikipedia.search(q, results=2)
                        if not hits:
                            continue
                        summary = wikipedia.summary(hits[0], sentences=_WIKI_MAX_SENTENCES)
                        title = hits[0]
                    except Exception as exc2:
                        last_exc = exc2
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
            except Exception as exc:
                last_exc = exc
                continue

        if chunks:
            print(f"[retrieval] wikipedia: retrieved {len(chunks)} chunks "
                  f"({n} queries attempted)")
        else:
            err = f" ({type(last_exc).__name__}: {last_exc})" if last_exc else ""
            print(f"[retrieval] wikipedia: 0 chunks after {n} queries{err}")
        return chunks


# ---------------------------------------------------------------------------
# WebRetriever
# ---------------------------------------------------------------------------

class WebRetriever(Retriever):
    """Retrieve evidence via DuckDuckGo + page text scraping.

    Tries the `duckduckgo_search` Python package first (which adapts to DDG's
    backend changes). If that's not installed or fails at runtime, falls back
    to scraping the HTML endpoint with requests + beautifulsoup4.

    The HTML scraper has historically broken whenever DDG changed their result
    class names (`a.result__url` → something else); the package is maintained
    against the live API and is the more reliable path.

    Respects a 30s total timeout budget. No API key required.
    """

    def retrieve(self, query: str, target_chars: int = 8000) -> list[CorpusChunk]:
        print(f"[retrieval] attempting web (DuckDuckGo) for query: {query!r}")
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError as exc:
            print(f"[retrieval] web: deps not installed ({exc}); "
                  f"install with `pip install requests beautifulsoup4` "
                  f"(also in requirements-colab.txt)")
            return []

        deadline = time.time() + _WEB_TOTAL_TIMEOUT_S

        # Search step: prefer the duckduckgo_search package, fall back to HTML.
        result_links, search_via, search_exc = self._search(query, deadline)
        if not result_links:
            cause = (f" ({type(search_exc).__name__}: {search_exc})"
                     if search_exc is not None else "")
            print(f"[retrieval] web: 0 URLs from DDG ({search_via}){cause}. "
                  f"Try `pip install duckduckgo-search` (in requirements-colab.txt).")
            return []
        print(f"[retrieval] web: {len(result_links)} URL(s) from DDG ({search_via})")

        # Scrape step: pull text from each URL, skip noise.
        chunks: list[CorpusChunk] = []
        last_exc: Optional[Exception] = None
        headers = {"User-Agent": "Mozilla/5.0 (academic research bot; not commercial)"}
        for i, url in enumerate(result_links):
            if time.time() >= deadline:
                print(f"[retrieval] web: hit {_WEB_TOTAL_TIMEOUT_S}s deadline "
                      f"after {i} URL(s); stopping early")
                break
            try:
                remaining = max(3.0, deadline - time.time())
                page_resp = requests.get(
                    url, headers=headers, timeout=min(8, remaining),
                )
                soup2 = BeautifulSoup(page_resp.text, "html.parser")
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
            except Exception as exc:
                last_exc = exc
                continue

        if chunks:
            print(f"[retrieval] web: retrieved {len(chunks)} chunks "
                  f"from {len(result_links)} candidate URLs")
        else:
            err = f" (last scrape error: {type(last_exc).__name__}: {last_exc})" \
                if last_exc else ""
            print(f"[retrieval] web: 0 chunks from {len(result_links)} "
                  f"candidates{err}")
        return chunks

    def _search(self, query: str, deadline: float) -> tuple[list[str], str, Optional[Exception]]:
        """Return (links, via, exc). `via` is 'ddg-package' or 'html-scrape'."""
        # Path A: duckduckgo_search package (preferred — maintained against DDG)
        try:
            from duckduckgo_search import DDGS  # type: ignore
            try:
                links: list[str] = []
                with DDGS(timeout=int(max(3, deadline - time.time()))) as ddgs:
                    for r in ddgs.text(query, max_results=_WEB_MAX_RESULTS):
                        href = r.get("href") or r.get("url") or ""
                        if href.startswith("http"):
                            links.append(href)
                        if len(links) >= _WEB_MAX_RESULTS:
                            break
                if links:
                    return links, "ddg-package", None
                return [], "ddg-package", None
            except Exception as exc:
                # Package present but call failed (rate limit, network, etc).
                # Fall through to HTML scraper.
                pkg_exc = exc
        except ImportError:
            pkg_exc = None

        # Path B: HTML scrape fallback.
        try:
            import requests
            from bs4 import BeautifulSoup
            ddg_url = "https://html.duckduckgo.com/html/"
            headers = {"User-Agent": "Mozilla/5.0 (academic research bot; not commercial)"}
            resp = requests.post(
                ddg_url, data={"q": query, "t": "h_"}, headers=headers,
                timeout=min(10, max(3, deadline - time.time())),
            )
            if resp.status_code != 200:
                err = RuntimeError(f"DDG HTML returned HTTP {resp.status_code}")
                return [], "html-scrape", pkg_exc or err
            soup = BeautifulSoup(resp.text, "html.parser")
            links: list[str] = []
            # DDG occasionally renames classes; try a few selectors.
            for selector in ("a.result__url", "a.result__a", "a[href*='http']"):
                for a in soup.select(selector):
                    href = a.get("href", "")
                    if href.startswith("http"):
                        links.append(href)
                if links:
                    break
            return links[:_WEB_MAX_RESULTS], "html-scrape", pkg_exc
        except Exception as exc:
            return [], "html-scrape", pkg_exc or exc


# ---------------------------------------------------------------------------
# CompositeRetriever
# ---------------------------------------------------------------------------

class CompositeRetriever(Retriever):
    """Wikipedia + Web → placeholder fallback.

    Unlike a strict failover ladder, this aggregates from both Wikipedia
    and Web until at least `_MIN_USEFUL_CHUNKS` chunks have been collected
    (or both sources are exhausted). Previously, a Wikipedia call that
    returned 1 short chunk would short-circuit and prevent Web retrieval
    from ever running — producing `corpus chunks: 1` for a 6-scout run.

    When neither source produces enough, prints a loud warning and falls
    back to the engineered placeholder corpus.
    """

    def __init__(self) -> None:
        self._wiki = WikipediaRetriever()
        self._web = WebRetriever()

    def retrieve(self, query: str, target_chars: int = 8000) -> list[CorpusChunk]:
        wiki_chunks: list[CorpusChunk] = []
        wiki_exc: Optional[Exception] = None
        try:
            wiki_chunks = self._wiki.retrieve(query, target_chars)
        except Exception as exc:
            wiki_exc = exc
            print(f"[retrieval] wikipedia retriever crashed: "
                  f"{type(exc).__name__}: {exc}")

        web_chunks: list[CorpusChunk] = []
        web_exc: Optional[Exception] = None
        # Always run web too if wiki returned fewer than _MIN_USEFUL_CHUNKS.
        # The old "wiki succeeded with N>=1 chunks → skip web" branch was the
        # specific path that produced the 1-chunk-degenerate-run pathology.
        if len(wiki_chunks) < _MIN_USEFUL_CHUNKS:
            try:
                web_chunks = self._web.retrieve(query, target_chars)
            except Exception as exc:
                web_exc = exc
                print(f"[retrieval] web retriever crashed: "
                      f"{type(exc).__name__}: {exc}")

        combined = wiki_chunks + web_chunks
        if combined:
            print(f"[retrieval] sources combined: wiki={len(wiki_chunks)} "
                  f"web={len(web_chunks)} total={len(combined)} "
                  f"(target>={_MIN_USEFUL_CHUNKS})")
            if len(combined) < _MIN_USEFUL_CHUNKS:
                print(f"[retrieval] WARNING: only {len(combined)} chunks "
                      f"retrieved; the partition will spread thin across scouts.")
            return combined

        # Last resort: engineered placeholder corpus. Banner the fallback so
        # nothing downstream can mistake the run for an empirical one.
        print("=" * 72)
        print("[retrieval] WARNING / FALLBACK: both wikipedia and web retrievers returned 0 chunks.")
        if wiki_exc:
            print(f"[retrieval]   wikipedia error: {type(wiki_exc).__name__}: {wiki_exc}")
        if web_exc:
            print(f"[retrieval]   web error:       {type(web_exc).__name__}: {web_exc}")
        print("[retrieval]   Common causes:")
        print("[retrieval]     - missing deps: `pip install -r requirements-colab.txt`")
        print("[retrieval]       (needs wikipedia, requests, beautifulsoup4, duckduckgo-search)")
        print("[retrieval]     - no network connectivity")
        print("[retrieval]     - DDG / Wikipedia rate-limited or blocked this IP")
        print("[retrieval]   Using the engineered placeholder corpus instead — diversity")
        print("[retrieval]   numbers from this run are NOT evidence.")
        print("=" * 72)
        from .intake import trivial_corpus_from_thesis, chunk_corpus
        text = trivial_corpus_from_thesis(query)
        raw_chunks = chunk_corpus(text, source_tag="placeholder_corpus")
        return raw_chunks


# ---------------------------------------------------------------------------
# CachedRetriever
# ---------------------------------------------------------------------------

class CachedRetriever(Retriever):
    """On-disk cache wrapping another retriever.

    Cache is keyed by SHA1(query). Each entry is a JSON file containing the
    list of CorpusChunk dicts. Cache directory resolution:

        1. constructor `cache_dir` argument
        2. SWARM_RETRIEVAL_CACHE_DIR env var
        3. `retrieval_cache/` in the current working directory (default)
    """

    def __init__(self, inner: Retriever, cache_dir: Optional[str] = None) -> None:
        self._inner = inner
        self._cache_dir = Path(
            cache_dir
            or os.environ.get("SWARM_RETRIEVAL_CACHE_DIR")
            or "retrieval_cache"
        )

    def retrieve(self, query: str, target_chars: int = 8000) -> list[CorpusChunk]:
        key = hashlib.sha1(query.encode("utf-8")).hexdigest()
        cache_path = self._cache_dir / f"{key}.json"

        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                cached = [
                    CorpusChunk(
                        chunk_id=d["chunk_id"],
                        text=d["text"],
                        source_tag=d["source_tag"],
                    )
                    for d in data
                ]
                # Reject thin caches (a previously-broken retrieval poisoned
                # this key with 0–3 chunks; serving it forever defeats the
                # point of running real retrieval). Re-fetch instead.
                if len(cached) >= _MIN_USEFUL_CHUNKS:
                    print(f"[retrieval] cache HIT: {len(cached)} chunks from {cache_path.name}")
                    return cached
                print(f"[retrieval] cache REJECTED (thin: {len(cached)} chunks < "
                      f"{_MIN_USEFUL_CHUNKS}); re-fetching")
            except Exception:
                pass  # corrupted cache; re-fetch

        chunks = self._inner.retrieve(query, target_chars)

        # Only persist substantial results — a 0–3 chunk write would pin
        # the cache to a thin result forever.
        if len(chunks) >= _MIN_USEFUL_CHUNKS:
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
        else:
            print(f"[retrieval] cache SKIP (thin: {len(chunks)} chunks); "
                  f"not persisting to {cache_path.name}")

        return chunks
