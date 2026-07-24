"""Agentic search tool — primary Tavily, fallback DuckDuckGo, last resort Cohere.

Phase 2A. The scout-partition retrieval (wiki-simple via Cohere) has three
problems this addresses at once:

  1. **Topic skew.** Cohere's wiki-simple index is biased to encyclopedic
     content and underweights current/contested topics.
  2. **Dedup gap.** Agents repeatedly retrieve the same passages with no
     awareness of what's already been queried.
  3. **Relevance threshold absence.** Scouts get partitions even when no
     useful evidence exists in the index.

Agents call `search_tool.search(query)` and receive `list[CorpusChunk]` with
`source_url` and `title` populated. Results are SHA(query)-keyed and cached
locally so repeated identical queries within and across runs hit disk, not
the network.

Backends (in order):
  - Tavily (`pip install tavily-python`, TAVILY_API_KEY env var, 1000/mo free)
  - DuckDuckGo (`pip install duckduckgo-search`, no key)
  - Cohere wiki-simple store (last resort, same dependency as core/retrieval.py)

Each backend returns up to `max_results` chunks. Failure of one backend falls
through to the next; final failure returns [].
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from .intake import CorpusChunk
from .config import (
    MAX_CHUNKS_PER_SOURCE, MMR_LAMBDA, HYBRID_RETRIEVAL, RRF_K,
)

_DEFAULT_CACHE_DIR = "/content/swarm_search_cache"
_MAX_CONTENT_CHARS = 2000
_DEFAULT_MAX_RESULTS = 8  # raised from 5 to allow more DDG leads
_MIN_FOLLOWUP_RESULTS = 3

# ---------------------------------------------------------------------------
# Retrieval timing stats (Tier-0 latency instrumentation).
#
# Retrieval is the dominant wall-clock cost (blocking HTTP). These process-
# global counters let summary.json report how much of a run was spent in
# search vs. inference. Guarded by a lock because searches now run in worker
# threads (asyncio.to_thread). Snapshot is read once at end of run.
# ---------------------------------------------------------------------------
import threading as _threading

_SEARCH_STATS_LOCK = _threading.Lock()
_SEARCH_STATS = {
    "calls": 0,           # search() invocations (excl. MOCK short-circuit)
    "empty_results": 0,   # invocations that returned no chunks
    "total_s": 0.0,       # cumulative wall time inside search()
    "max_s": 0.0,         # slowest single search()
    "fetch_calls": 0,     # page-content fetches attempted (incl. cache hits)
    "fetch_cache_hits": 0,  # page fetches served from the in-run URL cache
    "fetch_total_s": 0.0, # cumulative wall time fetching full pages (network only)
}

# Cross-worker page-content cache (Tier 1.2). Page text is immutable within a
# run, and many workers fetch the same top URLs across queries, so memoize
# url -> extracted text for the run. Reset at run start with the stats.
_PAGE_CACHE: dict[str, str] = {}
_PAGE_CACHE_LOCK = _threading.Lock()


def reset_search_stats() -> None:
    """Zero the retrieval counters + page cache (call at the start of a run)."""
    with _SEARCH_STATS_LOCK:
        _SEARCH_STATS.update(calls=0, empty_results=0, total_s=0.0, max_s=0.0,
                             fetch_calls=0, fetch_cache_hits=0, fetch_total_s=0.0)
    with _PAGE_CACHE_LOCK:
        _PAGE_CACHE.clear()


def search_stats_snapshot() -> dict:
    """Return a copy of the retrieval counters with derived averages."""
    with _SEARCH_STATS_LOCK:
        s = dict(_SEARCH_STATS)
    s["avg_s"] = round(s["total_s"] / s["calls"], 3) if s["calls"] else 0.0
    s["total_s"] = round(s["total_s"], 1)
    s["max_s"] = round(s["max_s"], 1)
    s["fetch_total_s"] = round(s["fetch_total_s"], 1)
    return s


def _fetch_page_text_cached(url: str, timeout: float) -> str:
    """`_fetch_page_text` with an in-run url->text memo (Tier 1.2).

    A cache hit returns instantly (no network) and is counted separately so
    fetch_total_s reflects only real network time.
    """
    if url:
        with _PAGE_CACHE_LOCK:
            if url in _PAGE_CACHE:
                with _SEARCH_STATS_LOCK:
                    _SEARCH_STATS["fetch_cache_hits"] += 1
                return _PAGE_CACHE[url]
    text = _fetch_page_text(url, timeout)
    if url:
        with _PAGE_CACHE_LOCK:
            _PAGE_CACHE[url] = text
    return text

_FOLLOWUP_MODIFIERS_BY_TASK: dict[str, list[str]] = {
    "debate": [
        "expert opinions",
        "analyst perspectives",
        "policy implications",
        "counterarguments",
    ],
    "analysis": [
        "statistics",
        "research findings",
        "case studies",
        "government reports",
    ],
    "problem_solving": [
        "best practices",
        "case studies",
        "pilot programs",
        "project reports",
    ],
    "creative": [
        "themes",
        "examples",
        "symbolism",
        "popular uses",
    ],
    "coding": [
        "python example",
        "implementation guide",
        "best practices",
        "common pitfalls",
    ],
}

_FOLLOWUP_DEFAULT = [
    "research",
    "statistics",
    "case studies",
    "report",
    "survey",
    "expert commentary",
]


def _cache_dir() -> Path:
    raw = os.environ.get("SWARM_SEARCH_CACHE_DIR", _DEFAULT_CACHE_DIR)
    p = Path(raw)
    # On non-Colab the /content prefix won't exist; fall back to a local dir.
    if not p.parent.exists() and raw.startswith("/content"):
        p = Path("search_cache")
    return p


def _cache_path(query: str, max_results: int) -> Path:
    key = hashlib.sha1(f"{query}||{max_results}".encode("utf-8")).hexdigest()[:16]
    return _cache_dir() / f"{key}.json"


def _sanitize_query(query: str) -> str:
    """Clean the outgoing web search query for DuckDuckGo."""
    if not query:
        return query
    query = query.replace("\n", " ").replace("\r", " ")
    query = re.sub(r"\s+", " ", query).strip()
    # Remove prompt-like artifacts and noisy punctuation that do not help web search.
    query = re.sub(r"(?i)\b(task|claim|evidence|answer|question|response)\b:\s*", "", query)
    query = re.sub(r"[\"'“”‘’<>\[\]]+", "", query)
    query = query.strip()
    if len(query) > 150:
        query = query[:150].rsplit(" ", 1)[0]
    return query


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    url = re.sub(r"^https?://", "", url, flags=re.I)
    url = re.sub(r"^www\.", "", url, flags=re.I)
    url = url.split("?", 1)[0].split("#", 1)[0]
    return url.rstrip("/").lower()


def _dedupe_chunks(chunks: list[CorpusChunk], max_results: int) -> list[CorpusChunk]:
    seen = set()
    deduped: list[CorpusChunk] = []
    for c in chunks:
        url = ""
        if "|" in c.source_tag:
            url = c.source_tag.rsplit("|", 1)[1].strip()
        norm = _normalize_url(url) or c.chunk_id
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(c)
        if len(deduped) >= max_results:
            break
    return deduped


# ---------------------------------------------------------------------------
# Fix S: diversity utilities (source cap, MMR, BM25, RRF)
# ---------------------------------------------------------------------------

_EMBEDDER_CACHE: Optional[object] = None


def _get_embedder():
    """Lazy singleton: ACTUALLY reuse the SignalStore's embedder.

    The docstring always claimed this; the body instantiated its own
    SentenceTransformer — a second ~90 MB copy of identical weights plus a
    second warmup. Delegating to the store's loader also inherits its
    transformers-AutoModel fallback (torchcodec/FFmpeg breakage on Colab)
    and its loud UNAVAILABLE diagnostics.
    """
    global _EMBEDDER_CACHE
    if _EMBEDDER_CACHE is None:
        try:
            from core.signal_store import _try_load_embedder
            _EMBEDDER_CACHE = _try_load_embedder()
        except Exception:
            pass
    return _EMBEDDER_CACHE


def _source_domain(chunk: CorpusChunk) -> str:
    """Extract the normalised domain from a chunk's source_tag."""
    tag = chunk.source_tag
    if "|" in tag:
        url = tag.rsplit("|", 1)[1].strip()
        norm = _normalize_url(url)
        return norm.split("/")[0]
    return _normalize_url(tag).split("/")[0]


def apply_source_cap(
    chunks: list[CorpusChunk],
    max_per_source: int = MAX_CHUNKS_PER_SOURCE,
) -> list[CorpusChunk]:
    """Keep at most max_per_source chunks per source domain (Fix S diversity floor)."""
    counts: dict[str, int] = {}
    result = []
    for c in chunks:
        domain = _source_domain(c) or c.chunk_id
        if counts.get(domain, 0) < max_per_source:
            counts[domain] = counts.get(domain, 0) + 1
            result.append(c)
    return result


def _bm25_rank(
    chunks: list[CorpusChunk],
    query: str,
) -> list[tuple[float, CorpusChunk]]:
    """Simple BM25 ranking (k1=1.5, b=0.75). Returns (score, chunk) pairs."""
    import math
    from collections import Counter

    if not chunks:
        return []
    query_terms = query.lower().split()
    texts = [c.text.lower() for c in chunks]
    N = len(texts)
    total_len = sum(len(t.split()) for t in texts)
    avg_dl = total_len / max(1, N)
    k1, b = 1.5, 0.75

    df: dict[str, int] = {}
    for text in texts:
        for term in set(query_terms):
            if term in text:
                df[term] = df.get(term, 0) + 1

    results = []
    for chunk, text in zip(chunks, texts):
        words = text.split()
        dl = len(words)
        tf_dict = Counter(words)
        score = 0.0
        for term in query_terms:
            tf = tf_dict.get(term, 0)
            idf = math.log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
            tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(1, avg_dl)))
            score += idf * tf_norm
        results.append((score, chunk))
    return results


def apply_rrf(
    ranked_lists: list[list[CorpusChunk]],
    k: int = RRF_K,
) -> list[CorpusChunk]:
    """Reciprocal Rank Fusion of multiple ranked lists of chunks (Fix S)."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, CorpusChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_map[cid] = chunk
    return [chunk_map[cid] for cid, _ in sorted(scores.items(),
                                                  key=lambda x: x[1], reverse=True)]


def apply_mmr(
    chunks: list[CorpusChunk],
    query: str,
    lam: float = MMR_LAMBDA,
) -> list[CorpusChunk]:
    """Reorder chunks via Maximal Marginal Relevance (Fix S).

    Score(c, selected) = lam * sim(c, query) - (1-lam) * max_sim(c, selected)
    Falls back to original order when no embedder is available.
    """
    if len(chunks) <= 1:
        return chunks
    embedder = _get_embedder()
    if embedder is None:
        return chunks
    try:
        import numpy as np
        q_emb = embedder.encode(query, normalize_embeddings=True)
        texts = [c.text[:500] for c in chunks]
        c_embs = embedder.encode(texts, normalize_embeddings=True)
        relevance = (c_embs @ q_emb).tolist()

        selected: list[int] = []
        remaining = list(range(len(chunks)))

        while remaining:
            if not selected:
                best = max(remaining, key=lambda i: relevance[i])
            else:
                sel_embs = np.array([c_embs[j] for j in selected])
                best = max(
                    remaining,
                    key=lambda i: lam * relevance[i] - (1 - lam) * float(
                        np.max(c_embs[i] @ sel_embs.T)
                    ),
                )
            selected.append(best)
            remaining.remove(best)

        return [chunks[i] for i in selected]
    except Exception:
        return chunks


# Fact-density scoring: digit tokens, percentages, currency, year-like
# numbers per 100 words. The downstream pipeline (particulars gate in
# DEVELOP/CHAIN, verification calibration) runs on fact-dense evidence;
# pure topicality ranking starves it.
_FACT_TOKEN_RE = re.compile(r"\b\d[\d,.]*%?\b|\$\d|€\d|£\d")


def _fact_density(text: str) -> float:
    """Fact tokens per 100 words, clamped to [0, 1]."""
    words = text.split()
    if not words:
        return 0.0
    n_facts = len(_FACT_TOKEN_RE.findall(text))
    return min(1.0, (n_facts * 100.0 / len(words)) / 10.0)


# Canonical developer sources for coding-task domain priors.
_CODING_DOMAINS = (
    "stackoverflow.com", "stackexchange.com", "github.com",
    "docs.python.org", "readthedocs.io", "developer.mozilla.org",
    "pypi.org", "docs.rs", "go.dev", "learn.microsoft.com",
)


def _domain_boost(chunk: CorpusChunk, task_type: Optional[str]) -> float:
    if task_type != "coding":
        return 0.0
    url = _url_of(chunk).lower()
    return 1.0 if any(d in url for d in _CODING_DOMAINS) else 0.0


def _quality_rerank(chunks: list[CorpusChunk],
                    task_type: Optional[str]) -> list[CorpusChunk]:
    """Stable re-sort by additive quality priors (fact density + coding
    domain). Applied after RRF (topical order preserved on ties via rank
    position) and before MMR."""
    from . import config as _cfg
    w_fact = getattr(_cfg, "SEARCH_FACT_DENSITY_WEIGHT", 0.0)
    w_dom = getattr(_cfg, "SEARCH_CODING_DOMAIN_BOOST", 0.0)
    if w_fact <= 0 and w_dom <= 0:
        return chunks
    # Fixed per-rank step: each incoming rank position costs 0.05, so the
    # default boosts can lift a chunk a bounded number of positions
    # (fact 0.15 → up to 3; coding domain 0.25 → up to 5) regardless of
    # list length. A proportional base would swamp boosts at small n.
    rank_step = 0.05
    scored = []
    for rank, c in enumerate(chunks):
        score = (-rank * rank_step
                 + w_fact * _fact_density(c.text)
                 + w_dom * _domain_boost(c, task_type))
        scored.append((score, rank, c))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored]


def _diversify(
    chunks: list[CorpusChunk],
    query: str,
    max_results: int,
    task_type: Optional[str] = None,
    enrich_pages: bool = False,
) -> list[CorpusChunk]:
    """Apply source cap + optional RRF + quality priors + MMR reranking.

    Called after backend search returns raw results. When `enrich_pages` is
    set (web/DDG snippets), full page content is fetched for the FINAL ranked
    survivors only — so we never pay to fetch pages that relevance-filtering or
    MMR then discards. Backends that already return full text (Tavily,
    Wikipedia) pass enrich_pages=False to avoid wasteful re-fetches.
    """
    chunks = apply_source_cap(chunks, MAX_CHUNKS_PER_SOURCE)
    chunks = _relevance_filter(chunks, query)   # drop off-topic noise before reranking

    if HYBRID_RETRIEVAL and len(chunks) > 1:
        bm25_pairs = _bm25_rank(chunks, query)
        bm25_ranked = [c for _, c in sorted(bm25_pairs, key=lambda x: x[0], reverse=True)]
        embedder = _get_embedder()
        if embedder is not None:
            try:
                import numpy as np
                q_emb = embedder.encode(query, normalize_embeddings=True)
                c_embs = embedder.encode([c.text[:500] for c in chunks],
                                         normalize_embeddings=True)
                dense_scores = (c_embs @ q_emb).tolist()
                dense_ranked = [c for _, c in sorted(
                    zip(dense_scores, chunks), key=lambda x: x[0], reverse=True
                )]
                chunks = apply_rrf([bm25_ranked, dense_ranked], k=RRF_K)
            except Exception:
                chunks = bm25_ranked
        else:
            chunks = bm25_ranked

    chunks = _quality_rerank(chunks, task_type)
    chunks = apply_mmr(chunks[:max_results * 2], query, lam=MMR_LAMBDA)
    chunks = chunks[:max_results]
    if enrich_pages:
        # Fetch full page content for the FINAL survivors only (top
        # SEARCH_FETCH_TOP_K of these). Previously this ran on the raw DDG
        # top-3 inside _ddg_search, wasting fetches on chunks dropped here.
        chunks = _enrich_with_pages(chunks)
    return chunks


# ---------------------------------------------------------------------------
# Search QUALITY: page-content fetch + relevance gate (the garbage-in fix)
# ---------------------------------------------------------------------------

def _url_of(chunk: CorpusChunk) -> str:
    """Recover the result URL from a chunk's source_tag ('title | url')."""
    tag = getattr(chunk, "source_tag", "") or ""
    return (tag.rsplit("|", 1)[-1] if "|" in tag else tag).strip()


def _fetch_page_text(url: str, timeout: float) -> str:
    """Fetch a URL and extract its main readable text (requests + BeautifulSoup).

    Returns '' on ANY failure so the caller keeps the snippet (can't regress).
    Strips script/style/nav/header/footer/aside; prefers <article>/<main>; keeps
    substantial <p>s; collapses whitespace; caps at _MAX_CONTENT_CHARS.
    """
    if not url:
        return ""
    if not url.lower().startswith(("http://", "https://")):
        url = "https://" + url
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return ""
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (compatible; SwarmBot/1.0)"})
        if resp.status_code != 200 or "html" not in resp.headers.get("content-type", "").lower():
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "noscript", "figure"]):
            tag.decompose()
        root = soup.find("article") or soup.find("main") or soup.body or soup
        paras = [p.get_text(" ", strip=True) for p in root.find_all("p")]
        text = " ".join(p for p in paras if len(p) > 40)
        if len(text) < 200:                       # sparse <p> markup — take all text
            text = root.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", text).strip()[:_MAX_CONTENT_CHARS]
    except Exception:
        return ""


def _enrich_with_pages(chunks: list[CorpusChunk]) -> list[CorpusChunk]:
    """Replace the top-K chunks' DDG snippet with fetched page content (parallel,
    budgeted). Snippet is kept wherever the fetch fails or returns too little."""
    from . import config as _cfg
    if not _cfg.SEARCH_FETCH_PAGES or not chunks:
        return chunks
    from concurrent.futures import ThreadPoolExecutor
    targets = chunks[:max(1, _cfg.SEARCH_FETCH_TOP_K)]
    timeout = _cfg.SEARCH_FETCH_TIMEOUT_S
    _fetch_t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=min(4, len(targets))) as ex:
            texts = list(ex.map(lambda c: _fetch_page_text_cached(_url_of(c), timeout), targets))
    except Exception:
        return chunks
    finally:
        with _SEARCH_STATS_LOCK:
            _SEARCH_STATS["fetch_calls"] += len(targets)
            _SEARCH_STATS["fetch_total_s"] += time.perf_counter() - _fetch_t0
    n = 0
    for c, t in zip(targets, texts):
        if t and len(t) >= _cfg.SEARCH_FETCH_MIN_CHARS and len(t) > len(c.text):
            c.text = t
            n += 1
    if n:
        print(f"[search] enriched {n}/{len(targets)} chunks with full page content")
    return chunks


def _relevance_filter(chunks: list[CorpusChunk], query: str) -> list[CorpusChunk]:
    """Drop chunks whose cosine sim to `query` is below the relevance gate
    (off-topic noise the reranker only reordered). Always keeps the 2 most
    relevant so a strict threshold never empties the result. No-op without an
    embedder or when the gate is disabled (<=0).

    The effective threshold is `max(SEARCH_RELEVANCE_MIN, top_sim - REL_MARGIN)`
    when SEARCH_RELEVANCE_REL_MARGIN > 0 (adaptive: prune results clearly worse
    than the best available), else the flat SEARCH_RELEVANCE_MIN floor. The
    adaptive term is default-off (margin 0.0) so default behaviour is unchanged.
    """
    from . import config as _cfg
    min_sim = _cfg.SEARCH_RELEVANCE_MIN
    if min_sim <= 0 or len(chunks) <= 2:
        return chunks
    embedder = _get_embedder()
    if embedder is None:
        return chunks
    try:
        q = embedder.encode(query, normalize_embeddings=True)
        cs = embedder.encode([c.text[:500] for c in chunks], normalize_embeddings=True)
        sims = (cs @ q).tolist()
    except Exception:
        return chunks
    ranked = sorted(zip(sims, chunks), key=lambda x: x[0], reverse=True)
    threshold = min_sim
    rel_margin = getattr(_cfg, "SEARCH_RELEVANCE_REL_MARGIN", 0.0)
    if rel_margin > 0 and ranked:
        top_sim = ranked[0][0]
        threshold = max(min_sim, top_sim - rel_margin)
    kept = [c for s, c in ranked if s >= threshold] or []
    if len(kept) < 2:                             # never drop below the 2 best
        kept = [c for _, c in ranked[:2]]
    if len(kept) < len(chunks):
        print(f"[search] relevance gate: kept {len(kept)}/{len(chunks)} (>= {threshold:.2f})")
    return kept


def _build_followup_query(query: str, task_type: Optional[str] = None) -> str:
    if not query:
        return ""
    modifiers = _FOLLOWUP_MODIFIERS_BY_TASK.get(task_type or "", _FOLLOWUP_DEFAULT)
    if not modifiers:
        return ""
    # Rotate the starting modifier by a hash of the query so different queries
    # get different follow-up angles, instead of every query getting the first
    # modifier (the "+ research" append) — that produced near-duplicate
    # follow-up searches on the same source pool (high self-BLEU symptom).
    start = int(hashlib.sha1(query.lower().encode("utf-8")).hexdigest(), 16) % len(modifiers)
    for k in range(len(modifiers)):
        modifier = modifiers[(start + k) % len(modifiers)]
        candidate = f"{query} {modifier}".strip()
        if candidate.lower() != query.lower():
            return candidate
    return ""


def _load_cache(query: str, max_results: int) -> Optional[list[CorpusChunk]]:
    p = _cache_path(query, max_results)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [
            CorpusChunk(
                chunk_id=d["chunk_id"],
                text=d["text"],
                source_tag=d.get("source_tag", ""),
            )
            for d in data
        ]
    except Exception:
        return None


def _save_cache(query: str, chunks: list[CorpusChunk], max_results: int) -> None:
    if not chunks:
        return
    p = _cache_path(query, max_results)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                [
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "source_tag": c.source_tag,
                    }
                    for c in chunks
                ],
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _tavily_search(query: str, max_results: int) -> list[CorpusChunk]:
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []
    try:
        from tavily import TavilyClient  # type: ignore
    except ImportError:
        return []
    try:
        client = TavilyClient(api_key=api_key)
        resp = client.search(query=query, max_results=max_results,
                             search_depth="basic")
    except Exception as exc:
        print(f"[search] tavily call failed: {type(exc).__name__}: {exc}")
        return []
    out: list[CorpusChunk] = []
    for i, r in enumerate(resp.get("results", []) or []):
        url = str(r.get("url", "") or "")
        title = str(r.get("title", "") or url)
        content = str(r.get("content", "") or "")[:_MAX_CONTENT_CHARS]
        if not content:
            continue
        cid = f"tavily_{i}_{hashlib.sha1(url.encode()).hexdigest()[:8]}"
        out.append(CorpusChunk(
            chunk_id=cid,
            text=content,
            source_tag=f"{title} | {url}",
        ))
    return out


def _ddg_search(query: str, max_results: int) -> list[CorpusChunk]:
    # `ddgs` is the maintained successor of `duckduckgo-search`. Try the new
    # package first, then fall back to the legacy name for environments that
    # haven't reinstalled yet. Same DDGS class shape.
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            return []
    raw: list[CorpusChunk] = []
    try:
        with DDGS(timeout=10) as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=max_results)):
                if i >= max_results:
                    break
                url = str(r.get("href") or r.get("url") or "")
                title = str(r.get("title", "") or url)
                body = str(r.get("body", "") or "")[:_MAX_CONTENT_CHARS]
                if not body:
                    continue
                cid = f"ddg_{i}_{hashlib.sha1(url.encode()).hexdigest()[:8]}"
                raw.append(CorpusChunk(
                    chunk_id=cid,
                    text=body,
                    source_tag=f"{title} | {url}",
                ))
    except Exception as exc:
        print(f"[search] ddg call failed: {type(exc).__name__}: {exc}")
        return []
    # Page enrichment is deferred to _diversify(enrich_pages=True) so we only
    # fetch full pages for the final ranked survivors, not the raw DDG top-K.
    return _dedupe_chunks(raw, max_results)


def _cohere_search(query: str, max_results: int) -> list[CorpusChunk]:
    # Cohere wiki-simple is opt-in only: it triggers a ~1 GB download on first
    # use and re-introduces the topic-skew the agentic search is meant to
    # escape. Enable explicitly with SWARM_SEARCH_USE_COHERE=1.
    if os.environ.get("SWARM_SEARCH_USE_COHERE", "").strip() in ("", "0", "false", "False"):
        return []
    try:
        from .corpus_store_cohere import get_store
    except ImportError:
        return []
    try:
        store = get_store()
        chunks = store.search(query, n_chunks=max_results)
    except Exception:
        return []
    return list(chunks[:max_results])


# ---------------------------------------------------------------------------
# Selection / startup banner
# ---------------------------------------------------------------------------

_PRIMARY_LOGGED = False


def _stackexchange_search(query: str, max_results: int) -> list[CorpusChunk]:
    """Stack Exchange API search (free, anonymous, no key; ~300 req/day/IP).

    Coding-task backend: question titles + answer-bearing bodies from
    Stack Overflow are far denser evidence than blogspam for
    implementation/pitfall queries. Fail-soft: returns [] on any error.
    """
    try:
        import requests
    except ImportError:
        return []
    try:
        resp = requests.get(
            "https://api.stackexchange.com/2.3/search/advanced",
            params={
                "order": "desc", "sort": "relevance", "q": query[:200],
                "site": "stackoverflow", "pagesize": min(max_results, 5),
                "filter": "withbody", "answers": 1,
            },
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SwarmBot/1.0)"},
        )
        if resp.status_code != 200:
            return []
        items = resp.json().get("items", [])
    except Exception as exc:
        print(f"[search] stackexchange failed: {type(exc).__name__}: {exc}")
        return []
    out: list[CorpusChunk] = []
    for i, it in enumerate(items):
        title = (it.get("title") or "").strip()
        body_html = it.get("body") or ""
        body = re.sub(r"<[^>]+>", " ", body_html)
        body = re.sub(r"\s+", " ", body).strip()[:1500]
        link = it.get("link") or ""
        if not (title and body):
            continue
        cid = f"se_{i}_{hashlib.sha1((link or title).encode()).hexdigest()[:8]}"
        out.append(CorpusChunk(
            chunk_id=cid,
            text=f"{title}. {body}",
            source_tag=f"{title[:120]} | {link}",
        ))
    if out:
        print(f"[search] stackexchange ok: {len(out)} results for {query!r}")
    return out


def _scholar_enabled() -> bool:
    """Semantic Scholar aux backend toggle (default ON; SWARM_SEARCH_SCHOLAR=0
    to disable)."""
    return os.environ.get("SWARM_SEARCH_SCHOLAR", "1").strip() \
        not in ("0", "false", "False")


def _semanticscholar_search(query: str, max_results: int) -> list[CorpusChunk]:
    """Semantic Scholar paper search (free, anonymous, no key).

    Aux backend for analysis/debate tasks — the same role Stack Exchange plays
    for coding. Paper abstracts are the densest free source of the quantified,
    citable particulars the swarm needs to out-evidence a flagship's parametric
    memory (DDG snippets are 300 chars of SEO text; abstracts carry effect
    sizes, sample counts, and named findings). Unauthenticated rate limits are
    shared-pool and 429s are common — short timeout, fail-soft [].
    """
    try:
        import requests
    except ImportError:
        return []
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query[:200],
                "limit": min(max_results, 5),
                "fields": "title,abstract,url,year,citationCount",
            },
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SwarmBot/1.0)"},
        )
        if resp.status_code != 200:
            return []
        items = resp.json().get("data", []) or []
    except Exception as exc:
        print(f"[search] semanticscholar failed: {type(exc).__name__}: {exc}")
        return []
    out: list[CorpusChunk] = []
    for i, it in enumerate(items):
        title = (it.get("title") or "").strip()
        abstract = re.sub(r"\s+", " ", (it.get("abstract") or "")).strip()
        if not (title and abstract):
            continue
        year = it.get("year") or ""
        url = it.get("url") or ""
        cid = f"s2_{i}_{hashlib.sha1((url or title).encode()).hexdigest()[:8]}"
        out.append(CorpusChunk(
            chunk_id=cid,
            text=f"{title} ({year}). {abstract[:1500]}",
            source_tag=f"{title[:120]} | {url}",
        ))
    if out:
        print(f"[search] semanticscholar ok: {len(out)} results for {query!r}")
    return out


def _wikipedia_search(query: str, max_results: int) -> list[CorpusChunk]:
    """Wikipedia summary search — the free, key-free fallback backend.

    In practice Tavily/Cohere are unconfigured, making DDG the only live
    backend; when DDG rate-limits, the run silently starves. Wikipedia's
    API is reliable and fact-dense (dates, figures, named entities).
    Fail-soft: returns [] on any error.
    """
    try:
        import wikipedia
    except ImportError:
        return []
    out: list[CorpusChunk] = []
    try:
        titles = wikipedia.search(query, results=min(max_results, 4))
        for i, title in enumerate(titles):
            try:
                summary = wikipedia.summary(title, sentences=5,
                                            auto_suggest=False)
            except Exception:
                continue
            if not summary or len(summary) < 80:
                continue
            cid = f"wiki_{i}_{hashlib.sha1(title.encode()).hexdigest()[:8]}"
            out.append(CorpusChunk(
                chunk_id=cid,
                text=summary[:1500],
                source_tag=f"{title} | https://en.wikipedia.org/wiki/"
                           f"{title.replace(' ', '_')}",
            ))
    except Exception as exc:
        print(f"[search] wikipedia failed: {type(exc).__name__}: {exc}")
        return out
    if out:
        print(f"[search] wikipedia ok: {len(out)} results for {query!r}")
    return out


def _ddg_available() -> bool:
    """Return True if either the ddgs or duckduckgo_search package is present."""
    try:
        from ddgs import DDGS  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS  # noqa: F401
        return True
    except ImportError:
        return False


def _log_primary_once() -> None:
    global _PRIMARY_LOGGED
    if _PRIMARY_LOGGED:
        return
    _PRIMARY_LOGGED = True
    if os.environ.get("TAVILY_API_KEY", "").strip():
        print("[search] tavily key found; primary=tavily")
    elif _ddg_available():
        print("[search] no TAVILY_API_KEY; primary=duckduckgo (fallback)")
    else:
        import warnings
        warnings.warn(
            "[search] WARNING: no web search backend available — "
            "neither TAVILY_API_KEY is set nor ddgs/duckduckgo_search is installed. "
            "Scouts will receive zero evidence and deposit ungrounded claims. "
            "Install: pip install ddgs   or set TAVILY_API_KEY.",
            RuntimeWarning,
            stacklevel=3,
        )
        print("[search] *** NO SEARCH BACKEND *** scouts will get empty evidence")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(query: str, max_results: int = _DEFAULT_MAX_RESULTS,
           task_type: Optional[str] = None) -> list[CorpusChunk]:
    """Run an agentic search (timed wrapper around `_search_impl`).

    Records wall time + result counts into `_SEARCH_STATS` so summary.json can
    report retrieval's share of the run. MOCK runs are not timed (they short-
    circuit instantly and would pollute real-run comparisons).

    `SWARM_DISABLE_LIVE_SEARCH=1` short-circuits to an empty result set before
    any backend is touched. Used by the `--corpus=pack:<path>` over-context
    eval mode (eval/packs.py, run_swarm.py assemble_partitions()) so the
    pre-built evidence pack is the ONLY evidence a run sees — per-action live
    web search would otherwise leak fresh evidence into the pack condition
    and make condition A vs condition F (single-call RAG over the same pack)
    an unfair comparison.
    """
    if os.environ.get("SWARM_DISABLE_LIVE_SEARCH", "").strip() not in ("", "0", "false", "False"):
        return []
    if os.environ.get("MOCK_LLM", "").strip() not in ("", "0", "false", "False"):
        return _search_impl(query, max_results, task_type)
    t0 = time.perf_counter()
    out: list[CorpusChunk] = []
    try:
        out = _search_impl(query, max_results, task_type)
        return out
    finally:
        dt = time.perf_counter() - t0
        with _SEARCH_STATS_LOCK:
            _SEARCH_STATS["calls"] += 1
            _SEARCH_STATS["total_s"] += dt
            if dt > _SEARCH_STATS["max_s"]:
                _SEARCH_STATS["max_s"] = dt
            if not out:
                _SEARCH_STATS["empty_results"] += 1


def _search_impl(query: str, max_results: int = _DEFAULT_MAX_RESULTS,
                 task_type: Optional[str] = None) -> list[CorpusChunk]:
    """Run an agentic search. Tavily → DDG → Cohere; cached by SHA(query).

    Returns up to `max_results` CorpusChunks; an empty list means every
    backend failed or returned nothing.
    """
    _log_primary_once()
    query = _sanitize_query(query)
    if not query:
        return []

    # MOCK_LLM short-circuit: yield deterministic placeholder chunks so
    # smoke tests don't pay 1 GB cohere-store downloads or hit live DDG
    # rate-limits. Pure plumbing path.
    if os.environ.get("MOCK_LLM", "").strip() not in ("", "0", "false", "False"):
        digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
        return [
            CorpusChunk(
                chunk_id=f"mock_{i}_{digest}",
                text=(f"Mock evidence chunk {i} for query {query!r}. "
                      f"This is placeholder content for offline plumbing tests."),
                source_tag=f"mock://search/{digest}/{i}",
            )
            for i in range(min(max_results, 3))
        ]

    cached = _load_cache(query, max_results)
    if cached is not None:
        return cached[:max_results]

    started = time.time()

    # Task-matched aux backend runs ALONGSIDE the web backend and merges.
    # Coding: Stack Exchange — SO answers are the densest implementation/
    # pitfall evidence available for free (the coding domain prior in
    # _quality_rerank then keeps them ahead of blogspam). Analysis/debate:
    # Semantic Scholar — paper abstracts carry the quantified particulars
    # (effect sizes, sample counts, named findings) that DDG snippets don't.
    aux_chunks: list[CorpusChunk] = []
    if task_type == "coding":
        aux_chunks = _stackexchange_search(query, max_results)
    elif task_type in ("analysis", "debate") and _scholar_enabled():
        aux_chunks = _semanticscholar_search(query, max_results)

    chunks = _tavily_search(query, max_results)
    if chunks:
        chunks = _dedupe_chunks(aux_chunks + chunks, max_results * 2)
        chunks = _diversify(chunks, query, max_results, task_type)
        _save_cache(query, chunks, max_results)
        print(f"[search] tavily ok: {len(chunks)} results for {query!r} "
              f"in {time.time() - started:.1f}s")
        return chunks

    chunks = _ddg_search(query, max_results)
    if chunks and len(chunks) >= _MIN_FOLLOWUP_RESULTS:
        chunks = _dedupe_chunks(aux_chunks + chunks, max_results * 2)
        chunks = _diversify(chunks, query, max_results, task_type, enrich_pages=True)
        _save_cache(query, chunks, max_results)
        print(f"[search] ddg ok: {len(chunks)} results for {query!r} "
              f"in {time.time() - started:.1f}s")
        return chunks

    if chunks or aux_chunks:
        followup = _build_followup_query(query, task_type)
        if followup:
            followup_chunks = _ddg_search(followup, max_results)
            if followup_chunks:
                merged = _dedupe_chunks(
                    aux_chunks + chunks + followup_chunks, max_results)
                merged = _diversify(merged, query, max_results, task_type, enrich_pages=True)
                _save_cache(query, merged, max_results)
                print(f"[search] ddg follow-up ok: {len(merged)} results for {query!r} "
                      f"(+{len(followup_chunks)} from follow-up {followup!r}) "
                      f"in {time.time() - started:.1f}s")
                return merged
        merged = _dedupe_chunks(aux_chunks + chunks, max_results)
        merged = _diversify(merged, query, max_results, task_type, enrich_pages=True)
        _save_cache(query, merged, max_results)
        print(f"[search] ddg ok (small batch): {len(merged)} results for {query!r} "
              f"in {time.time() - started:.1f}s")
        return merged

    # Wikipedia: the reliable key-free fallback. In practice Tavily/Cohere
    # are unconfigured, so when DDG rate-limits the run used to starve.
    chunks = _wikipedia_search(query, max_results)
    if chunks:
        chunks = _diversify(chunks, query, max_results, task_type)
        _save_cache(query, chunks, max_results)
        print(f"[search] wikipedia fallback: {len(chunks)} results for {query!r}")
        return chunks

    chunks = _cohere_search(query, max_results)
    if chunks:
        # Don't persist cohere fallback; it's bound to the same wiki-simple
        # bias the agentic search is trying to escape, and we don't want it
        # pinned in the cache.
        chunks = _diversify(chunks, query, max_results, task_type)
        print(f"[search] cohere fallback: {len(chunks)} results for {query!r}")
        return chunks

    # If the first query returned nothing, try a follow-up modifier to avoid
    # missing the same topic due to overly generic wording.
    followup = _build_followup_query(query, task_type)
    if followup:
        chunks = _ddg_search(followup, max_results)
        if chunks:
            # DDG snippets — enrich the final survivors (same as the other DDG
            # paths). Was implicitly enriched inside _ddg_search before the
            # enrichment move; without enrich_pages here this fallback would
            # hand agents 300-char snippets instead of full page content.
            chunks = _diversify(chunks, query, max_results, task_type, enrich_pages=True)
            _save_cache(query, chunks, max_results)
            print(f"[search] ddg follow-up only ok: {len(chunks)} results for {query!r} "
                  f"via follow-up {followup!r} in {time.time() - started:.1f}s")
            return chunks

    # Last rung: relax the query to its content terms and retry DDG, then
    # Wikipedia. Over-specific phrasings (planned atom queries, scout claim
    # fragments) routinely match zero pages while their keywords match
    # thousands; a keyword retry is nearly free compared to returning an
    # empty corpus slot. Local import: query_planner pulls in signal_store,
    # which search_tool must not require at module load.
    from .query_planner import relax_query
    relaxed = relax_query(query)
    if relaxed:
        chunks = _ddg_search(relaxed, max_results)
        if chunks:
            chunks = _diversify(chunks, query, max_results, task_type,
                                enrich_pages=True)
        else:
            chunks = _wikipedia_search(relaxed, max_results)
            if chunks:
                chunks = _diversify(chunks, query, max_results, task_type)
        if chunks:
            _save_cache(query, chunks, max_results)
            print(f"[search] relaxed-query rescue: {len(chunks)} results for "
                  f"{query!r} via {relaxed!r} in {time.time() - started:.1f}s")
            return chunks

    print(
        f"[RETRIEVAL] *** ALL BACKENDS FAILED *** for query {query!r}. "
        f"0 real chunks returned — this corpus slot will be empty."
    )
    return []


def summarize_for_signal(query: str, chunks: list[CorpusChunk],
                         max_chars: int = 800) -> str:
    """Render query + top-3 source titles/URLs for a SEARCH signal deposit.

    Other agents see this as a stigmergic trace: "this query has already been
    explored, and these were the top sources." It is not the evidence itself —
    scouts/foragers pull retrieved chunk text directly for their own deposits.
    """
    if not chunks:
        return f"QUERY: {query}\n(no results)"
    lines = [f"QUERY: {query}"]
    for c in chunks[:3]:
        lines.append(f"  - {c.source_tag[:200]}")
    # One content excerpt from the top result: makes the SEARCH trace
    # substantively informative (other workers' query planning mines these
    # traces; bare URLs carried no facts). Environment content, not agent
    # reasoning — no-leak safe.
    top_text = (chunks[0].text or "").strip()
    if top_text:
        lines.append(f"  TOP: {top_text[:220]}")
    out = "\n".join(lines)
    return out[:max_chars]
