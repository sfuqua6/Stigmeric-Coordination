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
    """Lazy singleton: reuse the same model as SignalStore."""
    global _EMBEDDER_CACHE
    if _EMBEDDER_CACHE is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER_CACHE = SentenceTransformer("all-MiniLM-L6-v2")
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


def _diversify(
    chunks: list[CorpusChunk],
    query: str,
    max_results: int,
) -> list[CorpusChunk]:
    """Apply source cap + optional RRF + MMR reranking (Fix S).

    Called after backend search returns raw results.
    """
    chunks = apply_source_cap(chunks, MAX_CHUNKS_PER_SOURCE)

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

    chunks = apply_mmr(chunks[:max_results * 2], query, lam=MMR_LAMBDA)
    return chunks[:max_results]


def _build_followup_query(query: str, task_type: Optional[str] = None) -> str:
    if not query:
        return ""
    modifiers = _FOLLOWUP_MODIFIERS_BY_TASK.get(task_type or "", _FOLLOWUP_DEFAULT)
    for modifier in modifiers:
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
    out = _dedupe_chunks(raw, max_results)
    return out


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


def _log_primary_once() -> None:
    global _PRIMARY_LOGGED
    if _PRIMARY_LOGGED:
        return
    _PRIMARY_LOGGED = True
    if os.environ.get("TAVILY_API_KEY", "").strip():
        print("[search] tavily key found; primary=tavily")
    else:
        # Probe for DDG package import. Tavily is unavailable so this is the
        # primary path.
        try:
            from duckduckgo_search import DDGS  # noqa: F401
            print("[search] no TAVILY_API_KEY; primary=duckduckgo (fallback)")
        except ImportError:
            print("[search] no TAVILY_API_KEY and duckduckgo_search not "
                  "installed; falling back to cohere store only")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(query: str, max_results: int = _DEFAULT_MAX_RESULTS,
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
    chunks = _tavily_search(query, max_results)
    if chunks:
        chunks = _diversify(chunks, query, max_results)
        _save_cache(query, chunks, max_results)
        print(f"[search] tavily ok: {len(chunks)} results for {query!r} "
              f"in {time.time() - started:.1f}s")
        return chunks

    # OLD: direct fallback chain without any secondary search query.
    # chunks = _ddg_search(query, max_results)
    # if chunks: ...
    # chunks = _cohere_search(query, max_results)
    # return []

    chunks = _ddg_search(query, max_results)
    if chunks and len(chunks) >= _MIN_FOLLOWUP_RESULTS:
        chunks = _diversify(chunks, query, max_results)
        _save_cache(query, chunks, max_results)
        print(f"[search] ddg ok: {len(chunks)} results for {query!r} "
              f"in {time.time() - started:.1f}s")
        return chunks

    if chunks:
        followup = _build_followup_query(query, task_type)
        if followup:
            followup_chunks = _ddg_search(followup, max_results)
            if followup_chunks:
                merged = _dedupe_chunks(chunks + followup_chunks, max_results)
                merged = _diversify(merged, query, max_results)
                _save_cache(query, merged, max_results)
                print(f"[search] ddg follow-up ok: {len(merged)} results for {query!r} "
                      f"(+{len(followup_chunks)} from follow-up {followup!r}) "
                      f"in {time.time() - started:.1f}s")
                return merged
        chunks = _diversify(chunks, query, max_results)
        _save_cache(query, chunks, max_results)
        print(f"[search] ddg ok (small batch): {len(chunks)} results for {query!r} "
              f"in {time.time() - started:.1f}s")
        return chunks

    chunks = _cohere_search(query, max_results)
    if chunks:
        # Don't persist cohere fallback; it's bound to the same wiki-simple
        # bias the agentic search is trying to escape, and we don't want it
        # pinned in the cache.
        chunks = _diversify(chunks, query, max_results)
        print(f"[search] cohere fallback: {len(chunks)} results for {query!r}")
        return chunks

    # If the first query returned nothing, try a follow-up modifier to avoid
    # missing the same topic due to overly generic wording.
    followup = _build_followup_query(query, task_type)
    if followup:
        chunks = _ddg_search(followup, max_results)
        if chunks:
            chunks = _diversify(chunks, query, max_results)
            _save_cache(query, chunks, max_results)
            print(f"[search] ddg follow-up only ok: {len(chunks)} results for {query!r} "
                  f"via follow-up {followup!r} in {time.time() - started:.1f}s")
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
    out = "\n".join(lines)
    return out[:max_chars]
