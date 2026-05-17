"""Pre-computed Cohere embeddings of Wikipedia as a retrieval corpus.

Loads `Cohere/wikipedia-22-12-simple-embeddings` (~1 GB, ~250k articles)
on first call, builds an in-memory FAISS index, and persists the index
plus the metadata DataFrame to `corpora/` (Drive on Colab, ./corpora/
locally). Subsequent runs reload from disk in <10s.

Why pre-computed embeddings: embedding 250k articles ourselves would
take ~15 min even on a T4. Cohere has already done it. We pay only the
download (~1 GB once) and a per-query embedding call (~10ms).

API key: requires COHERE_API_KEY for query embedding. Free tier is ~100
queries/min which is more than enough for a swarm run. The self-contained
BGE alternative (sentence-transformers/BAAI/bge-small-en-v1.5 on both
sides) is deferred — when it lands, this module gets an optional fallback
path in `_embed_query`.

Public surface:
    CohereWikiSimpleStore.search(query, n_chunks=20) -> list[CorpusChunk]

Wired into core/retrieval.py as CohereCorpusRetriever, the first item
in the CompositeRetriever fallback chain.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Optional

from .intake import CorpusChunk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DATASET_ID = "Cohere/wikipedia-22-12-simple-embeddings"
_COHERE_EMBED_MODEL = "embed-multilingual-v2.0"  # required for this dataset
_INDEX_FILENAME = "wiki_simple_faiss.index"
_META_FILENAME = "wiki_simple_meta.parquet"


def _corpora_dir() -> Path:
    """Resolve the corpora cache directory.

    Order: SWARM_CORPORA_DIR env var → Colab Drive standard location →
    local ./corpora/. Drive auto-detection looks for /content/drive/
    (created by drive.mount in the Colab setup notebook).
    """
    explicit = os.environ.get("SWARM_CORPORA_DIR")
    if explicit:
        return Path(explicit)
    drive_path = Path("/content/drive/MyDrive/swarm/corpora")
    if Path("/content/drive").exists():
        return drive_path
    return Path("corpora")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# The FAISS index + metadata DataFrame are ~1 GB resident. Re-loading per
# query would defeat the point. Hold one shared instance across all
# retrieve() calls within a process.

_STORE: Optional["CohereWikiSimpleStore"] = None


def get_store() -> "CohereWikiSimpleStore":
    global _STORE
    if _STORE is None:
        _STORE = CohereWikiSimpleStore()
    return _STORE


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


class CohereWikiSimpleStore:
    """In-memory FAISS over Cohere's pre-computed Simple-Wikipedia embeddings."""

    def __init__(self) -> None:
        self._index = None       # faiss.Index
        self._meta = None        # pandas.DataFrame [title, text, url]
        self._cohere_client = None
        self._corpora_dir = _corpora_dir()
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy init: download + build, or load from disk
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        index_path = self._corpora_dir / _INDEX_FILENAME
        meta_path = self._corpora_dir / _META_FILENAME

        if index_path.exists() and meta_path.exists():
            self._load_from_disk(index_path, meta_path)
        else:
            self._build_from_scratch(index_path, meta_path)
        self._loaded = True

    def _load_from_disk(self, index_path: Path, meta_path: Path) -> None:
        import faiss  # type: ignore
        import pandas as pd  # type: ignore

        t0 = time.time()
        self._index = faiss.read_index(str(index_path))
        self._meta = pd.read_parquet(meta_path)
        print(
            f"[cohere-store] loaded from disk: "
            f"{self._index.ntotal:,} vectors, {len(self._meta):,} rows, "
            f"{time.time() - t0:.1f}s ({index_path.parent})"
        )

    def _build_from_scratch(self, index_path: Path, meta_path: Path) -> None:
        from datasets import load_dataset  # type: ignore
        import faiss  # type: ignore
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore

        print(
            f"[cohere-store] first run — downloading {_DATASET_ID} "
            f"(~1 GB, ~3 min on Colab)..."
        )
        t0 = time.time()
        ds = load_dataset(_DATASET_ID, split="train")
        print(
            f"[cohere-store] downloaded {len(ds):,} rows in "
            f"{time.time() - t0:.1f}s"
        )

        t1 = time.time()
        embs = np.asarray(ds["emb"], dtype="float32")
        # Normalize so inner-product == cosine similarity.
        faiss.normalize_L2(embs)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        print(
            f"[cohere-store] built FAISS IndexFlatIP "
            f"({embs.shape[0]:,} × {embs.shape[1]}) in "
            f"{time.time() - t1:.1f}s"
        )

        self._index = index
        # Persist only the columns we render into CorpusChunk. The full
        # dataset has extra columns (id, paragraph_id, etc.) we don't need.
        self._meta = pd.DataFrame({
            "title": ds["title"],
            "text": ds["text"],
            "url": ds["url"],
        })

        try:
            self._corpora_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(index_path))
            self._meta.to_parquet(meta_path, index=False)
            size_gb = index_path.stat().st_size / 1e9
            print(
                f"[cohere-store] persisted to {index_path.parent} "
                f"({size_gb:.2f} GB index + meta parquet). "
                f"Next session reloads in <10s."
            )
        except Exception as exc:
            # Non-fatal: we already have the index in memory for this session.
            # Common failure: Drive not mounted on Colab, or no write
            # permission on the chosen corpora dir.
            print(
                f"[cohere-store] WARNING: could not persist index "
                f"({type(exc).__name__}: {exc}). This session will work but "
                f"next session will re-download. Check that {self._corpora_dir} "
                f"is writable (Drive mounted?)."
            )

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    def _embed_query(self, query: str):
        """Embed a single query via Cohere; returns shape (1, D) float32 normalized."""
        import numpy as np  # type: ignore
        import faiss  # type: ignore

        if self._cohere_client is None:
            api_key = os.environ.get("COHERE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "COHERE_API_KEY not set. Free key at https://cohere.com/. "
                    "Set in Colab via: import os; os.environ['COHERE_API_KEY']='...'. "
                    "BGE-self-contained fallback is deferred — see "
                    "core/corpus_store_cohere.py module docstring."
                )
            import cohere  # type: ignore
            self._cohere_client = cohere.Client(api_key)

        resp = self._cohere_client.embed(
            texts=[query],
            model=_COHERE_EMBED_MODEL,
            input_type="search_query",
        )
        q = np.asarray(resp.embeddings, dtype="float32")
        faiss.normalize_L2(q)
        return q

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    def search(self, query: str, n_chunks: int = 20) -> list[CorpusChunk]:
        """Return up to n_chunks CorpusChunk objects, ranked by cosine similarity.

        Raises RuntimeError when COHERE_API_KEY is missing. Wrapped in
        core/retrieval.py:CohereCorpusRetriever which catches and falls
        through to Wikipedia.
        """
        self._ensure_loaded()
        q_emb = self._embed_query(query)
        scores, ids = self._index.search(q_emb, k=n_chunks)

        chunks: list[CorpusChunk] = []
        for rank, idx in enumerate(ids[0]):
            idx = int(idx)
            if idx < 0:
                continue  # FAISS returns -1 for unfilled slots
            row = self._meta.iloc[idx]
            title = str(row["title"]) if row["title"] else ""
            text = str(row["text"]) if row["text"] else ""
            url = str(row.get("url", "")) if "url" in row else ""
            if not text:
                continue
            tag = title[:80] or url[:80] or "cohere_wiki_simple"
            chunks.append(CorpusChunk(
                chunk_id=f"cohere_{rank:02d}_{hashlib.sha1(title.encode('utf-8')).hexdigest()[:8]}",
                text=text,
                source_tag=tag,
            ))

        print(f"[retrieval] source=cohere_wiki_simple, N={len(chunks)}")
        return chunks
