"""Evidence-pack construction for the over-context eval (docs/OVERCONTEXT_EVAL_PLAN.md).

The decisive A-vs-F experiment needs condition A (swarm) and condition F
(single-call RAG) to see the EXACT SAME evidence, at three scales relative to
condition F's usable context (1x / 4x / 16x). This module builds that shared
evidence pack once per (prompt, scale) and persists it to disk so every
condition, and every re-run, reads the identical pack — fairness requires
determinism here.

Pack construction retrieves aggressively (CompositeRetriever, no MMR cut —
we want everything available, not a diversified top-K) across multiple facet
queries (core.facet_planner.FacetPlanner, degrading to its deterministic
fallback when no LLM is available — pack building must not require an LLM
call) until the target character budget is hit or sources are exhausted.

Packs are plain JSONL: one line per chunk, `{"text", "source_tag", "url"}`.
`load_pack()` reads them back as `core.intake.CorpusChunk` objects so
`run_swarm.py`'s `--corpus=pack:<path>` mode and `eval.ab_harness`'s
condition F can both consume them identically.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.intake import CorpusChunk  # noqa: E402

# Target char budgets, scaled off a local/7B-class model's usable context
# (~8K ctx -> ~24K/96K/384K chars at 1x/4x/16x; see OVERCONTEXT_EVAL_PLAN.md
# for the Groq llama-3.1-8b-instant 128K-ctx numbers, which are ~5x larger —
# pass `target_chars=` explicitly for those runs). These are DEFAULTS, always
# overridable via the `target_chars` argument / `--target-chars` CLI flag.
SCALE_CHARS: dict[str, int] = {
    "1x": 24_000,
    "4x": 96_000,
    "16x": 384_000,
}

# Safety cap on how many facet/query rounds a single pack build may issue —
# without this, an aggressive 16x pack against a retriever that never fills
# the budget (e.g. every source thin) could loop indefinitely.
_MAX_QUERY_ROUNDS = 40


def _slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower().strip())
    slug = slug.strip("_")[:max_len]
    return slug or "prompt"


def _default_pid(prompt_text: str) -> str:
    """Deterministic fallback pack id when the caller doesn't supply one
    (prompts.py Prompt.pid is the preferred id; this is only a fallback so
    build_pack() is usable standalone on arbitrary prompt text)."""
    h = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:10]
    return f"{_slugify(prompt_text, 24)}_{h}"


def pack_path_for(pid: str, scale: str, out_dir: Path) -> Path:
    return out_dir / f"{pid}_{scale}.jsonl"


def _gen_facets(prompt_text: str, n: int) -> list[str]:
    """Facet queries with NO LLM call — build_pack must work offline/in eval
    contexts where no llm handle is threaded through. FacetPlanner.generate()
    is async and degrades to its deterministic `_fallback()` whenever
    `llm is None`, so calling it with `llm=None` is exactly the "graceful
    degrade" path, not a hack around it."""
    from core.facet_planner import FacetPlanner
    import asyncio
    planner = FacetPlanner(llm=None)
    try:
        return asyncio.run(planner.generate(prompt_text, n=n))
    except Exception:
        # Should not happen (the fallback path has no I/O), but never let
        # pack building die on facet generation.
        return [prompt_text]


def build_pack(prompt_text: str, scale: str, out_dir: Path,
                pid: Optional[str] = None,
                target_chars: Optional[int] = None) -> Path:
    """Build (or reuse) an evidence pack for `prompt_text` at the given scale.

    Deterministic: if a pack already exists at the computed path, it is
    reused as-is (no re-retrieval) — required so condition A and condition F
    see the identical pack across repeated harness runs.

    Returns the path to the JSONL pack file.
    """
    if scale not in SCALE_CHARS:
        raise ValueError(f"unknown scale {scale!r}; use one of {sorted(SCALE_CHARS)}")
    budget = target_chars if target_chars is not None else SCALE_CHARS[scale]
    pid = pid or _default_pid(prompt_text)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = pack_path_for(pid, scale, out_dir)

    if path.exists():
        print(f"[packs] reuse existing pack: {path} (deterministic — not re-retrieving)")
        return path

    from core.retrieval import CachedRetriever, CompositeRetriever
    retriever = CachedRetriever(CompositeRetriever())

    # Query queue: the raw prompt first (best single query), then facet
    # queries (empirical/historical/economic/ethical/... angles), then — if
    # the budget still isn't hit — the facet queries again with generic
    # "more detail" suffixes so a 16x pack keeps pulling instead of stalling
    # once the ~8 canonical facets are exhausted.
    facets = _gen_facets(prompt_text, n=8)
    queries: list[str] = [prompt_text] + facets
    _extra_suffixes = ["detailed analysis", "statistics and data", "case studies",
                        "expert commentary", "history and background",
                        "criticism and counterarguments", "recent developments",
                        "policy and regulation"]
    for suf in _extra_suffixes:
        queries.append(f"{prompt_text} — {suf}")

    seen_texts: set[str] = set()
    chunks: list[dict] = []
    used_chars = 0
    rounds = 0
    qi = 0
    while used_chars < budget and rounds < _MAX_QUERY_ROUNDS:
        if qi >= len(queries):
            break  # exhausted the query queue; stop rather than loop forever
        query = queries[qi]
        qi += 1
        rounds += 1
        try:
            got = retriever.retrieve(query, target_chars=budget - used_chars)
        except Exception as exc:
            print(f"[packs] retrieval round {rounds} ({query[:60]!r}) failed: "
                  f"{type(exc).__name__}: {exc}")
            continue
        added = 0
        for ch in got:
            text = (getattr(ch, "text", "") or "").strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            tag = getattr(ch, "source_tag", "") or "chunk"
            url = tag if tag.startswith("http") else ""
            chunks.append({"text": text, "source_tag": tag, "url": url})
            used_chars += len(text)
            added += 1
            if used_chars >= budget:
                break
        print(f"[packs] round {rounds}/{_MAX_QUERY_ROUNDS} query={query[:60]!r} "
              f"+{added} chunks (total {len(chunks)}, {used_chars}/{budget} chars)")

    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[packs] built pack {path}: {len(chunks)} chunks, {used_chars} chars "
          f"(target {budget}, scale={scale})")
    return path


def load_pack(path: Path) -> list[CorpusChunk]:
    """Read a pack JSONL file back into CorpusChunk objects (no partition_id
    stamped — partitioning happens downstream via partition_for_scouts())."""
    path = Path(path)
    chunks: list[CorpusChunk] = []
    if not path.exists():
        print(f"[packs] WARNING: pack not found at {path}")
        return chunks
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunks.append(CorpusChunk(
                chunk_id=f"pack_{i:05d}",
                text=d.get("text", ""),
                source_tag=d.get("source_tag", "") or f"pack_chunk_{i}",
            ))
    return chunks
