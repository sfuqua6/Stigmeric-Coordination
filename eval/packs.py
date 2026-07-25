"""Evidence-pack construction for the over-context eval (docs/OVERCONTEXT_EVAL_PLAN.md).

The decisive A-vs-F experiment needs condition A (swarm) and condition F
(single-call RAG) to see the EXACT SAME evidence, at three scales relative to
condition F's usable context (1x / 4x / 16x). This module builds that shared
evidence pack once per (prompt, scale) and persists it to disk so every
condition, and every re-run, reads the identical pack — fairness requires
determinism here.

Pack construction retrieves aggressively (CompositeRetriever, no MMR cut —
we want everything available, not a diversified top-K) across multiple short,
keyword-style queries (topic phrase + facet term, see _gen_queries — no LLM
call, deterministic) until the target character budget is hit or the query
queue (capped at _MAX_QUERY_ROUNDS) is exhausted. A `<pid>_<scale>.meta.json`
sidecar records the achieved chars/fraction honestly — a pack that tops out
below target is reportable as achieved scale, not silently passed off as
full-scale.

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
# the budget (e.g. every source thin) could loop indefinitely. Also doubles
# as the per-pack Tavily request budget (see _gen_queries below): at most
# this many live retrieval calls are issued per (prompt, scale) build.
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


def _meta_path_for(pid: str, scale: str, out_dir: Path) -> Path:
    return out_dir / f"{pid}_{scale}.meta.json"


# ---------------------------------------------------------------------------
# Query construction — short, keyword-style queries (2026-07-24 rewrite).
#
# The original approach queried the retriever with the ENTIRE prompt sentence
# plus a short suffix appended (e.g. "<150-char prompt> — case studies").
# Verified empirically against a real TAVILY_API_KEY + GROQ_API_KEY run: the
# first such query (the bare prompt) retrieved well, but every subsequent
# facet/suffix variant collapsed to ~2 chunks each — real search backends
# return thin, low-diversity results for long run-on queries. A 16x pack
# topped out around 10% of the 384,000-char target as a result.
#
# Fix: derive a short topic phrase from the prompt (strip stopwords/question
# boilerplate, keep the content words) and pair it with a fixed list of short
# facet terms — ordinary keyword search queries, the kind a search engine
# actually retrieves well against. Entirely deterministic (fixed word list,
# no LLM, no randomness) so pack builds stay reproducible.
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "based", "on", "in", "of", "to", "and", "or", "but",
    "that", "this", "these", "those", "which", "what", "how", "why", "where",
    "when", "who", "whom", "would", "should", "could", "can", "will", "shall",
    "provided", "evidence", "evidence,", "reports", "report", "across",
    "from", "with", "for", "as", "it", "its", "their", "them", "they", "he",
    "she", "we", "you", "your", "our", "than", "then", "also", "not", "no",
    "if", "so", "such", "about", "into", "between", "among", "per", "via",
    "point", "including", "include", "identify",
})


def _extract_topic(prompt_text: str, max_words: int = 10) -> str:
    """Strip question boilerplate/stopwords, keep content words in order,
    dedup while preserving first occurrence. No LLM — pure string processing
    so pack builds stay deterministic and offline-safe."""
    words = re.findall(r"[A-Za-z][A-Za-z\-]*", prompt_text.lower())
    seen: set[str] = set()
    kept: list[str] = []
    for w in words:
        if w in _STOPWORDS or w in seen:
            continue
        seen.add(w)
        kept.append(w)
        if len(kept) >= max_words:
            break
    return " ".join(kept) if kept else prompt_text[:60]


# Short facet terms — the keyword-search equivalent of the old suffix list,
# extended so a 16x pack has enough distinct queries to page through before
# hitting _MAX_QUERY_ROUNDS. Order is fixed (determinism).
_FACET_TERMS: list[str] = [
    "statistics", "case study", "history and background", "economics and costs",
    "criticism and counterargument", "recent developments", "policy and regulation",
    "mechanism and explanation", "expert analysis", "data and evidence",
    "stakeholder perspectives", "comparative analysis", "risks and downsides",
    "benefits and upside", "regional variation", "long-term trends",
    "meta-analysis and systematic review", "policy debate", "empirical findings",
    "theoretical framework", "practical implications", "global comparison",
    "public opinion", "regulatory response", "market impact", "social impact",
    "environmental impact", "technological factors", "historical precedent",
    "future outlook", "counterexamples", "methodology", "funding and investment",
    "unintended consequences", "cost-benefit analysis", "ethical considerations",
    "implementation challenges", "success stories", "failure cases", "root causes",
]


def _gen_queries(prompt_text: str, max_queries: int = _MAX_QUERY_ROUNDS) -> list[str]:
    """Deterministic short keyword-style queries: the bare topic phrase, then
    topic + each facet term, capped at `max_queries` (the per-pack retrieval
    request budget)."""
    topic = _extract_topic(prompt_text)
    queries = [topic] + [f"{topic} {term}" for term in _FACET_TERMS]
    return queries[:max_queries]


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

    # Query queue: short keyword-style queries (topic + facet term), capped
    # at _MAX_QUERY_ROUNDS — see _gen_queries() docstring for why these
    # replaced the old full-sentence-plus-suffix queries.
    queries = _gen_queries(prompt_text, max_queries=_MAX_QUERY_ROUNDS)

    seen_texts: set[str] = set()
    chunks: list[dict] = []
    used_chars = 0
    rounds = 0
    qi = 0
    saw_placeholder = False
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
            if "placeholder_corpus" in tag:
                saw_placeholder = True
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

    # Sidecar metadata (does NOT alter the pack file format condition A/F
    # both read): honest record of how much of the target was actually
    # achieved, for reporting rather than silently passing off a thin pack
    # as full-scale.
    meta = {
        "pid": pid,
        "scale": scale,
        "target_chars": budget,
        "achieved_chars": used_chars,
        "achieved_fraction": round(used_chars / budget, 4) if budget else None,
        "n_chunks": len(chunks),
        "n_queries_issued": rounds,
        "n_queries_available": len(queries),
        "used_placeholder_fallback": saw_placeholder,
    }
    _meta_path_for(pid, scale, out_dir).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[packs] built pack {path}: {len(chunks)} chunks, {used_chars} chars "
          f"(target {budget}, scale={scale}, {rounds} queries issued)")
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
