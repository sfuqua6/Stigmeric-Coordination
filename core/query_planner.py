"""Query planning — natural-language, task-aware, deduplicated.

Post-mortem of "Does free will exist?" produced query examples like:

  "existence free remains contentious issue supported concepts moral evidence"
  "counterevidence to existence free remains contentious issue ..."
  "concept free highly debated Does free will exist?"

These all stem from the same root mistake: stripping stop-words to produce
"keyphrases", then concatenating them with the base prompt in both orderings,
then layering stance modifiers ("counterevidence to ...") on top of an already-
stripped fragment. The result is ungrammatical keyword soup that performs poorly
on DuckDuckGo and burns search budget on near-duplicates.

This rewrite enforces four rules:

  1. Queries are natural English. We do not strip stop-words from query
     strings. Keyphrase extraction is preserved only for *INTERNAL* matching
     against served-queries history; query strings themselves stay sentence-y.

  2. Topic anchors are pulled as short *sentence fragments* from existing
     INITIAL content (no stop-word stripping). These are used as standalone
     candidates, never blended with the base in both orderings.

  3. Stance modifiers are task-aware. Debate tasks see "arguments for/against",
     "philosophical positions on ..."; analysis tasks see "X explained",
     "X causes"; coding tasks see "X implementation". The old generic list
     ("economic analysis", "successes and failures") only fits some tasks.

  4. Deduplication is cross-worker. PoolState.served_queries tracks
     query -> n_results across the whole pool; find_cached_query() returns
     the served query a new candidate should reuse instead of refetching.
"""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import Optional

from .signal_store import SignalStore
from .signal_types import INITIAL
from .filters import strip_non_latin


# Queries with >= this many cached results are "well-served" — re-issuing
# them spends rate-limit budget on chunks we already have.
MIN_GOOD_RESULTS = 3
# Two queries with SequenceMatcher ratio >= this are duplicates for dedup.
DUP_RATIO = 0.85
# Top-k highest-strength INITIALs from which sentence fragments are extracted.
TOP_K_FOR_FRAGMENTS = 5
# Maximum words to keep when extracting a natural-language fragment.
FRAGMENT_MAX_WORDS = 10

# Per-iteration cap on search budget. The pool gates SCOUT/DEVELOP via this
# (see Worker._gather_target) so DDG rate-limits don't push response times
# past 4-5 s.
DEFAULT_SEARCH_BUDGET_PER_TICK = 4


# Topic-aware stance modifiers. Keys match TASK_PROMPTS in run_swarm.py.
_STANCE_BY_TASK: dict[str, list[str]] = {
    "debate": [
        "{q}",
        "arguments for {q}",
        "arguments against {q}",
        "philosophical positions on {q}",
        "common objections to {q}",
        "scholarly debate on {q}",
        "competing theories on {q}",
        "what experts say about {q}",
    ],
    "analysis": [
        "{q}",
        "{q} explained",
        "{q} causes",
        "{q} effects",
        "{q} mechanisms",
        "research on {q}",
        "case studies of {q}",
        "{q} history",
    ],
    "problem_solving": [
        "{q}",
        "solutions to {q}",
        "case studies of {q}",
        "{q} best practices",
        "{q} pilot programs",
        "policy options for {q}",
        "comparative approaches to {q}",
        "expert recommendations for {q}",
    ],
    "creative": [
        "{q}",
        "{q} themes",
        "{q} symbolism",
        "{q} in literature",
        "examples of {q}",
        "famous works about {q}",
    ],
    "coding": [
        "{q}",
        "{q} implementation",
        "{q} algorithm",
        "{q} edge cases",
        "{q} python example",
        "{q} best practices",
    ],
}
_STANCE_DEFAULT = _STANCE_BY_TASK["analysis"]


# Stop-words used ONLY for the internal dup-check fingerprint, NEVER for
# building query strings. Compare-only — never returned to the caller.
_STOP = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
    "and", "or", "but", "for", "on", "at", "by", "with", "as", "that",
    "this", "these", "those", "it", "its", "be", "been", "not", "no",
    "can", "will", "would", "should", "may", "might", "have", "has",
    "had", "do", "does", "did", "if", "then", "than", "we", "i", "you",
    "he", "she", "they", "our", "my", "your", "very", "more", "most",
    "also", "how", "what", "when", "why", "which", "who", "all", "any",
    "each", "from", "some", "many", "such", "while", "because",
    "however", "though", "about", "into", "over",
})


def _fingerprint(text: str) -> str:
    """Stop-word-stripped, lowercased token fingerprint for dup-detection only.

    Two queries with the same fingerprint are duplicates regardless of
    word order or punctuation. This is what we compare against
    served_queries; it is NEVER the query string we emit.
    """
    if not text:
        return ""
    words = [w.strip(".,;:?!\"'()[]{}").lower() for w in text.split()]
    keep = [w for w in words if w and w not in _STOP and len(w) > 2]
    return " ".join(sorted(keep))


def _extract_sentence_fragment(text: str, max_words: int = FRAGMENT_MAX_WORDS) -> str:
    """Pull a natural-language fragment from text, preserving stop-words.

    Takes the first clause up to `max_words` long. Strips CJK so a
    contaminated INITIAL can't poison future queries.
    """
    if not text:
        return ""
    text = strip_non_latin(text).strip()
    if not text:
        return ""
    # Split on sentence-final punctuation and clause boundaries.
    pieces = re.split(r"[.!?;:]\s+|,\s+(?=which|that|because|so|but|and\s+)", text)
    first = pieces[0].strip()
    words = first.split()
    if len(words) <= max_words:
        fragment = first
    else:
        fragment = " ".join(words[:max_words])
    # Lowercase the first character so it reads as a search query, not a sentence.
    # Preserve proper-noun-looking words (capitalised mid-fragment) untouched.
    if fragment and fragment[0].isupper():
        fragment = fragment[0].lower() + fragment[1:]
    return fragment.strip()


def _is_dup_of_existing(candidate: str, served: dict[str, int]) -> bool:
    """True if `candidate` overlaps a well-served prior query."""
    if not candidate:
        return True
    cand_fp = _fingerprint(candidate)
    cand_lower = candidate.lower()
    for served_q, n_results in served.items():
        if n_results < MIN_GOOD_RESULTS:
            continue
        if cand_lower == served_q.lower():
            return True
        if cand_fp and cand_fp == _fingerprint(served_q):
            return True
        if SequenceMatcher(None, cand_lower, served_q.lower()).ratio() >= DUP_RATIO:
            return True
    return False


def find_cached_query(candidate: str, served: dict[str, int]) -> Optional[str]:
    """Return the served query the caller should reuse instead of refetching.

    Used by Worker._gather_target BEFORE search() to skip the network call
    when another worker already fetched substantially the same thing.
    """
    if not candidate:
        return None
    cand_fp = _fingerprint(candidate)
    cand_lower = candidate.lower()
    best_ratio = 0.0
    best_q: Optional[str] = None
    for served_q, n_results in served.items():
        if n_results < MIN_GOOD_RESULTS:
            continue
        # Exact / fingerprint match is automatic.
        if cand_lower == served_q.lower():
            return served_q
        if cand_fp and cand_fp == _fingerprint(served_q):
            return served_q
        r = SequenceMatcher(None, cand_lower, served_q.lower()).ratio()
        if r > best_ratio:
            best_ratio = r
            best_q = served_q
    return best_q if best_q is not None and best_ratio >= DUP_RATIO else None


def _seed(*parts) -> int:
    """SHA1-derived integer seed; avoids modulo collapse across small periods."""
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha1(raw).digest()[:4], "big")


def _build_candidates(base: str, task_type: Optional[str],
                       fragments: list[str]) -> list[str]:
    """Compose stance-modified base queries followed by topic-anchor fragments.

    Returns the candidate pool in priority order:
      1. The base prompt itself.
      2. Stance-modified variants of the base (task-aware).
      3. Sentence-fragment topic anchors (each one used as a *standalone*
         search query — no concatenation with the base, no reverse-order).
    """
    modifiers = _STANCE_BY_TASK.get(task_type or "", _STANCE_DEFAULT)
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        if not q:
            return
        q = re.sub(r"\s+", " ", q).strip()
        if not q or q in seen:
            return
        seen.add(q)
        candidates.append(q)

    for tmpl in modifiers:
        _add(tmpl.format(q=base))
    for frag in fragments:
        # Fragments only ever appear standalone — never blended with the base.
        # That was the source of "concept free highly debated Does free will exist?"
        # in production.
        _add(frag)
    return candidates


def plan_scout_query(
    user_prompt: str,
    store: SignalStore,
    served_queries: dict[str, int],
    worker_id: int,
    iter_idx: int,
    own_history: list[str],
    task_type: Optional[str] = None,
) -> str:
    """Pick the next scout query — natural English, task-aware, deduplicated.

    The returned query is plain English fit for DuckDuckGo. Internal
    fingerprint matching catches "X causes Y" vs "what causes Y" duplicates;
    SequenceMatcher catches surface paraphrases; both are checked against
    served_queries with >= MIN_GOOD_RESULTS hits.
    """
    base = strip_non_latin(user_prompt or "").strip()
    if not base:
        return ""

    # Topic anchors from existing high-strength INITIALs (natural language,
    # no stop-word stripping). Only sampled when the swarm has produced
    # signals worth exploring — empty store means stance-only.
    initials = sorted(store.by_type(INITIAL), key=lambda s: s.strength,
                      reverse=True)[:TOP_K_FOR_FRAGMENTS]
    fragments: list[str] = []
    for sig in initials:
        frag = _extract_sentence_fragment(sig.content)
        if frag and len(frag.split()) >= 3:
            fragments.append(frag)

    candidates = _build_candidates(base, task_type, fragments)
    if not candidates:
        return base

    own_set = {q.lower() for q in own_history}
    seed = _seed(worker_id, iter_idx, base, task_type or "") % len(candidates)
    for offset in range(len(candidates)):
        q = candidates[(seed + offset) % len(candidates)]
        if q.lower() in own_set:
            continue
        if _is_dup_of_existing(q, served_queries):
            continue
        return q
    return candidates[seed]


def plan_develop_query(target_content: str, served_queries: dict[str, int],
                       task_type: Optional[str] = None) -> str:
    """Build a DEVELOP search query from an INITIAL's content.

    Prefers the natural-language sentence fragment ALONE — the prior
    "arguments supporting X" prefix produced ungrammatical pairings when
    X already contained "supported" or similar. Stance prefixes are
    fall-backs used only after the bare fragment is duplicate-served.
    """
    fragment = _extract_sentence_fragment(target_content or "", max_words=12)
    if not fragment or len(fragment.split()) < 3:
        return ""
    # First try the bare fragment — cleanest signal to DDG.
    if not _is_dup_of_existing(fragment, served_queries):
        return fragment
    # Fall-back stances: neutral framings that don't structurally clash
    # with content already inside the fragment.
    stance_options = {
        "debate":          ["perspectives on", "scholarly debate about", "philosophy of"],
        "analysis":        ["research on", "studies on", "data about"],
        "problem_solving": ["case studies of", "examples of", "approaches to"],
        "creative":        ["themes of", "examples of"],
        "coding":          ["implementations of", "examples of"],
    }
    stances = stance_options.get(task_type or "",
                                  ["research on", "perspectives on"])
    for stance in stances:
        alt = f"{stance} {fragment}"
        if not _is_dup_of_existing(alt, served_queries):
            return alt
    return ""


def plan_validate_query(target_content: str) -> str:
    """Query for the VALIDATE action — short, factual, natural language."""
    return _extract_sentence_fragment(target_content or "", max_words=10)
