"""Query planning — emergent refinement, not fixed rotation.

The original `_compose_scout_query` rotated through 8 fixed phrasings keyed
on worker_id+iter. With 24 workers across 1500+ iterations that produced
the "queries are nearly identical across 130+ iterations" pathology from
the Apollo-run post-mortem: minor rephrasing of the same claim, no
meaningful coverage of the topic space.

This module produces queries that *adapt* to what's already in the store:

  - The user prompt is the base.
  - Existing high-strength INITIAL signals contribute keyphrases that
    steer subsequent scouts toward unexplored angles raised by their
    predecessors (no-leak: we only read Signal.content, never reasoning).
  - Stance modifiers ("evidence for", "limitations of", ...) widen the
    framing space.
  - Queries that have already returned >= MIN_GOOD_RESULTS for an exact
    or near-exact match are dropped — we already have those chunks
    cached and re-querying spends rate-limit budget on duplicates.

The planner is pure (no I/O) and deterministic given the same inputs;
workers feed it (store, pool_state, worker_id, iter, history) and it
returns the next query string.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Optional

from .signal_store import SignalStore
from .signal_types import INITIAL, SUPPORT
from .filters import strip_non_latin


# Queries with >= this many cached results are "well-served" — re-issuing
# them spends rate-limit budget on chunks we already have.
MIN_GOOD_RESULTS = 3
# Two queries with SequenceMatcher ratio >= this are duplicates for dedup.
DUP_RATIO = 0.85
# Top-k highest-strength INITIALs from which keyphrases are extracted.
KEYPHRASE_INITIAL_TOP_K = 5


_STOP = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
    "and", "or", "but", "for", "on", "at", "by", "with", "as",
    "that", "this", "these", "those", "it", "its", "be", "been",
    "not", "no", "can", "will", "would", "should", "may", "might",
    "have", "has", "had", "do", "does", "did", "if", "then", "than",
    "we", "i", "you", "he", "she", "they", "our", "my", "your",
    "very", "more", "most", "also", "how", "what", "when", "why",
    "which", "who", "all", "any", "each", "from", "some", "many",
    "such", "while", "because", "however", "though",
})


# Stance modifiers — produce framing variation around any base query.
# Half are neutral, half adversarial — gives haters / critics something
# to chew on later in the lineage.
_STANCE_MODIFIERS = [
    "{q}",
    "{q} evidence",
    "{q} history",
    "{q} consequences",
    "{q} examples",
    "{q} case studies",
    "{q} criticism",
    "{q} limitations",
    "{q} alternatives",
    "counterarguments to {q}",
    "evidence against {q}",
    "recent research on {q}",
    "expert disagreement on {q}",
    "{q} successes and failures",
    "{q} unintended consequences",
    "{q} economic analysis",
    "{q} ethical analysis",
    "{q} comparative cases",
]


def _extract_keyphrases(text: str, max_phrases: int = 3,
                         max_words: int = 4) -> list[str]:
    """Pull short content keyphrases from a string.

    Returns up to `max_phrases` phrases of `max_words` non-stop words each.
    Deterministic, cheap, no NLP deps.
    """
    if not text:
        return []
    # Strip CJK so a contaminated INITIAL doesn't pollute future queries.
    text = strip_non_latin(text)
    words = [w.strip(".,;:?!\"'()[]{}").lower() for w in text.split()]
    keep = [w for w in words if w and w not in _STOP and len(w) > 2]
    if not keep:
        return []
    phrases: list[str] = []
    for i in range(0, len(keep), max_words):
        phrase = " ".join(keep[i:i + max_words])
        if phrase:
            phrases.append(phrase)
        if len(phrases) >= max_phrases:
            break
    return phrases


def _is_dup_of_existing(candidate: str, served: dict[str, int]) -> bool:
    """True if `candidate` is similar to a query that's already well-served."""
    cand_lower = candidate.lower()
    for served_q, n_results in served.items():
        if n_results < MIN_GOOD_RESULTS:
            continue
        if cand_lower == served_q.lower():
            return True
        ratio = SequenceMatcher(None, cand_lower, served_q.lower()).ratio()
        if ratio >= DUP_RATIO:
            return True
    return False


def find_cached_query(candidate: str, served: dict[str, int]) -> Optional[str]:
    """Return the served query whose result we should reuse, or None.

    Worker callers use this BEFORE search() to skip a network round-trip
    when another worker has already fetched substantially the same thing.
    """
    cand_lower = candidate.lower()
    best_ratio = 0.0
    best_q = None
    for served_q, n_results in served.items():
        if n_results < MIN_GOOD_RESULTS:
            continue
        r = SequenceMatcher(None, cand_lower, served_q.lower()).ratio()
        if r > best_ratio:
            best_ratio = r
            best_q = served_q
    if best_q is not None and best_ratio >= DUP_RATIO:
        return best_q
    return None


def plan_scout_query(
    user_prompt: str,
    store: SignalStore,
    served_queries: dict[str, int],
    worker_id: int,
    iter_idx: int,
    own_history: list[str],
) -> str:
    """Pick the next scout query.

    Pulls keyphrases from the strongest existing INITIALs (no-leak:
    content only, never reasoning) and combines them with stance
    modifiers. Skips queries that overlap with already-well-served ones.
    Falls through to base phrasings if nothing else fits.
    """
    base = strip_non_latin(user_prompt or "").strip()
    if not base:
        return ""

    # Gather candidates: stance-modified base, then keyphrase-blended.
    initials = sorted(store.by_type(INITIAL), key=lambda s: s.strength,
                      reverse=True)[:KEYPHRASE_INITIAL_TOP_K]
    keyphrases = []
    for sig in initials:
        keyphrases.extend(_extract_keyphrases(sig.content, max_phrases=2,
                                              max_words=4))
    # Deterministic deduplication preserving order.
    seen: set[str] = set()
    keyphrases = [k for k in keyphrases if not (k in seen or seen.add(k))]

    candidates: list[str] = []
    # First: stance-modified base.
    for tmpl in _STANCE_MODIFIERS:
        candidates.append(tmpl.format(q=base))
    # Second: keyphrase-blended (one keyphrase per query, paired with base).
    for kp in keyphrases[:6]:
        candidates.append(f"{base} {kp}")
        candidates.append(f"{kp} {base}")
    # Third: pure keyphrase queries — explore the angle on its own.
    candidates.extend(keyphrases[:4])

    own_set = set(own_history)
    # Hash the (worker_id, iter_idx) tuple via SHA1 so adjacent workers
    # don't collapse to the same index mod len(candidates). The previous
    # linear formula (worker_id*17 + iter_idx) had a periodicity bug:
    # any pair (w, i) and (w + len/17, i) hashed identically.
    import hashlib
    seed_bytes = hashlib.sha1(
        f"{worker_id}:{iter_idx}:{base}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(seed_bytes[:4], "big") % max(1, len(candidates))
    for offset in range(len(candidates)):
        q = candidates[(seed + offset) % len(candidates)]
        if q in own_set:
            continue
        if _is_dup_of_existing(q, served_queries):
            continue
        return q
    # Fall through: cycle through base + stance even when nothing's fresh.
    return candidates[seed]


def plan_develop_query(target_content: str, served_queries: dict[str, int]) -> str:
    """Query for the DEVELOP sparse-cluster search trigger.

    Builds a keyphrase-based query from the target INITIAL's content and
    rejects if it duplicates a well-served one (caller falls back to no
    search in that case).
    """
    target_content = strip_non_latin(target_content or "")
    phrases = _extract_keyphrases(target_content, max_phrases=1, max_words=8)
    if not phrases:
        return ""
    base_query = f"{phrases[0]} evidence"
    if _is_dup_of_existing(base_query, served_queries):
        # Try a stance variant before giving up.
        for stance in ("limitations of", "counterevidence to", "case studies of"):
            alt = f"{stance} {phrases[0]}"
            if not _is_dup_of_existing(alt, served_queries):
                return alt
        return ""
    return base_query


def plan_validate_query(target_content: str) -> str:
    """Query for the VALIDATE action — short, factual."""
    target_content = strip_non_latin(target_content or "")
    phrases = _extract_keyphrases(target_content, max_phrases=1, max_words=8)
    return phrases[0] if phrases else ""
