"""Validator — external grounding via Wikipedia.

The validator samples a signal, queries an external source (Wikipedia
summary by default; web fallback if a hook is registered), and deposits
a VERIFICATION signal as a child of the original. The VERIFICATION's
strength reflects how well the external source supports the claim.

VERIFICATION signals are special because the SignalStore's provenance
boost looks them up by ancestry — when a future child is deposited
under a verified lineage, that child gets a strength bonus.

Validator deliberately does NOT render external evidence text into
the prompts of other agents. It deposits its verification as a trace,
and downstream agents see it (or the boost it produces) only via the
signal store, never via direct hand-off.

Validator routing (Phase 1C)
------------------------------
Wikipedia lookups are expensive relative to the information they add.
Validators now prefer INITIALs that already have ≥ 2 direct SUPPORT
children — high-stakes targets where external grounding most changes
the survival outcome. If no well-supported INITIALs exist yet, the
validator falls back to its assigned sampling strategy.
"""

from __future__ import annotations

import random

from agents.base import BaseAgent
from core.signal_store import SignalStore, Signal
from core.signal_types import INITIAL, SUPPORT, VERIFICATION
from core.config import MAX_TOKENS_VALIDATOR
from core.sampling import SamplingStrategy

# Minimum SUPPORT children before an INITIAL is considered worth a Wikipedia lookup.
_MIN_SUPPORT_FOR_VALIDATION = 2


class Validator(BaseAgent):
    ROLE = "validator"
    OUTPUT_TYPE = VERIFICATION
    INPUT_TYPE = INITIAL
    MAX_TOKENS = MAX_TOKENS_VALIDATOR
    TEMPERATURE = 0.4
    DEFAULT_DEPOSIT_STRENGTH = 0.5

    def __init__(self, agent_id: str, llm, strategy: SamplingStrategy,
                 strategy_name: str, task_prompt: str):
        super().__init__(agent_id, llm)
        self.strategy = strategy
        self.strategy_name = strategy_name
        self.task_prompt = task_prompt

    def sample(self, store: SignalStore) -> list[Signal]:
        # Prefer high-stakes targets: INITIALs that already have ≥ 2 SUPPORT children.
        # Spending a Wikipedia lookup on an orphaned INITIAL that hasn't accumulated
        # support is wasteful — it's likely to be pruned before synthesis anyway.
        well_supported = store.signals_with_many_children_of_type(
            INITIAL, SUPPORT, _MIN_SUPPORT_FOR_VALIDATION
        )
        if well_supported:
            return [random.choice(well_supported)]
        # No well-supported INITIALs yet (early in run) — fall back to strategy.
        return self.strategy(store, self.INPUT_TYPE, 1)

    def build_prompt(self, samples: list[Signal], *,
                     store_count: int = 0, own_ids: tuple = ()) -> str:
        if not samples:
            return "No claim to verify.\n\nVERIFICATION:\nSCORE: 0.0"
        s = samples[0]
        count_hint = (
            f"There are currently {store_count} verification signals in the store. "
            f"Verify a distinct claim.\n"
        )
        external = _wiki_lookup(_extract_keyphrase(s.content))
        return (
            f"TASK: {self.task_prompt}\n\n"
            f"{count_hint}"
            f"Verify the following claim against the external snippet.\n\n"
            f"---CLAIM [{s.id}]---\n{s.content}\n---END CLAIM---\n\n"
            f"---EXTERNAL SNIPPET---\n{external}\n---END SNIPPET---\n\n"
            f"Does the snippet support, contradict, or fail to address "
            f"the claim? Reply with one short sentence followed by:\n\n"
            f"SCORE: <number in [0, 1]>"
        )

    def parse(self, raw: str) -> tuple[str, float]:
        text = raw.strip()
        score = 0.5
        # cheap parse — same as critic
        import re
        m = re.search(r"score\s*[:=]\s*([+-]?(?:\d+\.?\d*|\.\d+))", text, re.IGNORECASE)
        if m:
            try:
                score = max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                pass
        return text, score


# ---------------------------------------------------------------------------
# External lookup helpers
# ---------------------------------------------------------------------------

def _extract_keyphrase(text: str, max_words: int = 5) -> str:
    """Keyphrase: first N non-stopword content words, skipping first-person terms.

    Drops "I", "my", "we", "our", "you", "your" so first-person claims like
    "I think that renewable energy..." produce useful Wikipedia queries instead
    of "I think renewable energy".
    """
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
        "and", "or", "but", "for", "on", "at", "by", "with", "as",
        "that", "this", "these", "those", "it", "its",
        # first-person terms that produce useless queries
        "i", "my", "we", "our", "you", "your", "me", "us",
    }
    words = [w.strip(".,;:?!\"'()[]") for w in text.split()]
    keep = [w for w in words if w and w.lower() not in stop]
    return " ".join(keep[:max_words]) or text[:50]


def _wiki_lookup(query: str) -> str:
    """Best-effort Wikipedia lookup with graceful degradation.

    Returns a short snippet if available, or an explanatory placeholder
    otherwise. Never raises; verification proceeds even when offline.
    """
    try:
        import wikipedia  # type: ignore
        try:
            summary = wikipedia.summary(query, sentences=2, auto_suggest=True, redirect=True)
            return summary[:600]
        except Exception:
            try:
                hits = wikipedia.search(query, results=1)
                if hits:
                    summary = wikipedia.summary(hits[0], sentences=2)
                    return summary[:600]
            except Exception:
                pass
        return f"(no Wikipedia article found for query: {query!r})"
    except ImportError:
        return f"(wikipedia package not installed; query was: {query!r})"
