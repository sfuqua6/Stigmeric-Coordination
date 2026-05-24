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

import json
import os
import random
import re
from typing import Optional

from agents.base import BaseAgent
from core.signal_store import SignalStore, Signal
from core.signal_types import INITIAL, SUPPORT, VERIFICATION
from core.config import MAX_TOKENS_VALIDATOR
from core.sampling import SamplingStrategy

# Minimum SUPPORT children before an INITIAL is considered worth a Wikipedia lookup.
_MIN_SUPPORT_FOR_VALIDATION = 2

# Phase K diagnostic: when SWARM_VALIDATOR_DIAGNOSTIC=1, print the extracted
# keyphrase and the raw LLM output for every validator sample. Useful for
# verifying that the new JSON-output path actually receives parseable text.
_DIAGNOSTIC = os.environ.get("SWARM_VALIDATOR_DIAGNOSTIC", "").strip() not in ("", "0", "false", "False")


class Validator(BaseAgent):
    ROLE = "validator"
    OUTPUT_TYPE = VERIFICATION
    INPUT_TYPE = INITIAL
    MAX_TOKENS = MAX_TOKENS_VALIDATOR
    TEMPERATURE = 0.4
    DEFAULT_DEPOSIT_STRENGTH = 0.5

    def __init__(self, agent_id: str, llm, strategy: SamplingStrategy,
                 strategy_name: str, task_prompt: str,
                 cloud_provider: str = "none"):
        super().__init__(agent_id, llm)
        self.strategy = strategy
        self.strategy_name = strategy_name
        self.task_prompt = task_prompt
        self._cloud_provider = cloud_provider

    def _maybe_cloud_call(self, claim_content: str) -> Optional[str]:
        if self._cloud_provider == 'anthropic':
            # FUTURE-CLAUDE NOTE: To enable, set ANTHROPIC_API_KEY in env and
            # `pip install anthropic`. Costs ~$0.005 per run at Haiku 4.5
            # pricing. Disabled by default to preserve the fully-local
            # property of the project. Complete the implementation here.
            raise NotImplementedError('cloud validator wired but not enabled; see comment')
        if self._cloud_provider == 'gemini':
            # FUTURE-CLAUDE NOTE: Free tier at aistudio.google.com — 15 req/min,
            # 1500/day. Set GOOGLE_API_KEY in env and pip install
            # google-generativeai. Complete the implementation here.
            raise NotImplementedError('cloud validator wired but not enabled; see comment')
        return None

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
            return (
                "No claim to verify.\n\n"
                'Reply with this JSON: {"supports": false, "confidence": 0.0, "note": "no claim"}'
            )
        s = samples[0]
        count_hint = (
            f"There are currently {store_count} verification signals in the store. "
            f"Verify a distinct claim.\n"
        )
        keyphrase = _extract_keyphrase(s.content)
        external = _wiki_lookup(keyphrase)
        if _DIAGNOSTIC:
            print(f"[validator-diag {self.agent_id}] keyphrase={keyphrase!r}")
        # JSON output is far more reliably parseable on non-reasoning Instruct
        # models (Phase K). Falls back to "SCORE: X" regex if the model
        # ignores the JSON format.
        return (
            f"TASK: {self.task_prompt}\n\n"
            f"{count_hint}"
            f"Verify the following claim against the external snippet.\n\n"
            f"---CLAIM [{s.id}]---\n{s.content}\n---END CLAIM---\n\n"
            f"---EXTERNAL SNIPPET---\n{external}\n---END SNIPPET---\n\n"
            f"Reply with EXACTLY this JSON object (no other text):\n"
            f'{{"supports": true|false, "confidence": <number in [0,1]>, '
            f'"note": "<one short sentence>"}}\n\n'
            f"`supports` is true if the snippet plausibly backs the claim, "
            f"false if it contradicts or fails to address it. `confidence` "
            f"is your subjective certainty in the assessment."
        )

    def parse(self, raw: str) -> tuple[str, float]:
        text = raw.strip()
        if _DIAGNOSTIC:
            print(f"[validator-diag {self.agent_id}] raw={text[:300]!r}")
        # Phase K: try JSON first. If the model produced extra text around
        # the JSON object, extract the first {...} block.
        score = 0.5
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                supports = bool(parsed.get("supports", False))
                conf = float(parsed.get("confidence", 0.5))
                conf = max(0.0, min(1.0, conf))
                # Combine into a single score: supports=True → conf, supports=False → 1-conf
                # but clamp so a low-confidence "false" doesn't accidentally
                # land near 0.5 and read as a non-call.
                score = conf if supports else max(0.0, 1.0 - conf)
                note = str(parsed.get("note", "")).strip()
                if note:
                    text = note
                return text, score
            except (ValueError, TypeError):
                pass
        # Fallback: legacy SCORE: regex (kept for cases where the model
        # produces freeform text despite the JSON instruction).
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
    """External-snippet lookup via the agentic search_tool.

    Phase 2D: replaces the hard-coded Wikipedia path with the Tavily/DDG/Cohere
    cascade. The function name is preserved for callers; semantics are now
    "best 1-2 retrieved snippets, concatenated into ~600 chars." Returns a
    placeholder when every backend fails so validation proceeds offline.
    """
    try:
        from core.search_tool import search as _search
    except Exception:
        return f"(search tool unavailable; query was: {query!r})"
    chunks = _search(query, max_results=2, task_type="analysis")
    if not chunks:
        return f"(no external snippet found for query: {query!r})"
    pieces = []
    for c in chunks[:2]:
        tag = c.source_tag[:120]
        body = c.text[:400]
        pieces.append(f"[{tag}]\n{body}")
    out = "\n\n".join(pieces)
    return out[:600]
