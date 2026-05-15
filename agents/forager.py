"""Forager — develops INITIAL signals into SUPPORT signals.

Each forager is constructed with a different sampling strategy
(stratified extremes, medium-only, recent-only, under-visited-only,
weighted default). All foragers see the same signal store; none of
them see the same slice of it.

A forager renders only the sampled signal's content. It does not see
which scout deposited that signal, what the scout's reasoning was, or
what the signal's parent (if any) said.

Cluster-aware gap-filling (Phase 1A)
-------------------------------------
Before falling back to its assigned strategy, a forager first checks
which INITIALs have fewer than 2 direct SUPPORT children. These are
"underserved" clusters — they haven't yet accumulated enough corroboration
to cross the support_diversity threshold that lets them survive projection.
By sampling from underserved INITIALs first, foragers convert from broad
explorers into gap-fillers. Once every INITIAL has ≥ 2 SUPPORT children,
the strategy fallback resumes normal operation.
"""

from __future__ import annotations

import random

from agents.base import BaseAgent
from core.signal_store import SignalStore, Signal
from core.signal_types import INITIAL, SUPPORT
from core.config import MAX_TOKENS_FORAGER
from core.sampling import SamplingStrategy

# Minimum SUPPORT children before an INITIAL is considered "served".
_MIN_SUPPORT_PER_INITIAL = 2


class Forager(BaseAgent):
    ROLE = "forager"
    OUTPUT_TYPE = SUPPORT
    INPUT_TYPE = INITIAL
    MAX_TOKENS = MAX_TOKENS_FORAGER
    TEMPERATURE = 0.7
    DEFAULT_DEPOSIT_STRENGTH = 0.6

    def __init__(self, agent_id: str, llm, strategy: SamplingStrategy,
                 strategy_name: str, task_prompt: str):
        super().__init__(agent_id, llm)
        self.strategy = strategy
        self.strategy_name = strategy_name
        self.task_prompt = task_prompt

    def sample(self, store: SignalStore) -> list[Signal]:
        # Gap-fill first: target INITIALs with fewer than _MIN_SUPPORT_PER_INITIAL
        # direct SUPPORT children. This steers foragers toward underdeveloped claims
        # and raises support_diversity across the field without extra LLM calls.
        underserved = store.signals_with_few_children_of_type(
            INITIAL, SUPPORT, _MIN_SUPPORT_PER_INITIAL
        )
        if underserved:
            # Strength-weighted sample so stronger underserved signals get priority.
            weights = [max(0.01, s.strength) for s in underserved]
            return random.choices(underserved, weights=weights, k=1)
        # All INITIALs are well-served — fall back to assigned strategy for
        # deeper development of existing clusters.
        return self.strategy(store, self.INPUT_TYPE, 1)

    def build_prompt(self, samples: list[Signal], *,
                     store_count: int = 0, own_ids: tuple = ()) -> str:
        count_hint = (
            f"There are currently {store_count} SUPPORT signals in the store. "
            f"Produce something structurally distinct from them.\n"
        )
        own_hint = ""
        if own_ids:
            own_hint = (
                f"You have already deposited {len(own_ids)} signal(s) this round "
                f"(IDs: {', '.join(own_ids)}). Do not restate them.\n"
            )
        if not samples:
            return (
                f"TASK: {self.task_prompt}\n\n"
                f"{count_hint}{own_hint}"
                f"There are no initial signals to develop yet. Produce "
                f"one short observation that could serve as a useful "
                f"early-stage trace. One or two sentences.\n\n"
                f"OBSERVATION:"
            )
        s = samples[0]
        return (
            f"TASK: {self.task_prompt}\n\n"
            f"{count_hint}{own_hint}"
            f"You observe one deposited signal in the shared store. "
            f"You do not know which agent deposited it or how they "
            f"arrived at it.\n\n"
            f"---SIGNAL [{s.id}]---\n{s.content}\n---END SIGNAL---\n\n"
            f"Produce ONE concise development of this signal: a piece "
            f"of supporting evidence, an extension, or a refinement. "
            f"Do not restate the original. Do not speculate about who "
            f"deposited it or why. One or two sentences.\n\n"
            f"DEVELOPMENT:"
        )
