"""Developer — develops INITIAL signals into SUPPORT signals (formerly Forager).

The 'Forager' name was a metaphor mismatch: ant foragers forage for food, but
this role's job is *developing* existing signals — extending them with evidence,
anticipating objections, and producing SUPPORT that crosses the support_diversity
threshold. The rename to Developer makes the role's purpose explicit.

Each Developer is constructed with one of three sampling strategies:
    under_supported_clusters  (default gap-filler, highest priority)
    stratified_extremes       (spans weak + strong strength strata)
    recent_dissent_targeted   (targets INITIALs under active CRITIQUE_NEGATIVE/OBJECTION)

All Developers see the same signal store; none see the same slice of it.

Dissent-anticipation (§5 directive)
-------------------------------------
A Developer reads both the target INITIAL *and* the strongest
CRITIQUE_NEGATIVE or OBJECTION against that INITIAL (if any), stashed
during sample(). The prompt instructs it to produce SUPPORT that explicitly
anticipates the dissent. This converts SUPPORT from "extension" to
"defended extension", which reduces dissent_pressure and increases
the cluster's chance of surviving projection.
"""

from __future__ import annotations

import random
from typing import Optional

from agents.base import BaseAgent, type_parent_instruction
from core.signal_store import SignalStore, Signal
from core.signal_types import INITIAL, SUPPORT, CRITIQUE_NEGATIVE, OBJECTION, SEARCH
from core.config import MAX_TOKENS_FORAGER
from core.sampling import SamplingStrategy

# Minimum SUPPORT children before an INITIAL is considered "served".
_MIN_SUPPORT_PER_INITIAL = 2
# Phase 2D: support_diversity floor at which a developer issues an agentic
# search to bring fresh evidence into the prompt.
_SEARCH_TRIGGER_SUPPORT = 2


def recent_dissent_targeted(store: SignalStore, signal_type: str, n: int) -> list[Signal]:
    """Pick INITIALs with the highest combined CRITIQUE_NEGATIVE+OBJECTION
    strength but not enough SUPPORT to cross the survival threshold.

    Targets INITIALs under active challenge that haven't been sufficiently
    defended yet — the third developer strategy.
    """
    if signal_type != INITIAL:
        return store.sample_weighted(signal_type, n)

    initials = store.by_type(INITIAL)
    if not initials:
        return []

    def dissent_score(sig: Signal) -> float:
        children = [store.get(cid) for cid in store.by_parent(sig.id)]
        children = [c for c in children if c is not None]
        n_support = sum(1 for c in children if c.type == SUPPORT)
        if n_support >= _MIN_SUPPORT_PER_INITIAL:
            return -1.0  # already well-served; deprioritize
        return sum(
            c.strength for c in children
            if c.type in (CRITIQUE_NEGATIVE, OBJECTION)
        )

    return sorted(initials, key=dissent_score, reverse=True)[:n]


class Developer(BaseAgent):
    ROLE = "developer"
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
        # Stashed during sample() so build_prompt can include the strongest
        # dissent without needing direct store access.
        self._stashed_dissent: Optional[Signal] = None
        # Phase 2D: stashed retrieved chunks when the developer issued an
        # agentic search this iteration (sparse-support trigger).
        self._stashed_retrieval: list = []
        self._stashed_query: str = ""

    def sample(self, store: SignalStore) -> list[Signal]:
        # Gap-fill first: target INITIALs with fewer than _MIN_SUPPORT_PER_INITIAL
        # direct SUPPORT children. This steers developers toward underdeveloped
        # claims and raises support_diversity across the field.
        underserved = store.signals_with_few_children_of_type(
            INITIAL, SUPPORT, _MIN_SUPPORT_PER_INITIAL
        )
        if underserved:
            weights = [max(0.01, s.strength) for s in underserved]
            target = random.choices(underserved, weights=weights, k=1)
        else:
            target = self.strategy(store, self.INPUT_TYPE, 1)
            # Fallback: strategy may return [] when no signals fall in the
            # requested strength strata (e.g. stratified_extremes finds no
            # weak signals). Sample any INITIAL by weight so the iteration
            # stays productive and the partition_id invariant is preserved.
            if not target:
                target = store.sample_weighted(self.INPUT_TYPE, 1)

        # Stash the strongest dissent for the chosen INITIAL (for prompt building)
        self._stashed_dissent = None
        self._stashed_retrieval = []
        self._stashed_query = ""
        if target:
            children = [store.get(cid) for cid in store.by_parent(target[0].id)]
            children = [c for c in children if c is not None]
            dissent_sigs = [
                c for c in children
                if c.type in (CRITIQUE_NEGATIVE, OBJECTION)
            ]
            if dissent_sigs:
                self._stashed_dissent = max(dissent_sigs, key=lambda c: c.strength)

            # Phase 2D: when the cluster is sparsely supported, the developer
            # issues an agentic search and stashes the chunks for prompt
            # building. Deposits a SEARCH signal so other agents see the trail.
            n_support_children = sum(
                1 for c in children if c.type == SUPPORT
            )
            if n_support_children < _SEARCH_TRIGGER_SUPPORT:
                try:
                    from core.search_tool import search as _search, summarize_for_signal
                    # Build a query: 8-word excerpt of the target INITIAL keeps
                    # the search focused on the cluster, not the global task.
                    snippet = " ".join(target[0].content.split()[:8])
                    query = f"{snippet} evidence"
                    chunks = _search(query, max_results=3,
                                     task_type=getattr(self, "task_type", None))
                except Exception:
                    chunks = []
                    query = ""
                if chunks:
                    self._stashed_retrieval = chunks
                    self._stashed_query = query
                    store.deposit(
                        signal_type=SEARCH,
                        content=summarize_for_signal(query, chunks),
                        strength=0.4,
                        depositor=self.ROLE,
                        parent_id=target[0].id,
                        metadata={
                            "depositor_agent_id": self.agent_id,
                            "query": query,
                            "n_results": len(chunks),
                            "trigger": "sparse_support",
                        },
                    )

        return target

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

        dissent_block = ""
        if self._stashed_dissent is not None:
            d = self._stashed_dissent
            dissent_block = (
                f"\nThe following challenge has been raised against this signal "
                f"(strength={d.strength:.2f}). Your SUPPORT should anticipate "
                f"and address it:\n"
                f"  [{d.id}]: {d.content}\n"
            )

        retrieval_block = ""
        if self._stashed_retrieval:
            blocks = []
            for c in self._stashed_retrieval[:3]:
                tag = c.source_tag[:120]
                body = c.text[:400]
                blocks.append(f"[{tag}]\n{body}")
            retrieval_block = (
                f"\nThe cluster has thin support — you issued an agentic "
                f"search for {self._stashed_query!r} and retrieved:\n\n"
                + "\n\n".join(blocks)
                + "\n\nGround your development in the snippet that fits best.\n"
            )

        return (
            f"TASK: {self.task_prompt}\n\n"
            f"{count_hint}{own_hint}"
            f"You observe one deposited signal in the shared store. "
            f"You do not know which agent deposited it or how they "
            f"arrived at it.\n\n"
            f"---SIGNAL [{s.id}]---\n{s.content}\n---END SIGNAL---\n"
            f"{dissent_block}"
            f"{retrieval_block}\n"
            f"Produce ONE concise development of this signal: a piece "
            f"of supporting evidence, an extension, or a refinement that "
            f"anticipates the challenge above (if any). "
            f"Do not restate the original. Do not speculate about who "
            f"deposited it or why. One or two sentences.\n\n"
            f"{type_parent_instruction()}\n"
            f"DEVELOPMENT:"
        )
