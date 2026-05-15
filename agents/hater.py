"""Hater — challenges the consensus cluster.

The hater does NOT sample individual signals to attack one at a time.
That would be too local — and it would let the hater anchor on whatever
single signal it happened to draw, which is a stigmergic violation
disguised as adversarial pressure.

Instead, the hater receives a *consensus summary* from the signal store:
how many INITIAL signals exist, what their average strength is, and a
small set of representatives. This is a distributional view of the
pheromone field — the gradient, not the individual deposits' minds.
The hater is then asked to find a structural weakness that applies to
the cluster as a whole.

This is the cleanest expression of stigmergic-but-adversarial pressure:
the hater conditions on what the colony has been doing in aggregate,
not on what any one ant was thinking.
"""

from __future__ import annotations

from agents.base import BaseAgent
from core.signal_store import SignalStore, Signal
from core.signal_types import INITIAL, OBJECTION
from core.config import MAX_TOKENS_HATER


class Hater(BaseAgent):
    ROLE = "hater"
    OUTPUT_TYPE = OBJECTION
    INPUT_TYPE = INITIAL
    MAX_TOKENS = MAX_TOKENS_HATER
    TEMPERATURE = 0.85
    DEFAULT_DEPOSIT_STRENGTH = 0.6
    # Cap at 1 OBJECTION per round per hater. Two haters → 2 OBJECTIONs/round max.
    # Without this cap both haters burn all 8 iterations producing paraphrased dissent.
    MAX_DEPOSITS_PER_ROUND = 1

    def __init__(self, agent_id: str, llm, task_prompt: str, target_type: str = INITIAL):
        super().__init__(agent_id, llm)
        self.task_prompt = task_prompt
        self.target_type = target_type
        self._last_summary_ids: list[str] = []

    def sample(self, store: SignalStore) -> list[Signal]:
        # consume a consensus snapshot, not individual sampled signals
        summary = store.consensus_summary(self.target_type, k=3)
        rep_signals = [store.get(r["id"]) for r in summary["representatives"]]
        rep_signals = [s for s in rep_signals if s is not None]
        self._last_summary_ids = [s.id for s in rep_signals]
        # surface the representatives so they're recorded as consumed,
        # but don't render them as parent context — the prompt uses the
        # summary, not the individual signals' depositor metadata
        return rep_signals

    def build_prompt(self, samples: list[Signal], *,
                     store_count: int = 0, own_ids: tuple = ()) -> str:
        if not samples:
            return (
                f"TASK: {self.task_prompt}\n\n"
                f"No consensus has formed yet. Skip.\n\n"
                f"OBJECTION: (none)"
            )
        count_hint = (
            f"There are currently {store_count} objections in the store. "
            f"Produce a structurally distinct adversarial challenge.\n"
        )
        rep_lines = "\n".join(
            f"  - [{s.id}] strength={s.strength:.2f}: {s.content}"
            for s in samples
        )
        return (
            f"TASK: {self.task_prompt}\n\n"
            f"{count_hint}"
            f"You see a consensus cluster forming in the shared signal "
            f"store. The cluster is summarized by {len(samples)} "
            f"representative signals:\n\n"
            f"{rep_lines}\n\n"
            f"Find a structural weakness that applies to the CLUSTER AS "
            f"A WHOLE — not to any individual signal. What shared "
            f"assumption do these signals make that might be wrong? "
            f"What kind of evidence would they all collectively miss? "
            f"One or two sentences.\n\n"
            f"OBJECTION:"
        )

    def parent_id_for_deposit(self, samples: list[Signal]):
        # link to the strongest representative as a structural anchor
        if not samples:
            return None
        return max(samples, key=lambda s: s.strength).id
