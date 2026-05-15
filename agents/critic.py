"""Critic — evaluates a sampled signal as an artifact.

A critic samples one signal, treats it as a finished artifact (no
ancestry, no rendered chain, no awareness of the depositor), and
produces a CRITIQUE that includes a quality score in [0, 1]. The score
is parsed from the model's output and used as the deposit strength —
strong critiques (high quality assessment) deposit at higher strength.

Critics also adjust the strength of the signal they critiqued, via
amplify-or-decay based on the score, so well-evaluated signals rise and
poorly-evaluated ones decay faster.
"""

from __future__ import annotations

import re

from agents.base import BaseAgent
from core.signal_store import SignalStore, Signal
from core.signal_types import INITIAL, CRITIQUE
from core.config import MAX_TOKENS_CRITIC
from core.sampling import SamplingStrategy


_SCORE_RE = re.compile(r"score\s*[:=]\s*([+-]?(?:\d+\.?\d*|\.\d+))", re.IGNORECASE)


class Critic(BaseAgent):
    ROLE = "critic"
    OUTPUT_TYPE = CRITIQUE
    INPUT_TYPE = INITIAL
    MAX_TOKENS = MAX_TOKENS_CRITIC
    TEMPERATURE = 0.55
    DEFAULT_DEPOSIT_STRENGTH = 0.5

    def __init__(self, agent_id: str, llm, strategy: SamplingStrategy,
                 strategy_name: str, task_prompt: str):
        super().__init__(agent_id, llm)
        self.strategy = strategy
        self.strategy_name = strategy_name
        self.task_prompt = task_prompt

    def sample(self, store: SignalStore) -> list[Signal]:
        return self.strategy(store, self.INPUT_TYPE, 1)

    def build_prompt(self, samples: list[Signal], *,
                     store_count: int = 0, own_ids: tuple = ()) -> str:
        count_hint = (
            f"There are currently {store_count} CRITIQUE signals in the store. "
            f"Produce something structurally distinct from them.\n"
        )
        own_hint = ""
        if own_ids:
            own_hint = (
                f"You have already deposited {len(own_ids)} critique(s) this round "
                f"(IDs: {', '.join(own_ids)}). Do not restate them.\n"
            )
        if not samples:
            return (
                f"TASK: {self.task_prompt}\n\n"
                f"{count_hint}{own_hint}"
                f"No signals to critique yet. Skip.\n\n"
                f"CRITIQUE:\nSCORE: 0.0"
            )
        s = samples[0]
        return (
            f"TASK: {self.task_prompt}\n\n"
            f"{count_hint}{own_hint}"
            f"Evaluate the following deposited artifact on its merits. "
            f"You do not know who deposited it or what reasoning produced "
            f"it. Treat it as a finished claim.\n\n"
            f"---ARTIFACT [{s.id}]---\n{s.content}\n---END ARTIFACT---\n\n"
            f"Write a brief critique (one or two sentences) and assign "
            f"a quality score in [0, 1] reflecting how well the artifact "
            f"stands as a claim. Format your response exactly as:\n\n"
            f"CRITIQUE: <your critique>\n"
            f"SCORE: <number between 0 and 1>"
        )

    def parse(self, raw: str) -> tuple[str, float]:
        text = raw.strip()
        score = 0.5
        m = _SCORE_RE.search(text)
        if m:
            try:
                score = max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                pass
        # strength reflects the critique's assertive force, NOT its valence.
        # Critiques with confident scores (near 0 or near 1) deposit stronger.
        strength = 0.3 + 0.4 * abs(score - 0.5) * 2  # 0.3..0.7
        return text, strength

    # run() uses base implementation (no special mutation of parent strength)
