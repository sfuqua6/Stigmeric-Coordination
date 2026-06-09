"""Critic — evaluates a sampled signal as an artifact.

A critic samples one signal, treats it as a finished artifact (no
ancestry, no rendered chain, no awareness of the depositor), and
produces either a CRITIQUE_POSITIVE or CRITIQUE_NEGATIVE signal.

Valence-aware deposit (§5 directive)
--------------------------------------
The CRITIQUE signal type was split in §5 to fix the valence-collapse bug:
previously, a high-quality endorsement and a damning rejection both deposited
as CRITIQUE with identical structural weight in the projection.

    score >= 0.5  → CRITIQUE_POSITIVE (goes to support_set in projection)
    score <  0.5  → CRITIQUE_NEGATIVE (goes to dissent_set in projection)

Deposit strength is the raw score directly (preserves direction). This means:
    - A score of 0.9 → CRITIQUE_POSITIVE at strength 0.9 (strong endorsement)
    - A score of 0.1 → CRITIQUE_NEGATIVE at strength 0.1 (strong rejection)
    - A score of 0.5 → CRITIQUE_POSITIVE at strength 0.5 (neutral endorsement)

The prior formula (0.3 + 0.4 * |score - 0.5| * 2) collapsed direction into
magnitude only, so a 0.1 and a 0.9 produced equal-strength CRITIQUEs.
"""

from __future__ import annotations

import re

from agents.base import BaseAgent, type_parent_instruction
from core.signal_store import SignalStore, Signal
from core.signal_types import INITIAL, CRITIQUE_POSITIVE, CRITIQUE_NEGATIVE
from core.config import MAX_TOKENS_CRITIC
from core.sampling import SamplingStrategy


_SCORE_RE = re.compile(r"score\s*[:=]\s*([+-]?(?:\d+\.?\d*|\.\d+))", re.IGNORECASE)


class Critic(BaseAgent):
    ROLE = "critic"
    # OUTPUT_TYPE is determined at parse-time from score; set to CRITIQUE_NEGATIVE
    # as the default (overridden per-deposit in the run loop via parse()).
    OUTPUT_TYPE = CRITIQUE_NEGATIVE
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
        # Tracks the valence of the most recent parse() call so run() can
        # use the correct deposit type. Reset at each parse() call.
        self._last_deposit_type: str = CRITIQUE_NEGATIVE

    def sample(self, store: SignalStore) -> list[Signal]:
        return self.strategy(store, self.INPUT_TYPE, 1)

    def build_prompt(self, samples: list[Signal], *,
                     store_count: int = 0, own_ids: tuple = ()) -> str:
        count_hint = (
            f"There are currently {store_count} critique signals in the store. "
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
            f"stands as a claim. A score >= 0.5 means the claim is well-formed "
            f"(positive critique); < 0.5 means it has significant weaknesses "
            f"(negative critique). Format your response exactly as:\n\n"
            f"CRITIQUE: <your critique>\n"
            f"SCORE: <number between 0 and 1>\n\n"
            f"{type_parent_instruction()}"
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
        # §5: deposit type is determined by valence (direction), not magnitude.
        # Strength is the raw score — preserves both direction and magnitude.
        if score >= 0.5:
            self._last_deposit_type = CRITIQUE_POSITIVE
        else:
            self._last_deposit_type = CRITIQUE_NEGATIVE
        return text, score

    async def run(self, store: SignalStore, iterations: int):
        """Override run to use the valence-determined OUTPUT_TYPE per deposit."""
        from core.diversity import AgentContextRecord
        from agents.base import (
            AgentRunStats, strip_reasoning,
            parse_type_proposal, parse_parent_proposal, _strip_type_parent_lines,
        )
        from core.filters import is_junk_output

        stats = AgentRunStats(context_record=AgentContextRecord(
            agent_id=self.agent_id, role=self.ROLE,
        ))
        consecutive_dups = 0
        own_ids: list[str] = []
        own_embeddings: list[list[float]] = []

        for _ in range(iterations):
            if self.MAX_DEPOSITS_PER_ROUND is not None and stats.deposits >= self.MAX_DEPOSITS_PER_ROUND:
                break

            stats.iterations += 1
            samples = self.sample(store)
            stats.context_record.add_signals([s.id for s in samples])

            store_count = len(store.by_type(CRITIQUE_POSITIVE)) + len(store.by_type(CRITIQUE_NEGATIVE))
            prompt = self.build_prompt(samples, store_count=store_count, own_ids=tuple(own_ids))
            self._assert_no_leak(prompt, samples)

            raw = await self.llm.generate(
                prompt, role=self.ROLE,
                max_tokens=self.MAX_TOKENS, temperature=self.TEMPERATURE,
            )

            content, strength = self.parse(raw)
            content = strip_reasoning(content)
            if not content or is_junk_output(content):
                stats.rejected_dup += 1
                consecutive_dups += 1
                if consecutive_dups >= 3:
                    break
                continue

            # Phase 3A/3B: dynamic TYPE / PARENT — overrides valence-based
            # detection. Fall back to score-derived self._last_deposit_type
            # and the strongest-sample parent when not parseable.
            dyn_type = parse_type_proposal(raw)
            dyn_parent = parse_parent_proposal(raw, store)
            effective_type = dyn_type if dyn_type is not None else self._last_deposit_type
            if dyn_parent == "__NONE__":
                effective_parent = None
            elif dyn_parent is not None:
                effective_parent = dyn_parent
            else:
                effective_parent = self.parent_id_for_deposit(samples)
            content = _strip_type_parent_lines(content)
            if not content:
                continue

            deposit_meta = {
                "depositor_agent_id": self.agent_id,
                "score": round(strength, 4),
            }
            if dyn_type is not None:
                deposit_meta["proposed_type"] = dyn_type
            if dyn_parent is not None:
                deposit_meta["proposed_parent"] = (
                    None if dyn_parent == "__NONE__" else dyn_parent
                )

            sid = store.deposit(
                signal_type=effective_type,
                content=content,
                strength=strength,
                depositor=self.ROLE,
                parent_id=effective_parent,
                metadata=deposit_meta,
            )
            if sid is None:
                stats.rejected_dup += 1
                consecutive_dups += 1
                if consecutive_dups >= 3:
                    break
            else:
                consecutive_dups = 0
                stats.deposits += 1
                own_ids.append(sid)

                emb = store.get_embedding(sid)
                if emb is not None:
                    if own_embeddings:
                        n = len(emb)
                        centroid = [
                            sum(own_embeddings[j][i] for j in range(len(own_embeddings))) / len(own_embeddings)
                            for i in range(n)
                        ]
                        dot = sum(emb[i] * centroid[i] for i in range(n))
                        novelty = 1.0 - max(-1.0, min(1.0, dot))
                    else:
                        novelty = 1.0
                    stats.novelty_per_iter.append(round(novelty, 4))
                    own_embeddings.append(emb)

        return stats
