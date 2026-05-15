"""Scout — conditioned on a corpus partition only.

A scout is the only agent type that conditions on raw evidence. It receives
a ScoutPartition (an immutable slice of the corpus, never overlapping with
other scouts' slices) and deposits INITIAL signals derived from that slice.

A scout never reads the signal store. It cannot be influenced by what
other agents have already proposed, which is the source of its
independence.

Saturation cap (Phase 1B)
-------------------------
A scout's job is producing distinct INITIAL hypotheses from its partition.
Once SCOUT_MAX_DEPOSITS_PER_ROUND distinct INITIALs exist from this scout,
it stops — regardless of how many iterations remain. The ceiling
ITERATIONS_PER_ROUND still applies as an absolute cap.

Re-seed (Phase 1B)
------------------
When a scout has already made at least one successful deposit this round,
the next iteration's prompt includes a one-line excerpt of the most recent
deposit: "Previously you contributed: '...'. Produce something genuinely
different from that." This is intra-agent memory (own outputs only — no
other agents' content), which is safe under the no-leak rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agents.base import BaseAgent, AgentRunStats, strip_reasoning
from core.signal_store import SignalStore, Signal
from core.signal_types import INITIAL
from core.intake import ScoutPartition
from core.config import MAX_TOKENS_SCOUT, SCOUT_MAX_DEPOSITS_PER_ROUND, SCOUT_RESEED_CHARS
from core.diversity import AgentContextRecord
from core.filters import is_junk_output


@dataclass
class ScoutConfig:
    task_prompt: str
    partition: ScoutPartition


class Scout(BaseAgent):
    ROLE = "scout"
    OUTPUT_TYPE = INITIAL
    INPUT_TYPE = None
    MAX_TOKENS = MAX_TOKENS_SCOUT
    TEMPERATURE = 0.9
    DEFAULT_DEPOSIT_STRENGTH = 0.55

    def __init__(self, agent_id: str, llm, config: ScoutConfig):
        super().__init__(agent_id, llm)
        self.config = config

    # ---- run loop override (scouts have a custom saturation cap + re-seed) --

    async def run(self, store: SignalStore, iterations: int) -> AgentRunStats:
        stats = AgentRunStats(context_record=AgentContextRecord(
            agent_id=self.agent_id, role=self.ROLE,
        ))
        # Scouts' "context" is their corpus partition, recorded once for the
        # round. Per-iteration variation comes from temperature + chunk rotation.
        stats.context_record.add_chunks(self.config.partition.chunk_ids)
        n_chunks = max(1, len(self.config.partition.chunks))

        # Tracks content excerpts from own successful deposits this round, used
        # to build the re-seed hint for the next iteration.
        own_deposit_excerpts: list[str] = []
        consecutive_dups = 0

        for iter_idx in range(iterations):
            # Saturation cap: stop when we've hit SCOUT_MAX_DEPOSITS_PER_ROUND.
            # The outer ceiling (ITERATIONS_PER_ROUND) still applies.
            if len(own_deposit_excerpts) >= SCOUT_MAX_DEPOSITS_PER_ROUND:
                break

            stats.iterations += 1
            chunk_offset = iter_idx % n_chunks

            # Re-seed: give this iteration a nudge away from the most recent
            # deposit. Own content only — no other agents' work is injected.
            prior_own = (
                own_deposit_excerpts[-1][:SCOUT_RESEED_CHARS]
                if own_deposit_excerpts else None
            )
            prompt = self.build_prompt(
                samples=[], chunk_offset=chunk_offset, prior_own_content=prior_own
            )
            self._assert_no_leak(prompt, samples=[])

            raw = await self.llm.generate(
                prompt,
                role=self.ROLE,
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
            )
            content = strip_reasoning(raw.strip())
            if not content:
                continue

            # Junk filter: block first-person scratchpad before deposit.
            if is_junk_output(content):
                consecutive_dups += 1
                stats.rejected_dup += 1
                if consecutive_dups >= 3:
                    break
                continue

            sid = store.deposit(
                signal_type=self.OUTPUT_TYPE,
                content=content,
                strength=self.DEFAULT_DEPOSIT_STRENGTH,
                depositor=self.ROLE,
                parent_id=None,
                metadata={
                    "scout_agent_id": self.agent_id,
                    "depositor_agent_id": self.agent_id,
                    "chunk_ids": self.config.partition.chunk_ids,
                },
            )
            if sid is None:
                consecutive_dups += 1
                stats.rejected_dup += 1
                if consecutive_dups >= 3:
                    break
            else:
                consecutive_dups = 0
                stats.deposits += 1
                own_deposit_excerpts.append(content)

        return stats

    # ---- prompt ----------------------------------------------------------

    def sample(self, store: SignalStore) -> list[Signal]:
        return []

    def build_prompt(self, samples: list[Signal], *,
                     store_count: int = 0, own_ids: tuple = (),
                     chunk_offset: int = 0,
                     prior_own_content: Optional[str] = None) -> str:
        partition_text = self.config.partition.render(offset=chunk_offset)
        if not partition_text:
            partition_text = "(no corpus partition assigned to this scout)"

        reseed_hint = ""
        if prior_own_content:
            excerpt = prior_own_content.replace("\n", " ").strip()
            reseed_hint = (
                f'\nYou previously contributed: "{excerpt}"\n'
                f"Produce something genuinely different from that.\n"
            )

        return (
            f"TASK: {self.config.task_prompt}\n\n"
            f"You have been assigned the following evidence partition. "
            f"Other scouts have been assigned different partitions and "
            f"you cannot see theirs.\n\n"
            f"---EVIDENCE---\n{partition_text}\n---END EVIDENCE---\n"
            f"{reseed_hint}\n"
            f"Produce ONE concise initial claim or observation grounded "
            f"in this partition. Do not summarize the entire partition; "
            f"surface a single specific point worth depositing as a "
            f"first-order signal. One or two sentences only.\n\n"
            f"CLAIM:"
        )
