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

from dataclasses import dataclass, field
from typing import Optional, Tuple

from agents.base import BaseAgent, AgentRunStats, strip_reasoning
from core.signal_store import SignalStore, Signal
from core.signal_types import INITIAL, SEARCH
from core.intake import ScoutPartition
from core.config import MAX_TOKENS_SCOUT, SCOUT_MAX_DEPOSITS_PER_ROUND, SCOUT_RESEED_CHARS
from core.diversity import AgentContextRecord
from core.filters import is_junk_output


@dataclass
class ScoutConfig:
    task_prompt: str
    partition: ScoutPartition
    # Phase 2C: when True (default), scouts issue agentic queries via
    # core/search_tool.py and deposit a SEARCH signal per iteration. Set to
    # False (--corpus=placeholder regression mode) to fall back to the
    # contiguous-partition path.
    use_search: bool = True
    # The user prompt that scouts query around. Falls back to task_prompt
    # when not supplied.
    user_prompt: str = ""
    # Topology cell assignment from core/topology.assign_topology_cells().
    # When set, the scout's prompt includes a directional hint ("your assigned
    # region of the answer space is ...") and deposits carry
    # metadata["topology_coords"] for coverage tracking.
    topology_cell: Optional[tuple] = None
    # Human-readable description of the topology cell (pre-formatted).
    topology_cell_desc: str = ""


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
        # Scouts' "context" is their corpus partition (when partitioned) or
        # their query history (when agentic search is on).
        stats.context_record.add_chunks(self.config.partition.chunk_ids)
        n_chunks = max(1, len(self.config.partition.chunks))

        own_deposit_excerpts: list[str] = []
        consecutive_dups = 0

        # Per-scout query history (so re-seed can avoid repeating queries).
        prior_queries: list[str] = []
        # Last retrieved chunks for the current iteration's INITIAL prompt.
        last_retrieved: list = []

        for iter_idx in range(iterations):
            if len(own_deposit_excerpts) >= SCOUT_MAX_DEPOSITS_PER_ROUND:
                break

            stats.iterations += 1
            chunk_offset = iter_idx % n_chunks

            # Phase 2C: agentic search path — generate a query, retrieve,
            # deposit a SEARCH signal, then condition the INITIAL on retrieved
            # chunks. Fall through to the partition path on failure or when
            # use_search is False.
            retrieved = []
            if self.config.use_search:
                query = self._compose_query(iter_idx, prior_queries)
                if query:
                    try:
                        from core.search_tool import search as _search, summarize_for_signal
                        retrieved = _search(query, max_results=8,
                                            task_type=getattr(self, "task_type", None))
                    except Exception as exc:
                        print(f"[scout {self.agent_id}] search failed: "
                              f"{type(exc).__name__}: {exc}")
                        retrieved = []
                    if retrieved:
                        # Deposit SEARCH signal so other scouts/foragers see
                        # what's been queried.
                        search_content = summarize_for_signal(query, retrieved)
                        store.deposit(
                            signal_type=SEARCH,
                            content=search_content,
                            strength=0.4,
                            depositor=self.ROLE,
                            parent_id=None,
                            metadata={
                                "scout_agent_id": self.agent_id,
                                "depositor_agent_id": self.agent_id,
                                "query": query,
                                "n_results": len(retrieved),
                                "partition_id": self.config.partition.partition_id,
                            },
                        )
                        prior_queries.append(query)
                        last_retrieved = retrieved

            prior_own = (
                own_deposit_excerpts[-1][:SCOUT_RESEED_CHARS]
                if own_deposit_excerpts else None
            )
            # Topology cell hint: inject directional context without leaking
            # other agents' content (safe under no-leak rule — this is task
            # structure, not a signal's reasoning chain).
            topo_hint = None
            if self.config.topology_cell and self.config.topology_cell_desc:
                topo_hint = (
                    f"Your assigned region of the answer space is: "
                    f"{self.config.topology_cell_desc}. "
                    f"Generate a claim that lives in this region."
                )
            prompt = self.build_prompt(
                samples=[], chunk_offset=chunk_offset,
                prior_own_content=prior_own,
                retrieved_chunks=last_retrieved,
                extra_context=topo_hint,
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

            if is_junk_output(content):
                consecutive_dups += 1
                stats.rejected_dup += 1
                if consecutive_dups >= 3:
                    break
                continue

            # Track retrieved source URLs as chunk_ids for this deposit (in
            # search mode there is no partition; provenance is the URL set).
            source_chunk_ids = (
                [c.chunk_id for c in last_retrieved]
                if last_retrieved else self.config.partition.chunk_ids
            )
            deposit_meta = {
                "scout_agent_id": self.agent_id,
                "depositor_agent_id": self.agent_id,
                "chunk_ids": source_chunk_ids,
                "query": prior_queries[-1] if prior_queries else "",
                "partition_id": self.config.partition.partition_id,
            }
            if self.config.topology_cell is not None:
                deposit_meta["topology_coords"] = self.config.topology_cell
            sid = store.deposit(
                signal_type=self.OUTPUT_TYPE,
                content=content,
                strength=self.DEFAULT_DEPOSIT_STRENGTH,
                depositor=self.ROLE,
                parent_id=None,
                metadata=deposit_meta,
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

    # ---- query composition ---------------------------------------------------

    def _compose_query(self, iter_idx: int, prior_queries: list[str]) -> str:
        """Build a query phrasing for this scout iteration.

        Each scout starts from the user prompt and rotates through a small
        set of phrasings to bias toward different framings. The agent_id
        carries the round number and scout index so the phrasings vary by
        position naturally.
        """
        base = (self.config.user_prompt or self.config.task_prompt).strip()
        if not base:
            return ""
        # Lightweight phrasing rotation keyed on (scout_index, iter_idx).
        # Keep this deterministic so re-runs cache-hit.
        phrasings = [
            base,
            f"{base} evidence",
            f"{base} arguments",
            f"{base} criticism",
            f"{base} consequences",
            f"counterarguments to {base}",
            f"recent research on {base}",
            f"case studies of {base}",
        ]
        # Try to pick one that hasn't been queried yet by THIS scout.
        already = set(prior_queries)
        seed = iter_idx
        for offset in range(len(phrasings)):
            q = phrasings[(seed + offset) % len(phrasings)]
            if q not in already:
                return q
        return phrasings[seed % len(phrasings)]

    # ---- prompt ----------------------------------------------------------

    def sample(self, store: SignalStore) -> list[Signal]:
        return []

    def build_prompt(self, samples: list[Signal], *,
                     store_count: int = 0, own_ids: tuple = (),
                     chunk_offset: int = 0,
                     prior_own_content: Optional[str] = None,
                     retrieved_chunks: Optional[list] = None,
                     extra_context: Optional[str] = None) -> str:
        # Phase 2C: if scouts ran agentic search, condition on retrieved
        # chunks rather than the contiguous partition. Falls back to the
        # partition path when search returned nothing or is disabled.
        if retrieved_chunks:
            evidence_blocks = []
            for c in retrieved_chunks[:8]:
                tag = c.source_tag[:160]
                body = c.text[:800]
                evidence_blocks.append(f"[{tag}]\n{body}")
            evidence_text = "\n\n".join(evidence_blocks)
            evidence_intro = (
                "You issued an agentic search query and received the following "
                "unverified search leads. Use them to inform a specific claim, "
                "but do not treat them as proven facts. Other scouts have "
                "issued different queries and you cannot see what they retrieved."
            )
        else:
            evidence_text = self.config.partition.render(offset=chunk_offset)
            if not evidence_text:
                evidence_text = "(no corpus partition assigned to this scout)"
            evidence_intro = (
                "You have been assigned the following evidence partition. "
                "Other scouts have been assigned different partitions and "
                "you cannot see theirs."
            )

        reseed_hint = ""
        if prior_own_content:
            excerpt = prior_own_content.replace("\n", " ").strip()
            reseed_hint = (
                f'\nYou previously contributed: "{excerpt}"\n'
                f"Produce something genuinely different from that.\n"
            )

        topology_hint = f"\n{extra_context}\n" if extra_context else ""

        return (
            f"TASK: {self.config.task_prompt}\n\n"
            f"{evidence_intro}\n\n"
            f"---EVIDENCE---\n{evidence_text}\n---END EVIDENCE---\n"
            f"{reseed_hint}"
            f"{topology_hint}\n"
            f"Produce ONE concise initial claim or observation grounded "
            f"in this evidence. Do not summarize everything; surface a "
            f"single specific point worth depositing as a first-order "
            f"signal. One or two sentences only.\n\n"
            f"CLAIM:"
        )
