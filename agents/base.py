"""BaseAgent.

Contract:

    1. An agent's prompt is built from EXACTLY one of:
         - its own corpus partition (Scout only), or
         - signals it sampled from the store (everyone else),
       plus its role instruction. Nothing else.

    2. When rendering a sampled signal into a prompt, only the signal's
       `content` is included. Never `metadata['parent_content']` (the
       cleaned signal store does not store such a field, but enforce it
       at the agent layer too as belt-and-braces). Never any rendered
       walk of the provenance graph. The signal ID may be quoted for
       traceability but is opaque text to the model.

    3. The agent reports a context record for each iteration listing the
       chunk_ids and signal_ids it consumed. The diversity module
       aggregates these per round.

Subclasses implement:

    - build_prompt(self, samples: list[Signal]) -> str
    - parse(self, raw: str) -> tuple[content, strength]
    - sample(self, store) -> list[Signal]   (returns [] for scouts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.signal_store import Signal, SignalStore
from core.diversity import AgentContextRecord
from core.filters import is_junk_output

# ---------------------------------------------------------------------------
# Reasoning-block stripping (P0.2 / R3 / M3)
# ---------------------------------------------------------------------------
import re as _re

# Catches leading scratchpad sentences emitted by reasoning-tuned models:
#   DeepSeek-R1-Distill: "Alright, so I need to...", "Okay, so...", "Hmm, ..."
#   Generic chain-of-thought: "Step 1:", "Let me think", "Reasoning:", etc.
#   "Wait," self-corrections mid-generation.
# Pattern: match one or more consecutive lines that begin with a known marker,
# stopping at the first blank line (double newline) or end of string.
_SCRATCHPAD_RE = _re.compile(
    r"^(?:"
    r"step\s*\d+[:.)]"
    r"|let me(?:\s+think|\s+consider|\s+break|\s+start|\s+work|\s+re-?read|\s+re-?state)?"
    r"|let'?s\s+"
    r"|reasoning:"
    r"|chain of thought:"
    r"|first,?\s+"
    r"|second,?\s+"
    r"|third,?\s+"
    r"|alright,?\s+"
    r"|okay,?\s+"
    r"|ok,?\s+"
    r"|so\s+i\s+need"
    r"|wait,?\s+"
    r"|hmm,?\s+"
    r"|now,?\s+i(?:'ll|'m|\s+need|\s+want|\s+should|\s+will)"
    r").*?(?:\n\n|$)",
    _re.IGNORECASE | _re.MULTILINE,
)

def strip_reasoning(text: str) -> str:
    """Remove agent reasoning artifacts before deposit (preserves the no-leak rule).

    DeepSeek-R1-Distill emits <think>...</think> blocks and also standalone
    </think> tags when reasoning bleeds into the visible answer with no matching
    opener. Handle all three cases:
      1. Paired <think>…</think> — matched and removed.
      2. Orphan </think> — everything before the closing tag is reasoning;
         take only what comes after.
      3. Orphan <think> with no closing tag — strip from the <think> to EOT.
    Then remove any leading scratchpad sentences that slipped through.
    """
    if not text:
        return text
    out = text

    # Case 2: orphan </think> — everything before it is reasoning.
    # Use rsplit so we take the LAST </think> if multiple exist.
    if "</think>" in out:
        out = out.rsplit("</think>", 1)[1]

    # Case 1 & 3: paired <think>…</think> or orphan <think> with no closer.
    # The DOTALL flag makes .* consume newlines so paired blocks are collapsed.
    out = _re.sub(r"<think>.*", "", out, flags=_re.DOTALL | _re.IGNORECASE)

    # Strip leading scratchpad-style sentences (apply repeatedly until stable
    # to handle multi-paragraph preambles).
    prev = None
    while prev != out:
        prev = out
        out = _SCRATCHPAD_RE.sub("", out)
    return out.strip()



# ---------------------------------------------------------------------------

@dataclass
class AgentRunStats:
    deposits: int = 0
    rejected_dup: int = 0
    iterations: int = 0
    context_record: AgentContextRecord = field(default=None)  # type: ignore
    novelty_per_iter: list = field(default_factory=list)  # cosine dist to prior centroid per deposit


class BaseAgent:
    """Abstract agent.

    Concrete subclasses set:
        ROLE: str (used as depositor tag and prompt role)
        OUTPUT_TYPE: str (signal type this agent deposits)
        INPUT_TYPE: Optional[str] (signal type this agent samples; None for scouts)
        MAX_TOKENS: int
        TEMPERATURE: float
        DEFAULT_DEPOSIT_STRENGTH: float
    """

    ROLE: str = "agent"
    OUTPUT_TYPE: str = ""
    INPUT_TYPE: Optional[str] = None
    MAX_TOKENS: int = 120
    TEMPERATURE: float = 0.7
    DEFAULT_DEPOSIT_STRENGTH: float = 0.5
    # Optional cap on successful deposits per round. None = no cap (default).
    # Subclasses set this to limit spending (e.g., Hater = 1, Scout handled inline).
    MAX_DEPOSITS_PER_ROUND: Optional[int] = None

    def __init__(self, agent_id: str, llm):
        self.agent_id = agent_id
        self.llm = llm

    # ---- subclass extension points --------------------------------------

    def sample(self, store: SignalStore) -> list[Signal]:
        """Return the signals this agent will condition on this iteration."""
        raise NotImplementedError

    def build_prompt(self, samples: list[Signal], *,
                     store_count: int = 0, own_ids: tuple = ()) -> str:
        """Render samples + role instruction into a prompt.

        Renderers MUST only use signal.content (and optionally signal.id
        as opaque tag). They must never render parent content, ancestry,
        other agents' reasoning, or anything not present on the Signal
        artifact itself.

        store_count: how many signals of OUTPUT_TYPE currently exist (structural hint).
        own_ids: opaque IDs of signals this agent has already deposited this round
                 (intra-agent memory — safe because it's the agent's own output only).
        """
        raise NotImplementedError

    def parse(self, raw: str) -> tuple[str, float]:
        """Split LLM output into (content, deposit_strength).

        Default: return whole text with default strength. Subclasses
        override if they ask for a strength score in the prompt.
        """
        return raw.strip(), self.DEFAULT_DEPOSIT_STRENGTH

    def parent_id_for_deposit(self, samples: list[Signal]) -> Optional[str]:
        """The parent_id the deposit should link to. Default: first sample."""
        return samples[0].id if samples else None

    # ---- run loop --------------------------------------------------------

    async def run(self, store: SignalStore, iterations: int) -> AgentRunStats:
        stats = AgentRunStats(context_record=AgentContextRecord(
            agent_id=self.agent_id, role=self.ROLE,
        ))
        consecutive_dups = 0
        own_ids: list[str] = []
        own_embeddings: list[list[float]] = []  # for per-iteration novelty metric

        for _ in range(iterations):
            # Deposit cap: stop early if this role has a per-round ceiling and hit it.
            if self.MAX_DEPOSITS_PER_ROUND is not None and stats.deposits >= self.MAX_DEPOSITS_PER_ROUND:
                break

            stats.iterations += 1
            samples = self.sample(store)

            # record what we consumed (for the diversity metric)
            stats.context_record.add_signals([s.id for s in samples])

            store_count = len(store.by_type(self.OUTPUT_TYPE)) if self.OUTPUT_TYPE else 0
            prompt = self.build_prompt(samples, store_count=store_count,
                                       own_ids=tuple(own_ids))
            self._assert_no_leak(prompt, samples)

            raw = await self.llm.generate(
                prompt,
                role=self.ROLE,
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
            )

            content, strength = self.parse(raw)
            content = strip_reasoning(content)
            if not content:
                continue
            if is_junk_output(content):
                stats.rejected_dup += 1
                continue

            # Junk filter: reject first-person scratchpad that slipped past
            # strip_reasoning. Zero cost — one regex check per deposit attempt.
            if is_junk_output(content):
                stats.rejected_dup += 1
                consecutive_dups += 1
                if consecutive_dups >= 3:
                    break
                continue

            sid = store.deposit(
                signal_type=self.OUTPUT_TYPE,
                content=content,
                strength=strength,
                depositor=self.ROLE,
                parent_id=self.parent_id_for_deposit(samples),
                metadata={"depositor_agent_id": self.agent_id},
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

                # per-iteration semantic novelty (cosine dist to centroid of prior deposits)
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
                        novelty = 1.0  # first deposit — maximally novel relative to empty set
                    stats.novelty_per_iter.append(round(novelty, 4))
                    own_embeddings.append(emb)

        return stats

    # ---- enforcement -----------------------------------------------------

    def _assert_no_leak(self, prompt: str, samples: list[Signal]) -> None:
        """Fail loudly if a subclass accidentally smuggles forbidden text
        into the prompt. We can't check everything, but we can catch the
        common mistakes: dialogue threads, parent-content keys, chain-of-
        thought scaffolding.
        """
        forbidden_tokens = (
            "parent_content", "provenance_chain", "chain_of_thought",
            "previous reasoning", "prior reasoning", "agent reasoning",
            "responses:", "dialogue thread",
        )
        lowered = prompt.lower()
        for tok in forbidden_tokens:
            if tok in lowered:
                raise AssertionError(
                    f"[{self.agent_id}] no-leak rule violated: prompt contains "
                    f"forbidden token {tok!r}"
                )
