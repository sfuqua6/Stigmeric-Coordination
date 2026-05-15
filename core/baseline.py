"""Non-stigmergic baseline coordinator (§3 directive).

Runs the same number of independent LLM calls as the stigmergic pipeline but
without any inter-agent communication via the signal store. Every agent sees
the full corpus (no partitioning), calls the LLM once with a direct prompt,
and the results are deduplicated and concatenated into a final answer.

This baseline exists so diversity and quality metrics produced by the
stigmergic pipeline have a point of comparison. If stigmergic numbers don't
beat baseline numbers, the pheromone coordination is not adding value.

# FUTURE-CLAUDE NOTE:
# - Keep BaselineResult field names in sync with summary.json in run_swarm.py.
# - The baseline intentionally uses fewer moving parts than the stigmergic
#   pipeline. Do not add phase ordering, provenance boosts, or logit dynamics
#   here — that would defeat the purpose of having a baseline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BaselineResult:
    """Output of a BaselineCoordinator.run() call."""
    task_type: str
    user_prompt: str
    llm_backend: str
    responses: list[str]           # all unique non-empty responses
    final_answer: str
    n_calls: int                   # total LLM calls made
    n_unique: int                  # deduplicated response count
    elapsed_s: float
    corpus_chars: int = 0
    is_mock: bool = False

    def summary_dict(self) -> dict:
        """Return a dict matching the shape of run_swarm.py summary.json."""
        return {
            "mode": "baseline",
            "task_type": self.task_type,
            "user_prompt": self.user_prompt,
            "llm_backend": self.llm_backend,
            "total_llm_calls": self.n_calls,
            "total_elapsed_s": round(self.elapsed_s, 1),
            "n_unique_responses": self.n_unique,
            # Mirror stigmergic cluster keys with baseline equivalents
            "n_clusters": {
                "surviving": self.n_unique,
                "contested": 0,
                "weakly_supported": 0,
                "rejected_by_field": 0,
            },
            "corpus_chars": self.corpus_chars,
            "is_mock": self.is_mock,
        }


# ---------------------------------------------------------------------------
# String deduplication (no embeddings required)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _jaccard_sim(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def _deduplicate(responses: list[str], sim_threshold: float = 0.6) -> list[str]:
    """Remove near-duplicate responses (greedy, O(n²) but n is tiny here)."""
    unique: list[str] = []
    norms: list[str] = []
    for r in responses:
        n = _normalise(r)
        if not n:
            continue
        if any(_jaccard_sim(n, e) >= sim_threshold for e in norms):
            continue
        unique.append(r)
        norms.append(n)
    return unique


# ---------------------------------------------------------------------------
# Single agent call
# ---------------------------------------------------------------------------

_BASELINE_SYSTEM = (
    "You are one of several independent experts answering the same question. "
    "Give a single clear, evidence-backed claim or insight. "
    "Do not hedge excessively or repeat the question."
)


async def _baseline_call(
    llm,
    task_prompt: str,
    corpus_text: str,
    max_tokens: int = 200,
) -> str:
    """One independent LLM call with optional corpus context."""
    if corpus_text:
        prompt = (
            f"Context:\n{corpus_text[:3000]}\n\n"
            f"Task: {task_prompt}\n\n"
            "Respond with a single well-supported claim or insight:"
        )
    else:
        prompt = f"Task: {task_prompt}\n\nRespond with a single well-supported claim:"
    try:
        return await llm.generate(
            prompt, role="agent", max_tokens=max_tokens, temperature=0.8
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# BaselineCoordinator
# ---------------------------------------------------------------------------

class BaselineCoordinator:
    """Non-stigmergic baseline: N independent LLM calls, no shared store.

    Parameters
    ----------
    n_agents:    Total parallel LLM calls. Should match the stigmergic run's
                 (NUM_SCOUTS + NUM_FORAGERS) for a fair comparison.
    max_tokens:  Per-call token budget.
    """

    def __init__(self, n_agents: int = 8, max_tokens: int = 200) -> None:
        self.n_agents = n_agents
        self.max_tokens = max_tokens

    async def run(
        self,
        task_type: str,
        user_prompt: str,
        task_prompt: str,
        llm,
        corpus_text: str = "",
        output_dir: Optional[Path] = None,
        is_mock: bool = False,
    ) -> BaselineResult:
        t0 = time.time()

        # Fire all N_AGENTS calls concurrently.
        raw = await asyncio.gather(
            *(_baseline_call(llm, task_prompt, corpus_text, self.max_tokens)
              for _ in range(self.n_agents))
        )

        responses = _deduplicate([r.strip() for r in raw if r.strip()])
        n_unique = len(responses)

        # Simple synthesis: numbered list of unique responses.
        if responses:
            final_answer = "\n\n".join(
                f"[{i+1}] {r}" for i, r in enumerate(responses[:6])
            )
        else:
            final_answer = "(baseline produced no non-empty responses)"

        elapsed = time.time() - t0
        result = BaselineResult(
            task_type=task_type,
            user_prompt=user_prompt,
            llm_backend=getattr(llm, "name", "unknown"),
            responses=responses,
            final_answer=final_answer,
            n_calls=self.n_agents,
            n_unique=n_unique,
            elapsed_s=elapsed,
            corpus_chars=len(corpus_text),
            is_mock=is_mock,
        )

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "answer.txt").write_text(final_answer, encoding="utf-8")
            (output_dir / "summary.json").write_text(
                json.dumps(result.summary_dict(), indent=2), encoding="utf-8"
            )
            print(f"[baseline] outputs: {output_dir}")

        return result
