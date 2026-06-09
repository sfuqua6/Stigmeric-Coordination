"""FacetPlanner — generate N semantically distinct query facets per run.

Prevents scout divergence collapse by biasing each scout's retrieval
toward a different aspect of the thesis (empirical, historical, economic,
ethical, etc.). Each facet becomes the retrieval query for one partition
of the corpus.

The fast model is used (low-stakes, short output) so facet generation
does not eat into the per-round token budget. Falls back to deterministic
phrasings when the LLM is unavailable (mock mode, heterogeneous router
with no fast-path LLM, or on generation failure).
"""

from __future__ import annotations

import re
from typing import Optional

from .config import N_FACETS, USE_MOCK_LLM


_ANGLE_SUFFIXES = [
    "empirical evidence and data",
    "historical examples and precedents",
    "economic costs and incentives",
    "ethical implications and rights",
    "policy recommendations",
    "counterarguments and criticism",
    "case studies",
    "expert and stakeholder perspectives",
]


class FacetPlanner:
    """Generate N semantically distinct query facets for a thesis.

    Usage (from run_swarm.py, inside an async context):
        planner = FacetPlanner(llm=llm, task_type=task_type)
        facets = await planner.generate(user_prompt, n=N_FACETS)

    Returns a list of strings; each is a search query suitable for
    passing to search_tool.search().
    """

    def __init__(self, llm=None, task_type: str = "debate") -> None:
        self.llm = llm
        self.task_type = task_type

    async def generate(self, thesis: str, n: int = N_FACETS) -> list[str]:
        """Generate n facets. LLM path with deterministic fallback."""
        if USE_MOCK_LLM or self.llm is None:
            return self._fallback(thesis, n)
        try:
            facets = await self._llm_generate(thesis, n)
            return facets
        except Exception as exc:
            print(f"[FACET] generation failed ({type(exc).__name__}: {exc}), using fallback")
            return self._fallback(thesis, n)

    async def _llm_generate(self, thesis: str, n: int) -> list[str]:
        prompt = (
            f"Generate {n} distinct search queries to research different angles of:\n"
            f"  {thesis}\n\n"
            f"Cover different dimensions: empirical evidence, historical context, "
            f"economic analysis, ethical dimensions, policy implications, "
            f"counterarguments.\n"
            f"Output exactly {n} queries, one per line, no numbering or bullets.\n\n"
            f"QUERIES:"
        )
        raw = await self.llm.generate(
            prompt, role="fast", max_tokens=250, temperature=0.4
        )
        lines = [
            l.strip().lstrip("0123456789.-) ")
            for l in raw.split("\n")
            if l.strip() and not l.strip().startswith("#")
        ]
        lines = [l for l in lines if len(l) > 5][:n]
        for line in lines:
            print(f"[FACET] llm: {line!r}")

        if len(lines) < n:
            lines.extend(self._fallback(thesis, n - len(lines)))
        return lines[:n]

    def _fallback(self, thesis: str, n: int) -> list[str]:
        """Deterministic facets keyed on thesis + angle suffix."""
        facets = []
        for suffix in _ANGLE_SUFFIXES[:n]:
            facet = f"{thesis} — {suffix}"
            print(f"[FACET] fallback: {facet!r}")
            facets.append(facet)
        # Pad with bare thesis if angle list exhausted
        while len(facets) < n:
            facets.append(thesis)
        return facets[:n]

    @staticmethod
    def facet_slug(facet: str, max_len: int = 20) -> str:
        """Produce a short filesystem-safe slug from a facet string.

        Used as the suffix in "web_<slug>" partition IDs.
        """
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", facet.lower().strip())
        slug = slug.strip("_")[:max_len]
        return slug or "facet"
