"""Dated peer/sector relationship graph — the seed of the "community of experts".

Captures typed, DATED relationships between tickers so the swarm can reason
about peers ("Walmart and Dollar General are both grocery stores"). Edges carry
`learned_on` so every query is point-in-time: a run as of date T only sees
relationships learned on or before T. Without this, the graph would leak future
knowledge into a historical backtest (the same hazard as the KB — see
STOCK_SWARM_POC_PROMPT.md Part 3.6 / Stage 7).

This is a standalone module (it does NOT touch core/knowledge_base.py). Stage 7
can later fold these edges into the KB schema; for now the graph is independently
loadable/saveable and consumed by core/stock_compare.py.

Edge types:
    peer_of, same_sector, competitor_of, substitute_for   — undirected
    supplier_of                                            — directed (a supplies b)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

_UNDIRECTED = {"peer_of", "same_sector", "competitor_of", "substitute_for"}
_DIRECTED = {"supplier_of"}
_VALID_TYPES = _UNDIRECTED | _DIRECTED


@dataclass(frozen=True)
class RelationshipEdge:
    a: str
    b: str
    type: str
    label: str = ""          # e.g. "grocery" for a same_sector edge
    learned_on: Optional[date] = None
    source: str = ""

    def visible_at(self, as_of: Optional[date]) -> bool:
        if as_of is None or self.learned_on is None:
            return True
        return self.learned_on <= as_of


class RelationshipGraph:
    """A small typed, dated relationship graph over tickers.

    Queries scan the edge list (graphs are small). Every accessor takes an
    optional `as_of` and filters to edges visible at that date.
    """

    def __init__(self) -> None:
        self._edges: list[RelationshipEdge] = []

    # ---- mutation -------------------------------------------------------
    def add(self, a: str, b: str, type: str, *, label: str = "",
            learned_on: Optional[date] = None, source: str = "") -> None:
        if type not in _VALID_TYPES:
            raise ValueError(f"unknown relationship type {type!r}; "
                             f"valid: {sorted(_VALID_TYPES)}")
        self._edges.append(RelationshipEdge(
            a=a.upper(), b=b.upper(), type=type, label=label,
            learned_on=learned_on, source=source))

    # ---- queries (all point-in-time) ------------------------------------
    def _edges_visible(self, as_of: Optional[date]):
        return [e for e in self._edges if e.visible_at(as_of)]

    def peers_of(self, ticker: str, *, as_of: Optional[date] = None,
                 types: Optional[set] = None) -> list[str]:
        """Neighbours of `ticker` (undirected types + supplier_of in either
        direction), visible at `as_of`, optionally filtered to `types`."""
        t = ticker.upper()
        out: list[str] = []
        for e in self._edges_visible(as_of):
            if types is not None and e.type not in types:
                continue
            if e.a == t and e.b != t:
                out.append(e.b)
            elif e.b == t and e.a != t and (e.type in _UNDIRECTED or e.type in _DIRECTED):
                out.append(e.a)
        # dedupe preserving order
        seen: set = set()
        return [x for x in out if not (x in seen or seen.add(x))]

    def relationship_between(self, a: str, b: str, *,
                             as_of: Optional[date] = None) -> Optional[dict]:
        """The strongest relationship visible between a and b at as_of, or None.

        Returns {"type", "label"}. Undirected edges match either ordering;
        supplier_of matches a->b only."""
        a, b = a.upper(), b.upper()
        for e in self._edges_visible(as_of):
            if e.type in _UNDIRECTED and {e.a, e.b} == {a, b}:
                return {"type": e.type, "label": e.label}
            if e.type in _DIRECTED and e.a == a and e.b == b:
                return {"type": e.type, "label": e.label}
        return None

    def sector_of(self, ticker: str, *, as_of: Optional[date] = None) -> Optional[str]:
        """A sector label for `ticker` from any same_sector edge visible at as_of."""
        t = ticker.upper()
        for e in self._edges_visible(as_of):
            if e.type == "same_sector" and e.label and t in (e.a, e.b):
                return e.label
        return None

    # ---- persistence ----------------------------------------------------
    def to_list(self) -> list[dict]:
        return [
            {"a": e.a, "b": e.b, "type": e.type, "label": e.label,
             "learned_on": e.learned_on.isoformat() if e.learned_on else None,
             "source": e.source}
            for e in self._edges
        ]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_list(), indent=2), encoding="utf-8")

    @classmethod
    def from_list(cls, rows: list[dict]) -> "RelationshipGraph":
        g = cls()
        for r in rows:
            lo = r.get("learned_on")
            g.add(r["a"], r["b"], r["type"], label=r.get("label", ""),
                  learned_on=(datetime.strptime(lo[:10], "%Y-%m-%d").date() if lo else None),
                  source=r.get("source", ""))
        return g

    @classmethod
    def load(cls, path: str | Path) -> "RelationshipGraph":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_list(json.loads(p.read_text(encoding="utf-8")))
