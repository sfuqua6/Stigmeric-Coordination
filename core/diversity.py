"""Diversity metrics over agent context sets.

The leading indicator of whether information partitioning is working.

After each round, every agent reports the set of (chunk_ids, signal_ids)
it actually consumed during that round. We compute pairwise Jaccard
distance between these sets:

    d(A, B) = 1 - |A ∩ B| / |A ∪ B|

Distance near 1 means agents observed structurally disjoint information
during the round — the partition held. Distance near 0 means agents
were converging on the same evidence, which means the partition is
collapsing or the population is too small for the strategy library.

We aggregate to a single round-level scalar by averaging pairwise
distances within each role.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentContextRecord:
    """What one agent saw during one round."""
    agent_id: str
    role: str
    items: set[str] = field(default_factory=set)

    def add_chunks(self, chunk_ids: list[str]) -> None:
        self.items.update(chunk_ids)

    def add_signals(self, signal_ids: list[str]) -> None:
        self.items.update(signal_ids)


def jaccard_distance(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    inter = a & b
    return 1.0 - (len(inter) / len(union)) if union else 0.0


def role_diversity(records: list[AgentContextRecord]) -> dict[str, float]:
    """Average pairwise Jaccard distance, broken down by role.

    Returns a dict {role: avg_pairwise_distance}. A role with only one
    agent gets a distance of 0.0 (no comparison possible), which is
    correctly diagnostic — that agent's view is its own.
    """
    by_role: dict[str, list[AgentContextRecord]] = {}
    for r in records:
        by_role.setdefault(r.role, []).append(r)

    result: dict[str, float] = {}
    for role, recs in by_role.items():
        if len(recs) < 2:
            result[role] = 0.0
            continue
        dists = []
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                dists.append(jaccard_distance(recs[i].items, recs[j].items))
        result[role] = sum(dists) / len(dists) if dists else 0.0
    return result


def overall_diversity(records: list[AgentContextRecord]) -> float:
    """Average pairwise Jaccard across ALL agents regardless of role.

    Lower than per-role averages when roles see overlapping signals
    (e.g., critics and haters both sampling INITIAL signals). That's
    expected — cross-role overlap is part of the design, not a bug.
    """
    if len(records) < 2:
        return 0.0
    dists = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            dists.append(jaccard_distance(records[i].items, records[j].items))
    return sum(dists) / len(dists) if dists else 0.0


def format_report(records: list[AgentContextRecord]) -> str:
    """Pretty-print a diversity summary suitable for the round log."""
    by_role = role_diversity(records)
    lines = ["Diversity report (Jaccard distance over agent context sets):"]
    for role in sorted(by_role.keys()):
        lines.append(f"  {role:<12} avg_pairwise = {by_role[role]:.3f}")
    lines.append(f"  {'overall':<12} avg_pairwise = {overall_diversity(records):.3f}")
    return "\n".join(lines)
