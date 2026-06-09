"""Partition-overlap metrics and output-diversity hooks (§2 directive).

# FUTURE-CLAUDE NOTE — why the Jaccard metric was renamed
# --------------------------------------------------------
# The old "diversity" label was a lie: jaccard_distance over agent context
# SETS measures INPUT-SET OVERLAP (which corpus chunks and signal IDs each
# agent consumed). That is partition health, not output diversity. A run where
# every agent reads completely different chunks can still produce identical
# outputs (low output diversity). Conversely, a run with high partition overlap
# can still produce diverse outputs if the agents reason differently.
#
# The functions below are now named _partition_overlap_* to make this clear.
# They are still worth logging as a diagnostic — if partition overlap is high,
# the corpus is too small or the partition logic broke — but they must NOT be
# cited as evidence of swarm creativity or idea diversity. For that, use
# core/output_diversity.py (centroid cosine distance, Self-BLEU).
#
# The old public names (jaccard_distance, role_diversity, overall_diversity,
# format_report) are preserved as backward-compat aliases below. New callers
# must use the _partition_overlap_* names. Old callers will trigger no errors
# but should be migrated.
#
# --show-partition-overlap in run_swarm.py controls whether the Jaccard report
# is printed. Default: suppressed (not a headline metric).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentContextRecord:
    """What one agent saw and produced during one round."""
    agent_id: str
    role: str
    items: set[str] = field(default_factory=set)
    # Texts of signals this agent successfully deposited this round.
    # Used by core/output_diversity.py — not the Jaccard metric.
    deposit_contents: list[str] = field(default_factory=list)

    def add_chunks(self, chunk_ids: list[str]) -> None:
        self.items.update(chunk_ids)

    def add_signals(self, signal_ids: list[str]) -> None:
        self.items.update(signal_ids)


# ---------------------------------------------------------------------------
# Internal partition-overlap functions (formerly "diversity")
# ---------------------------------------------------------------------------

def _partition_overlap_jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard DISTANCE between two agent context sets.

    Returns 1.0 when disjoint (good partition health), 0.0 when identical.
    This measures INPUT OVERLAP, not output diversity - see module docstring.
    """
    if not a and not b:
        return 0.0
    union = a | b
    inter = a & b
    return 1.0 - (len(inter) / len(union)) if union else 0.0


def _role_partition_overlap(records: list[AgentContextRecord]) -> dict[str, float]:
    """Average pairwise Jaccard distance broken down by role.

    A role with only one agent gets 0.0 (no comparison possible).
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
                dists.append(_partition_overlap_jaccard(recs[i].items, recs[j].items))
        result[role] = sum(dists) / len(dists) if dists else 0.0
    return result


def _overall_partition_overlap(records: list[AgentContextRecord]) -> float:
    """Average pairwise Jaccard across ALL agents regardless of role."""
    if len(records) < 2:
        return 0.0
    dists = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            dists.append(_partition_overlap_jaccard(records[i].items, records[j].items))
    return sum(dists) / len(dists) if dists else 0.0


def format_partition_overlap_report(records: list[AgentContextRecord]) -> str:
    """Pretty-print a partition-overlap summary suitable for the round log."""
    by_role = _role_partition_overlap(records)
    lines = ["Partition overlap (Jaccard over agent context sets — input health, NOT output diversity):"]
    for role in sorted(by_role.keys()):
        lines.append(f"  {role:<12} avg_pairwise = {by_role[role]:.3f}")
    lines.append(f"  {'overall':<12} avg_pairwise = {_overall_partition_overlap(records):.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backward-compat aliases — migrate call-sites to _partition_overlap_* names
# ---------------------------------------------------------------------------

def jaccard_distance(a: set[str], b: set[str]) -> float:
    return _partition_overlap_jaccard(a, b)


def role_diversity(records: list[AgentContextRecord]) -> dict[str, float]:
    return _role_partition_overlap(records)


def overall_diversity(records: list[AgentContextRecord]) -> float:
    return _overall_partition_overlap(records)


def format_report(records: list[AgentContextRecord]) -> str:
    return format_partition_overlap_report(records)
