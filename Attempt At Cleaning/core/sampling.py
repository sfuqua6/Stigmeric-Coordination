"""Differentiated sampling strategies for downstream agents.

P1.8/M8 fix: previously this used a single rotating list with offset arithmetic,
which at the configured population sizes produced strategy *overlap* between
roles (validator and a critic both got recent_only; one foragers' strategy got
skipped entirely). The new design uses explicit per-role strategy lists so
coverage within a role is guaranteed and overlap across roles is intentional.
"""

from __future__ import annotations

from typing import Callable

from .signal_store import SignalStore, Signal
from .signal_types import (
    INITIAL, SUPPORT, VERIFICATION,
    CRITIQUE, CRITIQUE_POSITIVE, CRITIQUE_NEGATIVE, OBJECTION,
)


SamplingStrategy = Callable[[SignalStore, str, int], list[Signal]]


def stratified_extremes(store: SignalStore, signal_type: str, n: int) -> list[Signal]:
    half = max(1, n // 2)
    return store.sample_stratified(signal_type, weak=half, medium=0, strong=n - half)


def medium_only(store: SignalStore, signal_type: str, n: int) -> list[Signal]:
    return store.sample_stratified(signal_type, medium=n)


def recent_only(store: SignalStore, signal_type: str, n: int) -> list[Signal]:
    return store.sample_recent(signal_type, n=n, max_age_s=120.0)


def under_visited_only(store: SignalStore, signal_type: str, n: int) -> list[Signal]:
    return store.sample_under_visited(signal_type, n=n)


def weighted_default(store: SignalStore, signal_type: str, n: int) -> list[Signal]:
    return store.sample_weighted(signal_type, n=n)


def under_supported_clusters(store: SignalStore, signal_type: str, n: int) -> list[Signal]:
    """Sample INITIAL signals whose clusters have the FEWEST SUPPORT children.

    Drives foragers to extend weakly-supported clusters until they cross
    support_diversity >= 2. For non-INITIAL targets, falls back to
    strength-weighted sampling.
    """
    if signal_type != INITIAL:
        return store.sample_weighted(signal_type, n)

    initials = store.by_type(INITIAL)
    if not initials:
        return []

    def support_count(sig: Signal) -> int:
        return sum(
            1 for cid in store.by_parent(sig.id)
            if (c := store.get(cid)) and c.type == SUPPORT
        )

    # Sort ascending — fewest supports first (gap-filling priority).
    initials_sorted = sorted(initials, key=support_count)
    picked = initials_sorted[:n]
    for s in picked:
        s.visits += 1
    return picked


_EXAMINER_TYPES = frozenset({
    CRITIQUE, CRITIQUE_POSITIVE, CRITIQUE_NEGATIVE, OBJECTION, VERIFICATION,
})


def least_examined(store: SignalStore, signal_type: str, n: int) -> list[Signal]:
    """Sample INITIALs ordered by ascending count of CRITIQUE/OBJECTION/VERIFICATION children.

    Distributes critic and validator coverage toward INITIALs that have
    received the least adversarial or verification attention. Without this,
    the run pre-mortem showed 47 INITIALs but only 1 with dissent and 5
    with verification — coverage was Pareto-concentrated on a few signals.

    For non-INITIAL targets, falls back to strength-weighted sampling.
    """
    if signal_type != INITIAL:
        return store.sample_weighted(signal_type, n)

    initials = store.by_type(INITIAL)
    if not initials:
        return []

    def examiner_count(sig: Signal) -> int:
        return sum(
            1 for cid in store.by_parent(sig.id)
            if (c := store.get(cid)) and c.type in _EXAMINER_TYPES
        )

    # Stable secondary sort by id keeps test/diagnostic output deterministic
    # when many INITIALs tie at zero examiners (the common early-round case).
    initials_sorted = sorted(initials, key=lambda s: (examiner_count(s), s.id))
    picked = initials_sorted[:n]
    for s in picked:
        s.visits += 1
    return picked


def needs_verification(store: SignalStore, signal_type: str, n: int) -> list[Signal]:
    """Sample INITIAL signals with >= 2 SUPPORT children and 0 VERIFICATION children.

    Validators stop wasting Wikipedia lookups on orphaned or already-verified
    claims. For non-INITIAL targets, falls back to strength-weighted sampling.
    """
    if signal_type != INITIAL:
        return store.sample_weighted(signal_type, n)

    candidates: list[Signal] = []
    for sig in store.by_type(INITIAL):
        children = [store.get(cid) for cid in store.by_parent(sig.id)]
        children = [c for c in children if c is not None]
        n_supports = sum(1 for c in children if c.type == SUPPORT)
        n_verifications = sum(1 for c in children if c.type == VERIFICATION)
        if n_supports >= 2 and n_verifications == 0:
            candidates.append(sig)

    if not candidates:
        # Nothing meets the predicate yet — fall back to well-supported signals.
        return store.signals_with_many_children_of_type(INITIAL, SUPPORT, 2)[:n] or \
               store.sample_weighted(INITIAL, n)

    picked = candidates[:n]
    for s in picked:
        s.visits += 1
    return picked


# ---------------------------------------------------------------------------
# Role-specific strategy lists (P1.8 / M8 fix)
# ---------------------------------------------------------------------------
#
# Each role gets a list whose length is at least the role's configured agent
# count. The k-th agent in a role picks list[k]. Roles deliberately use
# different orderings so that, even where they share strategies, they don't
# overlap on the same k.

# §5 directive: strategy library shrinks to 3 for Developer role.
# recent_dissent_targeted is imported lazily to avoid circular import
# (agents.developer → core.sampling would be circular).
def _recent_dissent_targeted_proxy(
    store: "SignalStore", signal_type: str, n: int
) -> "list[Signal]":
    from agents.developer import recent_dissent_targeted
    return recent_dissent_targeted(store, signal_type, n)


FORAGER_STRATEGIES: list[tuple[str, SamplingStrategy]] = [
    ("under_supported_clusters",  under_supported_clusters),
    ("stratified_extremes",       stratified_extremes),
    ("recent_dissent_targeted",   _recent_dissent_targeted_proxy),
    # Legacy extras — still available if NUM_FORAGERS > 3
    ("medium_only",               medium_only),
    ("weighted_default",          weighted_default),
]

# Alias: DEVELOPER_STRATEGIES is the canonical name for this role's list.
DEVELOPER_STRATEGIES = FORAGER_STRATEGIES

CRITIC_STRATEGIES: list[tuple[str, SamplingStrategy]] = [
    ("weighted_default",    weighted_default),     # critic 0: strong+exploration
    ("stratified_extremes", stratified_extremes),  # critic 1: spans weak+strong
    ("least_examined",      least_examined),       # critic 2: targets neglected
                                                   # INITIALs (replaced under_visited_only)
]

VALIDATOR_STRATEGIES: list[tuple[str, SamplingStrategy]] = [
    ("needs_verification",  needs_verification),
    ("least_examined",      least_examined),       # replaced stratified_extremes:
                                                   # the targeted neglect signal is
                                                   # a better validator prior than
                                                   # strength-weighted sampling.
]


def strategy_for_forager(agent_index: int) -> tuple[str, SamplingStrategy]:
    return FORAGER_STRATEGIES[agent_index % len(FORAGER_STRATEGIES)]

# Canonical name for the renamed role (§5 directive).
strategy_for_developer = strategy_for_forager


def strategy_for_critic(agent_index: int) -> tuple[str, SamplingStrategy]:
    return CRITIC_STRATEGIES[agent_index % len(CRITIC_STRATEGIES)]


def strategy_for_validator(agent_index: int) -> tuple[str, SamplingStrategy]:
    return VALIDATOR_STRATEGIES[agent_index % len(VALIDATOR_STRATEGIES)]


# Backward compatibility: keep the old name as an alias so anything still
# importing it doesn't break. Prefer the role-specific helpers above.
DOWNSTREAM_STRATEGIES = FORAGER_STRATEGIES

def strategy_for(agent_index: int) -> tuple[str, SamplingStrategy]:
    return strategy_for_forager(agent_index)
