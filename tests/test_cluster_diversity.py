"""Fix D — Cluster-level support_diversity tests.

Verifies that support_diversity counts distinct (partition_id, author_role) pairs
from SUPPORT signals (not action-name strings), and that critic_endorsements
counts CRITIQUE_POSITIVE signals correctly.

Run with:
    pytest tests/test_cluster_diversity.py -v
"""

import sys
import os
import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.signal_store import SignalStore, _NULL_EMBEDDER
from core.projection import _aggregate_cluster, _compute_initial_metrics


def _make_store():
    return SignalStore(embedder=_NULL_EMBEDDER)


def _dep(store, signal_type, content, depositor, parent_id=None,
         partition_id="", metadata=None):
    meta = dict(metadata or {})
    if partition_id:
        meta["partition_id"] = partition_id
    elif signal_type in ("INITIAL", "SUPPORT") and "partition_id" not in meta:
        meta["partition_id"] = "partition_0"
    return store.deposit(
        signal_type=signal_type,
        content=content,
        strength=0.6,
        depositor=depositor,
        parent_id=parent_id,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# 1. Support_diversity: counts (partition_id, depositor) pairs
# ---------------------------------------------------------------------------

def test_support_diversity_two_partitions_same_role():
    """Two SUPPORT signals from the same role but different partitions → diversity=2."""
    store = _make_store()
    init_id = _dep(store, "INITIAL", "AI accelerates drug discovery.", "scout",
                   partition_id="partition_0")

    _dep(store, "SUPPORT", "AlphaFold predicts protein structures in hours.",
         "developer", parent_id=init_id, partition_id="partition_0")
    _dep(store, "SUPPORT", "GenAI identifies novel compound candidates rapidly.",
         "developer", parent_id=init_id, partition_id="partition_1")

    init_sig = store.get(init_id)
    metrics_map = {init_id: _compute_initial_metrics(init_sig, store)}
    cp = _aggregate_cluster(init_id, [init_id], metrics_map, store)

    assert cp.support_diversity == 2, (
        f"Expected support_diversity=2 (two distinct (partition, role) pairs), "
        f"got {cp.support_diversity}"
    )


def test_support_diversity_same_partition_same_role_counts_once():
    """Two SUPPORT signals from same partition + same role → diversity=1."""
    store = _make_store()
    init_id = _dep(store, "INITIAL", "Ocean acidification harms coral reefs.", "scout",
                   partition_id="partition_0")

    _dep(store, "SUPPORT", "pH levels dropped 0.1 since pre-industrial era.",
         "developer", parent_id=init_id, partition_id="partition_0")
    _dep(store, "SUPPORT", "Coral bleaching rates rose 50% since 1990.",
         "developer", parent_id=init_id, partition_id="partition_0")

    init_sig = store.get(init_id)
    metrics_map = {init_id: _compute_initial_metrics(init_sig, store)}
    cp = _aggregate_cluster(init_id, [init_id], metrics_map, store)

    assert cp.support_diversity == 1, (
        f"Same (partition, role) pair counts as 1, got {cp.support_diversity}"
    )


def test_support_diversity_three_distinct_pairs():
    """Three supports from three distinct (partition, role) pairs → diversity=3."""
    store = _make_store()
    init_id = _dep(store, "INITIAL", "Electrification of transport reduces emissions.",
                   "scout", partition_id="partition_0")

    _dep(store, "SUPPORT", "EV adoption rose 40% in 2023.",
         "developer", parent_id=init_id, partition_id="partition_0")
    _dep(store, "SUPPORT", "Grid decarbonization enables clean EV charging.",
         "developer", parent_id=init_id, partition_id="partition_1")
    _dep(store, "SUPPORT", "Battery costs fell 90% in a decade.",
         "critic", parent_id=init_id, partition_id="partition_0")  # different role

    init_sig = store.get(init_id)
    metrics_map = {init_id: _compute_initial_metrics(init_sig, store)}
    cp = _aggregate_cluster(init_id, [init_id], metrics_map, store)

    assert cp.support_diversity == 3, (
        f"Expected 3 distinct (partition, role) pairs, got {cp.support_diversity}"
    )


def test_support_diversity_no_support_signals():
    """No SUPPORT signals → support_diversity=0."""
    store = _make_store()
    init_id = _dep(store, "INITIAL", "Forests absorb CO2.", "scout",
                   partition_id="partition_0")
    init_sig = store.get(init_id)
    metrics_map = {init_id: _compute_initial_metrics(init_sig, store)}
    cp = _aggregate_cluster(init_id, [init_id], metrics_map, store)
    assert cp.support_diversity == 0


# ---------------------------------------------------------------------------
# 2. Critic endorsements: CRITIQUE_POSITIVE count
# ---------------------------------------------------------------------------

def test_critic_endorsements_counted_correctly():
    """CRITIQUE_POSITIVE signals appear in critic_endorsements count."""
    store = _make_store()
    init_id = _dep(store, "INITIAL", "Nuclear power is carbon-neutral.", "scout",
                   partition_id="partition_0")

    # One SUPPORT, two CRITIQUE_POSITIVE
    _dep(store, "SUPPORT", "IPCC cites nuclear in net-zero pathways.",
         "developer", parent_id=init_id, partition_id="partition_0")
    _dep(store, "CRITIQUE_POSITIVE", "Lifecycle emissions for nuclear are among lowest.",
         "critic", parent_id=init_id, partition_id="partition_0")
    _dep(store, "CRITIQUE_POSITIVE", "France's low-carbon grid demonstrates viability.",
         "critic", parent_id=init_id, partition_id="partition_1")

    init_sig = store.get(init_id)
    metrics_map = {init_id: _compute_initial_metrics(init_sig, store)}
    cp = _aggregate_cluster(init_id, [init_id], metrics_map, store)

    # CRITIQUE_POSITIVE are in support_set
    cp_endorsements = sum(
        1 for sid in cp.support_set
        for s in (store.get(sid),)
        if s is not None and s.type == "CRITIQUE_POSITIVE"
    )
    assert cp_endorsements == 2, (
        f"Expected 2 critic endorsements, got {cp_endorsements}"
    )


# ---------------------------------------------------------------------------
# 3. Cluster-level audit log fires (smoke)
# ---------------------------------------------------------------------------

def test_audit_log_prints(capsys):
    """_aggregate_cluster prints [CLUSTER AUDIT] line."""
    store = _make_store()
    init_id = _dep(store, "INITIAL", "Urbanization drives resource consumption.",
                   "scout", partition_id="partition_0")
    _dep(store, "SUPPORT", "Cities use 75% of global energy.",
         "developer", parent_id=init_id, partition_id="partition_0")

    init_sig = store.get(init_id)
    metrics_map = {init_id: _compute_initial_metrics(init_sig, store)}
    _aggregate_cluster(init_id, [init_id], metrics_map, store)

    captured = capsys.readouterr()
    assert "[CLUSTER AUDIT]" in captured.out, (
        "Expected [CLUSTER AUDIT] log line from _aggregate_cluster"
    )
    assert "support_diversity=" in captured.out
    assert "critic_endorsements=" in captured.out
    assert "members_in_cluster=" in captured.out


# ---------------------------------------------------------------------------
# 4. Cluster with multiple INITIAL members (merged cluster diversity)
# ---------------------------------------------------------------------------

def test_cluster_diversity_across_multiple_initials():
    """A merged cluster aggregates support_diversity across all member INITIALs."""
    store = _make_store()
    init0 = _dep(store, "INITIAL", "Solar panels reduce household energy bills.",
                 "scout", partition_id="partition_0")
    init1 = _dep(store, "INITIAL", "Home solar cuts electricity costs significantly.",
                 "scout", partition_id="partition_1")

    # Support for init0 from partition_0
    _dep(store, "SUPPORT", "Average US household saves $1200/year with rooftop solar.",
         "developer", parent_id=init0, partition_id="partition_0")
    # Support for init1 from partition_2
    _dep(store, "SUPPORT", "Solar payback period now under 7 years in most states.",
         "developer", parent_id=init1, partition_id="partition_2")

    init0_sig = store.get(init0)
    init1_sig = store.get(init1)
    metrics_map = {
        init0: _compute_initial_metrics(init0_sig, store),
        init1: _compute_initial_metrics(init1_sig, store),
    }
    cp = _aggregate_cluster(init0, [init0, init1], metrics_map, store)

    # Two distinct (partition, role) pairs: (p0, developer) and (p2, developer)
    assert cp.support_diversity == 2, (
        f"Merged cluster should have 2 distinct (partition, role) pairs, "
        f"got {cp.support_diversity}"
    )
