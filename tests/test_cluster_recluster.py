"""Deterministic proof of the corrective re-cluster: force a mixed blob, then
recluster_type() splits it into cohesive sub-clusters — and crucially groups the
minority sub-position TOGETHER (one cluster), not into dust singletons (which is
what the existing eject-on-reanchor would do).

No LLM/embedder — synthetic unit vectors.
"""

import math

import pytest

import core.config as cfg
from core.cluster_registry import ClusterRegistry
from core.signal_types import INITIAL

x = [1.0, 0.0, 0.0, 0.0]
y = [0.6, 0.8, 0.0, 0.0]   # dot(x, y) = 0.6  -> two genuinely distinct positions


def _mixed_blob(monkeypatch) -> ClusterRegistry:
    # Disable incremental reanchor during construction so we can build the blob
    # and test recluster_type() in isolation; merge everything with a low join bar.
    monkeypatch.setattr("core.cluster_registry.CLUSTER_REANCHOR_EVERY", 10_000)
    monkeypatch.setattr(cfg, "CLUSTER_JOIN_SIZE_PENALTY", 0.0)
    monkeypatch.setattr(cfg, "CLUSTER_JOIN_THRESHOLD", 0.5)
    reg = ClusterRegistry()
    cid = reg.create("x0", x, INITIAL)
    for i in range(1, 8):
        reg.try_join(f"x{i}", x, INITIAL)
    for i in range(8):
        reg.try_join(f"y{i}", y, INITIAL)
    # one blob of 16 (8 x-position + 8 y-position)
    assert len(reg.clusters_by_type(INITIAL)) == 1
    assert len(reg.get_cluster(cid).member_ids) == 16
    return reg


def test_recluster_splits_mixed_blob_into_two_cohesive_clusters(monkeypatch):
    monkeypatch.setattr(cfg, "CLUSTER_COHESION_MIN", 0.80)
    reg = _mixed_blob(monkeypatch)

    n_splits = reg.recluster_type(INITIAL)

    assert n_splits == 1
    clusters = reg.clusters_by_type(INITIAL)
    sizes = sorted(len(c.member_ids) for c in clusters)
    assert sizes == [8, 8]                       # balanced, NOT 1 cluster + 8 singletons

    # the two positions are cleanly separated
    for c in clusters:
        prefixes = {m[0] for m in c.member_ids}  # 'x' or 'y'
        assert prefixes in ({"x"}, {"y"})         # no cluster mixes positions


def test_recluster_is_idempotent_on_cohesive_clusters(monkeypatch):
    monkeypatch.setattr(cfg, "CLUSTER_COHESION_MIN", 0.80)
    reg = _mixed_blob(monkeypatch)
    assert reg.recluster_type(INITIAL) == 1       # first pass splits
    assert reg.recluster_type(INITIAL) == 0       # second pass: already cohesive, no-op


def test_recluster_preserves_lineage_largest_child_keeps_id(monkeypatch):
    monkeypatch.setattr(cfg, "CLUSTER_COHESION_MIN", 0.80)
    # y is dot=0.6 to x: a distinct minority only under a split bar above 0.6.
    # Pin it (production split is now ~0.55/0.42 for MiniLM, where 0.6 counts as
    # in-cluster); this test exercises the lineage-preservation mechanism.
    monkeypatch.setattr(cfg, "CLUSTER_SPLIT_THRESHOLD", 0.78)
    # make x-position the majority so it should keep the original cluster id
    monkeypatch.setattr("core.cluster_registry.CLUSTER_REANCHOR_EVERY", 10_000)
    monkeypatch.setattr(cfg, "CLUSTER_JOIN_SIZE_PENALTY", 0.0)
    monkeypatch.setattr(cfg, "CLUSTER_JOIN_THRESHOLD", 0.5)
    reg = ClusterRegistry()
    cid = reg.create("x0", x, INITIAL)
    for i in range(1, 10):
        reg.try_join(f"x{i}", x, INITIAL)   # 10 x
    for i in range(3):
        reg.try_join(f"y{i}", y, INITIAL)   # 3 y (minority)

    reg.recluster_type(INITIAL)
    kept = reg.get_cluster(cid)
    assert kept is not None
    assert len(kept.member_ids) == 10               # majority retained the original id
    assert all(m[0] == "x" for m in kept.member_ids)
