"""Deterministic proof that the size-penalized join threshold breaks the
"one mega-blob + dust" pathology — no LLM/embedder needed.

Scenario (synthetic unit vectors):
  - C_big:   16 identical vectors `b`  -> one large cluster (the vague attractor)
  - C_small: one distinct vector `s` (0.85 to b, so it won't join C_big)
  - query `q`: 0.95 to b, 0.90 to s

Old rule (fixed 0.88): q is closer to b -> swallowed by the 16-member blob.
New rule (size penalty): the 16-member cluster's bar is ~0.97, so q is redirected
to the specific small cluster. Same input, better-balanced clusters.
"""

import math

import pytest

import core.config as cfg
from core.cluster_registry import ClusterRegistry, _effective_join_threshold
from core.signal_types import INITIAL

b = [1.0, 0.0, 0.0, 0.0]
s = [0.85, math.sqrt(1.0 - 0.85 ** 2), 0.0, 0.0]            # dot(s,b)=0.85
q = [0.95, 0.1756, 0.2582, 0.0]                            # ~0.95 to b, ~0.90 to s


def test_vector_sims_are_as_designed():
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))
    assert abs(dot(q, b) - 0.95) < 1e-3
    assert abs(dot(q, s) - 0.90) < 2e-3
    assert abs(dot(s, b) - 0.85) < 1e-3


def _build(reg: ClusterRegistry):
    cid_big = reg.create("b0", b, INITIAL)
    for i in range(1, 16):
        assert reg.try_join(f"b{i}", b, INITIAL) == cid_big   # identical -> always join
    assert reg.try_join("s_probe", s, INITIAL) is None        # 0.85 < 0.88 -> no join
    cid_small = reg.create("s0", s, INITIAL)
    return cid_big, cid_small


def test_effective_threshold_grows_with_size(monkeypatch):
    monkeypatch.setattr(cfg, "CLUSTER_JOIN_THRESHOLD", 0.88)
    monkeypatch.setattr(cfg, "CLUSTER_JOIN_SIZE_PENALTY", 0.03)
    monkeypatch.setattr(cfg, "CLUSTER_JOIN_MAX_THRESHOLD", 0.97)
    assert _effective_join_threshold(1) == 0.88
    assert abs(_effective_join_threshold(2) - 0.91) < 1e-9      # +0.03*log2(2)
    assert abs(_effective_join_threshold(4) - 0.94) < 1e-9      # +0.03*2
    assert _effective_join_threshold(64) == 0.97               # capped (0.88+0.18 -> 0.97)


def test_penalty_zero_reproduces_fixed_threshold(monkeypatch):
    monkeypatch.setattr(cfg, "CLUSTER_JOIN_SIZE_PENALTY", 0.0)
    assert _effective_join_threshold(1000) == cfg.CLUSTER_JOIN_THRESHOLD


def test_old_rule_swallows_query_into_blob(monkeypatch):
    # Synthetic sims (0.85/0.90/0.95) are hand-calibrated to a 0.88 base, so
    # pin it here — this test exercises the size-penalty mechanism, not the
    # production threshold (which is now tier-aware ~0.72/0.55 for MiniLM).
    monkeypatch.setattr(cfg, "CLUSTER_JOIN_THRESHOLD", 0.88)
    monkeypatch.setattr(cfg, "CLUSTER_JOIN_SIZE_PENALTY", 0.0)   # old behaviour
    reg = ClusterRegistry()
    cid_big, cid_small = _build(reg)
    assert reg.try_join("q", q, INITIAL) == cid_big             # blob wins
    assert len(reg.get_cluster(cid_big).member_ids) == 17


def test_size_penalty_redirects_query_to_specific_cluster(monkeypatch):
    # See note above: pin the 0.88 base the synthetic sims were designed for.
    monkeypatch.setattr(cfg, "CLUSTER_JOIN_THRESHOLD", 0.88)
    monkeypatch.setattr(cfg, "CLUSTER_JOIN_SIZE_PENALTY", 0.03)  # the fix
    reg = ClusterRegistry()
    cid_big, cid_small = _build(reg)
    joined = reg.try_join("q", q, INITIAL)
    assert joined == cid_small                                  # redirected to the specific cluster
    assert len(reg.get_cluster(cid_big).member_ids) == 16       # blob did NOT grow
    assert len(reg.get_cluster(cid_small).member_ids) == 2
