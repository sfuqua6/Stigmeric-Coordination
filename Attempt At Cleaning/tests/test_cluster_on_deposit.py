"""Fix C' — ClusterRegistry at-deposit clustering tests.

Verifies that ClusterRegistry assigns cluster_ids at deposit time,
that near-duplicate signals join existing clusters instead of being rejected,
that VERIFICATION inherits parent cluster_id, and that projection.py
reads ClusterRegistry clusters.

Run with:
    pytest tests/test_cluster_on_deposit.py -v
"""

import sys
import os
import math
import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.cluster_registry import ClusterRegistry, _dot, _l2_normalize
from core.signal_store import SignalStore
from core.config import CLUSTER_JOIN_THRESHOLD, CLUSTER_SPLIT_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(n: int, idx: int) -> list[float]:
    """One-hot unit vector of dimension n with 1.0 at idx."""
    v = [0.0] * n
    v[idx] = 1.0
    return v


def _make_store() -> SignalStore:
    """SignalStore with embedder disabled (we'll inject embeddings manually)."""
    from core.signal_store import _NULL_EMBEDDER
    return SignalStore(embedder=_NULL_EMBEDDER)


def _deposit_with_emb(store: SignalStore, content: str, signal_type: str,
                      partition_id: str, emb: list[float]) -> str:
    """Deposit a signal, then manually inject its embedding into the store."""
    sid = store.deposit(
        signal_type=signal_type,
        content=content,
        strength=0.6,
        depositor="scout" if signal_type == "INITIAL" else "developer",
        parent_id=None,
        metadata={"partition_id": partition_id},
    )
    if sid and emb:
        # Inject embedding manually (mimics the real embedder path)
        store._embeddings[sid] = emb
        # Manually wire into registry too, since embedder was disabled
        cr = store._cluster_registry
        cluster_id = cr.try_join(sid, emb, signal_type)
        if cluster_id is None:
            cluster_id = cr.create(sid, emb, signal_type)
        store._signals[sid].cluster_id = cluster_id
    return sid


# ---------------------------------------------------------------------------
# 1. ClusterRegistry — unit tests
# ---------------------------------------------------------------------------

class TestClusterRegistryUnit:
    def test_create_new_cluster(self):
        cr = ClusterRegistry()
        emb = _unit_vec(8, 0)
        cid = cr.create("sig_0", emb, "INITIAL")
        assert cid is not None
        assert cid.startswith("cluster_")
        assert cr.get_cluster_id("sig_0") == cid

    def test_try_join_above_threshold(self):
        cr = ClusterRegistry()
        emb0 = _unit_vec(8, 0)
        cid0 = cr.create("sig_0", emb0, "INITIAL")
        # Very similar vector (cos sim = 1.0 for identical)
        cid1 = cr.try_join("sig_1", emb0, "INITIAL")
        assert cid1 == cid0, "Identical vector must join existing cluster"
        assert cr.get_cluster_id("sig_1") == cid0

    def test_try_join_below_threshold(self):
        cr = ClusterRegistry()
        emb0 = _unit_vec(8, 0)
        cr.create("sig_0", emb0, "INITIAL")
        # Orthogonal vector — cosine sim = 0.0
        emb1 = _unit_vec(8, 1)
        result = cr.try_join("sig_1", emb1, "INITIAL")
        assert result is None, "Orthogonal vector must not join existing cluster"

    def test_type_isolation(self):
        """Signals of different types don't cluster together."""
        cr = ClusterRegistry()
        emb = _unit_vec(8, 0)
        cid_init = cr.create("sig_init", emb, "INITIAL")
        # Same embedding but different type — should NOT join INITIAL cluster
        result = cr.try_join("sig_sup", emb, "SUPPORT")
        assert result is None, "Different signal types must not join the same cluster"

    def test_clusters_by_type(self):
        cr = ClusterRegistry()
        emb0 = _unit_vec(8, 0)
        emb1 = _unit_vec(8, 1)
        cr.create("a", emb0, "INITIAL")
        cr.create("b", emb1, "INITIAL")
        clusters = cr.clusters_by_type("INITIAL")
        assert len(clusters) == 2

    def test_centroid_updates_on_join(self):
        cr = ClusterRegistry()
        emb0 = _unit_vec(4, 0)  # [1,0,0,0]
        cid = cr.create("sig_0", emb0, "INITIAL")
        emb1 = _unit_vec(4, 0)  # identical
        cr.try_join("sig_1", emb1, "INITIAL")
        cl = cr.get_cluster(cid)
        # Centroid must be L2-normalized
        mag = math.sqrt(sum(x * x for x in cl.centroid))
        assert abs(mag - 1.0) < 1e-6, f"Centroid must be unit-length, got mag={mag}"

    def test_non_clusterable_type_skipped(self):
        cr = ClusterRegistry()
        emb = _unit_vec(8, 0)
        result = cr.try_join("ver_0", emb, "VERIFICATION")
        assert result is None, "VERIFICATION is not in CLUSTERING_ENABLED_TYPES"


# ---------------------------------------------------------------------------
# 2. SignalStore deposit — cluster_id assigned
# ---------------------------------------------------------------------------

def test_initial_gets_cluster_id_with_embedder():
    """Full-stack: real embedder assigns cluster_id to INITIAL at deposit."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence_transformers not installed")

    store = SignalStore(embedder=None)  # auto-loads embedder
    sid = store.deposit(
        signal_type="INITIAL",
        content="Urban heat islands increase cooling energy demand significantly.",
        strength=0.6,
        depositor="scout",
        parent_id=None,
        metadata={"partition_id": "partition_0"},
    )
    assert sid is not None
    sig = store.get(sid)
    assert sig.cluster_id is not None, (
        "INITIAL signal should have cluster_id assigned by ClusterRegistry"
    )
    assert sig.cluster_id.startswith("cluster_"), (
        f"cluster_id should start with 'cluster_', got {sig.cluster_id!r}"
    )


def test_near_duplicate_joins_cluster_instead_of_being_rejected():
    """Near-duplicate INITIAL signals deposit (not rejected) and join same cluster."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence_transformers not installed")

    store = SignalStore(embedder=None)
    text1 = "Renewable energy reduces greenhouse gas emissions substantially."
    text2 = "Renewable energy greatly decreases greenhouse gas emissions."

    sid1 = store.deposit(
        signal_type="INITIAL",
        content=text1,
        strength=0.6,
        depositor="scout",
        parent_id=None,
        metadata={"partition_id": "partition_0"},
    )
    sid2 = store.deposit(
        signal_type="INITIAL",
        content=text2,
        strength=0.6,
        depositor="scout",
        parent_id=None,
        metadata={"partition_id": "partition_1"},
    )
    assert sid1 is not None, "First deposit should succeed"
    assert sid2 is not None, (
        "Near-duplicate should be deposited (not rejected) after Fix C'"
    )
    sig1 = store.get(sid1)
    sig2 = store.get(sid2)
    assert sig1.cluster_id is not None
    assert sig2.cluster_id is not None
    assert sig1.cluster_id == sig2.cluster_id, (
        "Near-duplicate signals should share a cluster_id"
    )


def test_verification_inherits_cluster_id():
    """VERIFICATION signal inherits parent's cluster_id."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence_transformers not installed")

    store = SignalStore(embedder=None)
    init_id = store.deposit(
        signal_type="INITIAL",
        content="Deforestation drives species extinction globally.",
        strength=0.6,
        depositor="scout",
        parent_id=None,
        metadata={"partition_id": "partition_0"},
    )
    init_sig = store.get(init_id)
    original_cluster_id = init_sig.cluster_id

    ver_id = store.deposit(
        signal_type="VERIFICATION",
        content="Wikipedia: Deforestation rates confirmed by IPBES data.",
        strength=0.75,
        depositor="validator",
        parent_id=init_id,
        metadata={},
    )
    ver_sig = store.get(ver_id)
    assert ver_sig.cluster_id == original_cluster_id, (
        f"VERIFICATION should inherit parent cluster_id={original_cluster_id!r}, "
        f"got {ver_sig.cluster_id!r}"
    )


# ---------------------------------------------------------------------------
# 3. SignalStore.get_live_clusters — returns ClusterRegistry clusters
# ---------------------------------------------------------------------------

def test_get_live_clusters_without_embedder():
    """Without embedder, no clusters exist → get_live_clusters returns []."""
    from core.signal_store import _NULL_EMBEDDER
    store = SignalStore(embedder=_NULL_EMBEDDER)
    store.deposit(
        signal_type="INITIAL",
        content="Some claim without embedder.",
        strength=0.6,
        depositor="scout",
        parent_id=None,
        metadata={"partition_id": "partition_0"},
    )
    result = store.get_live_clusters("INITIAL")
    assert result == [], "No clusters expected when embedder is disabled"


def test_get_live_clusters_with_injected_embeddings():
    """Inject embeddings manually; get_live_clusters returns correct structure."""
    from core.signal_store import _NULL_EMBEDDER
    store = SignalStore(embedder=_NULL_EMBEDDER)

    sid0 = _deposit_with_emb(store, "Climate change accelerates ice melt.",
                              "INITIAL", "partition_0", _unit_vec(8, 0))
    sid1 = _deposit_with_emb(store, "Rapid ice melt is caused by climate change.",
                              "INITIAL", "partition_1", _unit_vec(8, 0))  # same cluster
    sid2 = _deposit_with_emb(store, "Urban transit reduces car dependency.",
                              "INITIAL", "partition_2", _unit_vec(8, 3))  # different cluster

    clusters = store.get_live_clusters("INITIAL")
    assert len(clusters) >= 1, "Should have at least one cluster"

    all_rep_ids = [rep for rep, _ in clusters]
    all_member_ids = [mid for _, members in clusters for mid in members]
    assert sid0 in all_member_ids
    assert sid1 in all_member_ids
    assert sid2 in all_member_ids

    # The two same-embedding signals should be in the same cluster
    cluster_of_0 = store.get(sid0).cluster_id
    cluster_of_1 = store.get(sid1).cluster_id
    assert cluster_of_0 == cluster_of_1, "Identical vectors should be in same cluster"


def test_get_live_clusters_excludes_pruned_signals():
    """Pruned signals should not appear in get_live_clusters()."""
    from core.signal_store import _NULL_EMBEDDER
    store = SignalStore(embedder=_NULL_EMBEDDER)
    sid = _deposit_with_emb(store, "Claim to be pruned.", "INITIAL",
                             "partition_0", _unit_vec(8, 0))
    # Force strength below prune threshold
    sig = store.get(sid)
    sig.strength = 0.0
    sig._logit = -999.0
    store.prune_weak()
    assert store.get(sid) is None, "Signal should be pruned"

    clusters = store.get_live_clusters("INITIAL")
    for rep, members in clusters:
        assert sid not in members, "Pruned signal should not appear in live clusters"


# ---------------------------------------------------------------------------
# 4. Projection uses ClusterRegistry (smoke test with injected embeddings)
# ---------------------------------------------------------------------------

def test_projection_uses_registry_clusters():
    """build_projection prefers ClusterRegistry over greedy single-linkage."""
    from core.signal_store import _NULL_EMBEDDER
    from core.projection import build_projection

    store = SignalStore(embedder=_NULL_EMBEDDER)
    sid0 = _deposit_with_emb(store, "Biodiversity loss threatens food security.",
                              "INITIAL", "partition_0", _unit_vec(8, 0))
    sid1 = _deposit_with_emb(store, "Food security is threatened by biodiversity loss.",
                              "INITIAL", "partition_1", _unit_vec(8, 0))

    proj = build_projection(store, has_validators=False)
    # With two INITIAL signals in the same cluster, projection should see them.
    all_member_ids = []
    for cp in (proj.surviving + proj.contested + proj.weakly_supported
               + proj.rejected_by_field + proj.unverified):
        all_member_ids.extend(cp.member_ids)

    assert sid0 in all_member_ids, "First INITIAL should appear in projection"
    assert sid1 in all_member_ids, "Second INITIAL should appear in projection"


# ---------------------------------------------------------------------------
# 5. Hater carries targets_cluster_id in OBJECTION metadata
# ---------------------------------------------------------------------------

def test_hater_extra_metadata_includes_cluster_id():
    """Hater.extra_deposit_metadata() returns targets_cluster_id after sample()."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from agents.hater import Hater
    from core.signal_store import _NULL_EMBEDDER

    store = SignalStore(embedder=_NULL_EMBEDDER)
    sid = _deposit_with_emb(store, "Corporations drive climate inaction.",
                             "INITIAL", "partition_0", _unit_vec(8, 0))

    llm = MagicMock()
    llm.generate = AsyncMock(return_value="This ignores state actors.\nOBJECTION: state actors also contribute.")

    hater = Hater("hater_0", llm, task_prompt="Climate action debate")
    hater.sample(store)  # sets _last_cluster_id
    meta = hater.extra_deposit_metadata()

    assert "targets_cluster_id" in meta, (
        "Hater.extra_deposit_metadata() must include targets_cluster_id"
    )
    assert meta["targets_cluster_id"] == store.get(sid).cluster_id


def test_hater_no_cluster_id_when_no_registry_data():
    """When registry is empty, extra_deposit_metadata returns {} for targets_cluster_id."""
    from unittest.mock import AsyncMock, MagicMock
    from agents.hater import Hater
    from core.signal_store import _NULL_EMBEDDER

    store = SignalStore(embedder=_NULL_EMBEDDER)
    # Deposit without injecting embeddings → no registry clusters
    store.deposit(
        signal_type="INITIAL",
        content="Some claim.",
        strength=0.6,
        depositor="scout",
        parent_id=None,
        metadata={"partition_id": "partition_0"},
    )

    llm = MagicMock()
    llm.generate = AsyncMock(return_value="No cluster.")

    hater = Hater("hater_0", llm, task_prompt="Test")
    hater.sample(store)
    meta = hater.extra_deposit_metadata()
    # No cluster → targets_cluster_id either absent or empty
    assert meta.get("targets_cluster_id", "") == "", (
        "Should have no targets_cluster_id when registry has no clusters"
    )
