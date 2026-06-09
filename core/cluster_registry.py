"""ClusterRegistry — incremental signal clustering at deposit time.

Clusters are maintained incrementally: each deposit either joins an existing
cluster (cosine sim to centroid >= CLUSTER_JOIN_THRESHOLD) or creates a new one.
The centroid is updated as a running L2-normalized mean. Every
CLUSTER_REANCHOR_EVERY deposits into a cluster, the true medoid is recomputed
(member with highest mean cosine sim to all others) and replaces the approximate
centroid. A split check follows reanchor: members whose cosine sim to the new
centroid drops below CLUSTER_SPLIT_THRESHOLD are ejected to their own new cluster.

This replaces the greedy single-linkage post-hoc clustering in projection.py
(which ran once on the final store state) with continuous at-deposit clustering
so projection.py can read pre-built clusters rather than compute them.

CLUSTERING_ENABLED_TYPES = {"INITIAL", "SUPPORT", "OBJECTION"}
VERIFICATION and SEARCH are NOT clustered here — they inherit parent cluster_id
via SignalStore.deposit() for provenance tracking.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .config import (
    CLUSTER_JOIN_THRESHOLD,
    CLUSTER_SPLIT_THRESHOLD,
    CLUSTER_REANCHOR_EVERY,
    CLUSTERING_ENABLED_TYPES,
)

# Print cluster-join LOUD logs for the first this-many total registry deposits.
_MAX_LOUD_DEPOSITS = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _l2_normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-12:
        return v[:]
    return [x / n for x in v]


def _effective_join_threshold(size: int) -> float:
    """Similarity required to join a cluster of `size` members.

    Grows with size so an already-large cluster can't keep swallowing the field
    (anti-mega-blob). Read from config live so tests/ablation can flip the
    penalty to 0 (which reproduces the old fixed-threshold behaviour). See
    CLUSTER_JOIN_SIZE_PENALTY in core/config.py.
    """
    from . import config as _cfg
    base = _cfg.CLUSTER_JOIN_THRESHOLD
    penalty = _cfg.CLUSTER_JOIN_SIZE_PENALTY
    if penalty <= 0.0 or size <= 1:
        return base
    return min(_cfg.CLUSTER_JOIN_MAX_THRESHOLD, base + penalty * math.log2(size))


# ---------------------------------------------------------------------------
# Internal cluster record
# ---------------------------------------------------------------------------

@dataclass
class _Cluster:
    cluster_id: str
    signal_type: str
    member_ids: list[str] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)
    # Snapshot of the centroid at the moment the cluster was created (the
    # founding member's embedding, L2-normalized). Used by the genome's
    # Phenotype to compute centroid_drift across the run.
    centroid_at_formation: list[float] = field(default_factory=list)
    deposit_count: int = 0   # deposits INTO this cluster; triggers reanchor


# ---------------------------------------------------------------------------
# ClusterRegistry
# ---------------------------------------------------------------------------

class ClusterRegistry:
    """Maintains semantic clusters of signals at deposit time.

    Thread safety: NOT thread-safe on its own. SignalStore holds the lock
    for its _lock (RLock) around all deposit() calls, so ClusterRegistry
    methods are always called under that lock. Do NOT acquire a separate
    lock inside ClusterRegistry.
    """

    def __init__(self) -> None:
        self._clusters: dict[str, _Cluster] = {}
        self._signal_to_cluster: dict[str, str] = {}
        self._by_type: dict[str, list[str]] = {}
        self._member_embeddings: dict[str, list[float]] = {}
        self._total_deposits: int = 0   # global counter for LOUD log gate

    # ---- public API --------------------------------------------------------

    def try_join(
        self,
        signal_id: str,
        embedding: list[float],
        signal_type: str,
    ) -> Optional[str]:
        """Attempt to join an existing cluster for signal_type.

        Scans all clusters of signal_type and joins the one the signal most
        clearly clears the bar for, where each cluster's bar is SIZE-ADJUSTED
        (larger clusters demand higher similarity — see _effective_join_threshold).
        This lets a distinct claim nucleate / join a small specific cluster
        instead of being swallowed by a large vague attractor. Returns the
        cluster_id joined, or None (caller then calls create()).
        """
        if signal_type not in CLUSTERING_ENABLED_TYPES:
            return None
        cluster_ids = self._by_type.get(signal_type, [])
        best_cid: Optional[str] = None
        best_margin = 0.0   # require sim >= the cluster's size-adjusted threshold
        for cid in cluster_ids:
            cl = self._clusters.get(cid)
            if cl is None or not cl.centroid:
                continue
            sim = _dot(embedding, cl.centroid)
            margin = sim - _effective_join_threshold(len(cl.member_ids))
            if margin >= 0.0 and (best_cid is None or margin > best_margin):
                best_cid = cid
                best_margin = margin
        if best_cid is not None:
            self._join(best_cid, signal_id, embedding)
            return best_cid
        return None

    def create(
        self,
        signal_id: str,
        embedding: list[float],
        signal_type: str,
    ) -> str:
        """Create a new single-member cluster. Returns the new cluster_id."""
        cid = "cluster_" + uuid.uuid4().hex[:8]
        cl = _Cluster(
            cluster_id=cid,
            signal_type=signal_type,
            member_ids=[signal_id],
            centroid=embedding[:],
            centroid_at_formation=embedding[:],
            deposit_count=1,
        )
        self._clusters[cid] = cl
        self._signal_to_cluster[signal_id] = cid
        self._by_type.setdefault(signal_type, []).append(cid)
        self._member_embeddings[signal_id] = embedding[:]
        self._total_deposits += 1
        if self._total_deposits <= _MAX_LOUD_DEPOSITS:
            print(
                f"[CLUSTER] NEW  cid={cid} type={signal_type} "
                f"first_member={signal_id}"
            )
        return cid

    def get_cluster_id(self, signal_id: str) -> Optional[str]:
        """Return the cluster_id for a given signal, or None."""
        return self._signal_to_cluster.get(signal_id)

    def get_cluster(self, cluster_id: str) -> Optional[_Cluster]:
        return self._clusters.get(cluster_id)

    def clusters_by_type(self, signal_type: str) -> list[_Cluster]:
        """All clusters (including empty ones) for a signal_type."""
        cids = self._by_type.get(signal_type, [])
        return [self._clusters[cid] for cid in cids if cid in self._clusters]

    # ---- periodic divisive re-cluster (corrective) --------------------------

    def recluster_type(self, signal_type: str, cohesion_min: Optional[float] = None,
                       min_size: int = 2) -> int:
        """Force-split incoherent clusters of `signal_type` so similarity re-decides
        the partition (the "break apart, then measure" fix). Idempotent on cohesive
        clusters. The LARGER child keeps the parent cluster_id (preserves genome
        lineage / trajectory); smaller children spin off as new clusters (fission,
        which the genome is designed to survive). Returns the number of splits.
        """
        if signal_type not in CLUSTERING_ENABLED_TYPES:
            return 0
        from . import config as _cfg
        cmin = cohesion_min if cohesion_min is not None else _cfg.CLUSTER_COHESION_MIN
        splits = 0
        for cid in list(self._by_type.get(signal_type, [])):
            splits += self._split_recursive(cid, signal_type, cmin, min_size)
        return splits

    # ---- internals ---------------------------------------------------------

    def _members_with_emb(self, cluster_id: str) -> list[tuple[str, list[float]]]:
        cl = self._clusters.get(cluster_id)
        if cl is None:
            return []
        return [(m, self._member_embeddings[m]) for m in cl.member_ids
                if m in self._member_embeddings]

    def _medoid(self, embs: list[tuple[str, list[float]]]) -> tuple[int, float]:
        """(index of medoid, its mean cosine sim to the others) = cohesion proxy."""
        if len(embs) < 2:
            return 0, 1.0
        best_avg, best_i = -1.0, 0
        for i, (_, ei) in enumerate(embs):
            avg = sum(_dot(ei, ej) for j, (_, ej) in enumerate(embs) if j != i) / (len(embs) - 1)
            if avg > best_avg:
                best_avg, best_i = avg, i
        return best_i, best_avg

    def _farthest_pair(self, embs: list[tuple[str, list[float]]]) -> tuple[int, int]:
        """Indices of the least-similar (min cosine) member pair — the bisect seeds."""
        worst, wi, wj = 2.0, 0, 1
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                d = _dot(embs[i][1], embs[j][1])
                if d < worst:
                    worst, wi, wj = d, i, j
        return wi, wj

    def _split_recursive(self, cluster_id: str, signal_type: str,
                         cmin: float, min_size: int) -> int:
        cl = self._clusters.get(cluster_id)
        if cl is None:
            return 0
        embs = self._members_with_emb(cluster_id)
        if len(embs) < 2 * min_size:
            return 0
        mi, cohesion = self._medoid(embs)
        medoid_emb = embs[mi][1]
        min_sim = min(_dot(e, medoid_emb) for _, e in embs)
        from . import config as _cfg
        # Cohesive iff the mean is tight AND no member is an outlier. The min-sim
        # test catches a distinct MINORITY embedded in a big cluster (which a mean
        # metric drowns out) — and recluster groups it, vs reanchor's eject-to-dust.
        if cohesion >= cmin and min_sim >= _cfg.CLUSTER_SPLIT_THRESHOLD:
            return 0   # keep as-is

        ai, bi = self._farthest_pair(embs)
        seed_a, seed_b = embs[ai][1], embs[bi][1]
        group_a, group_b = [], []
        for mid, e in embs:
            (group_a if _dot(e, seed_a) >= _dot(e, seed_b) else group_b).append((mid, e))
        if not group_a or not group_b:
            return 0
        if len(group_a) < len(group_b):       # larger group keeps the id (lineage)
            group_a, group_b = group_b, group_a

        no_emb = [m for m in cl.member_ids if m not in self._member_embeddings]
        cl.member_ids = [m for m, _ in group_a] + no_emb
        cl.centroid = group_a[self._medoid(group_a)[0]][1][:]

        new_cid = "cluster_" + uuid.uuid4().hex[:8]
        b_centroid = group_b[self._medoid(group_b)[0]][1][:]
        self._clusters[new_cid] = _Cluster(
            cluster_id=new_cid, signal_type=signal_type,
            member_ids=[m for m, _ in group_b], centroid=b_centroid,
            centroid_at_formation=b_centroid, deposit_count=len(group_b))
        self._by_type.setdefault(signal_type, []).append(new_cid)
        for m, _ in group_b:
            self._signal_to_cluster[m] = new_cid

        if self._total_deposits <= _MAX_LOUD_DEPOSITS:
            print(f"[CLUSTER] RECLUSTER-SPLIT {cluster_id}(n={len(cl.member_ids)}) "
                  f"/ {new_cid}(n={len(group_b)}) cohesion={cohesion:.3f} < {cmin}")

        # either half may still be a blob — recurse
        return (1
                + self._split_recursive(cluster_id, signal_type, cmin, min_size)
                + self._split_recursive(new_cid, signal_type, cmin, min_size))

    def _join(self, cluster_id: str, signal_id: str, embedding: list[float]) -> None:
        cl = self._clusters[cluster_id]
        n_before = len(cl.member_ids)
        cl.member_ids.append(signal_id)
        self._signal_to_cluster[signal_id] = cluster_id
        self._member_embeddings[signal_id] = embedding[:]

        # Update running centroid: weighted sum of old centroid (n_before members)
        # and new embedding (1 member), then L2-normalize.
        n = n_before + 1
        if cl.centroid:
            raw = [cl.centroid[i] * n_before + embedding[i] for i in range(len(embedding))]
            cl.centroid = _l2_normalize(raw)
        else:
            cl.centroid = _l2_normalize(embedding[:])

        cl.deposit_count += 1
        self._total_deposits += 1

        if self._total_deposits <= _MAX_LOUD_DEPOSITS:
            sim = _dot(embedding, cl.centroid)
            print(
                f"[CLUSTER] JOIN cid={cluster_id} type={cl.signal_type} "
                f"signal={signal_id} size={n} centroid_sim={sim:.3f}"
            )

        if cl.deposit_count % CLUSTER_REANCHOR_EVERY == 0:
            self._reanchor(cluster_id)

    def _reanchor(self, cluster_id: str) -> None:
        """Recompute centroid as true medoid; eject split-threshold violators."""
        cl = self._clusters.get(cluster_id)
        if cl is None or len(cl.member_ids) < 2:
            return

        # Gather available (member_id, embedding) pairs.
        emb_pairs = [
            (mid, self._member_embeddings.get(mid))
            for mid in cl.member_ids
        ]
        emb_pairs = [(mid, e) for mid, e in emb_pairs if e is not None]
        if len(emb_pairs) < 2:
            return

        # Find medoid: member with highest mean cosine sim to all others.
        best_avg = -1.0
        best_mid = emb_pairs[0][0]
        best_emb = emb_pairs[0][1]
        for i, (mid_i, emb_i) in enumerate(emb_pairs):
            others = [emb_j for mid_j, emb_j in emb_pairs if mid_j != mid_i]
            avg_sim = sum(_dot(emb_i, ej) for ej in others) / len(others)
            if avg_sim > best_avg:
                best_avg = avg_sim
                best_mid = mid_i
                best_emb = emb_i
        cl.centroid = best_emb[:]
        print(
            f"[CLUSTER] REANCHOR cid={cluster_id} medoid={best_mid} "
            f"members={len(cl.member_ids)} avg_medoid_sim={best_avg:.3f}"
        )

        # Split check: eject members whose cosine sim to new centroid < threshold.
        remaining: list[str] = []
        ejected: list[tuple[str, list[float]]] = []
        for mid, emb in emb_pairs:
            if mid == best_mid:
                remaining.append(mid)
                continue
            sim = _dot(emb, cl.centroid)
            if sim < CLUSTER_SPLIT_THRESHOLD:
                ejected.append((mid, emb))
            else:
                remaining.append(mid)

        # Also keep any members whose embeddings weren't available (preserve them).
        no_emb = [mid for mid in cl.member_ids if self._member_embeddings.get(mid) is None]
        remaining.extend(no_emb)

        if not ejected:
            return

        cl.member_ids = remaining
        for mid, emb in ejected:
            self._signal_to_cluster.pop(mid, None)
            new_cid = "cluster_" + uuid.uuid4().hex[:8]
            new_cl = _Cluster(
                cluster_id=new_cid,
                signal_type=cl.signal_type,
                member_ids=[mid],
                centroid=emb[:],
                centroid_at_formation=emb[:],
                deposit_count=1,
            )
            self._clusters[new_cid] = new_cl
            self._signal_to_cluster[mid] = new_cid
            self._by_type.setdefault(cl.signal_type, []).append(new_cid)
            print(
                f"[CLUSTER] SPLIT mid={mid} from {cluster_id} → new {new_cid} "
                f"(sim was below {CLUSTER_SPLIT_THRESHOLD})"
            )
