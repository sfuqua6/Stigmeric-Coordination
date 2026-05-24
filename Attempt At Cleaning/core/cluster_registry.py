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


# ---------------------------------------------------------------------------
# Internal cluster record
# ---------------------------------------------------------------------------

@dataclass
class _Cluster:
    cluster_id: str
    signal_type: str
    member_ids: list[str] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)
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

        Scans all clusters of signal_type, picks the one with the highest
        centroid similarity. Returns cluster_id if sim >= CLUSTER_JOIN_THRESHOLD,
        else None (caller should then call create()).
        """
        if signal_type not in CLUSTERING_ENABLED_TYPES:
            return None
        cluster_ids = self._by_type.get(signal_type, [])
        best_sim = -1.0
        best_cid: Optional[str] = None
        for cid in cluster_ids:
            cl = self._clusters.get(cid)
            if cl is None or not cl.centroid:
                continue
            sim = _dot(embedding, cl.centroid)
            if sim > best_sim:
                best_sim = sim
                best_cid = cid
        if best_cid is not None and best_sim >= CLUSTER_JOIN_THRESHOLD:
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

    # ---- internals ---------------------------------------------------------

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
                deposit_count=1,
            )
            self._clusters[new_cid] = new_cl
            self._signal_to_cluster[mid] = new_cid
            self._by_type.setdefault(cl.signal_type, []).append(new_cid)
            print(
                f"[CLUSTER] SPLIT mid={mid} from {cluster_id} → new {new_cid} "
                f"(sim was below {CLUSTER_SPLIT_THRESHOLD})"
            )
