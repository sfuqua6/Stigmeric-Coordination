# FUTURE-CLAUDE NOTE: Strength dynamics in this file are gated on the
# USE_LOGIT_DYNAMICS flag from core/config.py. The default (True) path
# updates an internal `_logit` field additively and projects strength back
# via the sigmoid. This is what fixed three real bugs:
#   1. Saturation crash: repeated multiplicative amplify pinned strengths at
#      1.0 and destroyed ranking among "strong" signals.
#   2. Contrarian drift: the old decay_all gave CRITIQUE/OBJECTION an exact
#      anti-decay factor, so dissent_pressure accumulated forever.
#   3. Order dependence: multiplicative amplify×decay was commutative only
#      below saturation; logit-space additive updates are exactly
#      commutative.
# The legacy (multiplicative) code path is preserved for one release as an
# escape hatch. Do not remove it without also removing the
# USE_LOGIT_DYNAMICS=False branch in tests/test_logit_dynamics.py.
"""Stigmergic signal store: typed DAG with strength dynamics.

Design contract — the no-leak rule
-----------------------------------
This store exposes signals as artifacts. Consumers receive:

    Signal.id         (str)   — identifier
    Signal.type       (str)   — universal type from signal_types
    Signal.content    (str)   — the deposited trace itself
    Signal.strength   (float) — current strength in [0, 1]
    Signal.parent_id  (str?)  — provenance edge, ID only
    Signal.metadata   (dict)  — sanitized: contains scores and counts, never
                                rendered text from other agents

Methods that walk the provenance graph return SHAPES (lists of signal IDs,
counts of typed ancestors), not rendered strings of ancestor content. There
is no `get_dialogue_thread`, no `responses` field, no `parent_content` in
metadata. Agents that want context must re-sample from the store and reason
from the artifacts they receive.

Strength dynamics
-----------------
- Decay: per round, every signal s_v ← s_v * (1 - DECAY_RATE).
- Amplify: corroboration multiplies by AMPLIFY_FACTOR, capped at 1.
- Prune: signals below PRUNE_THRESHOLD are removed.
- Provenance boost: at deposit time, if the parent's ancestry contains
  VERIFICATION signals with avg strength >= BOOST_THRESHOLD, the new
  signal's initial strength is boosted by 1 + BOOST_BETA * avg, capped at 1.

Similarity dedup
----------------
At deposit, the new content is compared against recent same-type signals
using sentence-transformer embeddings (with string-similarity fallback).
If similarity >= DIVERSITY_THRESHOLD, the existing signal is amplified and
the deposit is rejected. This is similarity over the *trace*, not over any
agent's reasoning.
"""

from __future__ import annotations

import math
import time
import random
from threading import RLock
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from .config import (
    DECAY_RATE,
    PRUNE_THRESHOLD,
    AMPLIFY_FACTOR,
    EXPLORATION_BONUS,
    DIVERSITY_THRESHOLD,
    BOOST_THRESHOLD,
    BOOST_BETA,
    USE_LOGIT_DYNAMICS,
    DELTA_DECAY,
    DELTA_DECAY_CONTRARIAN,
    DELTA_AMPLIFY,
    DELTA_DEDUP_AMPLIFY,
    DELTA_BOOST_BETA,
    CLUSTERING_ENABLED_TYPES,
    PEER_PRUNE_MIN_MEMBERS,
    PEER_PRUNE_FACTOR,
)
from .signal_types import VERIFICATION, CONTRARIAN_TYPES
from .cluster_registry import ClusterRegistry


# ---------------------------------------------------------------------------
# Logit-space helpers
# ---------------------------------------------------------------------------

_LOGIT_EPS = 1e-6


def _to_logit(s: float, eps: float = _LOGIT_EPS) -> float:
    """Map strength in [0,1] to logit in R, clamped to avoid ±inf at endpoints."""
    s = min(1.0 - eps, max(eps, s))
    return math.log(s / (1.0 - s))


def _from_logit(x: float) -> float:
    """Sigmoid: map logit back to strength in (0,1)."""
    # Stable sigmoid for large |x|
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Signal artifact
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """An artifact deposited in the signal store.

    Consumers see only the fields below. There is intentionally no `responses`
    list, no `is_response_to`, and metadata never contains parent_content.
    """
    id: str
    type: str
    content: str
    strength: float
    timestamp: float
    depositor: str               # role tag (e.g. "scout", "forager") — not an agent ID,
                                 # to discourage agent-identity-based reasoning
    parent_id: Optional[str] = None
    partition_id: str = ""    # stamped from INITIAL metadata; inherited by SUPPORT
    cluster_id: Optional[str] = None  # assigned by ClusterRegistry at deposit time
    visits: int = 0
    # Iteration counter at deposit time. Set by SignalStore.deposit() from
    # its own ._current_iter, which is bumped by the worker pool each tick
    # via set_iteration(). Used by interaction-density age weighting in
    # core/projection.py and by the youth-grace gate in decay_all.
    # When SignalStore was never told about iterations (legacy round-based
    # path that doesn't call set_iteration), iter_at_deposit stays 0 for
    # every signal and the interaction-based metrics fall back to the
    # timestamp half-life calculation.
    iter_at_deposit: int = 0
    metadata: dict = field(default_factory=dict)
    # Internal: logit(strength), used only when USE_LOGIT_DYNAMICS is True.
    # Kept in sync with `strength` at every mutation point in SignalStore.
    # NOT part of the public artifact contract; consumers see `strength`.
    _logit: float = field(default=0.0, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.strength = max(0.0, min(1.0, self.strength))
        self._logit = _to_logit(self.strength)


# ---------------------------------------------------------------------------
# Optional embedding backend
# ---------------------------------------------------------------------------

class _DisabledEmbedder:
    """Explicit sentinel to disable the embedder without passing None.

    Passing ``embedder=False`` used to work by accident: ``_encode`` would
    call ``False.encode(text)`` which raises AttributeError inside the
    try/except, returning None.  This sentinel makes the intent explicit and
    won't silently break if _encode is ever tightened.
    """
    pass


_NULL_EMBEDDER = _DisabledEmbedder()


# Process-wide embedder cache + load-status. Loaded once (the model is ~90 MB
# and stateless), logged once. _EMBEDDER_STATUS is "all-MiniLM-L6-v2" on success
# or a short failure reason — surfaced in summary.json so a silently-degraded run
# is visible. A None embedder disables semantic clustering AND dedup, which makes
# every claim its own singleton cluster (see groqrun1.txt: 0 merges all run).
_EMBEDDER_SENTINEL = "unset"
_EMBEDDER_CACHE = _EMBEDDER_SENTINEL
_EMBEDDER_STATUS = "unset"


def _try_load_embedder():
    global _EMBEDDER_CACHE, _EMBEDDER_STATUS
    if _EMBEDDER_CACHE is not _EMBEDDER_SENTINEL:
        return _EMBEDDER_CACHE
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER_CACHE = SentenceTransformer("all-MiniLM-L6-v2")
        _EMBEDDER_STATUS = "all-MiniLM-L6-v2"
        print("[store] embedder loaded: all-MiniLM-L6-v2 "
              "(semantic clustering + dedup ON)")
    except Exception as exc:
        _EMBEDDER_CACHE = None
        _EMBEDDER_STATUS = f"UNAVAILABLE: {type(exc).__name__}: {str(exc)[:120]}"
        print(f"[store] *** embedder {_EMBEDDER_STATUS} *** — semantic clustering "
              f"AND dedup are OFF; every claim becomes its own singleton cluster. "
              f"Install sentence-transformers and ensure all-MiniLM-L6-v2 can "
              f"download (HF network/rate-limit).")
    return _EMBEDDER_CACHE


# ---------------------------------------------------------------------------
# SignalStore
# ---------------------------------------------------------------------------

class SignalStore:
    """Thread-safe trace medium with strength dynamics."""

    def __init__(self, embedder=None):
        self._lock = RLock()
        self._signals: dict[str, Signal] = {}
        self._next_id = 0
        # Interaction counter. The worker pool calls set_iteration() each
        # tick. When it stays 0 (legacy round-based path / tests that don't
        # plumb it), iter_at_deposit on every signal is 0 and downstream
        # interaction-density code falls back to its timestamp path.
        self._current_iter: int = 0

        # secondary indexes
        self._by_type: dict[str, set[str]] = {}
        self._by_parent: dict[str, set[str]] = {}

        # embedding cache (for dedup only — not rendered to agents)
        # Pass embedder=_NULL_EMBEDDER (or the module-level sentinel) to
        # explicitly disable embeddings; pass None to auto-detect.
        if embedder is _NULL_EMBEDDER or isinstance(embedder, _DisabledEmbedder):
            self._embedder = None
        elif embedder is False:
            # legacy: False was accidentally accepted; honour it with a warning
            import warnings
            warnings.warn(
                "SignalStore(embedder=False) is deprecated; "
                "use embedder=_NULL_EMBEDDER instead",
                DeprecationWarning, stacklevel=2,
            )
            self._embedder = None
        elif embedder is None:
            self._embedder = _try_load_embedder()
        else:
            self._embedder = embedder
        self._embeddings: dict[str, list[float]] = {}

        # Fix C': deposit-time clustering (replaces post-hoc greedy single-linkage)
        self._cluster_registry = ClusterRegistry()

    # ---- deposit ----------------------------------------------------------

    def deposit(
        self,
        signal_type: str,
        content: str,
        strength: float,
        depositor: str,
        parent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        """Deposit a new signal. Returns the new ID, or None if rejected as
        a near-duplicate (in which case the existing twin is amplified).
        """
        content = content.strip()
        if not content:
            return None

        with self._lock:
            # similarity dedup against recent same-type signals (last 5 min)
            cutoff = time.time() - 300
            same_type = [
                s for s in self._signals.values()
                if s.type == signal_type and s.timestamp >= cutoff
            ]

            # String pre-screen: check the 3 most-recent same-type signals with
            # SequenceMatcher before encoding. Near-identical text (ratio > 0.95)
            # is rejected without calling the embedder — saves one encode per dup.
            recent_3 = sorted(same_type, key=lambda s: s.timestamp, reverse=True)[:3]
            for existing in recent_3:
                quick_ratio = SequenceMatcher(
                    None, content.lower(), existing.content.lower()
                ).ratio()
                if quick_ratio > 0.95:
                    self._apply_dedup_amplify(existing, new_strength=strength)
                    existing.visits += 1
                    return None

            # Encode for clustering (Fix C') and provenance boost.
            # The old embedding-similarity rejection path (cosine >= DIVERSITY_THRESHOLD
            # → reject + amplify) is removed: near-duplicates now join the same cluster
            # instead of being rejected. The string pre-screen above still catches
            # truly identical text (ratio > 0.95).
            new_emb = self._encode(content)

            # provenance boost (clamped sanitized metadata)
            meta = dict(metadata) if metadata else {}
            if parent_id and parent_id in self._signals:
                avg_ver = self._avg_verification_strength(parent_id)
                if avg_ver >= BOOST_THRESHOLD:
                    if USE_LOGIT_DYNAMICS:
                        # additive logit boost; preserves ranking without saturating
                        new_logit = _to_logit(strength) + DELTA_BOOST_BETA * avg_ver
                        boosted = _from_logit(new_logit)
                        meta["provenance_boost_delta"] = round(DELTA_BOOST_BETA * avg_ver, 4)
                    else:
                        factor = 1.0 + BOOST_BETA * avg_ver
                        boosted = min(1.0, strength * factor)
                        meta["provenance_boost"] = round(factor, 4)
                    meta["original_strength"] = round(strength, 4)
                    strength = boosted

            # Resolve partition_id: explicit > inherited from parent > empty.
            # SUPPORT signals inherit from their parent (INITIAL or SUPPORT)
            # so the partition trace flows through the full chain.
            pid = meta.get("partition_id", "")
            if not pid and parent_id:
                parent_sig = self._signals.get(parent_id)
                if parent_sig and parent_sig.partition_id:
                    pid = parent_sig.partition_id

            sid = f"{signal_type}_{self._next_id:05d}"
            self._next_id += 1

            sig = Signal(
                id=sid,
                type=signal_type,
                content=content,
                strength=strength,
                timestamp=time.time(),
                depositor=depositor,
                parent_id=parent_id,
                partition_id=pid,
                iter_at_deposit=self._current_iter,
                metadata=meta,
            )

            if signal_type in ("INITIAL", "SUPPORT") and not sig.partition_id:
                raise AssertionError(
                    f"[PARTITION LEAK] {sig.depositor} deposited {sig.id} without "
                    f"partition_id. Scout dispatch is not propagating partitions."
                )

            self._signals[sid] = sig
            self._by_type.setdefault(signal_type, set()).add(sid)
            if parent_id:
                self._by_parent.setdefault(parent_id, set()).add(sid)
            if new_emb is not None:
                self._embeddings[sid] = new_emb

            # Fix C': assign cluster_id via ClusterRegistry for clusterable types.
            if new_emb is not None and signal_type in CLUSTERING_ENABLED_TYPES:
                cluster_id = self._cluster_registry.try_join(sid, new_emb, signal_type)
                if cluster_id is None:
                    cluster_id = self._cluster_registry.create(sid, new_emb, signal_type)
                sig.cluster_id = cluster_id
            elif signal_type == VERIFICATION and parent_id:
                # VERIFICATION inherits the parent's cluster_id for provenance tracking.
                parent_sig = self._signals.get(parent_id)
                if parent_sig and parent_sig.cluster_id:
                    sig.cluster_id = parent_sig.cluster_id

            # Gap 2: trail amplification for SUPPORT deposits
            from .signal_types import SUPPORT as _SUPPORT
            if signal_type == _SUPPORT:
                self._amplify_cluster_trail(sid, parent_id)

            return sid

    # ---- read accessors --------------------------------------------------

    def get(self, signal_id: str) -> Optional[Signal]:
        with self._lock:
            return self._signals.get(signal_id)

    def by_type(self, signal_type: str) -> list[Signal]:
        with self._lock:
            ids = self._by_type.get(signal_type, ())
            return [self._signals[i] for i in ids if i in self._signals]

    def all(self) -> list[Signal]:
        with self._lock:
            return list(self._signals.values())

    def stats(self) -> dict:
        with self._lock:
            counts: dict[str, int] = {}
            for s in self._signals.values():
                counts[s.type] = counts.get(s.type, 0) + 1
            strengths = [s.strength for s in self._signals.values()]
            return {
                "total": len(self._signals),
                "by_type": counts,
                "avg_strength": (sum(strengths) / len(strengths)) if strengths else 0.0,
                "max_strength": max(strengths) if strengths else 0.0,
            }

    # ---- checkpoint serialization (phase-isolated execution) -------------
    # Round-trips the store to/from a JSON file so subprocess-per-phase runs
    # can hand off state. Embeddings are NOT persisted; they are rebuilt
    # lazily on next deposit/similarity check.

    def save_state(self, path) -> None:
        import json
        from pathlib import Path
        with self._lock:
            payload = {
                "_next_id": self._next_id,
                "signals": [
                    {
                        "id": s.id,
                        "type": s.type,
                        "content": s.content,
                        "strength": s.strength,
                        "_logit": s._logit,
                        "timestamp": s.timestamp,
                        "depositor": s.depositor,
                        "parent_id": s.parent_id,
                        "partition_id": s.partition_id,
                        "cluster_id": s.cluster_id,
                        "visits": s.visits,
                        "metadata": s.metadata,
                    }
                    for s in self._signals.values()
                ],
            }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_state(self, path) -> None:
        """Replace this store's contents with the state at `path`. In-place."""
        import json
        from pathlib import Path
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        with self._lock:
            self._signals.clear()
            self._by_type.clear()
            self._by_parent.clear()
            self._embeddings.clear()
            self._next_id = int(data.get("_next_id", 0))
            for sd in data.get("signals", []):
                sig = Signal(
                    id=sd["id"],
                    type=sd["type"],
                    content=sd["content"],
                    strength=float(sd["strength"]),
                    timestamp=float(sd.get("timestamp", time.time())),
                    depositor=sd.get("depositor", "unknown"),
                    parent_id=sd.get("parent_id"),
                    partition_id=sd.get("partition_id", ""),
                    cluster_id=sd.get("cluster_id"),
                    visits=int(sd.get("visits", 0)),
                    metadata=dict(sd.get("metadata", {})),
                )
                # Restore canonical _logit if present (preserves USE_LOGIT_DYNAMICS state).
                if "_logit" in sd:
                    sig._logit = float(sd["_logit"])
                self._signals[sig.id] = sig
                self._by_type.setdefault(sig.type, set()).add(sig.id)
                if sig.parent_id:
                    self._by_parent.setdefault(sig.parent_id, set()).add(sig.id)

    # ---- sampling --------------------------------------------------------

    def sample_weighted(self, signal_type: str, n: int = 1) -> list[Signal]:
        """Strength-weighted sampling with exploration bonus.

        Bumps `visits` on every sample so the exploration bonus actually
        decays over time for popular signals.
        """
        with self._lock:
            cands = self.by_type(signal_type)
            if not cands:
                return []
            max_v = max((s.visits for s in cands), default=0) or 1
            weights = []
            for s in cands:
                base = s.strength + 0.1
                bonus = EXPLORATION_BONUS * (1.0 - s.visits / max_v)
                weights.append(base + bonus)
            k = min(n, len(cands))
            picked = random.choices(cands, weights=weights, k=k)
            # m2 fix: random.choices is sampling-with-replacement; dedupe visit bumps
            for s in {id(x): x for x in picked}.values():
                s.visits += 1
            return picked

    def sample_stratified(
        self,
        signal_type: str,
        weak: int = 0,
        medium: int = 0,
        strong: int = 0,
        weak_t: float = 0.4,
        strong_t: float = 0.7,
    ) -> list[Signal]:
        """Sample by strength stratum without bias toward strong signals."""
        with self._lock:
            cands = self.by_type(signal_type)
            ws = [s for s in cands if s.strength < weak_t]
            ms = [s for s in cands if weak_t <= s.strength < strong_t]
            ss = [s for s in cands if s.strength >= strong_t]
            out: list[Signal] = []
            for pool, k in ((ws, weak), (ms, medium), (ss, strong)):
                if pool and k > 0:
                    out.extend(random.sample(pool, min(k, len(pool))))
            for s in out:
                s.visits += 1
            return out

    def sample_recent(self, signal_type: str, n: int, max_age_s: float = 60.0) -> list[Signal]:
        with self._lock:
            cutoff = time.time() - max_age_s
            cands = [s for s in self.by_type(signal_type) if s.timestamp >= cutoff]
            cands.sort(key=lambda s: s.timestamp, reverse=True)
            picked = cands[:n]
            for s in picked:
                s.visits += 1
            return picked

    def sample_under_visited(self, signal_type: str, n: int) -> list[Signal]:
        with self._lock:
            cands = self.by_type(signal_type)
            cands.sort(key=lambda s: s.visits)
            picked = cands[:n]
            for s in picked:
                s.visits += 1
            return picked

    def sample_from_clusters(
        self,
        signal_type: str,
        n: int = 1,
        worker_centroid: Optional[list[float]] = None,
    ) -> list[Signal]:
        """Stigmergic cluster-trail sampling (Gap 1 fix).

        Selects a cluster first (weighted by total member strength, optionally
        biased toward the worker's current semantic position), then samples
        within that cluster weighted by individual signal strength.

        Falls back to sample_weighted() when:
          - USE_CLUSTER_AWARE_SAMPLING is False
          - ClusterRegistry has no clusters for this signal_type
          - No embeddings are available (worker_centroid is None and no embs)
        """
        from .config import USE_CLUSTER_AWARE_SAMPLING
        if not USE_CLUSTER_AWARE_SAMPLING:
            return self.sample_weighted(signal_type, n)

        with self._lock:
            raw_clusters = self._cluster_registry.clusters_by_type(signal_type)
            live_clusters = []
            for cl in raw_clusters:
                members = [self._signals[mid] for mid in cl.member_ids
                           if mid in self._signals]
                if members:
                    live_clusters.append((cl, members))

            if not live_clusters:
                return self.sample_weighted(signal_type, n)

            # Weight each cluster: base = sum of member strengths.
            # If worker has a centroid, multiply by cosine sim to cluster centroid.
            # Diversity penalty: divide by log(1 + size) so dominant clusters
            # don't monopolise sampling as they grow — each new member adds
            # diminishing marginal weight. Without this, trail amplification +
            # cluster-aware sampling creates a winner-takes-all feedback loop.
            cluster_weights = []
            for cl, members in live_clusters:
                base_w = sum(s.strength for s in members) + 0.1
                if worker_centroid is not None and cl.centroid:
                    sim = float(sum(a * b for a, b in zip(worker_centroid, cl.centroid)))
                    # sim in [-1, 1]; shift to [0.1, 1.1] so no cluster is zeroed out
                    base_w *= max(0.1, (sim + 1.0) / 2.0 + 0.1)
                size_penalty = math.log1p(len(members))
                cluster_weights.append(max(0.01, base_w / size_penalty))

            # Pick a cluster
            chosen_cl, chosen_members = random.choices(
                live_clusters, weights=cluster_weights, k=1
            )[0]

            # Sample within the cluster weighted by strength + exploration bonus
            max_v = max((s.visits for s in chosen_members), default=0) or 1
            member_weights = []
            for s in chosen_members:
                base = s.strength + 0.1
                bonus = EXPLORATION_BONUS * (1.0 - s.visits / max_v)
                member_weights.append(base + bonus)

            k = min(n, len(chosen_members))
            picked = random.choices(chosen_members, weights=member_weights, k=k)
            for s in {id(x): x for x in picked}.values():
                s.visits += 1
            return picked

    # ---- live gap-fill helpers (for foragers and validators) ------------

    def signals_with_few_children_of_type(
        self, signal_type: str, child_type: str, max_children: int
    ) -> list[Signal]:
        """Return signals of `signal_type` with fewer than `max_children`
        direct children of `child_type`.

        Used by foragers: sample INITIALs that haven't yet attracted enough
        SUPPORT to cross the support_diversity threshold. This converts
        foragers from broad explorers to gap-fillers.
        """
        with self._lock:
            result = []
            for sig in self.by_type(signal_type):
                child_count = sum(
                    1 for cid in self._by_parent.get(sig.id, ())
                    if self._signals.get(cid) and self._signals[cid].type == child_type
                )
                if child_count < max_children:
                    result.append(sig)
            return result

    def signals_with_many_children_of_type(
        self, signal_type: str, child_type: str, min_children: int
    ) -> list[Signal]:
        """Return signals of `signal_type` with at least `min_children`
        direct children of `child_type`.

        Used by validators: target high-stakes INITIALs that already have
        accumulated support, so Wikipedia lookups are spent on claims that
        are likely to survive rather than on orphaned one-offs.
        """
        with self._lock:
            result = []
            for sig in self.by_type(signal_type):
                child_count = sum(
                    1 for cid in self._by_parent.get(sig.id, ())
                    if self._signals.get(cid) and self._signals[cid].type == child_type
                )
                if child_count >= min_children:
                    result.append(sig)
            return result

    # ---- consensus-cluster view (for haters) ----------------------------

    def consensus_summary(self, signal_type: str, k: int = 3) -> dict:
        """Distributional view of the signal field for one type.

        Returns counts and representative *content traces* — never any
        agent's reasoning. Hater agents condition on this to challenge
        the cluster, not on individual depositors' minds.
        """
        with self._lock:
            cands = self.by_type(signal_type)
            if not cands:
                return {"count": 0, "avg_strength": 0.0, "representatives": []}
            cands.sort(key=lambda s: s.strength, reverse=True)
            reps = cands[:k]
            avg = sum(s.strength for s in cands) / len(cands)
            return {
                "count": len(cands),
                "avg_strength": round(avg, 4),
                "representatives": [
                    {"id": s.id, "content": s.content, "strength": round(s.strength, 4)}
                    for s in reps
                ],
            }

    # ---- DBSCAN clustering (for haters) ------------------------------------

    def cluster_signals_dbscan(
        self, signal_type: str, eps: float = 0.35
    ) -> list[list["Signal"]]:
        """Cosine-DBSCAN clustering over cached embeddings for one signal type.

        Uses a simplified DBSCAN: two signals are neighbours if their cosine
        similarity >= (1 - eps). Core-points are signals with >= 1 neighbour
        (minPts=2 including self). Noise points form singleton clusters.

        Returns a list of clusters (each a list of Signal objects), sorted
        by cluster size descending. Used by Hater to identify the dominant
        consensus cluster to challenge.

        If no embeddings are available, falls back to one cluster per signal
        (degenerate case; hater then sees each signal as its own cluster).
        """
        with self._lock:
            sigs = self.by_type(signal_type)
            if not sigs:
                return []

            # Gather embeddings
            embs = [(s, self._embeddings.get(s.id)) for s in sigs]
            has_embs = any(e is not None for _, e in embs)

            if not has_embs:
                return [[s] for s in sigs]

            n = len(sigs)
            visited = [False] * n
            cluster_ids = [-1] * n

            def neighbours(i: int) -> list[int]:
                ei = embs[i][1]
                if ei is None:
                    return []
                result = []
                for j, (_, ej) in enumerate(embs):
                    if i == j or ej is None:
                        continue
                    sim = float(sum(a * b for a, b in zip(ei, ej)))
                    if sim >= (1.0 - eps):
                        result.append(j)
                return result

            cid = 0
            for i in range(n):
                if visited[i]:
                    continue
                visited[i] = True
                nbrs = neighbours(i)
                if not nbrs:
                    cluster_ids[i] = -1  # noise
                else:
                    cluster_ids[i] = cid
                    queue = list(nbrs)
                    while queue:
                        j = queue.pop()
                        if not visited[j]:
                            visited[j] = True
                            j_nbrs = neighbours(j)
                            if j_nbrs:
                                queue.extend(j_nbrs)
                        if cluster_ids[j] == -1:
                            cluster_ids[j] = cid
                    cid += 1

            # Group by cluster
            clusters: dict[int, list] = {}
            for i, (s, _) in enumerate(embs):
                ckey = cluster_ids[i]
                if ckey == -1:
                    ckey = cid  # each noise point is its own cluster
                    cid += 1
                clusters.setdefault(ckey, []).append(s)

            return sorted(clusters.values(), key=len, reverse=True)

    # ---- amplify / decay / prune ----------------------------------------

    def amplify(self, signal_id: str, factor: float = AMPLIFY_FACTOR) -> bool:
        with self._lock:
            s = self._signals.get(signal_id)
            if not s:
                return False
            if USE_LOGIT_DYNAMICS:
                # `factor` is ignored in logit mode; the delta is a config constant.
                # This is the intentional design: amplification is a fixed step in
                # information space, not a per-call multiplier.
                s._logit += DELTA_AMPLIFY
                s.strength = _from_logit(s._logit)
            else:
                s.strength = min(1.0, s.strength * factor)
                s._logit = _to_logit(s.strength)
            s.visits += 1
            return True

    def decay_all(self, factor: float = 1.0) -> None:
        """Apply decay scaled by `factor`.

        Logit mode (default, USE_LOGIT_DYNAMICS=True):
            non-contrarian:  logit += DELTA_DECAY              × factor
            contrarian:      logit += DELTA_DECAY_CONTRARIAN   × factor

        `factor=1.0` is the legacy "one round" decay magnitude. Callers in
        the continuous worker pool pass a smaller factor (e.g. 0.2) so a
        60 s decay tick applies 1/5 of a legacy-round's decay rather than a
        full round's worth — without this scaling, continuous-pool runs
        apply decay 5–25× more often than the dynamics were calibrated for
        and prune SUPPORT bases out from under CHAIN before depth can
        accumulate.

        Legacy mode (USE_LOGIT_DYNAMICS=False) preserves the old
        multiplicative path with exact anti-decay for contrarians; kept as
        an escape hatch for one release. factor scales the (1 - DECAY_RATE)
        step toward 1.0 — factor=0.0 means no decay; factor=1.0 means full.
        """
        # Lazy import: MIN_INTERACTIONS_BEFORE_DECAY is a config knob and
        # carries calibration meaning, so we don't bake it into a constant
        # at module-load — env overrides applied by config.py would miss.
        try:
            from .config import MIN_INTERACTIONS_BEFORE_DECAY as _MIN_INT
        except ImportError:
            _MIN_INT = 0
        with self._lock:
            current_iter = self._current_iter
            if USE_LOGIT_DYNAMICS:
                d_main = DELTA_DECAY * factor
                d_contra = DELTA_DECAY_CONTRARIAN * factor
                for s in self._signals.values():
                    # Youth grace: signals younger than _MIN_INT iterations
                    # skip decay entirely. Removes the "deposited late and
                    # immediately punished" pathology while the rest of the
                    # field continues to consolidate. When current_iter is 0
                    # (no worker pool wiring) iter_age is 0 for everyone, so
                    # this branch effectively no-ops and behavior matches
                    # the pre-pivot path.
                    if _MIN_INT > 0 and current_iter > 0:
                        if (current_iter - s.iter_at_deposit) < _MIN_INT:
                            continue
                    if s.type in CONTRARIAN_TYPES:
                        s._logit += d_contra
                    else:
                        s._logit += d_main
                    s.strength = _from_logit(s._logit)
                return
            decay = DECAY_RATE * factor
            anti_decay = 1.0 / (1.0 - decay) if decay < 1.0 else 1.0
            for s in self._signals.values():
                if _MIN_INT > 0 and current_iter > 0:
                    if (current_iter - s.iter_at_deposit) < _MIN_INT:
                        continue
                s.strength *= (1.0 - decay)
                if s.type in CONTRARIAN_TYPES:
                    s.strength = min(1.0, s.strength * anti_decay)
                s._logit = _to_logit(s.strength)

    # ---- logit-update helpers (internal) --------------------------------

    def _apply_dedup_amplify(self, sig: Signal,
                              new_strength: Optional[float] = None) -> None:
        """Bump strength when a near-duplicate deposit arrives.

        For VERIFICATION signals specifically: when the new deposit's strength
        is higher than the existing one, REPLACE the strength (high-water-
        mark) rather than additively amplify. Otherwise a stale 0.0
        verification deposited early in the run dominates over fresh higher-
        score validators that happen to land as dedup-matches — the prior
        run's verification floor stayed at zero because of this.

        For other signal types, additive amplify (the original semantics):
        repeated deposit of the same content = stigmergic reinforcement.

        Logit mode: additive delta (DELTA_DEDUP_AMPLIFY).
        Legacy mode: multiplicative ×1.05 (matches prior behaviour).
        """
        if sig.type == VERIFICATION and new_strength is not None and new_strength > sig.strength:
            new_strength = min(1.0, max(0.0, new_strength))
            sig.strength = new_strength
            sig._logit = _to_logit(new_strength)
            return
        if USE_LOGIT_DYNAMICS:
            sig._logit += DELTA_DEDUP_AMPLIFY
            sig.strength = _from_logit(sig._logit)
        else:
            sig.strength = min(1.0, sig.strength * 1.05)
            sig._logit = _to_logit(sig.strength)

    def _amplify_cluster_trail(self, deposited_id: str, parent_id: Optional[str]) -> None:
        """Reinforce the cluster neighbourhood when a SUPPORT lands (Gap 2 fix).

        Must be called INSIDE the store lock (called from deposit()).
        """
        from .config import USE_TRAIL_AMPLIFICATION, DELTA_CLUSTER_TRAIL
        if not USE_TRAIL_AMPLIFICATION:
            return
        if not USE_LOGIT_DYNAMICS:
            return

        lookup_id = parent_id if parent_id else deposited_id
        cluster_id = self._cluster_registry.get_cluster_id(lookup_id)
        if cluster_id is None:
            return

        cl = self._cluster_registry.get_cluster(cluster_id)
        if cl is None or not cl.member_ids:
            return

        live_members = [self._signals[mid] for mid in cl.member_ids
                        if mid in self._signals and mid != deposited_id]
        if not live_members:
            return

        delta = DELTA_CLUSTER_TRAIL / max(1, len(live_members))
        for s in live_members:
            s._logit += delta
            s.strength = _from_logit(s._logit)

    def recluster(self, signal_type: str = "INITIAL") -> int:
        """Run a divisive re-cluster pass (corrective): force-split incohesive
        cluster blobs so similarity re-decides the partition. Intended to be
        called every CLUSTER_RECLUSTER_EVERY iterations and once before
        projection. Returns the number of splits made. Held under the store lock.

        Hook (run loop): after decay_all()/prune_weak() each round, or on an
        iteration counter — e.g. `if iter % CLUSTER_RECLUSTER_EVERY == 0: store.recluster()`.
        """
        with self._lock:
            return self._cluster_registry.recluster_type(signal_type)

    def prune_weak(self) -> int:
        """Remove signals below the prune threshold.

        Peer-relative protection: clusters with >= PEER_PRUNE_MIN_MEMBERS live
        member signals have their prune floor halved (PRUNE_THRESHOLD *
        PEER_PRUNE_FACTOR). This prevents the richest cluster from being
        uniformly decayed below the absolute floor while isolated weak signals
        survive by never accumulating enough decay to matter.
        """
        with self._lock:
            # Count live members per cluster for peer-relative protection.
            cluster_live: dict = {}
            for s in self._signals.values():
                if s.cluster_id:
                    cluster_live[s.cluster_id] = cluster_live.get(s.cluster_id, 0) + 1
            protected = {
                cid for cid, n in cluster_live.items()
                if n >= PEER_PRUNE_MIN_MEMBERS
            }

            doomed = []
            for sid, s in self._signals.items():
                floor = PRUNE_THRESHOLD
                if s.cluster_id in protected:
                    floor = PRUNE_THRESHOLD * PEER_PRUNE_FACTOR
                if s.strength < floor:
                    doomed.append(sid)
            for sid in doomed:
                self._delete_locked(sid)
            return len(doomed)

    # ---- provenance walks (return SHAPES, not rendered text) ------------

    def ancestor_ids(self, signal_id: str, target_type: Optional[str] = None) -> list[str]:
        """Walk parent links upward. Returns IDs only.

        Agents do not get rendered ancestor *content*; provenance is
        structural information, not an excuse to read prior reasoning.
        """
        with self._lock:
            seen: set[str] = set()
            out: list[str] = []
            cur = self._signals.get(signal_id)
            while cur and cur.parent_id and cur.parent_id not in seen:
                seen.add(cur.parent_id)
                parent = self._signals.get(cur.parent_id)
                if not parent:
                    break
                if target_type is None or parent.type == target_type:
                    out.append(parent.id)
                cur = parent
            return out

    def descendant_count(self, signal_id: str, target_type: Optional[str] = None) -> int:
        """Count descendants of a given type. Returns a number, not content."""
        with self._lock:
            seen: set[str] = set()
            queue: list[str] = list(self._by_parent.get(signal_id, set()))
            count = 0
            while queue:
                sid = queue.pop()
                if sid in seen:
                    continue
                seen.add(sid)
                s = self._signals.get(sid)
                if not s:
                    continue
                if target_type is None or s.type == target_type:
                    count += 1
                queue.extend(self._by_parent.get(sid, set()))
            return count

    # ---- public helpers (used by projection and knowledge base) ----------

    def verification_strength(self, signal_id: str) -> float:
        """Public wrapper around _avg_verification_strength."""
        with self._lock:
            return self._avg_verification_strength(signal_id)

    def get_embedding(self, signal_id: str) -> Optional[list[float]]:
        """Return cached unit-normalized embedding for a signal, or None."""
        with self._lock:
            return self._embeddings.get(signal_id)

    def embedder_status(self) -> str:
        """'all-MiniLM-L6-v2' if semantic clustering/dedup is active, else a
        short reason it isn't. Surfaced in summary.json so a silently-degraded
        run (None embedder -> all-singleton clusters) is visible."""
        if self._embedder is None:
            # Distinguish an explicitly-disabled store from a failed auto-load.
            return _EMBEDDER_STATUS if _EMBEDDER_STATUS != "unset" else "disabled"
        return _EMBEDDER_STATUS if _EMBEDDER_STATUS != "unset" else "active"

    def by_parent(self, parent_id: str) -> set[str]:
        """Return IDs of direct children of parent_id (read-only copy)."""
        with self._lock:
            return set(self._by_parent.get(parent_id, set()))

    def get_live_clusters(self, signal_type: str) -> list[tuple[str, list[str]]]:
        """Return (representative_id, [member_ids]) for live ClusterRegistry clusters.

        Only includes signal_ids currently in the store (pruned signals excluded).
        Representative = strongest live member. Clusters sorted by size descending
        (same order as cluster_signals_dbscan() output so callers are interchangeable).
        Returns empty list when ClusterRegistry has no clusters for this type.
        """
        with self._lock:
            raw_clusters = self._cluster_registry.clusters_by_type(signal_type)
            result = []
            for cl in raw_clusters:
                live = [mid for mid in cl.member_ids if mid in self._signals]
                if not live:
                    continue
                rep_id = max(live, key=lambda mid: self._signals[mid].strength)
                result.append((rep_id, live))
            result.sort(key=lambda x: len(x[1]), reverse=True)
            return result

    def set_iteration(self, n: int) -> None:
        """Advance the store's internal iteration counter.

        Called by the worker pool each tick. Every deposit() that follows
        stamps `iter_at_deposit = n`. Setting backward is allowed but
        unusual; the counter is monotonic in practice.
        """
        with self._lock:
            self._current_iter = int(n)

    def get_iteration(self) -> int:
        """Read the current iteration counter."""
        with self._lock:
            return self._current_iter

    def iter_age(self, signal_id: str) -> int:
        """Iterations elapsed since this signal was deposited.

        Used by the interaction-density age weighting in projection.py
        and by decay_all's youth-grace gate. Returns 0 for signals
        deposited before set_iteration() was wired (legacy path).
        """
        with self._lock:
            s = self._signals.get(signal_id)
            if s is None:
                return 0
            return max(0, self._current_iter - s.iter_at_deposit)

    def mark_read(self, signal_id: str, n: int = 1) -> None:
        """Bump `visits` by n on a signal that was rendered into a prompt.

        `visits` was previously only incremented by the random-sample helpers
        (sample_strength_biased, sample_recent, sample_under_visited). That
        missed every signal pulled in as context via the dissent-routing,
        REFINE-rebuttal, or synthesizer-traverse paths — so OBJECTION /
        CRITIQUE_NEGATIVE signals showed visits=0 across full runs even when
        they actually informed downstream generation.

        Callers should invoke this *after* successfully reading the content
        into the prompt body, so visits reflects only real prompt inclusion,
        not arbitrary lookups like provenance walks.
        """
        if n <= 0:
            return
        with self._lock:
            s = self._signals.get(signal_id)
            if s is not None:
                s.visits += int(n)

    def mark_many_read(self, signal_ids) -> None:
        """Batch helper for callers that consume several signals at once."""
        with self._lock:
            for sid in signal_ids:
                s = self._signals.get(sid)
                if s is not None:
                    s.visits += 1

    # ---- internals -------------------------------------------------------

    def _avg_verification_strength(self, signal_id: str) -> float:
        """Signed-average strength of VERIFICATION signals on the lineage.

        Walks both the signal itself AND every ancestor, collecting any
        VERIFICATION signal that is (a) the node itself, or (b) a direct
        child of the node. Deduplicates by ID.

        §5 validator fix: strength below 0.3 counts as anti-evidence.
        The signed contribution is (strength - 0.3) for each signal,
        normalised by count. This means a VERIFICATION at strength 0.0
        contributes -0.3 per signal (strong contradiction), while one at
        strength 1.0 contributes +0.7 (strong support).

        Return is in (-0.3, 0.7]. Callers (provenance boost in deposit())
        use this value directly; BOOST_THRESHOLD (0.7) is calibrated
        against this range so only genuine positive verifications trigger.
        """
        seen_ver: dict[str, "Signal"] = {}
        chain = [signal_id] + self.ancestor_ids(signal_id)
        for sid in chain:
            node = self._signals.get(sid)
            if node and node.type == VERIFICATION and node.id not in seen_ver:
                seen_ver[node.id] = node
            for child_id in self._by_parent.get(sid, ()):
                child = self._signals.get(child_id)
                if child and child.type == VERIFICATION and child.id not in seen_ver:
                    seen_ver[child.id] = child
        if not seen_ver:
            return 0.0
        # Signed average: strength < 0.3 is anti-evidence
        signed_sum = sum((v.strength - 0.3) for v in seen_ver.values())
        return signed_sum / len(seen_ver)

    def _delete_locked(self, signal_id: str) -> None:
        s = self._signals.pop(signal_id, None)
        if not s:
            return
        if s.type in self._by_type:
            self._by_type[s.type].discard(signal_id)
        if s.parent_id and s.parent_id in self._by_parent:
            self._by_parent[s.parent_id].discard(signal_id)
        self._by_parent.pop(signal_id, None)  # clean up own child-list entry
        self._embeddings.pop(signal_id, None)

    def _encode(self, text: str) -> Optional[list[float]]:
        if self._embedder is None:
            return None
        try:
            import numpy as np
            v = self._embedder.encode(text)
            arr = np.asarray(v, dtype="float32")
            n = float((arr ** 2).sum() ** 0.5)
            if n > 0:
                arr = arr / n
            return arr.tolist()
        except Exception:
            return None

    def _similarity(self, a: str, b: str, ea, eb) -> float:
        if ea is not None and eb is not None:
            # already normalized — cosine = dot
            return float(sum(x * y for x, y in zip(ea, eb)))
        if self._embedder is not None and (ea is not None or eb is not None):
            # only one side has an embedding; encode the other
            other = self._encode(b if ea is not None else a)
            if other is not None:
                ea2, eb2 = (ea, other) if ea is not None else (other, eb)
                return float(sum(x * y for x, y in zip(ea2, eb2)))
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
