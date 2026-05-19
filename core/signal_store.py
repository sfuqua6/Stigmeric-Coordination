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
)
from .signal_types import VERIFICATION, CONTRARIAN_TYPES


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
    visits: int = 0
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


def _try_load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SignalStore
# ---------------------------------------------------------------------------

class SignalStore:
    """Thread-safe trace medium with strength dynamics."""

    def __init__(self, embedder=None):
        self._lock = RLock()
        self._signals: dict[str, Signal] = {}
        self._next_id = 0

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

            # Encode for dedup comparison; always encode so the embedding is
            # available for downstream clustering even when same_type is empty.
            new_emb = self._encode(content)

            for existing in same_type:
                sim = self._similarity(content, existing.content,
                                       new_emb, self._embeddings.get(existing.id))
                if sim >= DIVERSITY_THRESHOLD:
                    self._apply_dedup_amplify(existing, new_strength=strength)
                    existing.visits += 1
                    return None

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
                metadata=meta,
            )
            self._signals[sid] = sig
            self._by_type.setdefault(signal_type, set()).add(sid)
            if parent_id:
                self._by_parent.setdefault(parent_id, set()).add(sid)
            if new_emb is not None:
                self._embeddings[sid] = new_emb

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
        with self._lock:
            if USE_LOGIT_DYNAMICS:
                d_main = DELTA_DECAY * factor
                d_contra = DELTA_DECAY_CONTRARIAN * factor
                for s in self._signals.values():
                    if s.type in CONTRARIAN_TYPES:
                        s._logit += d_contra
                    else:
                        s._logit += d_main
                    s.strength = _from_logit(s._logit)
                return
            decay = DECAY_RATE * factor
            anti_decay = 1.0 / (1.0 - decay) if decay < 1.0 else 1.0
            for s in self._signals.values():
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

    def prune_weak(self) -> int:
        with self._lock:
            doomed = [sid for sid, s in self._signals.items() if s.strength < PRUNE_THRESHOLD]
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

    def by_parent(self, parent_id: str) -> set[str]:
        """Return IDs of direct children of parent_id (read-only copy)."""
        with self._lock:
            return set(self._by_parent.get(parent_id, set()))

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
