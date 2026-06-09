# Claude Code Prompt: Stigmergic Coordination Upgrade

## Context

You are working in the `Attempt At Cleaning/` subdirectory of an LLM multi-agent swarm
codebase. The system is a stigmergic signal store where agents deposit typed signals
(INITIAL, SUPPORT, CRITIQUE, OBJECTION, VERIFICATION) into a shared store, with
strength dynamics (logit-space decay/amplify/prune) and a continuous worker pool.

The codebase already has:
- `core/signal_store.py` — typed DAG with RLock, logit-space dynamics, embedding cache
- `core/cluster_registry.py` — incremental centroid clustering at deposit time
- `core/worker_pool.py` — 24 async workers with action selection and SAFE retrieval
- `core/sampling.py` — per-role sampling strategies (stratified, under-visited, etc.)
- `core/config.py` — all tunables with tier-detection

The system CLAIMS to be stigmergic but currently operates as a blackboard. The four
specific gaps are:

1. **No cluster-aware sampling** — `_sample_initial()` pulls from a flat list of all
   INITIAL signals, ignoring the ClusterRegistry entirely. Agents can't follow trails.

2. **No trail amplification** — when a SUPPORT deposit succeeds against a cluster,
   the cluster's other members are not reinforced. Traversing a trail leaves no trace.

3. **Inverted action causality** — `choose_action()` picks an action type from global
   share counters BEFORE the worker reads the field. In stigmergy, local field state
   at the agent's position should bias which action fires.

4. **Workers have no semantic position** — `_own_scout_excerpts` is a list of strings.
   Without a position in embedding space, workers can't specialize into niches and
   there is no gradient to follow.

## Task

Implement the four changes below. Each is independently gated by a new config flag
so any one of them can be disabled without touching the others. After each change,
run `pytest tests/ -x -q` and fix any failures before moving to the next.

---

## Change 1: Cluster-Aware Sampling

### 1a. Add to `core/config.py`

```python
# Stigmergy upgrade — Gap 1
USE_CLUSTER_AWARE_SAMPLING: bool = True
```

Add it near the other `USE_*` feature flags.

### 1b. Add to `core/signal_store.py`

Add a new method `sample_from_clusters()` to `SignalStore`, placed after
`sample_under_visited()` and before `signals_with_few_children_of_type()`:

```python
def sample_from_clusters(
    self,
    signal_type: str,
    n: int = 1,
    worker_centroid: Optional[list[float]] = None,
) -> list[Signal]:
    """Stigmergic cluster-trail sampling.

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
        cluster_weights = []
        for cl, members in live_clusters:
            base_w = sum(s.strength for s in members) + 0.1
            if worker_centroid is not None and cl.centroid:
                sim = float(sum(a * b for a, b in zip(worker_centroid, cl.centroid)))
                # sim in [-1, 1]; shift to [0.1, 1.1] so no cluster is zeroed out
                base_w *= max(0.1, (sim + 1.0) / 2.0 + 0.1)
            cluster_weights.append(max(0.01, base_w))

        # Pick a cluster
        import random as _random
        chosen_cl, chosen_members = _random.choices(
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
        picked = _random.choices(chosen_members, weights=member_weights, k=k)
        for s in {id(x): x for x in picked}.values():
            s.visits += 1
        return picked
```

### 1c. Replace `_sample_initial()` and `_sample_underserved_initial()` in `core/worker_pool.py`

In `_sample_initial()`, replace the body with:

```python
def _sample_initial(store: SignalStore, recent: Counter,
                    rng: random.Random,
                    worker_centroid=None) -> Optional[Signal]:
    results = store.sample_from_clusters(INITIAL, n=1,
                                         worker_centroid=worker_centroid)
    if not results:
        return None
    s = results[0]
    if recent.get(s.id, 0) > 0:
        # Got a recently-visited signal; try once more then accept it
        alt = store.sample_from_clusters(INITIAL, n=1,
                                          worker_centroid=worker_centroid)
        if alt and recent.get(alt[0].id, 0) == 0:
            return alt[0]
    return s
```

In `_sample_underserved_initial()`, add a `worker_centroid=None` parameter and call:

```python
underserved = store.signals_with_few_children_of_type(INITIAL, SUPPORT, 2)
if underserved:
    results = store.sample_from_clusters(INITIAL, n=1,
                                          worker_centroid=worker_centroid)
    # Prefer underserved members; if cluster sample landed on a well-served
    # signal, check if the cluster has underserved members and pick one.
    if results and results[0] in underserved:
        return results[0]
    # Fallback: pick the least-served from the underserved list
    weights = [max(0.05, s.strength) for s in underserved]
    return rng.choices(underserved, weights=weights, k=1)[0]
pool_list = store.by_type(INITIAL)
if not pool_list:
    return None
weights = [max(0.05, s.strength) for s in pool_list]
return rng.choices(pool_list, weights=weights, k=1)[0]
```

---

## Change 2: Trail Amplification

### 2a. Add to `core/config.py`

```python
# Stigmergy upgrade — Gap 2
USE_TRAIL_AMPLIFICATION: bool = True
# Fraction of DELTA_AMPLIFY applied to cluster siblings when a SUPPORT lands.
# Scaled by 1/cluster_size so large clusters don't compound unboundedly.
# Start conservative — can be raised once empirical runs confirm the effect.
DELTA_CLUSTER_TRAIL: float = 0.03
```

### 2b. Add `_amplify_cluster_trail()` to `SignalStore` in `core/signal_store.py`

Add this private method after `_apply_dedup_amplify()`:

```python
def _amplify_cluster_trail(self, deposited_id: str, parent_id: Optional[str]) -> None:
    """Reinforce the cluster neighbourhood when a SUPPORT lands (Gap 2 fix).

    When a non-duplicate SUPPORT signal is deposited, all live members of the
    parent signal's cluster receive a small logit-space boost. The boost is
    divided by cluster size so reinforcement is intensive on small clusters
    (building a new trail) and diffuse on large ones (trail already established).

    Only fires when USE_TRAIL_AMPLIFICATION is True and embeddings are present
    (if no ClusterRegistry clusters exist, silently no-ops).

    Must be called INSIDE the store lock (called from deposit()).
    """
    from .config import USE_TRAIL_AMPLIFICATION, DELTA_CLUSTER_TRAIL
    if not USE_TRAIL_AMPLIFICATION:
        return
    if not USE_LOGIT_DYNAMICS:
        return  # only defined for logit path; skip on legacy path

    # Find the cluster the parent belongs to
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
```

### 2c. Call `_amplify_cluster_trail()` from `deposit()` in `core/signal_store.py`

In `SignalStore.deposit()`, immediately after the line `return sid` (i.e., after the
signal is registered but before the method returns), insert — inside the `with self._lock`
block — a call to `_amplify_cluster_trail` only for SUPPORT signals:

Find the line:
```python
            return sid
```

Replace it with:
```python
            # Gap 2: trail amplification for SUPPORT deposits
            from .signal_types import SUPPORT as _SUPPORT
            if signal_type == _SUPPORT:
                self._amplify_cluster_trail(sid, parent_id)

            return sid
```

---

## Change 3: Field-Induced Local Action Bias

### 3a. Add to `core/config.py`

```python
# Stigmergy upgrade — Gap 3
USE_LOCAL_ACTION_BIAS: bool = True
# Multipliers applied to base action weights when local cluster state matches.
# Values > 1.0 increase the action's chance; < 1.0 decrease it.
LOCAL_BIAS_DEVELOP_LOW_DIVERSITY: float = 2.0   # cluster support_diversity < 2
LOCAL_BIAS_REFINE_HAS_DISSENT: float = 2.0      # cluster has OBJECTION/CRITIQUE_NEG
LOCAL_BIAS_VALIDATE_NO_VERIFY: float = 2.0      # cluster has no VERIFICATION
LOCAL_BIAS_CHAIN_DEEP_SUPPORT: float = 1.5      # cluster support_depth >= 3
```

### 3b. Add `_local_action_biases()` to `Worker` in `core/worker_pool.py`

Add this method to the `Worker` class, after `_compute_disabled_actions()`:

```python
def _local_action_biases(
    self, store: SignalStore, target: Optional[Signal]
) -> dict:
    """Compute local field-state action multipliers for the target's cluster.

    Returns a dict of {action: multiplier} for actions that should be
    up- or down-weighted based on the local cluster state around `target`.
    Returns empty dict when USE_LOCAL_ACTION_BIAS is False, when no target
    is available, or when no cluster information exists for this target.

    These multipliers are applied IN ADDITION to the global share pressure
    in choose_action(); they reflect what this specific part of the field
    needs, not the global pool balance.
    """
    from core.config import (
        USE_LOCAL_ACTION_BIAS,
        LOCAL_BIAS_DEVELOP_LOW_DIVERSITY,
        LOCAL_BIAS_REFINE_HAS_DISSENT,
        LOCAL_BIAS_VALIDATE_NO_VERIFY,
        LOCAL_BIAS_CHAIN_DEEP_SUPPORT,
    )
    if not USE_LOCAL_ACTION_BIAS or target is None:
        return {}

    biases: dict = {}

    # Count local cluster signals
    children = [store.get(cid) for cid in store.by_parent(target.id)]
    children = [c for c in children if c is not None]

    support_count = sum(1 for c in children
                        if c.type in ("SUPPORT", "CRITIQUE_POSITIVE"))
    dissent_count = sum(1 for c in children
                        if c.type in ("CRITIQUE_NEGATIVE", "OBJECTION"))
    verify_count = sum(1 for c in children if c.type == "VERIFICATION")

    # Low support diversity → push DEVELOP
    if support_count < 2:
        biases[DEVELOP] = LOCAL_BIAS_DEVELOP_LOW_DIVERSITY

    # Dissent present → push REFINE (rebuttal path)
    if dissent_count >= 1:
        biases[REFINE] = LOCAL_BIAS_REFINE_HAS_DISSENT

    # No verification on a supported cluster → push VALIDATE
    if support_count >= 2 and verify_count == 0:
        biases[VALIDATE] = LOCAL_BIAS_VALIDATE_NO_VERIFY

    # Well-developed cluster (support_count >= 3) → push CHAIN for depth
    if support_count >= 3:
        biases[CHAIN] = LOCAL_BIAS_CHAIN_DEEP_SUPPORT

    return biases
```

### 3c. Thread local biases into `choose_action()` in `core/worker_pool.py`

Modify `choose_action()` signature to accept an optional `local_biases` dict:

```python
def choose_action(field_state: FieldState, worker_history: deque,
                  pool_state: PoolState, rng: random.Random,
                  disabled_actions: Optional[set] = None,
                  local_biases: Optional[dict] = None) -> Optional[str]:
```

In the weights loop (after the share-pressure block, before the recency penalty),
apply local biases:

```python
        # Local field bias (Gap 3 fix): multiply by cluster-local state signal
        if local_biases and a in local_biases:
            w *= local_biases[a]
```

In `Worker.iterate()`, after `target` is resolved (after the cooldown re-pick),
compute biases and pass them to the SECOND `choose_action` call and, when possible,
the action already chosen:

```python
        local_biases = self._local_action_biases(store, target)
```

Pass `local_biases=local_biases` to both `choose_action` calls in `iterate()`.

---

## Change 4: Worker Semantic Position

### 4a. Add to `core/config.py`

```python
# Stigmergy upgrade — Gap 4
USE_WORKER_POSITION: bool = True
# Window size for the running centroid average (number of recent deposits).
WORKER_POSITION_WINDOW: int = 8
```

### 4b. Modify `Worker.__init__()` in `core/worker_pool.py`

Replace:
```python
        self._own_scout_excerpts: list[str] = []
```

With:
```python
        self._own_scout_excerpts: list[str] = []
        # Gap 4: semantic position as running centroid of own deposit embeddings.
        self._position_centroid: Optional[list[float]] = None
        self._position_embs: list[list[float]] = []   # recent window
```

### 4c. Update position after each successful deposit in `Worker.iterate()`

After the block that reads:
```python
        if action == SCOUT:
            self._own_scout_excerpts.append(content)
        return action
```

Insert:
```python
        # Gap 4: update semantic position from this deposit's embedding
        from core.config import USE_WORKER_POSITION, WORKER_POSITION_WINDOW
        if USE_WORKER_POSITION:
            emb = store.get_embedding(sid)
            if emb is not None:
                self._position_embs.append(emb)
                if len(self._position_embs) > WORKER_POSITION_WINDOW:
                    self._position_embs.pop(0)
                if self._position_embs:
                    n = len(self._position_embs[0])
                    centroid = [
                        sum(e[i] for e in self._position_embs) / len(self._position_embs)
                        for i in range(n)
                    ]
                    # L2 normalize
                    norm = sum(x * x for x in centroid) ** 0.5
                    if norm > 1e-9:
                        self._position_centroid = [x / norm for x in centroid]
```

### 4d. Pass `worker_centroid` into sampling calls

In `_gather_target()`, for the `DEVELOP` and `CRITIQUE` action branches, pass
`worker_centroid=self._position_centroid` to `_sample_initial()` and
`_sample_underserved_initial()`. Both were updated in Change 1 to accept this
parameter.

For the `OBJECT` action branch, when falling back to `consensus_summary` (no DBSCAN
clusters), pass `worker_centroid=self._position_centroid` to a cluster-aware lookup
if you add one. If not, leave the OBJECT branch as-is — OBJECT targets the dominant
consensus cluster globally by design.

---

## Constraints to Respect

- **No-leak rule is untouched.** None of these changes render signal content into
  prompts differently. Trail amplification only modifies `strength` and `_logit` on
  existing signals — it does not add content, metadata, or parent links. The
  `_assert_no_leak` check in `Worker` is not affected.

- **All changes are config-gated.** Setting `USE_CLUSTER_AWARE_SAMPLING = False`,
  `USE_TRAIL_AMPLIFICATION = False`, `USE_LOCAL_ACTION_BIAS = False`, and
  `USE_WORKER_POSITION = False` in `core/config.py` must exactly reproduce the
  pre-change behavior. This is the escape hatch if any change regresses a real run.

- **No new external dependencies.** Everything uses the existing embedding cache,
  `ClusterRegistry`, and `RLock` already in the store.

- **Graceful degradation when embeddings are absent.** Every new code path that uses
  embeddings must fall back cleanly when `store.get_embedding()` returns `None` or
  when `ClusterRegistry` has no clusters. The system must run correctly with
  `MOCK_LLM=1` (which produces degenerate embeddings or none at all).

- **`_amplify_cluster_trail()` must be called inside the existing `self._lock`.**
  It is a private method that mutates signal strengths and must not acquire the lock
  itself (it's already held by `deposit()`).

- **`asyncio` single-threaded constraint.** No threading primitives are added. The
  store's `RLock` is not replaced with an asyncio lock. Position centroid updates in
  `iterate()` happen outside the store lock (the worker owns its own state).

---

## Testing

After all four changes:

```bash
cd "Attempt At Cleaning"
pytest tests/ -x -q
```

Expected: all existing tests pass. The new behavior (cluster sampling, trail
amplification, local biases, worker position) operates on live store state with
real embeddings, so it cannot be validated by mock tests. If any test fails:

1. Check if it tests sampling behavior directly (e.g., `test_partition_propagation.py`,
   `test_cluster_on_deposit.py`) — these may need updating if they assert exact
   signal counts that are now affected by trail amplification.

2. Trail amplification changes signal strengths of cluster siblings. Tests that assert
   exact strength values after a deposit may need to relax to `>=` comparisons or
   to disable `USE_TRAIL_AMPLIFICATION` for the test setup.

3. If `test_no_leak_real_patterns.py` fails, check that `_amplify_cluster_trail()`
   does not appear in any prompt-building code path.

---

## What a Correct Implementation Looks Like

In a `MOCK_LLM=1` run (`python run_swarm.py debate "Test" --mode=pool`), the output
should show:

```
[CLUSTER] JOIN ...          ← ClusterRegistry working (already present)
[worker_000] ...            ← workers still active
[decay] applied 1 tick ...  ← decay loop still working
```

And in `round_log.json`, the `action_log` share distribution should look different
from a pre-change run when clusters have formed — DEVELOP share should rise when
underserved clusters exist, REFINE should rise when dissent is present, because local
biases are now influencing action selection per-worker rather than only the global
share enforcement doing all the work.

In a real-LLM run, `outputs/<task>_<timestamp>/signals.json` should show non-trivial
`strength` values on cluster members that received trail amplification — specifically,
strong clusters should have member signals with strengths slightly above their initial
deposit strength even if they were never explicitly `amplify()`-ed.
