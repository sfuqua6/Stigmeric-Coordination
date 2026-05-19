"""Continuous worker pool — replaces the round/phase scheduler.

A pool of `n_workers` async Worker tasks runs against a live SignalStore
until a convergence criterion fires. Each iteration each worker:

  1. Snapshots field state.
  2. Picks an action via choose_action(field_state, history) — respects
     preconditions, cold-start restrictions, share floors/ceilings, and
     the worker's own recent target history (cooldown).
  3. Samples a target signal if the action needs one (DEVELOP/CHAIN/...).
     If the target precondition fails at sample time, re-picks a fresh
     action with a new snapshot (no firing on stale targets).
  4. Builds the prompt via core/actions, calls the LLM, parses the
     response, deposits.

vLLM's `max_num_seqs=32` is the actual parallelism cap; `n_workers` should
exceed it slightly so workers are always queued. The continuous batcher
schedules across them.

No-leak rule still applies: every prompt path through core/actions only
renders Signal.content (plus retrieved chunks). The Worker class itself
does not touch ancestor text or other agents' history.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

from .signal_store import SignalStore, Signal
from .signal_types import (
    INITIAL, SUPPORT, CRITIQUE_POSITIVE, CRITIQUE_NEGATIVE,
    OBJECTION, VERIFICATION, SEARCH,
)
from . import actions as A
from .actions import (
    ACTIONS, ACTION_REGISTRY, ACTION_PRECONDITIONS,
    FieldState, ParsedDeposit,
    SCOUT, DEVELOP, CHAIN, CRITIQUE, OBJECT, VALIDATE, REFINE,
)
from .filters import is_junk_output

# Phase 5: validator raw-output log. Module-level so the run_swarm
# orchestrator can set the destination path once. Writer is best-effort.
_VALIDATOR_RAW_LOG_PATH: Optional[str] = None
_VALIDATOR_RAW_COUNT = {"n": 0}
_VALIDATOR_RAW_LIMIT = 5  # log this many full validator outputs per run


def set_validator_raw_log(path) -> None:
    """Wire a per-run path that the worker pool writes validator raw output to.

    Called by run_continuous_pipeline before run_pool. Pass None to disable.
    """
    global _VALIDATOR_RAW_LOG_PATH
    _VALIDATOR_RAW_LOG_PATH = str(path) if path is not None else None
    _VALIDATOR_RAW_COUNT["n"] = 0


def _log_validator_raw(agent_id: str, query: str, raw: str) -> None:
    if _VALIDATOR_RAW_LOG_PATH is None:
        return
    if _VALIDATOR_RAW_COUNT["n"] >= _VALIDATOR_RAW_LIMIT:
        return
    _VALIDATOR_RAW_COUNT["n"] += 1
    try:
        with open(_VALIDATOR_RAW_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(
                f"=== validator #{_VALIDATOR_RAW_COUNT['n']} "
                f"agent={agent_id} query={query!r} ===\n"
                f"{raw}\n\n"
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cold-start + share configuration
# ---------------------------------------------------------------------------

COLD_START_ITERATIONS = 30
COLD_START_INITIAL_FLOOR = 8
COLD_START_ACTIONS = {SCOUT: 0.7, DEVELOP: 0.3}

# Base weights when no cold-start / share pressure is active.
# CHAIN bumped 0.8 -> 1.1 because depth regressed (support_depth_max 4 -> 3)
# when share fell to 16% — CHAIN now competes more strongly with DEVELOP for
# turns once chains exist (share floor / ceiling still bound the long-term mix).
_BASE_WEIGHTS = {
    SCOUT:    1.0,
    DEVELOP:  1.5,
    CHAIN:    1.1,
    CRITIQUE: 1.0,
    OBJECT:   0.7,
    VALIDATE: 1.2,
    REFINE:   0.7,
}

ACTION_SHARE_TARGETS = {
    SCOUT:    {"min": 0.10, "max": 0.40},
    DEVELOP:  {"min": 0.20, "max": 0.50},
    CRITIQUE: {"min": 0.10, "max": 0.30},
    OBJECT:   {"min": 0.05, "max": 0.20},
    VALIDATE: {"min": 0.10, "max": 0.25},
    # CHAIN ceiling raised 0.20 -> 0.30 so it can deepen lineage without
    # bumping into the soft ceiling at 20% share.
    CHAIN:    {"min": 0.05, "max": 0.30},
    REFINE:   {"min": 0.05, "max": 0.20},
}

# Look-back window for share enforcement.
_SHARE_WINDOW = 50

# Recent-target bias: clusters touched recently across all workers get
# down-weighted in sampling so coverage spreads.
_RECENT_TARGET_PENALTY = 0.5

# Per-worker cooldown: forbid the immediately-preceding target.
_WORKER_COOLDOWN_DEPTH = 3

# Search budget across the pool. Each SCOUT/DEVELOP/VALIDATE that would
# call the live backend instead checks this counter first; if the per-
# 5-second window has been exhausted, the worker tries to reuse a cached
# query via find_cached_query (often available) and otherwise skips the
# search for this iteration. Stops the per-iter SCOUT storm from pushing
# DDG response times to 4-5s.
SEARCH_BUDGET_PER_WINDOW = 6
SEARCH_WINDOW_S = 5.0


# ---------------------------------------------------------------------------
# Pool-level shared state
# ---------------------------------------------------------------------------

@dataclass
class PoolState:
    """Mutable shared state across all workers in the pool."""
    iteration_counter: int = 0
    started_at: float = field(default_factory=time.time)
    action_log: deque = field(default_factory=lambda: deque(maxlen=_SHARE_WINDOW))
    recent_targets: Counter = field(default_factory=Counter)
    # Set to True once cold-start exits (one-way; never re-enters).
    cold_start_done: bool = False
    # Cached field-state snapshot (computed once per orchestrator tick to
    # avoid repeated store.stats() across workers).
    last_snapshot: Optional[FieldState] = None
    last_snapshot_iter: int = -1
    # Query history (Phase: emergent refinement). Maps the literal query
    # string -> n_results returned by the backend. Other workers consult
    # this before issuing a new query so we don't spend rate-limit budget
    # re-fetching the same thing.
    served_queries: dict = field(default_factory=dict)
    # Rolling search budget. Each entry is the unix timestamp of a live
    # backend call; entries older than SEARCH_WINDOW_S are pruned at
    # check time. Caps DDG hits so response times stay sub-second.
    search_timestamps: deque = field(default_factory=lambda: deque(maxlen=64))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def try_reserve_search(self) -> bool:
        """Return True iff a live search call fits in the current window."""
        now = time.time()
        cutoff = now - SEARCH_WINDOW_S
        # Prune expired entries.
        while self.search_timestamps and self.search_timestamps[0] < cutoff:
            self.search_timestamps.popleft()
        if len(self.search_timestamps) >= SEARCH_BUDGET_PER_WINDOW:
            return False
        self.search_timestamps.append(now)
        return True

    def record_action(self, action: str, target_id: Optional[str]) -> int:
        """Append a successful action to the running log. iteration_counter
        is incremented by the worker_loop itself on every attempt, so this
        method only logs deposits."""
        self.action_log.append(action)
        if target_id is not None:
            self.recent_targets[target_id] += 1
        return self.iteration_counter

    def share(self, action: str) -> float:
        if not self.action_log:
            return 0.0
        n = sum(1 for a in self.action_log if a == action)
        return n / len(self.action_log)

    def shares(self) -> dict[str, float]:
        if not self.action_log:
            return {a: 0.0 for a in ACTIONS}
        total = len(self.action_log)
        c = Counter(self.action_log)
        return {a: c.get(a, 0) / total for a in ACTIONS}


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def choose_action(field_state: FieldState, worker_history: deque,
                  pool_state: PoolState, rng: random.Random) -> Optional[str]:
    """Pick an action for one worker iteration.

    Returns None when nothing is available (only SCOUT is precondition-free,
    so this should be rare — happens only if SCOUT itself is somehow masked).

    Rules:
      1. Cold-start phase: restrict to SCOUT / DEVELOP until threshold met.
      2. Preconditions: filter actions whose precondition fails.
      3. Share floors/ceilings: bias weight ×1.5 below min, ×0.3 above max.
      4. Recency penalty: a worker's recent action gets ×0.7 to avoid loops.
    """
    cold_start = (
        not pool_state.cold_start_done
        and pool_state.iteration_counter < COLD_START_ITERATIONS
        and field_state.n_initials < COLD_START_INITIAL_FLOOR
    )
    if cold_start:
        # Restricted set: only SCOUT and DEVELOP, with stated weights.
        candidates = []
        for a, w in COLD_START_ACTIONS.items():
            if ACTION_PRECONDITIONS[a](field_state):
                candidates.append((a, w))
        if not candidates:
            return SCOUT  # bootstrap path; SCOUT precondition is always True
        names, weights = zip(*candidates)
        return rng.choices(names, weights=weights, k=1)[0]

    # Cold-start has ended (or never engaged) — full action set.
    if pool_state.iteration_counter >= COLD_START_ITERATIONS and not pool_state.cold_start_done:
        pool_state.cold_start_done = True

    available = [a for a in ACTIONS if ACTION_PRECONDITIONS[a](field_state)]
    if not available:
        return SCOUT  # SCOUT is unconditionally available
    weights = []
    for a in available:
        w = _BASE_WEIGHTS.get(a, 1.0)
        # Share pressure
        target = ACTION_SHARE_TARGETS.get(a)
        if target:
            share = pool_state.share(a)
            if share < target["min"]:
                w *= 1.5
            elif share > target["max"]:
                w *= 0.3
        # Recency penalty: if this worker just did this action, dampen.
        if worker_history and a in list(worker_history)[-2:]:
            w *= 0.7
        weights.append(max(0.001, w))
    return rng.choices(available, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Target sampling for actions that need one
# ---------------------------------------------------------------------------

def _sample_initial(store: SignalStore, recent: Counter,
                    rng: random.Random) -> Optional[Signal]:
    initials = store.by_type(INITIAL)
    if not initials:
        return None
    weights = []
    for s in initials:
        w = max(0.05, s.strength)
        if recent.get(s.id, 0) > 0:
            w *= _RECENT_TARGET_PENALTY
        weights.append(w)
    return rng.choices(initials, weights=weights, k=1)[0]


def _sample_underserved_initial(store: SignalStore, recent: Counter,
                                 rng: random.Random) -> Optional[Signal]:
    underserved = store.signals_with_few_children_of_type(INITIAL, SUPPORT, 2)
    pool_list = underserved or store.by_type(INITIAL)
    if not pool_list:
        return None
    weights = []
    for s in pool_list:
        w = max(0.05, s.strength)
        if recent.get(s.id, 0) > 0:
            w *= _RECENT_TARGET_PENALTY
        weights.append(w)
    return rng.choices(pool_list, weights=weights, k=1)[0]


def _sample_support(store: SignalStore, recent: Counter,
                    rng: random.Random) -> Optional[Signal]:
    supports = store.by_type(SUPPORT)
    if not supports:
        return None
    weights = []
    for s in supports:
        w = max(0.05, s.strength)
        if recent.get(s.id, 0) > 0:
            w *= _RECENT_TARGET_PENALTY
        weights.append(w)
    return rng.choices(supports, weights=weights, k=1)[0]


def _sample_well_supported_cluster_head(
    store: SignalStore, recent: Counter, rng: random.Random,
) -> Optional[Signal]:
    pool_list = store.signals_with_many_children_of_type(INITIAL, SUPPORT, 2)
    if not pool_list:
        return None
    weights = []
    for s in pool_list:
        w = max(0.05, s.strength)
        if recent.get(s.id, 0) > 0:
            w *= _RECENT_TARGET_PENALTY
        weights.append(w)
    return rng.choices(pool_list, weights=weights, k=1)[0]


def _strongest_dissent(store: SignalStore, parent_id: str) -> Optional[Signal]:
    children = [store.get(cid) for cid in store.by_parent(parent_id)]
    children = [c for c in children if c is not None and c.type in
                (CRITIQUE_NEGATIVE, OBJECTION)]
    if not children:
        return None
    return max(children, key=lambda c: c.strength)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class Worker:
    """Generic, role-agnostic. Picks an action each iteration."""

    def __init__(self, worker_id: int, llm, task_prompt: str,
                 user_prompt: str, rng_seed: Optional[int] = None,
                 task_type: Optional[str] = None):
        self.worker_id = worker_id
        self.agent_id = f"worker_{worker_id:03d}"
        self.llm = llm
        self.task_prompt = task_prompt
        self.user_prompt = user_prompt
        # Task type drives query-stance selection and survival profile
        # (None = analysis defaults).
        self.task_type = task_type
        self.recent_actions: deque = deque(maxlen=8)
        self.recent_targets: deque = deque(maxlen=_WORKER_COOLDOWN_DEPTH)
        self._rng = random.Random(rng_seed if rng_seed is not None else worker_id)
        # Per-worker scout query history (for cache-friendly variation).
        self._query_history: list[str] = []
        # Track own deposits so SCOUT re-seed can avoid restating them.
        self._own_scout_excerpts: list[str] = []

    async def iterate(self, store: SignalStore, pool_state: PoolState,
                      field_state: FieldState) -> Optional[str]:
        """Run one iteration. Returns the action name on deposit, None on skip."""
        action = choose_action(field_state, self.recent_actions, pool_state,
                               self._rng)
        if action is None:
            return None

        # Target sampling. If precondition fails at sample time (race with
        # other workers), re-pick action once with a fresh snapshot.
        target, retrieved_chunks, query, dissent = await self._gather_target(
            action, store, pool_state, field_state,
        )

        # If the action needs a target and none is available, re-snapshot
        # and re-pick. Cap re-pick at 1 to avoid livelock.
        if target is None and action not in (SCOUT,):
            fresh = FieldState.from_store(
                store, iteration=pool_state.iteration_counter,
                elapsed_s=time.time() - pool_state.started_at,
            )
            action = choose_action(fresh, self.recent_actions, pool_state,
                                   self._rng) or SCOUT
            target, retrieved_chunks, query, dissent = await self._gather_target(
                action, store, pool_state, fresh,
            )
            if target is None and action != SCOUT:
                # Bail this iteration; let other workers move the field.
                return None

        # Cooldown: skip the immediately-preceding target for this worker.
        if target is not None and target.id in self.recent_targets:
            # Down-weighted re-pick once; if still hitting the cooldown
            # target, accept it (don't spin).
            new_target, *_rest = await self._gather_target(
                action, store, pool_state, field_state,
            )
            if new_target is not None and new_target.id != target.id:
                target = new_target

        # Build prompt
        prompt = self._build_prompt(action, target, retrieved_chunks, query,
                                    dissent)
        if prompt is None:
            return None

        # No-leak guard
        self._assert_no_leak(prompt)

        spec = ACTION_REGISTRY[action]
        raw = await self.llm.generate(
            prompt, role=action.lower(),
            max_tokens=spec.max_tokens, temperature=spec.temperature,
        )

        # Phase 5: log the first N validator raw outputs to validator_raw.log
        # so a failed JSON parse vs. failed prompt is diagnosable post-run.
        if action == VALIDATE:
            _log_validator_raw(self.agent_id, query or "", raw or "")

        # Parse. validate_parse is task-aware (engages/quality schema for
        # non-factual tasks); other parsers take only `raw`.
        try:
            if action == VALIDATE:
                parsed = spec.parse(raw, task_type=self.task_type)
            else:
                parsed = spec.parse(raw)
        except Exception as exc:
            print(f"[worker {self.agent_id}] parse failed for {action}: "
                  f"{type(exc).__name__}: {exc}")
            return None

        from agents.base import strip_reasoning
        content = strip_reasoning(parsed.content or "")
        if not content or is_junk_output(content):
            return None

        # Dynamic TYPE/PARENT overrides (Phase 3A/3B compatibility)
        from core.actions import _parse_type, _parse_parent
        dyn_type = _parse_type(raw)
        dyn_parent = _parse_parent(raw, store)
        effective_type = dyn_type if dyn_type is not None else parsed.signal_type
        if action == CHAIN and target is not None:
            # CHAIN's whole point is depth — force parent to the SUPPORT we
            # sampled, regardless of what the model emitted. Otherwise the
            # LLM routinely returns PARENT: <some_initial_id> and the chain
            # flattens back to depth 2.
            effective_parent = target.id
            effective_type = SUPPORT  # also lock the type for safety
        elif dyn_parent == "__NONE__":
            effective_parent = None
        elif dyn_parent is not None:
            effective_parent = dyn_parent
        else:
            effective_parent = parsed.parent_id_override or (
                target.id if target is not None else None
            )

        meta = dict(parsed.metadata or {})
        meta["depositor_agent_id"] = self.agent_id
        meta["action"] = action
        if dyn_type is not None:
            meta["proposed_type"] = dyn_type
        if dyn_parent is not None:
            meta["proposed_parent"] = (
                None if dyn_parent == "__NONE__" else dyn_parent
            )
        if query:
            meta["query"] = query
        if action == SCOUT:
            # Preserve scout-style metadata for projection's partition_origin parsing.
            meta["scout_agent_id"] = self.agent_id

        sid = store.deposit(
            signal_type=effective_type,
            content=content,
            strength=parsed.strength,
            depositor=action.lower(),
            parent_id=effective_parent,
            metadata=meta,
        )
        if sid is None:
            return None

        # Bookkeeping
        self.recent_actions.append(action)
        if target is not None:
            self.recent_targets.append(target.id)
        async with pool_state.lock:
            pool_state.record_action(action, target.id if target else None)
        if action == SCOUT:
            self._own_scout_excerpts.append(content)
        return action

    # ---- target gathering --------------------------------------------------

    async def _gather_target(
        self, action: str, store: SignalStore, pool_state: PoolState,
        field_state: FieldState,
    ) -> tuple[Optional[Signal], list, str, Optional[Signal]]:
        """Sample target signal, retrieved chunks, and dissent context."""
        retrieved: list = []
        query = ""
        dissent: Optional[Signal] = None
        if action == SCOUT:
            # Emergent query refinement: planner consults the live store
            # for high-strength INITIAL keyphrases and the pool's served-
            # queries history to avoid loops.
            from core.query_planner import plan_scout_query, find_cached_query
            from core.search_tool import search as _search, summarize_for_signal
            try:
                query = plan_scout_query(
                    self.user_prompt, store, pool_state.served_queries,
                    self.worker_id, len(self._query_history),
                    self._query_history,
                    task_type=self.task_type,
                )
                cached_q = find_cached_query(query, pool_state.served_queries) if query else None
                if cached_q is not None and cached_q != query:
                    # Skip the network round-trip — another worker already
                    # fetched substantially the same thing. The cached
                    # result is in the search_tool's on-disk cache; reuse it.
                    query = cached_q
                if not query:
                    retrieved = []
                elif cached_q is not None:
                    # Cached hit — no budget needed; search_tool's on-disk
                    # cache returns instantly.
                    retrieved = _search(query, max_results=5)
                elif pool_state.try_reserve_search():
                    retrieved = _search(query, max_results=5)
                else:
                    # Budget exhausted; skip the live call this iteration.
                    retrieved = []
            except Exception as exc:
                print(f"[scout {self.agent_id}] query plan failed: "
                      f"{type(exc).__name__}: {exc}")
                retrieved = []
            if retrieved:
                try:
                    store.deposit(
                        signal_type=SEARCH,
                        content=summarize_for_signal(query, retrieved),
                        strength=0.4,
                        depositor="scout",
                        parent_id=None,
                        metadata={
                            "depositor_agent_id": self.agent_id,
                            "scout_agent_id": self.agent_id,
                            "query": query,
                            "n_results": len(retrieved),
                        },
                    )
                    self._query_history.append(query)
                    # Record in shared served-queries (no lock — dict assign
                    # is atomic in CPython; max() handles racing workers).
                    prev = pool_state.served_queries.get(query, 0)
                    pool_state.served_queries[query] = max(prev, len(retrieved))
                except Exception:
                    pass
            return None, retrieved, query, None

        if action == DEVELOP:
            target = _sample_underserved_initial(
                store, pool_state.recent_targets, self._rng,
            )
            if target is None:
                return None, [], "", None
            dissent = _strongest_dissent(store, target.id)
            # Sparse-cluster search hook: <2 SUPPORT children → query.
            n_support = sum(
                1 for cid in store.by_parent(target.id)
                if store.get(cid) and store.get(cid).type == SUPPORT
            )
            if n_support < 2:
                try:
                    from core.search_tool import search as _search, summarize_for_signal
                    from core.query_planner import plan_develop_query, find_cached_query
                    query = plan_develop_query(target.content, pool_state.served_queries,
                                                task_type=self.task_type)
                    if query:
                        cached_q = find_cached_query(query, pool_state.served_queries)
                        if cached_q is not None and cached_q != query:
                            query = cached_q
                            retrieved = _search(query, max_results=3)
                        elif pool_state.try_reserve_search():
                            retrieved = _search(query, max_results=3)
                        else:
                            retrieved = []
                        if retrieved:
                            store.deposit(
                                signal_type=SEARCH,
                                content=summarize_for_signal(query, retrieved),
                                strength=0.4,
                                depositor="develop",
                                parent_id=target.id,
                                metadata={
                                    "depositor_agent_id": self.agent_id,
                                    "query": query,
                                    "n_results": len(retrieved),
                                    "trigger": "sparse_support",
                                },
                            )
                            prev = pool_state.served_queries.get(query, 0)
                            pool_state.served_queries[query] = max(prev, len(retrieved))
                except Exception:
                    retrieved = []
            return target, retrieved, query, dissent

        if action == CHAIN:
            target = _sample_support(store, pool_state.recent_targets, self._rng)
            return target, [], "", None

        if action == CRITIQUE:
            target = _sample_initial(store, pool_state.recent_targets, self._rng)
            return target, [], "", None

        if action == OBJECT:
            # Use DBSCAN cluster heads when available
            clusters = store.cluster_signals_dbscan(INITIAL, eps=0.35)
            if clusters and len(clusters[0]) > 0:
                largest = clusters[0]
                reps = sorted(largest, key=lambda s: s.strength, reverse=True)[:3]
            else:
                summary = store.consensus_summary(INITIAL, k=3)
                reps = [store.get(r["id"]) for r in summary["representatives"]]
                reps = [s for s in reps if s is not None]
            target = reps[0] if reps else None
            # We hand the FULL rep list to object_prompt via a sentinel:
            # attach reps to the worker so _build_prompt sees them.
            self._object_reps = reps
            return target, [], "", None

        if action == VALIDATE:
            target = _sample_well_supported_cluster_head(
                store, pool_state.recent_targets, self._rng,
            )
            if target is None:
                return None, [], "", None
            try:
                from core.search_tool import search as _search
                from core.query_planner import plan_validate_query, find_cached_query
                query = plan_validate_query(target.content)
                if query:
                    cached_q = find_cached_query(query, pool_state.served_queries)
                    if cached_q is not None and cached_q != query:
                        query = cached_q
                        hits = _search(query, max_results=2)
                    elif pool_state.try_reserve_search():
                        hits = _search(query, max_results=2)
                    else:
                        hits = []
                else:
                    hits = []
                if hits:
                    prev = pool_state.served_queries.get(query, 0)
                    pool_state.served_queries[query] = max(prev, len(hits))
                blocks = []
                for c in hits[:2]:
                    tag = (getattr(c, "source_tag", "") or "")[:120]
                    body = (getattr(c, "text", "") or "")[:400]
                    blocks.append(f"[{tag}]\n{body}")
                external = "\n\n".join(blocks) if blocks else (
                    f"(no external snippet found for {query!r})"
                )
            except Exception:
                external = f"(search failed; query was {query!r})"
            # Stash external snippet on the worker for _build_prompt to read.
            self._validate_external = external
            return target, [], query, None

        if action == REFINE:
            # Pick an unverified INITIAL with at least some support.
            target = _sample_underserved_initial(
                store, pool_state.recent_targets, self._rng,
            )
            return target, [], "", None

        return None, [], "", None

    # ---- prompt building ---------------------------------------------------

    def _build_prompt(self, action: str, target: Optional[Signal],
                      retrieved: list, query: str,
                      dissent: Optional[Signal]) -> Optional[str]:
        if action == SCOUT:
            prior_own = (
                self._own_scout_excerpts[-1][:175]
                if self._own_scout_excerpts else None
            )
            return A.scout_prompt(self.task_prompt, retrieved, prior_own)
        if target is None:
            return None
        if action == DEVELOP:
            return A.develop_prompt(self.task_prompt, target, dissent,
                                    retrieved, query)
        if action == CHAIN:
            return A.chain_prompt(self.task_prompt, target)
        if action == CRITIQUE:
            return A.critique_prompt(self.task_prompt, target)
        if action == OBJECT:
            reps = getattr(self, "_object_reps", [target])
            if not reps:
                return None
            return A.object_prompt(self.task_prompt, reps)
        if action == VALIDATE:
            external = getattr(self, "_validate_external", "")
            return A.validate_prompt(self.task_prompt, target, external,
                                     task_type=self.task_type)
        if action == REFINE:
            return A.refine_prompt(self.task_prompt, target)
        return None

    def _assert_no_leak(self, prompt: str) -> None:
        forbidden = (
            "parent_content", "provenance_chain", "chain_of_thought",
            "previous reasoning", "prior reasoning", "agent reasoning",
            "responses:", "dialogue thread",
        )
        lowered = prompt.lower()
        for tok in forbidden:
            if tok in lowered:
                raise AssertionError(
                    f"[{self.agent_id}] no-leak rule violated: prompt contains "
                    f"forbidden token {tok!r}"
                )


# ---------------------------------------------------------------------------
# Pool runner
# ---------------------------------------------------------------------------

async def worker_loop(worker: Worker, store: SignalStore,
                      pool_state: PoolState, stop_event: asyncio.Event) -> None:
    """One worker's run loop. Spins until stop_event fires."""
    while not stop_event.is_set():
        # Snapshot field state (cached per-tick if available, else re-read).
        snapshot = pool_state.last_snapshot
        if snapshot is None or (
            pool_state.iteration_counter - pool_state.last_snapshot_iter > 4
        ):
            snapshot = FieldState.from_store(
                store, iteration=pool_state.iteration_counter,
                elapsed_s=time.time() - pool_state.started_at,
            )
            pool_state.last_snapshot = snapshot
            pool_state.last_snapshot_iter = pool_state.iteration_counter

        try:
            await worker.iterate(store, pool_state, snapshot)
        except Exception as exc:
            print(f"[worker {worker.agent_id}] iteration error: "
                  f"{type(exc).__name__}: {exc}")
        # Iterations count attempts, not deposits — otherwise the
        # convergence detector's MIN_ITERATIONS floor is unreachable
        # when dedup or junk-filter rejects a string of attempts.
        async with pool_state.lock:
            pool_state.iteration_counter += 1
        await asyncio.sleep(0)


async def run_pool(store: SignalStore, llm, task_prompt: str,
                   user_prompt: str, stop_event: asyncio.Event,
                   n_workers: int = 24,
                   task_type: Optional[str] = None) -> PoolState:
    """Spin up `n_workers` and run until `stop_event` fires.

    Returns the PoolState (action log, iteration count, etc.) for the
    orchestrator's summary.
    """
    pool_state = PoolState()
    workers = [
        Worker(i, llm, task_prompt, user_prompt, rng_seed=i, task_type=task_type)
        for i in range(n_workers)
    ]
    tasks = [
        asyncio.create_task(worker_loop(w, store, pool_state, stop_event))
        for w in workers
    ]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    return pool_state


# ---------------------------------------------------------------------------
# Time-based decay/prune background task (Phase 3)
# ---------------------------------------------------------------------------

# Wall-clock baseline for a single "round" of decay. DELTA_DECAY in
# core/config.py is calibrated for one round, so a continuous-pool tick
# should apply (interval_s / DECAY_ROUND_BASELINE_S) of that magnitude.
# 300 s = 5 minutes — the rough wall-clock cost of one round in the
# legacy scheduler.
DECAY_ROUND_BASELINE_S = 300.0


async def decay_loop(store: SignalStore, stop_event: asyncio.Event,
                     interval_s: float = 60.0) -> int:
    """Periodically decay + prune until stop_event fires. Returns prune count.

    Wall-clock-scaled decay: each tick applies (interval_s / DECAY_ROUND_BASELINE_S)
    of one legacy round's decay. At 60 s interval with 300 s baseline that's
    0.2× per tick, so a signal at strength 0.6 takes ~50 ticks (~50 minutes)
    to reach PRUNE_THRESHOLD=0.30 — long enough for CHAIN and DEVELOP to keep
    the field consolidating without prune storms.

    Without scaling, prior runs saw 60-80 clusters pruned per cycle (the
    "Pool oscillates between ~65-100 weakly supported and never consolidates
    upward" pathology) because DELTA_DECAY was calibrated for 1 round but
    fired every 30-60 s.
    """
    total_pruned = 0
    factor = max(0.0, interval_s / DECAY_ROUND_BASELINE_S)
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass
        if stop_event.is_set():
            break
        try:
            store.decay_all(factor=factor)
            pruned = store.prune_weak()
            total_pruned += pruned
            print(f"[decay] applied 1 tick (factor={factor:.2f}); pruned {pruned}")
        except Exception as exc:
            print(f"[decay] error: {type(exc).__name__}: {exc}")
    return total_pruned
