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
import re
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
#
# OBJECT raised 0.7 -> 1.0 and REFINE raised 0.7 -> 1.0 after the m13 audit
# showed pro:contra signals at 5.7:1 in a debate task. REFINE is now the
# rebuttal path (consumes objections, produces SUPPORTs that engage them),
# so its base weight is bumped in tandem with OBJECT — otherwise objections
# generate but never get answered.
_BASE_WEIGHTS = {
    SCOUT:    1.0,
    DEVELOP:  1.5,
    CHAIN:    1.1,
    CRITIQUE: 1.0,
    OBJECT:   1.0,
    VALIDATE: 1.2,
    REFINE:   1.0,
}

ACTION_SHARE_TARGETS = {
    SCOUT:    {"min": 0.10, "max": 0.40},
    DEVELOP:  {"min": 0.20, "max": 0.45},   # ceiling 0.50 -> 0.45 to free up
                                             # iterations for OBJECT + REFINE
    CRITIQUE: {"min": 0.10, "max": 0.30},
    # OBJECT floor 0.05 -> 0.12: in m13 audit OBJECT *actions* fired ~18%
    # of iterations but produced only 7% of signals (44 vs 304 supports).
    # Raising the floor compensates for higher rejection rate on objection
    # outputs, and the ceiling rises with it so OBJECT can dominate when
    # the field is heavily skewed.
    OBJECT:   {"min": 0.12, "max": 0.30},
    VALIDATE: {"min": 0.10, "max": 0.25},
    # CHAIN ceiling raised 0.20 -> 0.30 so it can deepen lineage without
    # bumping into the soft ceiling at 20% share.
    CHAIN:    {"min": 0.05, "max": 0.30},
    # REFINE floor 0.05 -> 0.10 and ceiling 0.20 -> 0.25 — needed for the
    # rebuttal-mode REFINE to consume the new OBJECT volume.
    REFINE:   {"min": 0.10, "max": 0.25},
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
                  pool_state: PoolState, rng: random.Random,
                  disabled_actions: Optional[set] = None,
                  local_biases: Optional[dict] = None) -> Optional[str]:
    """Pick an action for one worker iteration.

    Returns None when nothing is available (only SCOUT is precondition-free,
    so this should be rare — happens only if SCOUT itself is somehow masked).

    Rules:
      1. Cold-start phase: restrict to SCOUT / DEVELOP until threshold met.
      2. Preconditions: filter actions whose precondition fails.
      3. Disabled actions (bundle-specific, e.g. OBJECT/VALIDATE on
         "creative") are removed from the candidate set entirely.
      4. Share floors/ceilings: bias weight ×1.5 below min, ×0.3 above max.
      5. Local field bias (Gap 3): cluster-local state multipliers per action.
      6. Recency penalty: a worker's recent action gets ×0.7 to avoid loops.
    """
    disabled = disabled_actions or frozenset()
    cold_start = (
        not pool_state.cold_start_done
        and pool_state.iteration_counter < COLD_START_ITERATIONS
        and field_state.n_initials < COLD_START_INITIAL_FLOOR
    )
    if cold_start:
        # Restricted set: only SCOUT and DEVELOP, with stated weights.
        candidates = []
        for a, w in COLD_START_ACTIONS.items():
            if a in disabled:
                continue
            if ACTION_PRECONDITIONS[a](field_state):
                candidates.append((a, w))
        if not candidates:
            return SCOUT  # bootstrap path; SCOUT precondition is always True
        names, weights = zip(*candidates)
        return rng.choices(names, weights=weights, k=1)[0]

    # Cold-start has ended (or never engaged) — full action set.
    if pool_state.iteration_counter >= COLD_START_ITERATIONS and not pool_state.cold_start_done:
        pool_state.cold_start_done = True

    available = [
        a for a in ACTIONS
        if a not in disabled and ACTION_PRECONDITIONS[a](field_state)
    ]
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
        # Local field bias (Gap 3 fix): multiply by cluster-local state signal
        if local_biases and a in local_biases:
            w *= local_biases[a]
        # Recency penalty: if this worker just did this action, dampen.
        if worker_history and a in list(worker_history)[-2:]:
            w *= 0.7
        weights.append(max(0.001, w))
    return rng.choices(available, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Target sampling for actions that need one
# ---------------------------------------------------------------------------

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


def _sample_underserved_initial(store: SignalStore, recent: Counter,
                                 rng: random.Random,
                                 worker_centroid=None) -> Optional[Signal]:
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


def _sample_contested_initial(
    store: SignalStore, recent: Counter,
    rng: random.Random,
) -> Optional[Signal]:
    """Sample an INITIAL that has accumulated dissent (cluster-aware).

    Returns None when no INITIAL in the field has any contrarian children
    on its lineage — REFINE then falls back to the underserved-initial path
    so it still polishes claims when there's no dissent to rebut.
    """
    contested: list[Signal] = []
    for sig in store.by_type(INITIAL):
        if _cluster_dissent(store, sig.id) is not None:
            contested.append(sig)
    if not contested:
        return None
    weights = []
    for s in contested:
        w = max(0.05, s.strength)
        if recent.get(s.id, 0) > 0:
            w *= _RECENT_TARGET_PENALTY
        weights.append(w)
    return rng.choices(contested, weights=weights, k=1)[0]


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
    """Strongest CRITIQUE_NEGATIVE / OBJECTION attached to a single INITIAL.

    Kept for back-compat with REFINE and the legacy DEVELOP path. New code
    should prefer _cluster_dissent which walks the whole cluster (objections
    are typically deposited against cluster reps, not against every sibling
    INITIAL that talks about the same idea).
    """
    children = [store.get(cid) for cid in store.by_parent(parent_id)]
    children = [c for c in children if c is not None and c.type in
                (CRITIQUE_NEGATIVE, OBJECTION)]
    if not children:
        return None
    return max(children, key=lambda c: c.strength)


def _cluster_dissent(
    store: SignalStore, target_id: str,
    sim_threshold: float = 0.65,
) -> Optional[Signal]:
    """Strongest dissent against the whole *cluster* the target belongs to.

    Previously DEVELOP only saw objections deposited as direct children of
    the *specific* INITIAL it sampled. But OBJECT-action deposits target
    cluster representatives — typically the highest-strength INITIAL — so
    a sibling INITIAL that talks about the same idea (different framing /
    different scout) saw zero dissent.

    This walks both: (1) direct children of target_id, and (2) direct
    children of every sibling INITIAL with embedding similarity above
    sim_threshold. Returns the strongest contrarian among the union.

    Threshold 0.65 is slightly below the default cluster threshold (0.72)
    so we catch loosely-related dissent without dragging in objections
    from an entirely different conceptual cluster.
    """
    target = store.get(target_id)
    if target is None:
        return None
    target_emb = store.get_embedding(target_id)

    # Start with direct children — same set the legacy walk found.
    candidates: list[Signal] = []
    for cid in store.by_parent(target_id):
        c = store.get(cid)
        if c is not None and c.type in (CRITIQUE_NEGATIVE, OBJECTION):
            candidates.append(c)

    # Add dissent from sibling INITIALs in the same conceptual cluster.
    if target_emb is not None:
        from .signal_types import INITIAL
        for sib in store.by_type(INITIAL):
            if sib.id == target_id:
                continue
            sib_emb = store.get_embedding(sib.id)
            if sib_emb is None:
                continue
            sim = sum(a * b for a, b in zip(target_emb, sib_emb))
            if sim < sim_threshold:
                continue
            for cid in store.by_parent(sib.id):
                c = store.get(cid)
                if c is not None and c.type in (CRITIQUE_NEGATIVE, OBJECTION):
                    candidates.append(c)

    # Dedupe by id (same dissent could be reachable from multiple siblings
    # in dense clusters — though objections only have one parent each).
    by_id = {c.id: c for c in candidates}
    if not by_id:
        return None
    return max(by_id.values(), key=lambda c: c.strength)


# ---------------------------------------------------------------------------
# SAFE retrieval helpers (SAFE + Step-Back + HyDE per-atom verification)
# ---------------------------------------------------------------------------

# FLARE uncertainty threshold: confidence below this triggers retrieval.
_FLARE_TAU = 0.6

# Non-factual task types: use engages/quality schema rather than supports/confidence.
_NON_FACTUAL_TASKS = {"debate", "analysis", "problem_solving", "creative"}


async def _safe_decompose(content: str, llm, max_atoms: int = 3) -> list[dict]:
    """Decompose a claim into atomic facts with centrality weights.

    Each atom is a dict: {"text": str, "weight": float}.  Falls back to a
    single atom (the whole content) when the LLM response is unparseable.
    """
    prompt = (
        f"Break this claim into at most {max_atoms} independently verifiable "
        f"atomic facts. For each fact write one line:\n"
        f"ATOM: <one-sentence proposition> | WEIGHT: <centrality 0.0-1.0>\n\n"
        f"WEIGHT reflects how central the proposition is to the main assertion "
        f"(1.0 = load-bearing; 0.1 = incidental background).\n\n"
        f"Claim: {content[:400]}"
    )
    try:
        raw = await llm.generate(
            prompt, role="validator", max_tokens=220, temperature=0.2
        )
        atoms: list[dict] = []
        for line in raw.strip().splitlines():
            upper = line.upper()
            if "ATOM:" not in upper:
                continue
            parts = line.split("|")
            atom_text = parts[0].split(":", 1)[-1].strip().strip('"\'')
            weight = 1.0
            if len(parts) > 1 and "WEIGHT:" in parts[1].upper():
                try:
                    weight = max(0.1, min(1.0, float(parts[1].split(":", 1)[-1].strip())))
                except ValueError:
                    pass
            if atom_text and len(atom_text.split()) >= 3:
                atoms.append({"text": atom_text, "weight": weight})
        if atoms:
            return atoms[:max_atoms]
    except Exception:
        pass
    # Fallback: whole content as single atom.
    return [{"text": content[:200], "weight": 1.0}]


async def _safe_score_atom(atom_text: str, snippet: str, llm,
                            task_type: str = None) -> float:
    """Score one atomic fact against a retrieved snippet.

    Uses the engages/quality schema for non-factual tasks and
    supports/confidence for factual tasks.  Returns 0.5 (abstain) on failure.
    """
    if task_type in _NON_FACTUAL_TASKS:
        prompt = (
            f"Does this snippet substantively engage with the following claim?\n"
            f"Claim: {atom_text}\n"
            f"Snippet: {snippet[:350]}\n"
            f"Reply with only: SCORE: X.X  (1.0=fully engages, 0.0=irrelevant)"
        )
    else:
        prompt = (
            f"Does this snippet support the following factual claim?\n"
            f"Claim: {atom_text}\n"
            f"Snippet: {snippet[:350]}\n"
            f"Reply with only: SCORE: X.X  "
            f"(1.0=strongly supports, 0.5=neutral/no result, 0.0=contradicts)"
        )
    try:
        raw = await llm.generate(
            prompt, role="validator", max_tokens=10, temperature=0.1
        )
        m = re.search(r"SCORE\s*[:=]\s*([0-9.]+)", raw, re.IGNORECASE)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
    except Exception:
        pass
    return 0.5  # abstain


def _format_safe_external(atom_results: list[dict]) -> str:
    """Render atom-level verification results as the external-snippet block.

    The main validate_prompt sees this string as its 'external_snippet',
    allowing the LLM to produce a one-sentence overall assessment grounded
    in the per-atom evidence that was already gathered.
    """
    lines = ["=== ATOMIC VERIFICATION RESULTS ===\n"]
    total_weight = sum(a.get("weight", 1.0) for a in atom_results) or 1.0
    agg = sum(a.get("score", 0.5) * a.get("weight", 1.0) for a in atom_results) / total_weight
    for i, a in enumerate(atom_results, 1):
        lines.append(
            f"ATOM {i} [weight={a.get('weight', 1.0):.2f}]: {a['text']}\n"
            f"  QUERY: {a.get('query', '')}\n"
            f"  SOURCE: {a.get('snippet_tag', '(no result)')}\n"
            f"  ATOM SCORE: {a.get('score', 0.5):.2f}\n"
        )
    lines.append(f"WEIGHTED AGGREGATE SCORE: {agg:.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class Worker:
    """Generic, role-agnostic. Picks an action each iteration."""

    def __init__(self, worker_id: int, llm, task_prompt: str,
                 user_prompt: str, rng_seed: Optional[int] = None,
                 task_type: Optional[str] = None,
                 router=None):
        self.worker_id = worker_id
        self.agent_id = f"worker_{worker_id:03d}"
        # When `router` is set, llm is unused — every generate() call routes
        # through router.engine_for(role). The single-llm path is kept so
        # MockLLM / homogeneous single-engine runs work unchanged.
        self.llm = llm
        self.router = router
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
        # Gap 4: semantic position as running centroid of own deposit embeddings.
        self._position_centroid: Optional[list[float]] = None
        self._position_embs: list[list[float]] = []   # recent window
        # Genome cache: cluster_id (= representative_id) -> ClusterGenome.
        # Populated externally by AdaptiveLLMPool.update_genome_cache() after
        # each round's build_projection() call. Empty until then (genome=None
        # falls back to the standard single-signal prompt in action builders).
        self._genome_cache: dict = {}
        # Bundle-disabled actions (cached). Roles set to None in the bundle's
        # ROLE_TO_ENGINE map cause their action to drop out of choose_action.
        self._disabled_actions: frozenset = self._compute_disabled_actions()

    def _compute_disabled_actions(self) -> frozenset:
        if self.router is None:
            return frozenset()
        from core.config import ACTION_TO_ROLE
        out = set()
        for act, role in ACTION_TO_ROLE.items():
            if self.router.role_disabled(role):
                out.add(act)
        return frozenset(out)

    def _local_action_biases(
        self, store: SignalStore, target: Optional[Signal]
    ) -> dict:
        """Compute local field-state action multipliers for the target's cluster (Gap 3 fix).

        Returns a dict of {action: multiplier} reflecting what this specific
        part of the field needs, applied on top of global share pressure.
        Returns empty dict when USE_LOCAL_ACTION_BIAS is False or target is None.
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

        children = [store.get(cid) for cid in store.by_parent(target.id)]
        children = [c for c in children if c is not None]

        support_count = sum(1 for c in children
                            if c.type in (SUPPORT, CRITIQUE_POSITIVE))
        dissent_count = sum(1 for c in children
                            if c.type in (CRITIQUE_NEGATIVE, OBJECTION))
        verify_count = sum(1 for c in children if c.type == VERIFICATION)

        if support_count < 2:
            biases[DEVELOP] = LOCAL_BIAS_DEVELOP_LOW_DIVERSITY

        if dissent_count >= 1:
            biases[REFINE] = LOCAL_BIAS_REFINE_HAS_DISSENT

        if support_count >= 2 and verify_count == 0:
            biases[VALIDATE] = LOCAL_BIAS_VALIDATE_NO_VERIFY

        if support_count >= 3:
            biases[CHAIN] = LOCAL_BIAS_CHAIN_DEEP_SUPPORT

        return biases

    def _llm_for_action(self, action: str):
        """Resolve the right LLM backend for this action.

        When a router is wired, look up the role implied by the action via
        ACTION_TO_ROLE and ask the router for that role's engine. Otherwise
        fall back to the single self.llm.
        """
        if self.router is None:
            return self.llm
        from core.config import ACTION_TO_ROLE
        role = ACTION_TO_ROLE.get(action, "forager")
        return self.router.engine_for(role)

    async def iterate(self, store: SignalStore, pool_state: PoolState,
                      field_state: FieldState) -> Optional[str]:
        """Run one iteration. Returns the action name on deposit, None on skip."""
        local_biases: dict = {}  # populated after first gather_target
        action = choose_action(field_state, self.recent_actions, pool_state,
                               self._rng,
                               disabled_actions=self._disabled_actions,
                               local_biases=local_biases)
        if action is None:
            return None

        # Target sampling. If precondition fails at sample time (race with
        # other workers), re-pick action once with a fresh snapshot.
        target, retrieved_chunks, query, dissent = await self._gather_target(
            action, store, pool_state, field_state,
        )

        # Gap 3: update local biases from first available target.
        local_biases = self._local_action_biases(store, target)

        # If the action needs a target and none is available, re-snapshot
        # and re-pick. Cap re-pick at 1 to avoid livelock.
        if target is None and action not in (SCOUT,):
            fresh = FieldState.from_store(
                store, iteration=pool_state.iteration_counter,
                elapsed_s=time.time() - pool_state.started_at,
            )
            action = choose_action(fresh, self.recent_actions, pool_state,
                                   self._rng,
                                   disabled_actions=self._disabled_actions,
                                   local_biases=local_biases) or SCOUT
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
        chosen_llm = self._llm_for_action(action)
        raw = await chosen_llm.generate(
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

        # SAFE post-parse override: substitute pre-computed atom scores and
        # extended metadata when _gather_target ran the SAFE decomposition path.
        if action == VALIDATE:
            _atoms = getattr(self, "_validate_atoms", None)
            _agg = getattr(self, "_validate_agg_score", None)
            if _atoms and _agg is not None:
                parsed = ParsedDeposit(
                    signal_type=parsed.signal_type,
                    content=parsed.content,
                    strength=round(_agg, 4),
                    parent_id_override=parsed.parent_id_override,
                    metadata={
                        **(parsed.metadata or {}),
                        "atoms": _atoms,
                        "aggregation": "weighted_mean",
                        "atom_count": len(_atoms),
                        "score": round(_agg, 4),
                    },
                )
                self._validate_atoms = None
                self._validate_agg_score = None

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
        elif action in (DEVELOP, REFINE, CRITIQUE, VALIDATE) and target is not None:
            # Force parent to the sampled target. DEVELOP/REFINE fix (6.11)
            # extended to CRITIQUE and VALIDATE: when the model emits PARENT: none,
            # CRITIQUE_POS/NEG and VERIFICATION signals land orphaned — invisible
            # to projection's BFS — so dissent_pressure and verification_score
            # never accumulate. The sampled target is always the intended parent.
            effective_parent = target.id
        elif dyn_parent == "__NONE__":
            # Only allow parentless deposits for SCOUT (no sampled target).
            # For OBJECT (shows multiple reps; model may legitimately point to
            # any of them) fall back to target.id rather than None so the
            # OBJECTION is at least linked to the cluster representative.
            effective_parent = None if target is None else target.id
        elif dyn_parent is not None:
            effective_parent = dyn_parent
        else:
            effective_parent = parsed.parent_id_override or (
                target.id if target is not None else None
            )

        meta = dict(parsed.metadata or {})
        meta["depositor_agent_id"] = self.agent_id
        meta["action"] = action

        # Atom targeting: stamp which atom this DEVELOP/REFINE deposit addresses.
        # Enables atom_graph edge construction in _build_genomes and future
        # REFINE-targeted prompting to know which atom was strengthened.
        if action in (DEVELOP, REFINE) and target is not None and target.cluster_id:
            genome = self._genome_cache.get(target.cluster_id)
            if genome is not None and genome.atoms:
                from core.actions import _atom_for_develop as _afd
                targeted = _afd(genome)
                if targeted and "targets_atom" not in meta:
                    meta["targets_atom"] = targeted.atom_id
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
            # Each worker is its own partition in the continuous pool (independent
            # query histories). Required by the INITIAL partition_id assertion.
            if "partition_id" not in meta:
                meta["partition_id"] = self.agent_id

        if action in (DEVELOP, CHAIN, REFINE):
            # Use the depositing worker's own agent_id as partition_id.
            # All DEVELOP deposits for the same INITIAL would otherwise inherit
            # the same partition_id from the parent, giving
            #   support_partitions = {(parent_partition, "develop")}
            # → support_diversity = 1 regardless of how many workers contribute.
            # Worker IDs are unique, so each worker's deposit produces a distinct
            # (partition_id, depositor) pair → support_diversity grows correctly.
            meta["partition_id"] = self.agent_id

        # Belt-and-suspenders for SUPPORT deposits: if the sampled target signal
        # was pruned between _gather_target() and here, the store's parent-lookup
        # inheritance returns None and partition_id would be empty, firing the
        # assertion. Carry partition_id forward from the target we already sampled.
        if (effective_type == "SUPPORT"
                and "partition_id" not in meta
                and target is not None
                and target.partition_id):
            meta["partition_id"] = target.partition_id

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

        # Gap 4: update semantic position from this deposit's embedding
        from core.config import USE_WORKER_POSITION, WORKER_POSITION_WINDOW
        if USE_WORKER_POSITION:
            emb = store.get_embedding(sid)
            if emb is not None:
                self._position_embs.append(emb)
                if len(self._position_embs) > WORKER_POSITION_WINDOW:
                    self._position_embs.pop(0)
                if self._position_embs:
                    n_dims = len(self._position_embs[0])
                    centroid = [
                        sum(e[i] for e in self._position_embs) / len(self._position_embs)
                        for i in range(n_dims)
                    ]
                    norm = sum(x * x for x in centroid) ** 0.5
                    if norm > 1e-9:
                        self._position_centroid = [x / norm for x in centroid]

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
                # Step-Back + HyDE (FLARE-light): once past cold-start, lift the
                # query to a higher abstraction level so the vocabulary matches
                # confirming documents rather than the question itself.  Skip
                # if the resulting query is already well-served (FLARE-light dedup).
                if query and pool_state.cold_start_done:
                    from core.query_planner import plan_step_back, plan_hyde_query
                    _scout_llm = self._llm_for_action(SCOUT)
                    try:
                        _step_back = await plan_step_back(query, _scout_llm)
                        _hyde_q = await plan_hyde_query(_step_back, _scout_llm)
                        # Prefer HyDE query; fall back to step-back phrase; then keep original.
                        if _hyde_q and find_cached_query(_hyde_q, pool_state.served_queries) is None:
                            query = _hyde_q
                        elif _step_back and find_cached_query(_step_back, pool_state.served_queries) is None:
                            query = _step_back
                        # else: original query is fine (already cached or HyDE failed)
                    except Exception:
                        pass
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
                    retrieved = _search(query, max_results=8,
                                        task_type=getattr(self, "task_type", None))
                elif pool_state.try_reserve_search():
                    retrieved = _search(query, max_results=8,
                                        task_type=getattr(self, "task_type", None))
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
                worker_centroid=self._position_centroid,
            )
            if target is None:
                return None, [], "", None
            # Cluster-aware dissent walk: objections deposited against
            # cluster reps now reach DEVELOP even when DEVELOP sampled a
            # sibling INITIAL. mark_read bumps visits on the dissent the
            # prompt actually renders.
            dissent = _cluster_dissent(store, target.id)
            if dissent is not None:
                store.mark_read(dissent.id)
                store.mark_read(target.id)
            # Sparse-cluster search hook: <2 SUPPORT children → query.
            n_support = sum(
                1 for cid in store.by_parent(target.id)
                if store.get(cid) and store.get(cid).type == SUPPORT
            )
            # FLARE-strict: structural trigger (< 2 supports) is necessary but
            # not sufficient — only retrieve when the LM self-rates as uncertain.
            if n_support < 2:
                try:
                    from core.search_tool import search as _search, summarize_for_signal
                    from core.query_planner import (
                        plan_develop_query, find_cached_query,
                        rate_confidence, plan_step_back, plan_hyde_query,
                    )
                    _dev_llm = self._llm_for_action(DEVELOP)
                    _conf = await rate_confidence(target.content, _dev_llm)
                    if _conf >= _FLARE_TAU:
                        # LM is confident — skip retrieval for this iteration.
                        pass
                    else:
                        # Low confidence: build HyDE query from step-back of target.
                        _step_back = await plan_step_back(target.content, _dev_llm)
                        _hyde_q = await plan_hyde_query(_step_back, _dev_llm)
                        # Fall back to stance-based query if HyDE returned nothing clean.
                        query = (
                            _hyde_q
                            if _hyde_q
                            else plan_develop_query(
                                target.content, pool_state.served_queries,
                                task_type=self.task_type,
                            )
                        )
                        if query:
                            cached_q = find_cached_query(query, pool_state.served_queries)
                            if cached_q is not None and cached_q != query:
                                query = cached_q
                                retrieved = _search(
                                    query, max_results=5,
                                    task_type=getattr(self, "task_type", None),
                                )
                            elif pool_state.try_reserve_search():
                                retrieved = _search(
                                    query, max_results=5,
                                    task_type=getattr(self, "task_type", None),
                                )
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
                                        "trigger": "sparse_support_flare",
                                        "flare_confidence": round(_conf, 3),
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
            target = _sample_initial(store, pool_state.recent_targets, self._rng,
                                     worker_centroid=self._position_centroid)
            return target, [], "", None

        if action == OBJECT:
            # Genome-aware target selection: prefer influential but weakly-grounded
            # clusters (high composite_fitness × low grounding = worth attacking).
            # Falls back to DBSCAN field-size selection when no genome cache.
            genome_target: Optional[Signal] = None
            if self._genome_cache:
                # Score each cached genome: fitness/grounding ratio identifies clusters
                # that have survived selection but lack external backing — highest
                # priority for adversarial objection.
                best_score = -1.0
                for rep_id, genome in self._genome_cache.items():
                    grounding = genome.fitness_breakdown.get("grounding", 0.0)
                    # Vulnerability = fitness (influence) / (grounding + ε)
                    vuln = genome.composite_fitness / max(0.05, grounding)
                    if vuln > best_score:
                        sig = store.get(rep_id)
                        if sig is not None:
                            best_score = vuln
                            genome_target = sig

            if genome_target is not None:
                # genome_target is the cluster representative; the genome cache
                # has one entry per cluster so there are no other members to add.
                reps = [genome_target]
            else:
                # Legacy fallback: DBSCAN cluster heads
                dbscan_clusters = store.cluster_signals_dbscan(INITIAL, eps=0.35)
                if dbscan_clusters and len(dbscan_clusters[0]) > 0:
                    largest = dbscan_clusters[0]
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
            # Genome-aware target selection: prefer high-fitness clusters whose
            # atoms have the lowest mean verification_score — highest marginal
            # value from adding an external verification pass. Falls back to
            # the existing well-supported cluster head sampler.
            validate_target: Optional[Signal] = None
            if self._genome_cache:
                best_val = -1.0
                for rep_id, genome in self._genome_cache.items():
                    if not genome.atoms:
                        continue
                    mean_unverified = sum(
                        1.0 - a.verification_score for a in genome.atoms
                    ) / len(genome.atoms)
                    val = genome.composite_fitness * mean_unverified
                    if val > best_val:
                        sig = store.get(rep_id)
                        if sig is not None and not pool_state.recent_targets.get(sig.id, 0):
                            best_val = val
                            validate_target = sig
            target = validate_target or _sample_well_supported_cluster_head(
                store, pool_state.recent_targets, self._rng,
            )
            if target is None:
                return None, [], "", None
            # SAFE: decompose the cluster representative into atomic facts,
            # issue step-back + HyDE queries per atom, score each independently,
            # and aggregate with centrality weights.
            chosen_llm = self._llm_for_action(VALIDATE)
            query = ""
            try:
                from core.search_tool import search as _search
                from core.query_planner import (
                    plan_step_back, plan_hyde_query, find_cached_query,
                )
                atoms = await _safe_decompose(target.content, chosen_llm)
                atom_results: list[dict] = []
                for atom in atoms:
                    atom_text = atom["text"]
                    atom_weight = atom.get("weight", 1.0)
                    atom_query = ""
                    snippet = ""
                    snippet_tag = "(no result)"
                    try:
                        step_back = await plan_step_back(atom_text, chosen_llm)
                        hyde_q = await plan_hyde_query(step_back, chosen_llm)
                        # Dedup: reuse a cached query if semantically equivalent.
                        atom_query = hyde_q or step_back
                        cached_q = find_cached_query(atom_query, pool_state.served_queries) if atom_query else None
                        if cached_q is not None:
                            atom_query = cached_q
                            hits = _search(atom_query, max_results=2)
                        elif atom_query and pool_state.try_reserve_search():
                            hits = _search(atom_query, max_results=2)
                        else:
                            hits = []
                        if hits and atom_query:
                            prev = pool_state.served_queries.get(atom_query, 0)
                            pool_state.served_queries[atom_query] = max(prev, len(hits))
                            if not query:
                                query = atom_query
                    except Exception:
                        hits = []
                    if hits:
                        tag = (getattr(hits[0], "source_tag", "") or "")[:120]
                        body = (getattr(hits[0], "text", "") or "")[:300]
                        snippet = f"[{tag}]\n{body}"
                        snippet_tag = tag or "(unnamed source)"
                    else:
                        snippet = f"(no result for: {atom_query!r})"
                    # Score this atom against its snippet.
                    try:
                        atom_score = await _safe_score_atom(
                            atom_text, snippet, chosen_llm,
                            task_type=self.task_type,
                        )
                    except Exception:
                        atom_score = 0.5
                    atom_results.append({
                        "text": atom_text,
                        "weight": atom_weight,
                        "query": atom_query,
                        "score": atom_score,
                        "snippet_tag": snippet_tag,
                    })
                # Centrality-weighted mean across atoms.
                total_w = sum(a["weight"] for a in atom_results) or 1.0
                agg_score = sum(a["score"] * a["weight"] for a in atom_results) / total_w
                self._validate_atoms = atom_results
                self._validate_agg_score = agg_score
                self._validate_external = _format_safe_external(atom_results)
            except Exception as exc:
                print(f"[validate {self.agent_id}] SAFE pass failed: "
                      f"{type(exc).__name__}: {exc}; falling back to single-query path")
                # Legacy fallback: single keyphrase → single snippet.
                self._validate_atoms = None
                self._validate_agg_score = None
                try:
                    from core.query_planner import plan_validate_query, find_cached_query
                    from core.search_tool import search as _search
                    query = plan_validate_query(target.content)
                    if query:
                        cached_q = find_cached_query(query, pool_state.served_queries)
                        if cached_q is not None and cached_q != query:
                            query = cached_q
                        if pool_state.try_reserve_search():
                            hits = _search(query, max_results=3)
                        else:
                            hits = []
                    else:
                        hits = []
                    if hits and query:
                        pool_state.served_queries[query] = max(
                            pool_state.served_queries.get(query, 0), len(hits)
                        )
                    blocks = []
                    for c in hits[:2]:
                        tag = (getattr(c, "source_tag", "") or "")[:120]
                        body = (getattr(c, "text", "") or "")[:400]
                        blocks.append(f"[{tag}]\n{body}")
                    self._validate_external = (
                        "\n\n".join(blocks) if blocks
                        else f"(no snippet for {query!r})"
                    )
                except Exception:
                    self._validate_external = "(search failed)"
            return target, [], query, None

        if action == REFINE:
            # Prefer INITIALs that have accumulated dissent — REFINE now
            # serves as the rebuttal/deliberation step. Falls back to the
            # underserved-initial path when no dissent exists anywhere.
            target = _sample_contested_initial(
                store, pool_state.recent_targets, self._rng,
            )
            if target is None:
                target = _sample_underserved_initial(
                    store, pool_state.recent_targets, self._rng,
                    worker_centroid=self._position_centroid,
                )
                if target is None:
                    return None, [], "", None
            # Pull cluster-aware dissent so the prompt can render a rebuttal.
            dissent = _cluster_dissent(store, target.id)
            if dissent is not None:
                store.mark_read(dissent.id)
                store.mark_read(target.id)
            return target, [], "", dissent

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
        # Genome-aware: look up the cluster genome for this target signal.
        # Falls back to None (standard prompt) when the cache is empty or the
        # cluster has no genome yet (early iterations, or mock mode).
        genome = self._genome_cache.get(target.cluster_id) if target.cluster_id else None
        if action == DEVELOP:
            return A.develop_prompt(self.task_prompt, target, dissent,
                                    retrieved, query, genome=genome)
        if action == CHAIN:
            return A.chain_prompt(self.task_prompt, target)
        if action == CRITIQUE:
            return A.critique_prompt(self.task_prompt, target, genome=genome)
        if action == OBJECT:
            reps = getattr(self, "_object_reps", [target])
            if not reps:
                return None
            return A.object_prompt(self.task_prompt, reps, task_type=self.task_type,
                                   genome=genome)
        if action == VALIDATE:
            external = getattr(self, "_validate_external", "")
            return A.validate_prompt(self.task_prompt, target, external,
                                     task_type=self.task_type)
        if action == REFINE:
            # `dissent` is forwarded from _gather_target when the chosen
            # target has cluster-level dissent; A.refine_prompt switches
            # to its rebuttal variant when dissent is non-None.
            return A.refine_prompt(self.task_prompt, target, dissent=dissent,
                                   genome=genome)
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
            # Mirror the counter into the store so newly-deposited signals
            # stamp iter_at_deposit correctly. set_iteration is cheap (one
            # lock-and-assign). The store uses this for both decay's
            # youth-grace gate and projection's age-density weighting.
            try:
                store.set_iteration(pool_state.iteration_counter)
            except AttributeError:
                # SignalStore old enough to not have set_iteration — skip.
                # The fallback in projection.py handles iter_age == 0
                # gracefully.
                pass
        await asyncio.sleep(0)


async def run_pool(store: SignalStore, llm, task_prompt: str,
                   user_prompt: str, stop_event: asyncio.Event,
                   n_workers: int = 24,
                   task_type: Optional[str] = None,
                   router=None,
                   genome_cache: Optional[dict] = None) -> PoolState:
    """Spin up `n_workers` and run until `stop_event` fires.

    Returns the PoolState (action log, iteration count, etc.) for the
    orchestrator's summary.

    `router` (optional MultiEngineRouter): when provided, each Worker
    routes generate() per-action via ACTION_TO_ROLE → router.engine_for.
    The single `llm` argument is still accepted as the fallback for
    actions whose role is missing from the bundle.

    `genome_cache` (optional dict): cluster_id -> ClusterGenome, populated
    externally by run_swarm.py after each build_projection() call. Workers
    share a reference to this dict — updates made between rounds are visible
    immediately. Falls back to empty (no genome-aware prompting) when None.
    """
    pool_state = PoolState()
    _gc = genome_cache if genome_cache is not None else {}
    workers = [
        Worker(i, llm, task_prompt, user_prompt, rng_seed=i,
               task_type=task_type, router=router)
        for i in range(n_workers)
    ]
    for w in workers:
        w._genome_cache = _gc
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
