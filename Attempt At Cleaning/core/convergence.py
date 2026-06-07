"""ConvergenceDetector — replaces the fixed-NUM_ROUNDS halting condition.

The continuous worker pool runs until this detector flips `satisfied()` to
True. There are three halt paths, evaluated in order:

  1. Quality gate (preferred outcome): at least one cluster has
       - support_diversity >= 4
       - dissent_pressure < 0.5
       - TWO independent validator deposits, each with score >= 0.7
     Then run >= 30 more iterations to let dissent surface; if dissent
     pressure rises in that window, un-flip and continue.
  2. Saturation: 60 iterations without a new surviving cluster AND
     |Δ max_strength over last 60| < 0.02.
  3. Cap: hit MAX_ITERATIONS or MAX_TIME.

Hard floors prevent premature halt:
  - iterations >= MIN_ITERATIONS (default 50)
  - elapsed >= MIN_TIME_S (default 60)
  - n_initials >= MIN_INITIALS_FOR_HALT (default 6)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import os

from .signal_store import SignalStore
from .signal_types import INITIAL, VERIFICATION
from .projection import build_projection
from .config import SURVIVAL_TASK_PROFILES, SURVIVAL_DEFAULT_PROFILE


def _float_env(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _int_env(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


# Hard floors — never halt before these are met (any halt path).
# Override via env vars for fast smoke tests (e.g. SWARM_MIN_TIME_S=0).
MIN_ITERATIONS = _int_env("SWARM_MIN_ITERATIONS", 50)
MIN_TIME_S = _float_env("SWARM_MIN_TIME_S", 60.0)
MIN_INITIALS_FOR_HALT = _int_env("SWARM_MIN_INITIALS_FOR_HALT", 6)
# Require at least this many inter-cluster edges before an EARLY (quality/
# saturation) halt. DEFAULT NOW 0 (disabled): this floor was tuned in the
# mega-blob era, when one giant cluster generated lots of (mostly spurious,
# empty-dissent) edges. The cluster size-penalty fix removed the blob, so
# cohesive distinct clusters legitimately form FEW or ZERO edges — and this
# floor was then blocking convergence forever (see forevergroq: 4.2 h /
# 3000+ iters, quality_met=True the whole time, 0 edges). Set >0 only if you
# re-introduce edge-based gating. (Never blocks the absolute caps either way.)
MIN_INTER_CLUSTER_EDGES = _int_env("SWARM_MIN_INTER_CLUSTER_EDGES", 0)

# Quality gate thresholds.
QUALITY_SUPPORT_DIV = 4
QUALITY_DISSENT_PRESSURE_MAX = 0.5
QUALITY_VER_SCORE_MIN = 0.7
QUALITY_DUAL_VALIDATORS = 2          # require this many independent VERIFICATIONs
QUALITY_HOLD_ITERATIONS = 30         # wait this many iters after quality_met flip

# Saturation thresholds.
SAT_NO_NEW_SURVIVING = _int_env("SWARM_SAT_NO_NEW_SURVIVING", 60)
SAT_STRENGTH_DELTA = 0.02
SAT_WINDOW = 60

# Hard caps.
MAX_ITERATIONS = _int_env("SWARM_MAX_ITERATIONS", 2000)
MAX_TIME_S = _float_env("SWARM_MAX_TIME_S", 900.0)


@dataclass
class DetectorState:
    """Public snapshot of detector internals for logging."""
    quality_met: bool = False
    quality_met_at_iter: Optional[int] = None
    iterations_since_quality: int = 0
    iterations_since_new_surviving: int = 0
    n_surviving_last: int = 0
    n_inter_cluster_edges_last: int = 0  # from last tick; used for MIN_INTER_CLUSTER_EDGES floor
    strength_history: deque = field(default_factory=lambda: deque(maxlen=SAT_WINDOW))
    reason: str = ""


class ConvergenceDetector:
    """Live convergence detector. Read pool state via `tick()`.

    Usage:
        detector = ConvergenceDetector(store)
        while not detector.satisfied(iteration_counter, elapsed_s):
            await asyncio.sleep(2)
            detector.tick(iteration_counter, elapsed_s)
        # detector.reason is one of "quality" | "saturation" | "cap"
    """

    def __init__(self, store: SignalStore,
                 min_iterations: int = MIN_ITERATIONS,
                 min_time_s: float = MIN_TIME_S,
                 min_initials_for_halt: int = MIN_INITIALS_FOR_HALT,
                 max_iterations: int = MAX_ITERATIONS,
                 max_time_s: float = MAX_TIME_S,
                 task_type: Optional[str] = None,
                 tick_interval: int = 10):
        self.store = store
        self.min_iterations = min_iterations
        self.min_time_s = min_time_s
        self.min_initials_for_halt = min_initials_for_halt
        self.max_iterations = max_iterations
        self.max_time_s = max_time_s
        self.task_type = task_type
        # Task profile picks the quality-gate variant. When task_type is
        # None we fall back to the strict factual gate (two independent
        # validators) so existing callers don't change behavior.
        if task_type is None:
            self.task_profile = {"requires_verification": True,
                                  "credibility_chain_depth": 999}
        else:
            self.task_profile = SURVIVAL_TASK_PROFILES.get(
                task_type, SURVIVAL_DEFAULT_PROFILE,
            )
        self.tick_interval = max(1, tick_interval)
        self.state = DetectorState()

    # ------------------------------------------------------------------
    # Live update
    # ------------------------------------------------------------------

    def tick(self, iteration_counter: int, elapsed_s: float) -> None:
        """Update state from the live signal store.

        build_projection is expensive (full DAG walk); only run it every
        tick_interval calls. Strength history is updated every call so
        saturation detection stays smooth.
        """
        # Always update strength history (cheap store stat).
        max_strength = self.store.stats().get("max_strength", 0.0) or 0.0
        self.state.strength_history.append(max_strength)

        if iteration_counter % self.tick_interval != 0:
            return

        proj = build_projection(self.store, has_validators=True,
                                 task_type=self.task_type)
        n_surviving = len(proj.surviving)

        # New surviving cluster?
        if n_surviving > self.state.n_surviving_last:
            self.state.iterations_since_new_surviving = 0
        else:
            self.state.iterations_since_new_surviving += 1
        self.state.n_surviving_last = n_surviving
        self.state.n_inter_cluster_edges_last = len(proj.inter_cluster_edges)

        # Quality gate
        prev_quality = self.state.quality_met
        self.state.quality_met = self._evaluate_quality(proj)
        if self.state.quality_met and not prev_quality:
            self.state.quality_met_at_iter = iteration_counter
            self.state.iterations_since_quality = 0
        elif self.state.quality_met:
            self.state.iterations_since_quality += 1
        else:
            # Quality lost — un-flip and reset the hold counter.
            self.state.quality_met_at_iter = None
            self.state.iterations_since_quality = 0

    def _evaluate_quality(self, proj) -> bool:
        """At least one cluster passes the quality gate.

        Factual task profile (requires_verification=True):
            support_diversity >= 4, dissent_pressure < 0.5, and TWO
            independent validator deposits with strength >= 0.7.

        Non-factual task profile (requires_verification=False):
            support_diversity >= 4, dissent_pressure < 0.5, and
            support_depth >= credibility_chain_depth (default 3).
            External verification is honoured if present but not required —
            web sources can't corroborate philosophical or interpretive
            claims cleanly, so demanding it gates everything out.
        """
        requires_ver = self.task_profile.get("requires_verification", True)
        chain_floor = int(self.task_profile.get("credibility_chain_depth", 999))
        for cp in proj.surviving:
            if cp.support_diversity < QUALITY_SUPPORT_DIV:
                continue
            if cp.dissent_pressure >= QUALITY_DISSENT_PRESSURE_MAX:
                continue
            if not requires_ver:
                # Internal-coherence gate: chain depth is the quality signal.
                if cp.support_depth >= chain_floor:
                    return True
                continue
            # Factual gate: count independent validator deposits with score
            # >= threshold. Independence = distinct depositor_agent_id.
            ver_signals = [
                self.store.get(vid) for vid in cp.verification_set
            ]
            ver_signals = [v for v in ver_signals if v is not None]
            strong = [v for v in ver_signals
                      if v.strength >= QUALITY_VER_SCORE_MIN]
            distinct_validators = {
                v.metadata.get("depositor_agent_id", "")
                for v in strong
            }
            distinct_validators.discard("")
            if len(distinct_validators) >= QUALITY_DUAL_VALIDATORS:
                return True
        return False

    # ------------------------------------------------------------------
    # Halt decision
    # ------------------------------------------------------------------

    def satisfied(self, iteration_counter: int, elapsed_s: float) -> bool:
        # ABSOLUTE caps FIRST — a run must NEVER exceed these, regardless of the
        # soft floors below. (forevergroq bug: the inter-cluster-edge floor sat
        # ABOVE the cap checks and returned False whenever no edge formed, so the
        # pool ran 4+ hours / 3000+ iters past MAX_TIME_S. A cap that a floor can
        # veto is not a cap.)
        if iteration_counter >= self.max_iterations:
            self.state.reason = "cap_iterations"
            return True
        if elapsed_s >= self.max_time_s:
            self.state.reason = "cap_time"
            return True

        # Hard floors — prevent halting EARLY (below the caps) before these hold.
        if iteration_counter < self.min_iterations:
            return False
        if elapsed_s < self.min_time_s:
            return False
        if self.state.n_surviving_last == 0:
            n_initials = sum(
                1 for _ in self.store.by_type(INITIAL)
            )
            if n_initials < self.min_initials_for_halt:
                return False

        # Inter-cluster edge floor: ensure the field has had enough cross-cluster
        # activity to populate at least one typed edge before an EARLY halt.
        # (Does NOT block the absolute caps above.)
        if (MIN_INTER_CLUSTER_EDGES > 0
                and self.state.n_inter_cluster_edges_last < MIN_INTER_CLUSTER_EDGES):
            return False

        # Quality halt (preferred outcome)
        if (self.state.quality_met
                and self.state.iterations_since_quality
                >= QUALITY_HOLD_ITERATIONS):
            self.state.reason = "quality"
            return True

        # Saturation halt
        if (self.state.iterations_since_new_surviving >= SAT_NO_NEW_SURVIVING
                and self._strength_delta() < SAT_STRENGTH_DELTA):
            self.state.reason = "saturation"
            return True

        return False

    def _strength_delta(self) -> float:
        h = self.state.strength_history
        if len(h) < 2:
            return 1.0  # not enough data; treat as "still changing"
        return abs(h[-1] - h[0])
