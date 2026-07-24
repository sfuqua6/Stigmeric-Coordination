"""ConvergenceDetector — replaces the fixed-NUM_ROUNDS halting condition.

The continuous worker pool runs until this detector flips `satisfied()` to
True. There are three halt paths, evaluated in order:

  1. Quality gate (preferred outcome): at least one cluster has
       - support_diversity >= 4
       - dissent_pressure < 0.5
       - factual profiles (coding/stock/None): TWO independent validator
         deposits, each with score >= 0.7
       - non-factual profiles: support_depth >= credibility_chain_depth;
         debate/analysis (requires_grounding) additionally need ONE
         verification signal >= QUALITY_GROUNDING_VER_MIN (0.55)
     Then run >= QUALITY_HOLD_ITERATIONS more iterations to let dissent
     surface; if dissent pressure rises in that window, un-flip and continue.
  2. Render-set stability: the top-RENDER_K cluster set the composer would
     read (chosen by build_plan) has identical membership AND evidence
     signatures for RENDER_STABLE_ITERS iterations. Robust to dust-cluster
     fragmentation, which resets the other counters.
  3. Saturation: 60 iterations without a new surviving cluster AND
     |Δ max_strength over last 60| < 0.02.
  4. Cap: hit MAX_ITERATIONS or MAX_TIME.

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
# Grounding floor for requires_grounding profiles (debate/analysis): ONE
# verification signal above the validator abstain plateau (~0.5) on the
# qualifying cluster. Deliberately softer than the factual dual-validator
# gate — the goal is "the halt says something about grounding", not
# "philosophical claims need proofs".
QUALITY_GROUNDING_VER_MIN = _float_env("SWARM_QUALITY_GROUNDING_VER_MIN", 0.55)
# Wait this many iters of held quality before the preferred "quality" halt fires.
# Lowered 30 -> 20: real runs (see outputs/kb/run3.txt) hold quality_met=True but
# the hold counter plateaus ~25 and never reaches 30 before the wall-clock cap, so
# the run dies on cap_time instead of converging on quality. 20 lets a genuinely
# held gate complete. Env-overridable for tuning.
QUALITY_HOLD_ITERATIONS = _int_env("SWARM_QUALITY_HOLD_ITERATIONS", 20)

# Saturation thresholds.
SAT_NO_NEW_SURVIVING = _int_env("SWARM_SAT_NO_NEW_SURVIVING", 60)
SAT_STRENGTH_DELTA = 0.02
SAT_WINDOW = 60

# Novelty-saturation halt (the cheap "nothing new is being learned" probe).
# When the last NOVELTY_SAT_WINDOW clusterable deposits opened essentially no
# new idea-regions (store.novelty_rate below the floor) AND at least one cluster
# already survives, the field has stopped exploring — halt without waiting for
# the strict quality/strength windows. Independent of the quality gate, so it
# also catches runs whose quality_met flickers and never completes its hold.
NOVELTY_SAT_WINDOW = _int_env("SWARM_NOVELTY_SAT_WINDOW", 40)
NOVELTY_SAT_FLOOR = _float_env("SWARM_NOVELTY_SAT_FLOOR", 0.05)

# Render-set stability halt (the compute-waste fix, 2026-07-01).
#
# Why the other halts never fire on real runs: near-duplicate claims in the
# 0.65-0.85 similarity band open NEW dust clusters instead of joining existing
# ones, so every "no new surviving cluster" / novelty counter keeps resetting —
# the field's failure mode (duplicate spray) masquerades as perpetual
# exploration, and runs die on cap_time (see hey fable.md: every substantive
# run). This halt watches the only thing the readout actually consumes: the
# top-RENDER_K render set chosen by build_plan(). When the render set AND the
# evidence signatures of its clusters (members/support/dissent/verification
# counts) are unchanged for RENDER_STABLE_ITERS iterations, further iterations
# cannot change the answer — dust cluster #57 opening somewhere in the tail
# does not reset this counter. 0 disables.
RENDER_STABLE_ITERS = _int_env("SWARM_RENDER_STABLE_ITERS", 40)

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
    novelty_rate_last: float = 1.0       # fraction of recent deposits opening new regions
    # Render-set stability: signature of the current build_plan render set
    # (rep_id + evidence counts per cluster) and how many iterations it has
    # been unchanged. See RENDER_STABLE_ITERS.
    render_signature_last: tuple = ()
    iterations_render_stable: int = 0
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
        # Iteration at which the expensive projection body last ran. The body
        # is gated on iteration PROGRESS (not poll-time modulo), so the hold
        # counters advance in true iteration units regardless of poll cadence.
        # None = body has never run; the first body run advances counters by 0
        # (the old `-tick_interval` seed handed every windowed counter ~10
        # phantom iterations, so e.g. saturation fired at 50-55 real
        # iterations against its nominal 60 window).
        self._last_body_iter: Optional[int] = None
        self.state = DetectorState()

    # ------------------------------------------------------------------
    # Live update
    # ------------------------------------------------------------------

    def tick(self, iteration_counter: int, elapsed_s: float) -> None:
        """Update state from the live signal store.

        build_projection is expensive (full DAG walk); run it at most once per
        tick_interval ITERATIONS of pool progress. The orchestrator polls this
        on a wall-clock timer (~every 2s), so gating on `iteration_counter %
        tick_interval` (the old code) fired the body only on the rare poll that
        happened to land on a multiple of tick_interval — and made the hold
        counters advance in an erratic poll-unit, never reaching their
        thresholds (every kb run died on cap_time despite quality_met). Gating
        on iteration *progress* and advancing the counters by the real
        iteration delta makes QUALITY_HOLD_ITERATIONS / SAT_NO_NEW_SURVIVING
        mean what their names say.
        """
        # Always update strength history (cheap store stat).
        max_strength = self.store.stats().get("max_strength", 0.0) or 0.0
        self.state.strength_history.append(max_strength)

        if self._last_body_iter is None:
            iters_advanced = 0          # first body run: initialize, no credit
        else:
            iters_advanced = iteration_counter - self._last_body_iter
            if iters_advanced < self.tick_interval:
                return
        self._last_body_iter = iteration_counter

        proj = build_projection(self.store, has_validators=True,
                                 task_type=self.task_type)
        n_surviving = len(proj.surviving)

        # Cheap novelty probe (no projection needed; read alongside it).
        self.state.novelty_rate_last = self.store.novelty_rate(NOVELTY_SAT_WINDOW)

        # Render-set stability: advance only while the signature is non-empty
        # and unchanged; any change (different clusters selected OR new
        # evidence landing on a selected cluster) resets the counter. A
        # transient build_plan failure (sig=None) HOLDS the accumulated
        # stability instead of erasing it — an exception is not evidence
        # that the render set moved.
        sig = self._render_signature(proj)
        if sig is None:
            pass
        elif sig and sig == self.state.render_signature_last:
            self.state.iterations_render_stable += iters_advanced
            self.state.render_signature_last = sig
        else:
            self.state.iterations_render_stable = 0
            self.state.render_signature_last = sig

        # New surviving cluster?
        if n_surviving > self.state.n_surviving_last:
            self.state.iterations_since_new_surviving = 0
        else:
            self.state.iterations_since_new_surviving += iters_advanced
        self.state.n_surviving_last = n_surviving
        self.state.n_inter_cluster_edges_last = len(proj.inter_cluster_edges)

        # Quality gate
        prev_quality = self.state.quality_met
        self.state.quality_met = self._evaluate_quality(proj)
        if self.state.quality_met and not prev_quality:
            self.state.quality_met_at_iter = iteration_counter
            self.state.iterations_since_quality = 0
        elif self.state.quality_met:
            self.state.iterations_since_quality += iters_advanced
        else:
            # Quality lost — un-flip and reset the hold counter.
            self.state.quality_met_at_iter = None
            self.state.iterations_since_quality = 0

    def _render_signature(self, proj) -> tuple:
        """Signature of what the synthesizer would read if the run halted now.

        build_plan is the same deterministic selector the readout uses, so
        this is not a proxy — it IS the render set. Per selected cluster the
        signature carries the evidence-shape counts; a SUPPORT/OBJECT/
        VERIFICATION landing on a rendered cluster changes the signature and
        resets stability, so a run halts only when the part of the field that
        reaches the page has genuinely stopped moving.

        Keys on the registry cluster_id where available (stable across
        representative flips — strength jitter from dedup/trail amplification
        can swap which member is strongest without changing anything the
        reader sees), falling back to representative_id. Returns None on a
        build_plan failure so the caller HOLDS accumulated stability rather
        than resetting it (never blocks other halt paths)."""
        try:
            from .projection import build_plan
            plan = build_plan(proj, self.store)
        except Exception:
            return None
        by_rep = {
            cp.representative_id: cp
            for cp in (proj.surviving + proj.contested + proj.unverified)
        }

        def _stable_key(rep_id: str) -> str:
            try:
                cid = self.store._cluster_registry.get_cluster_id(rep_id)
                if cid:
                    return cid
            except Exception:
                pass
            return rep_id

        sig = []
        for rid in plan.render_clusters:
            cp = by_rep.get(rid)
            if cp is None:
                sig.append((_stable_key(rid), -1, -1, -1, -1))
            else:
                sig.append((_stable_key(rid), len(cp.member_ids),
                            len(cp.support_set), len(cp.dissent_set),
                            len(cp.verification_set)))
        return tuple(sorted(sig))

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
            Profiles with requires_grounding=True (debate/analysis — the task
            types that actually run the Validator role) additionally need ONE
            verification signal above the abstain plateau on the qualifying
            cluster: without this, the flagship halt asks nothing about
            grounding and fires ~10s after the MIN_TIME floor lifts.
        """
        requires_ver = self.task_profile.get("requires_verification", True)
        requires_grounding = self.task_profile.get("requires_grounding", False)
        chain_floor = int(self.task_profile.get("credibility_chain_depth", 999))
        for cp in proj.surviving:
            if cp.support_diversity < QUALITY_SUPPORT_DIV:
                continue
            if cp.dissent_pressure >= QUALITY_DISSENT_PRESSURE_MAX:
                continue
            if not requires_ver:
                # Internal-coherence gate: chain depth is the quality signal.
                if cp.support_depth >= chain_floor:
                    if not requires_grounding:
                        return True
                    grounded = any(
                        v is not None and v.strength >= QUALITY_GROUNDING_VER_MIN
                        for v in (self.store.get(vid)
                                  for vid in cp.verification_set)
                    )
                    if grounded:
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

        # Render-set stability halt: the top-RENDER_K set the composer would
        # read is unchanged (same clusters, same evidence counts) for a full
        # window. Dust clusters opening in the tail cannot reset this — only
        # changes to what actually reaches the page do. This is the primary
        # defense against paying for post-saturation churn (cap_time endings).
        if (RENDER_STABLE_ITERS > 0
                and self.state.n_surviving_last > 0
                and self.state.render_signature_last
                and self.state.iterations_render_stable >= RENDER_STABLE_ITERS):
            self.state.reason = "render_set_stable"
            return True

        # Saturation halt
        if (self.state.iterations_since_new_surviving >= SAT_NO_NEW_SURVIVING
                and self._strength_delta() < SAT_STRENGTH_DELTA):
            self.state.reason = "saturation"
            return True

        # Novelty-saturation halt (cheap probe). The field has stopped opening
        # new idea-regions: the last full window of clusterable deposits all
        # fell onto existing clusters. Requires a survivor so the synthesizer
        # has something to read, and a full window so a brief early lull can't
        # trip it. Independent of the quality gate — catches flickering runs.
        if (self.state.n_surviving_last > 0
                and self.store.novelty_sample_count() >= NOVELTY_SAT_WINDOW
                and self.state.novelty_rate_last < NOVELTY_SAT_FLOOR):
            self.state.reason = "novelty_saturation"
            return True

        return False

    def _strength_delta(self) -> float:
        h = self.state.strength_history
        if len(h) < 2:
            return 1.0  # not enough data; treat as "still changing"
        return abs(h[-1] - h[0])
