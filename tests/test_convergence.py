"""ConvergenceDetector halt-ordering — the forevergroq fix.

The bug: the inter-cluster-edge floor was checked BEFORE the hard caps, so a run
with zero edges (which the cluster size-penalty now makes normal) never halted —
4.2 hours / 3000+ iters past MAX_TIME_S. Caps must be absolute.
"""

import core.convergence as cv
from core.convergence import ConvergenceDetector
from core.signal_store import SignalStore


def _det(max_time=100.0, max_iter=2000):
    return ConvergenceDetector(SignalStore(), max_time_s=max_time,
                               max_iterations=max_iter, task_type="debate")


def test_time_cap_fires_even_with_zero_edges():
    d = _det(max_time=100.0)
    d.state.n_inter_cluster_edges_last = 0     # the forevergroq condition
    d.state.n_surviving_last = 3
    assert d.satisfied(iteration_counter=60, elapsed_s=101.0) is True
    assert d.state.reason == "cap_time"


def test_iter_cap_fires_even_with_zero_edges():
    d = _det(max_iter=2000)
    d.state.n_inter_cluster_edges_last = 0
    d.state.n_surviving_last = 3
    assert d.satisfied(iteration_counter=2000, elapsed_s=50.0) is True
    assert d.state.reason == "cap_iterations"


def test_quality_halt_fires_when_met_and_held_even_with_zero_edges():
    # The good outcome: quality met + held -> halt (floor default 0 no longer
    # blocks it). This is exactly what forevergroq SHOULD have done.
    d = _det(max_time=900.0)   # high cap so the condition under test is reached, not the cap
    d.state.n_surviving_last = 3
    d.state.quality_met = True
    d.state.iterations_since_quality = cv.QUALITY_HOLD_ITERATIONS
    d.state.n_inter_cluster_edges_last = 0
    assert d.satisfied(iteration_counter=60, elapsed_s=200.0) is True
    assert d.state.reason == "quality"


def test_edge_floor_still_blocks_EARLY_halt_when_enabled(monkeypatch):
    # Below the caps, with the floor explicitly re-enabled, zero edges blocks an
    # early quality halt (but NOT the caps — covered above).
    monkeypatch.setattr(cv, "MIN_INTER_CLUSTER_EDGES", 1)
    d = _det(max_time=900.0)   # high cap so the condition under test is reached, not the cap
    d.state.n_surviving_last = 3
    d.state.quality_met = True
    d.state.iterations_since_quality = cv.QUALITY_HOLD_ITERATIONS
    d.state.n_inter_cluster_edges_last = 0
    assert d.satisfied(iteration_counter=60, elapsed_s=200.0) is False


def test_min_floors_prevent_early_halt():
    d = _det()
    # below MIN_ITERATIONS / MIN_TIME_S and below caps -> not satisfied
    assert d.satisfied(iteration_counter=1, elapsed_s=1.0) is False


class _FakeProj:
    """Minimal projection stand-in for tick() unit tests."""
    def __init__(self, n_surviving=0):
        self.surviving = [object()] * n_surviving
        self.inter_cluster_edges = []


def test_quality_hold_advances_in_iteration_units(monkeypatch):
    """The kb-run bug: tick() is polled on a 2s wall-clock timer, so the old
    `iteration_counter % tick_interval` body-gate fired only on the rare poll
    landing on a multiple of 10, and the hold counter crawled in poll-units —
    never reaching QUALITY_HOLD_ITERATIONS even when quality was stably met for
    80s+ (every kb run died on cap_time). Now the counter advances by the real
    iteration delta and the quality halt actually fires."""
    d = ConvergenceDetector(SignalStore(), max_time_s=1e9, max_iterations=10_000,
                            task_type="debate", min_iterations=50, min_time_s=0.0)
    monkeypatch.setattr(cv, "build_projection", lambda *a, **k: _FakeProj(n_surviving=1))
    monkeypatch.setattr(d, "_evaluate_quality", lambda proj: True)

    start = 60                            # above min_iterations=50 floor
    d.tick(start, 10.0)                    # first body run: quality flips True
    assert d.state.quality_met is True
    assert d.state.iterations_since_quality == 0

    d.tick(start + cv.QUALITY_HOLD_ITERATIONS, 50.0)   # one jump past the hold
    assert d.state.iterations_since_quality == cv.QUALITY_HOLD_ITERATIONS
    assert d.satisfied(start + cv.QUALITY_HOLD_ITERATIONS, 100.0) is True
    assert d.state.reason == "quality"


def test_tick_body_runs_once_per_interval_not_per_poll(monkeypatch):
    """Repeated polls at the SAME iteration (slow pool, fast timer) must not
    each advance the hold counter — the body runs once per tick_interval of
    iteration progress."""
    d = ConvergenceDetector(SignalStore(), max_time_s=1e9, task_type="debate",
                            min_iterations=50, min_time_s=0.0)
    monkeypatch.setattr(cv, "build_projection", lambda *a, **k: _FakeProj(n_surviving=1))
    monkeypatch.setattr(d, "_evaluate_quality", lambda proj: True)
    d.tick(12, 10.0)
    assert d.state.iterations_since_quality == 0
    for _ in range(5):                    # five more polls, iteration unchanged
        d.tick(12, 12.0)
    assert d.state.iterations_since_quality == 0   # body did not re-run


def test_novelty_saturation_halts():
    """Cheap probe: a full window of redundant deposits (no new regions) +
    a survivor -> halt, independent of the quality gate."""
    d = _det(max_time=1e9, max_iter=10_000)
    d.state.n_surviving_last = 2
    d.store._deposit_outcomes.extend([False] * cv.NOVELTY_SAT_WINDOW)
    d.state.novelty_rate_last = 0.0
    assert d.satisfied(iteration_counter=60, elapsed_s=200.0) is True
    assert d.state.reason == "novelty_saturation"


def test_novelty_saturation_waits_for_full_window():
    """A brief early lull (partial window) must not trip the novelty halt."""
    d = _det(max_time=1e9, max_iter=10_000)
    d.state.n_surviving_last = 2
    d.store._deposit_outcomes.extend([False] * (cv.NOVELTY_SAT_WINDOW - 1))
    d.state.novelty_rate_last = 0.0
    assert d.satisfied(iteration_counter=60, elapsed_s=200.0) is False


def test_novelty_saturation_needs_a_survivor():
    """No surviving cluster -> nothing for the synthesizer to read -> don't
    halt on novelty alone."""
    d = _det(max_time=1e9, max_iter=10_000)
    d.state.n_surviving_last = 0
    d.store._deposit_outcomes.extend([False] * cv.NOVELTY_SAT_WINDOW)
    d.state.novelty_rate_last = 0.0
    # n_surviving==0 also triggers the min_initials floor; both keep it running.
    assert d.satisfied(iteration_counter=60, elapsed_s=200.0) is False


def test_oscillating_surviving_count_halts_via_cap():
    """Boundary-oscillation: surviving count flickers 6↔8 so saturation never
    fires (iterations_since_new_surviving is reset each tick). The cap must
    still fire — this is the forevergroq scenario."""
    d = _det(max_time=900.0, max_iter=100)
    d.state.n_surviving_last = 6
    # Simulate oscillation: alternately set to 8 and back to 6; each tick
    # resets iterations_since_new_surviving to 0 so saturation never fires.
    for i in range(60, 100):
        d.state.n_surviving_last = 8 if i % 2 == 0 else 6
        d.state.iterations_since_new_surviving = 0   # reset as if new cluster appeared
        assert d.satisfied(iteration_counter=i, elapsed_s=float(i)) is False, \
            f"should not halt early at iter={i}"
    # At cap: must halt regardless of oscillation
    assert d.satisfied(iteration_counter=100, elapsed_s=500.0) is True
    assert d.state.reason == "cap_iterations"
