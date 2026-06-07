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
