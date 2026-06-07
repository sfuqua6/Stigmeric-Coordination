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
