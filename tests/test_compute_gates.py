"""Compute program part 1: render-set stability halt + pre-call scout gate.

Both mechanisms attack the same waste: real runs die on cap_time because
dust-cluster fragmentation resets every saturation counter, and novelty is
filtered AFTER the LLM call is spent. These tests pin:

  1. ConvergenceDetector "render_set_stable" halt — fires when the top-K
     render set (what the composer reads) is unchanged for
     RENDER_STABLE_ITERS; resets when the set or its evidence counts change.
  2. worker_pool.scout_gate_engaged — pre-call gate semantics (floors,
     window, disable switch).

No LLM, no network: stores/projections are faked.
"""

import types
from collections import deque

from core import convergence as conv
from core.convergence import ConvergenceDetector
from core.worker_pool import scout_gate_engaged
from core import config as cfg


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeStore:
    """Just enough store surface for detector.tick()/satisfied()."""

    def __init__(self, novelty_rate=0.5, sample_count=100):
        self._novelty_rate = novelty_rate
        self._sample_count = sample_count

    def stats(self):
        return {"max_strength": 0.5}

    def novelty_rate(self, window=40):
        return self._novelty_rate

    def novelty_sample_count(self):
        return self._sample_count

    def by_type(self, _t):
        return iter(())


def _detector(store=None, **kw):
    kw.setdefault("min_iterations", 0)
    kw.setdefault("min_time_s", 0.0)
    kw.setdefault("tick_interval", 1)
    return ConvergenceDetector(store or _FakeStore(), **kw)


def _fake_proj(n_surviving=1):
    # support_diversity=0 makes each cluster fail the quality gate's first
    # check, so _evaluate_quality is exercised but stays False.
    cp = lambda: types.SimpleNamespace(support_diversity=0)
    return types.SimpleNamespace(
        surviving=[cp() for _ in range(n_surviving)],
        contested=[], unverified=[],
        inter_cluster_edges=[],
    )


# ---------------------------------------------------------------------------
# Render-set stability halt
# ---------------------------------------------------------------------------

def test_render_stable_halt_fires(monkeypatch):
    det = _detector()
    det.state.n_surviving_last = 1
    det.state.render_signature_last = (("INITIAL_1", 3, 4, 0, 1),)
    det.state.iterations_render_stable = conv.RENDER_STABLE_ITERS
    assert det.satisfied(iteration_counter=100, elapsed_s=100.0)
    assert det.state.reason == "render_set_stable"


def test_render_stable_requires_survivor_and_signature():
    det = _detector()
    det.state.iterations_render_stable = conv.RENDER_STABLE_ITERS + 10
    # no survivor -> the min_initials floor path runs, and no halt fires
    det.state.n_surviving_last = 0
    det.state.render_signature_last = (("INITIAL_1", 3, 4, 0, 1),)
    assert not det.satisfied(iteration_counter=100, elapsed_s=100.0)
    # survivor but empty signature (plan failed) -> no halt
    det.state.n_surviving_last = 1
    det.state.render_signature_last = ()
    assert not det.satisfied(iteration_counter=100, elapsed_s=100.0)


def test_tick_accumulates_and_resets_stability(monkeypatch):
    det = _detector()
    sigs = iter([
        (("A", 1, 1, 0, 0),),   # tick 1: baseline (differs from initial ())
        (("A", 1, 1, 0, 0),),   # tick 2: unchanged -> accumulate
        (("A", 1, 2, 0, 0),),   # tick 3: new SUPPORT on A -> reset
    ])
    monkeypatch.setattr(ConvergenceDetector, "_render_signature",
                        lambda self, proj: next(sigs))
    monkeypatch.setattr(conv, "build_projection",
                        lambda *a, **k: _fake_proj(n_surviving=1))
    det.tick(iteration_counter=10, elapsed_s=10.0)
    assert det.state.iterations_render_stable == 0     # sig changed from ()
    det.tick(iteration_counter=20, elapsed_s=20.0)
    assert det.state.iterations_render_stable == 10    # unchanged, +10 iters
    det.tick(iteration_counter=30, elapsed_s=30.0)
    assert det.state.iterations_render_stable == 0     # evidence count moved


def test_dust_cluster_does_not_reset_stability(monkeypatch):
    """New tail clusters (the fragmentation failure mode) leave the render
    signature unchanged, so stability keeps accumulating — the whole point."""
    det = _detector()
    fixed_sig = (("A", 2, 3, 0, 0),)
    monkeypatch.setattr(ConvergenceDetector, "_render_signature",
                        lambda self, proj: fixed_sig)
    n_surv = iter([1, 2, 3])   # a "new surviving" dust cluster every tick
    monkeypatch.setattr(conv, "build_projection",
                        lambda *a, **k: _fake_proj(n_surviving=next(n_surv)))
    det.state.render_signature_last = fixed_sig
    det._last_body_iter = 0        # anchor so each tick advances exactly 10
    det.tick(10, 10.0)
    det.tick(20, 20.0)
    det.tick(30, 30.0)
    # iterations_since_new_surviving reset every tick (dust)...
    assert det.state.iterations_since_new_surviving == 0
    # ...but render stability accumulated through it.
    assert det.state.iterations_render_stable == 30


def test_caps_still_dominate():
    det = _detector(max_iterations=50)
    det.state.n_surviving_last = 1
    det.state.render_signature_last = (("A", 1, 1, 0, 0),)
    det.state.iterations_render_stable = 0
    assert det.satisfied(iteration_counter=50, elapsed_s=1.0)
    assert det.state.reason == "cap_iterations"


# ---------------------------------------------------------------------------
# Pre-call scout gate
# ---------------------------------------------------------------------------

def _fs(n_initials=20):
    return types.SimpleNamespace(n_initials=n_initials)


def test_scout_gate_fires_on_saturated_field():
    store = _FakeStore(novelty_rate=0.02, sample_count=100)
    assert scout_gate_engaged(store, _fs(n_initials=20))


def test_scout_gate_holds_while_field_young():
    # Not enough INITIALs yet
    store = _FakeStore(novelty_rate=0.0, sample_count=100)
    assert not scout_gate_engaged(store, _fs(n_initials=cfg.SCOUT_GATE_MIN_INITIALS - 1))
    # Not a full novelty window yet
    store = _FakeStore(novelty_rate=0.0, sample_count=cfg.SCOUT_GATE_WINDOW - 1)
    assert not scout_gate_engaged(store, _fs(n_initials=20))
    # Field still novel
    store = _FakeStore(novelty_rate=0.5, sample_count=100)
    assert not scout_gate_engaged(store, _fs(n_initials=20))


def test_scout_gate_disable_switch(monkeypatch):
    monkeypatch.setattr(cfg, "SCOUT_GATE_NOVELTY_FLOOR", 0.0)
    store = _FakeStore(novelty_rate=0.0, sample_count=100)
    assert not scout_gate_engaged(store, _fs(n_initials=50))
