"""Tests for agents/stock_roles.py — construction wiring + the gradable
prediction heuristic (the artifact the backtest grades).

Uses lightweight fakes so no LLM/GPU/store machinery is needed. The prediction
SIGN/direction is the robust contract here; magnitude is a tunable.
"""

from datetime import date

from agents.stock_roles import (
    build_stock_agents, EquityBriefSynthesizer, LensScout, DataValidator,
)
from core.stock_data import Snapshot, LENSES


def _snap(**kw):
    return Snapshot(ticker="NVDA", as_of=date(2024, 1, 15), **kw)


# ---- construction helper -------------------------------------------------

def test_build_stock_agents_structure():
    snap = _snap(pe=28.0, fwd_pe=25.0, rev_growth_yoy=22.0)
    agents = build_stock_agents(lambda role: object(), snap, "Assess NVDA",
                                horizon_days=21)
    # one scout per lens
    assert len(agents["scout"]) == len(LENSES)
    assert all(isinstance(s, LensScout) for s in agents["scout"])
    assert all(isinstance(v, DataValidator) for v in agents["validator"])
    synth = agents["synthesizer"]
    assert isinstance(synth, EquityBriefSynthesizer)
    assert synth._snapshot is snap
    assert synth._horizon_days == 21
    # scouts carry distinct lens partition ids
    assert {s.lens for s in agents["scout"]} == set(LENSES)


# ---- gradable prediction heuristic --------------------------------------

class _CP:
    def __init__(self, rid, support_diversity, verification_score):
        self.representative_id = rid
        self.support_diversity = support_diversity
        self.verification_score = verification_score


class _Sig:
    def __init__(self, sid, content):
        self.id = sid
        self.content = content


class _Store:
    def __init__(self, contents):
        self._c = contents

    def get(self, sid):
        return _Sig(sid, self._c[sid]) if sid in self._c else None

    def by_type(self, _t):
        return []


class _Proj:
    def __init__(self, surviving):
        self.surviving = surviving


def _synth():
    s = EquityBriefSynthesizer(object(), "Assess NVDA")
    return s


def test_prediction_bullish_field_goes_long():
    snap = _snap(pe=28.0)
    proj = _Proj([_CP("INITIAL_1", 5, 0.8), _CP("INITIAL_2", 4, 0.6)])
    store = _Store({
        "INITIAL_1": "Undervalued with strong growth and expanding margins",
        "INITIAL_2": "Revenue accelerating; analysts raised targets — upside",
    })
    pred = _synth().build_prediction(proj, store, snap, "NVDA", 21)
    assert pred["direction"] == "long"
    assert pred["predicted_return_pct"] > 0
    assert 0.0 <= pred["confidence"] <= 1.0
    assert pred["schema"] == "stock_prediction_v1"
    assert pred["as_of_date"] == "2024-01-15"


def test_prediction_bearish_field_avoids():
    snap = _snap(pe=28.0)
    proj = _Proj([_CP("INITIAL_1", 5, 0.2), _CP("INITIAL_2", 4, 0.0)])
    store = _Store({
        "INITIAL_1": "Overvalued and expensive; decelerating growth",
        "INITIAL_2": "Margin compression and regulatory risk — downside",
    })
    pred = _synth().build_prediction(proj, store, snap, "NVDA", 21)
    # short is disabled in the POC, so a bearish field becomes 'avoid'
    assert pred["direction"] == "avoid"
    assert pred["predicted_return_pct"] < 0  # the sign still reflects the lean


def test_prediction_neutral_field_avoids():
    snap = _snap(pe=28.0)
    proj = _Proj([_CP("INITIAL_1", 3, 0.0)])
    store = _Store({"INITIAL_1": "The company operates in several segments."})
    pred = _synth().build_prediction(proj, store, snap, "NVDA", 21)
    assert pred["direction"] == "avoid"
    assert pred["predicted_return_pct"] == 0.0


def test_prediction_empty_field_is_safe():
    snap = _snap()
    pred = _synth().build_prediction(_Proj([]), _Store({}), snap, "NVDA", 21)
    assert pred["direction"] == "avoid"
    assert pred["n_surviving_clusters"] == 0
