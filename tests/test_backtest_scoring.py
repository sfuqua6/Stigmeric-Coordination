"""Tests for eval/backtest.py scoring + eval/ground_truth.py realized returns.

The grader is the "make money" contract; its arithmetic must be exactly right
(direction, P&L, baseline edge), or a losing strategy can look like a winner.
"""

from datetime import date, timedelta

import pytest

from eval.backtest import Prediction, score_case, aggregate
from eval.ground_truth import PriceSeries, realized_return


# ---- realized return -----------------------------------------------------

def _series(start: date, closes):
    """Build a business-dayish series (skip weekends) from a list of closes."""
    pts = []
    d = start
    for c in closes:
        while d.weekday() >= 5:  # Sat/Sun
            d += timedelta(days=1)
        pts.append((d, c))
        d += timedelta(days=1)
    return PriceSeries(pts)


def test_realized_return_basic():
    s = _series(date(2024, 1, 1), [100.0] + [0.0] * 9 + [110.0] + [0.0] * 30)
    # entry on 2024-01-01 (or next trading day) close 100; 10 trading days later 110
    # fill intermediate with non-zero to avoid accidental entry/exit on 0
    s = _series(date(2024, 1, 1), [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    r = realized_return(s, date(2024, 1, 1), 10)
    assert r == pytest.approx(0.10)


def test_realized_return_entry_on_or_after():
    # start falls on a weekend -> entry is next trading day (rolls forward)
    s = _series(date(2024, 1, 1), [100, 101, 102, 103, 104, 105, 106, 107])
    # 2024-01-06 is a Saturday; entry should be 2024-01-08 (Mon), not earlier
    r = realized_return(s, date(2024, 1, 6), 1)
    assert r is not None
    assert r > 0  # 106 -> 107


def test_realized_return_none_when_window_exceeds_series():
    s = _series(date(2024, 1, 1), [100, 101, 102])
    assert realized_return(s, date(2024, 1, 1), 10) is None


# ---- per-case scoring ----------------------------------------------------

def _pred(direction, pct, conf=0.6):
    return Prediction("NVDA", "2024-01-15", 21, direction, pct, conf)


def test_score_long_win():
    c = score_case(_pred("long", 8.0), 0.10)
    assert c.direction_correct is True
    assert c.pnl_pct == pytest.approx(10.0)
    assert c.magnitude_error == pytest.approx(2.0)


def test_score_long_loss():
    c = score_case(_pred("long", 8.0), -0.05)
    assert c.direction_correct is False
    assert c.pnl_pct == pytest.approx(-5.0)


def test_score_avoid_correct_when_drop():
    c = score_case(_pred("avoid", -3.0), -0.05)
    assert c.direction_correct is True   # right to skip
    assert c.pnl_pct == 0.0              # sat in cash


def test_score_short_win():
    c = score_case(_pred("short", -8.0), -0.08)
    assert c.direction_correct is True
    assert c.pnl_pct == pytest.approx(8.0)


# ---- aggregate scorecard -------------------------------------------------

def test_aggregate_hit_rate_and_edge():
    cases = [
        score_case(_pred("long", 5.0), 0.10),   # win, pnl +10
        score_case(_pred("long", 5.0), -0.04),  # loss, pnl -4
        score_case(_pred("long", 5.0), 0.06),   # win, pnl +6
    ]
    card = aggregate(cases)
    assert card.n == 3
    assert card.hit_rate == pytest.approx(2 / 3)
    # mean P&L = (10 - 4 + 6)/3 = 4.0 ; buy&hold = same here (all long) = 4.0
    assert card.mean_pnl_pct == pytest.approx(4.0)
    assert card.baseline_buyhold_pct == pytest.approx(4.0)


def test_aggregate_avoid_beats_buyhold_when_it_dodges_a_loss():
    # buy&hold eats the -10; the swarm avoided it -> positive edge.
    cases = [
        score_case(_pred("long", 5.0), 0.08),    # +8
        score_case(_pred("avoid", -2.0), -0.10),  # 0 (dodged the drop)
    ]
    card = aggregate(cases)
    assert card.baseline_buyhold_pct == pytest.approx((8.0 + -10.0) / 2)  # -1.0
    assert card.mean_pnl_pct == pytest.approx((8.0 + 0.0) / 2)            # 4.0
    assert card.edge_vs_buyhold_pct > 0


def test_aggregate_equity_multiple_compounds():
    cases = [score_case(_pred("long", 0.0), 0.10),
             score_case(_pred("long", 0.0), 0.10)]
    card = aggregate(cases)
    assert card.equity_multiple == pytest.approx(1.1 * 1.1)


def test_aggregate_empty_safe():
    card = aggregate([])
    assert card.n == 0
    assert card.hit_rate == 0.0
