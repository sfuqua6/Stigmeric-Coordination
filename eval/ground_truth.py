"""Grader-only "reality": realized forward returns.

THIS MODULE IS THE FUTURE. It computes what actually happened after the as-of
date — the value the backtest grades predictions against. No agent, role,
scout, synthesizer, or anything under `agents/` or in the live swarm path may
import it. If they could, the swarm would be able to peek at the answer, which
is look-ahead bias of the worst kind.

The boundary is enforced two ways:
  1. tests/test_ground_truth_boundary.py greps the agents/ package and the
     swarm modules and fails if any of them import eval.ground_truth.
  2. _forbid_agent_import() below does a best-effort runtime check of the call
     stack at import time.

The price-series math (entry/exit selection over trading days, return) is REAL
and tested. The live price-history fetch (`load_price_series`) is a Stage-5
scaffold; the grader can also be fed a frozen price series from the historical
DB, which is the look-ahead-safe path.
"""

from __future__ import annotations

import inspect
from bisect import bisect_left
from datetime import date, datetime
from typing import Optional, Sequence


def _forbid_agent_import() -> None:
    """Best-effort: refuse to be imported from agent/swarm code."""
    for frame in inspect.stack():
        mod = frame.frame.f_globals.get("__name__", "")
        if mod.startswith("agents.") or mod in ("run_swarm", "agents"):
            raise ImportError(
                "eval.ground_truth must not be imported from the swarm path "
                f"(imported from {mod!r}). Realized returns are grader-only — "
                "exposing them to agents is look-ahead bias."
            )


_forbid_agent_import()


# ---------------------------------------------------------------------------
# Price series
# ---------------------------------------------------------------------------

class PriceSeries:
    """An ordered close-price series keyed by trading date.

    Built from (date, close) pairs. `entry/exit` selection uses the first
    trading day on or after the requested date, so weekends/holidays are
    handled without look-ahead (we never reach backward in time).
    """

    def __init__(self, points: Sequence[tuple[date, float]]):
        pts = sorted((_d(d), float(c)) for d, c in points)
        self._dates = [d for d, _ in pts]
        self._closes = [c for _, c in pts]

    def __len__(self) -> int:
        return len(self._dates)

    def close_on_or_after(self, d: date) -> Optional[tuple[date, float]]:
        d = _d(d)
        i = bisect_left(self._dates, d)
        if i >= len(self._dates):
            return None
        return self._dates[i], self._closes[i]

    def n_trading_days_after(self, d: date, n: int) -> Optional[tuple[date, float]]:
        """The close `n` trading days after the entry day (entry = on/after d)."""
        d = _d(d)
        i = bisect_left(self._dates, d)
        j = i + n
        if i >= len(self._dates) or j >= len(self._dates):
            return None
        return self._dates[j], self._closes[j]


def realized_return(series: PriceSeries, start: date, horizon_days: int) -> Optional[float]:
    """Realized fractional return from entry (on/after `start`) to `horizon_days`
    trading days later. Returns None if the series doesn't cover the window.

    This is the GAIN OR LOSS the prediction is graded against.
    """
    entry = series.close_on_or_after(start)
    if entry is None:
        return None
    exit_pt = series.n_trading_days_after(start, horizon_days)
    if exit_pt is None:
        return None
    _, entry_close = entry
    _, exit_close = exit_pt
    if entry_close <= 0:
        return None
    return exit_close / entry_close - 1.0


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_price_series_frozen(points: Sequence[tuple[str, float]]) -> PriceSeries:
    """Build a PriceSeries from frozen (iso_date, close) pairs — the
    look-ahead-safe path (historical DB supplies the full series; we only ever
    index forward from `start`)."""
    return PriceSeries([(_d(d), c) for d, c in points])


def load_price_series_yfinance(symbol: str, start: date, end: date) -> PriceSeries:
    """SCAFFOLD (Stage 5): fetch daily closes [start, end] via yfinance.

    yfinance *prices* are genuinely point-in-time (unlike its fundamentals), so
    this is acceptable for the grader. Implement with:
        yf.Ticker(symbol).history(start=start, end=end)['Close']
    and pass the result through load_price_series_frozen-style construction.
    """
    raise NotImplementedError(
        "load_price_series_yfinance is a Stage-5 scaffold; wire yfinance price "
        "history here. Until then feed the grader frozen series from the DB."
    )


def _d(v) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return datetime.strptime(str(v)[:10], "%Y-%m-%d").date()
