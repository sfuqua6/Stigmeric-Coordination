"""Tests for YFinanceProvider._raw_to_snapshot unit conversions + news parsing.

These run OFFLINE (synthetic raw dicts) because Stage 0 showed Yahoo rate-limits
live requests. The unit conversions are the silent-error trap: yfinance gives
margins/growth as fractions, debtToEquity as a percent-like number, and a
dividendYield field whose unit has changed across versions. If any of these is
wrong, every snapshot is quietly corrupt and the backtest is meaningless.
"""

from datetime import date, datetime, timezone

import pytest

from core.stock_data import (
    YFinanceProvider, assert_no_lookahead, _normalise_news_item,
)

AS_OF = date(2024, 6, 1)


def _map(info=None, fast=None, news=None):
    prov = YFinanceProvider()
    raw = {"info": info or {}, "fast_info": fast or {}, "news": news or []}
    return prov._raw_to_snapshot("AAPL", AS_OF, raw)


# ---- unit conversions ----------------------------------------------------

def test_margins_and_growth_fraction_to_pp():
    s = _map(info={"grossMargins": 0.45, "operatingMargins": 0.30,
                   "profitMargins": 0.25, "revenueGrowth": 0.22,
                   "earningsGrowth": 0.18})
    assert s.gross_margin == pytest.approx(45.0)
    assert s.operating_margin == pytest.approx(30.0)
    assert s.net_margin == pytest.approx(25.0)
    assert s.rev_growth_yoy == pytest.approx(22.0)
    assert s.eps_growth_yoy == pytest.approx(18.0)


def test_debt_to_equity_percent_to_ratio():
    # yfinance reports 150.0 meaning 150% -> snapshot stores the 1.5 ratio
    s = _map(info={"debtToEquity": 150.0})
    assert s.debt_to_equity == pytest.approx(1.5)


def test_dividend_yield_from_rate_and_price_unambiguous():
    s = _map(info={"dividendRate": 2.0, "currentPrice": 100.0})
    assert s.div_yield == pytest.approx(2.0)   # 2/100 = 2%


def test_dividend_yield_fallback_fraction():
    s = _map(info={"dividendYield": 0.005})    # legacy fraction form
    assert s.div_yield == pytest.approx(0.5)


def test_dividend_yield_fallback_already_percent():
    s = _map(info={"dividendYield": 2.5})      # newer percent form
    assert s.div_yield == pytest.approx(2.5)


def test_ratios_and_prices_passthrough():
    s = _map(info={"trailingPE": 28.5, "forwardPE": 25.0, "trailingPegRatio": 1.8,
                   "priceToSalesTrailing12Months": 7.2, "enterpriseToEbitda": 20.0,
                   "trailingEps": 6.0, "targetMeanPrice": 180.0, "marketCap": 3e12,
                   "currentPrice": 150.0, "fiftyDayAverage": 170.0,
                   "twoHundredDayAverage": 160.0, "fiftyTwoWeekHigh": 200.0,
                   "fiftyTwoWeekLow": 120.0})
    assert s.pe == 28.5 and s.fwd_pe == 25.0 and s.peg == 1.8
    assert s.ps == 7.2 and s.ev_ebitda == 20.0 and s.eps == 6.0
    assert s.analyst_target == 180.0 and s.market_cap == 3e12 and s.price == 150.0
    assert s.sma50 == 170.0 and s.sma200 == 160.0
    assert s.week52_high == 200.0 and s.week52_low == 120.0


def test_fcf_margin_derived():
    s = _map(info={"freeCashflow": 28.0, "totalRevenue": 100.0})
    assert s.fcf_margin == pytest.approx(28.0)


def test_nan_values_dropped():
    s = _map(info={"grossMargins": float("nan"), "trailingPE": 28.0})
    assert s.gross_margin is None
    assert s.pe == 28.0


def test_fast_info_fallback_when_info_missing():
    s = _map(info={}, fast={"last_price": 150.0, "market_cap": 2.4e12,
                            "year_high": 199.0, "year_low": 121.0})
    assert s.price == 150.0 and s.market_cap == 2.4e12
    assert s.week52_high == 199.0 and s.week52_low == 121.0


# ---- point-in-time news --------------------------------------------------

def test_future_news_dropped_past_kept():
    s = _map(news=[
        {"title": "kept", "published": date(2024, 5, 30), "url": "", "source": ""},
        {"title": "future leak", "published": date(2024, 7, 1), "url": "", "source": ""},
        {"title": "undated", "published": None, "url": "", "source": ""},
    ])
    titles = [n.title for n in s.news]
    assert titles == ["kept"]


def test_mapped_snapshot_passes_lookahead_guard():
    s = _map(info={"trailingPE": 28.0},
             news=[{"title": "ok", "published": date(2024, 5, 1)}])
    assert_no_lookahead(s)   # provenance stamped == as_of, news <= as_of


# ---- news schema normalisation -------------------------------------------

def test_normalise_news_old_schema_epoch():
    epoch = int(datetime(2024, 5, 20, tzinfo=timezone.utc).timestamp())
    n = _normalise_news_item({"title": "old", "providerPublishTime": epoch,
                              "link": "http://x", "publisher": "Reuters"})
    assert n["title"] == "old"
    assert n["published"] == date(2024, 5, 20)
    assert n["source"] == "Reuters"


def test_normalise_news_new_schema_iso():
    n = _normalise_news_item({"content": {
        "title": "new", "pubDate": "2024-05-21T10:00:00Z",
        "canonicalUrl": {"url": "http://y"},
        "provider": {"displayName": "Bloomberg"}}})
    assert n["title"] == "new"
    assert n["published"] == date(2024, 5, 21)
    assert n["url"] == "http://y" and n["source"] == "Bloomberg"


def test_normalise_news_undated_returns_none():
    n = _normalise_news_item({"title": "no date"})
    assert n["published"] is None


# ---- live smoke (skips under rate limit / offline) -----------------------

def test_live_fetch_smoke():
    import os
    if os.environ.get("RUN_YF_LIVE", "").strip() in ("", "0", "false", "False"):
        pytest.skip("live yfinance test gated behind RUN_YF_LIVE=1 (Yahoo rate-limits)")
    yf = pytest.importorskip("yfinance")
    prov = YFinanceProvider()
    try:
        snap = prov.get_snapshot("AAPL", date.today())
    except Exception as exc:  # rate limited / network — Stage 0 documented this
        pytest.skip(f"yfinance live fetch unavailable: {type(exc).__name__}: {str(exc)[:80]}")
    assert snap.ticker == "AAPL"
    assert_no_lookahead(snap)
