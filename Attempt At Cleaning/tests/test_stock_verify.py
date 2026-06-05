"""Tests for core/stock_verify.py — the numeric-claim verification core.

These guard the D4 fix: if extraction or closeness silently breaks,
verification_score becomes meaningless and the backtest is worthless.
"""

import math

import pytest

from core.stock_verify import (
    extract_numeric_claims, closeness, verify_claim, UNRESOLVED_STRENGTH,
)
from core.stock_data import Snapshot
from datetime import date


def _snap(**metrics):
    return Snapshot(ticker="TEST", as_of=date(2024, 1, 15), **metrics)


# ---- extraction ----------------------------------------------------------

def test_extract_forward_pe_multiple():
    claims = extract_numeric_claims("NVDA trades at 34x forward earnings")
    metrics = {c.metric: c.value for c in claims}
    assert metrics.get("fwd_pe") == 34.0


def test_extract_trailing_pe():
    claims = extract_numeric_claims("It has a P/E of 28.5 right now")
    assert claims[0].metric == "pe"
    assert claims[0].value == 28.5


def test_extract_percent_growth_stays_in_pp():
    claims = extract_numeric_claims("Revenue grew 22% YoY")
    m = {c.metric: c.value for c in claims}
    assert m.get("rev_growth_yoy") == 22.0  # percentage points, not 0.22


def test_extract_margin_percent():
    claims = extract_numeric_claims("gross margin of 45%")
    m = {c.metric: c.value for c in claims}
    assert m.get("gross_margin") == 45.0


def test_extract_money_suffix():
    claims = extract_numeric_claims("market cap of $1.2T")
    m = {c.metric: c.value for c in claims}
    assert m.get("market_cap") == pytest.approx(1.2e12)


def test_extract_debt_to_equity_percent_normalised():
    # D/E quoted as a percent must normalise to the ratio the snapshot stores.
    claims = extract_numeric_claims("debt-to-equity of 50%")
    m = {c.metric: c.value for c in claims}
    assert m.get("debt_to_equity") == pytest.approx(0.5)


def test_extract_none_when_no_number():
    assert extract_numeric_claims("the artifact presents a clear thesis") == []


# ---- closeness -----------------------------------------------------------

def test_closeness_exact_ratio():
    assert closeness(34.0, 34.0, "fwd_pe") == 1.0


def test_closeness_within_tolerance_ratio():
    # 34 vs 34.5 -> 1.47% rel err, within 3% full-credit band.
    assert closeness(34.0, 34.5, "fwd_pe") == 1.0


def test_closeness_partial_ratio():
    # 34 vs 40 -> 15% rel err; between 3% and 30% -> linear.
    s = closeness(34.0, 40.0, "fwd_pe")
    assert 0.0 < s < 1.0
    expected = 1.0 - (0.15 - 0.03) / (0.30 - 0.03)
    assert s == pytest.approx(round(expected, 4), abs=1e-3)


def test_closeness_zero_when_far():
    assert closeness(10.0, 40.0, "fwd_pe") == 0.0  # 75% rel err > 30%


def test_closeness_percent_absolute_band():
    # margins compared in percentage points: 45 vs 46 -> 1pp -> full credit.
    assert closeness(45.0, 46.0, "gross_margin") == 1.0
    # 45 vs 50 -> 5pp, between 1pp and 10pp -> partial.
    s = closeness(45.0, 50.0, "gross_margin")
    assert 0.0 < s < 1.0


def test_closeness_actual_zero_safe():
    # must not divide by zero
    s = closeness(5.0, 0.0, "fwd_pe")
    assert 0.0 <= s <= 1.0


# ---- verify_claim end to end --------------------------------------------

def test_verify_claim_match():
    snap = _snap(fwd_pe=34.0)
    r = verify_claim("Trading at 34x forward earnings — fair", snap)
    assert r.metric == "fwd_pe"
    assert r.strength == 1.0
    assert r.actual == 34.0


def test_verify_claim_mismatch_low_strength():
    snap = _snap(fwd_pe=20.0)
    r = verify_claim("Trading at 34x forward earnings", snap)
    assert r.metric == "fwd_pe"
    assert r.strength < 0.5


def test_verify_claim_unresolved_is_neutral():
    snap = _snap(pe=28.0)   # snapshot has no margin
    r = verify_claim("gross margin of 45%", snap)
    assert r.strength == UNRESOLVED_STRENGTH


def test_verify_claim_no_number_is_neutral():
    snap = _snap(pe=28.0)
    r = verify_claim("this stock looks like a strong long-term hold", snap)
    assert r.metric is None
    assert r.strength == UNRESOLVED_STRENGTH


def test_as_atom_shape():
    snap = _snap(pe=28.0)
    r = verify_claim("P/E of 28", snap)
    atom = r.as_atom()
    assert set(atom) == {"metric", "claimed", "actual", "strength", "text"}
