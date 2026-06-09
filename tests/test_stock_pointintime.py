"""Tests for core/stock_data.py — point-in-time discipline and lens partitioning.

The look-ahead guard is the existential-risk assertion for the whole backtest:
if it lets future-dated data through, every "make money" number is fake.
"""

import json
from datetime import date

import pytest

from core.stock_data import (
    Snapshot, NewsItem, LENS_FIELDS, assert_no_lookahead, LookaheadError,
    FrozenSnapshotProvider, snapshot_to_dict, snapshot_from_dict,
)


def _snap(**kw):
    base = dict(ticker="NVDA", as_of=date(2024, 1, 15))
    base.update(kw)
    return Snapshot(**base)


# ---- look-ahead guard ----------------------------------------------------

def test_no_lookahead_passes_for_past_dates():
    s = _snap(pe=28.0,
              provenance={"pe": date(2024, 1, 10)},
              news=[NewsItem("ok", date(2024, 1, 12))])
    assert_no_lookahead(s)  # should not raise


def test_lookahead_news_raises():
    s = _snap(news=[NewsItem("future leak", date(2024, 2, 1))])
    with pytest.raises(LookaheadError):
        assert_no_lookahead(s)


def test_lookahead_provenance_raises():
    s = _snap(pe=28.0, provenance={"pe": date(2024, 3, 1)})
    with pytest.raises(LookaheadError):
        assert_no_lookahead(s)


def test_provenance_exactly_as_of_is_allowed():
    s = _snap(pe=28.0, provenance={"pe": date(2024, 1, 15)})
    assert_no_lookahead(s)


# ---- lens partitioning ---------------------------------------------------

def test_lens_facts_only_shows_lens_fields():
    s = _snap(pe=28.0, fwd_pe=25.0, rev_growth_yoy=22.0, gross_margin=45.0)
    val = s.lens_facts("valuation")
    assert "pe" in val and "fwd_pe" in val
    # growth-only fields must NOT leak into the valuation lens
    assert "rev_growth_yoy" not in val
    assert "gross_margin" not in val


def test_lens_facts_growth_shows_growth():
    s = _snap(rev_growth_yoy=22.0, pe=28.0)
    g = s.lens_facts("growth")
    assert "rev_growth_yoy" in g
    assert "pe " not in g  # valuation field absent


def test_every_lens_renders_without_crash():
    s = _snap(pe=28.0, rev_growth_yoy=10.0, sma50=100.0, debt_to_equity=0.5,
              news=[NewsItem("headline", date(2024, 1, 1))])
    for lens in LENS_FIELDS:
        assert isinstance(s.lens_facts(lens), str)


# ---- serialization round-trip --------------------------------------------

def test_snapshot_dict_roundtrip():
    s = _snap(pe=28.0, fwd_pe=25.0, market_cap=1.2e12,
              provenance={"pe": date(2024, 1, 10)},
              news=[NewsItem("h", date(2024, 1, 11), url="http://x", source="src")])
    d = snapshot_to_dict(s)
    # must be JSON-serialisable
    d2 = json.loads(json.dumps(d))
    s2 = snapshot_from_dict(d2)
    assert s2.ticker == "NVDA"
    assert s2.pe == 28.0
    assert s2.as_of == date(2024, 1, 15)
    assert s2.news[0].published == date(2024, 1, 11)
    assert s2.provenance["pe"] == date(2024, 1, 10)


# ---- frozen provider (the historical-DB reader) --------------------------

def test_frozen_provider_reads_and_validates(tmp_path):
    root = tmp_path / "db"
    (root / "NVDA").mkdir(parents=True)
    s = _snap(pe=28.0, provenance={"pe": date(2024, 1, 10)})
    (root / "NVDA" / "2024-01-15.json").write_text(
        json.dumps(snapshot_to_dict(s)), encoding="utf-8")
    prov = FrozenSnapshotProvider(root)
    got = prov.get_snapshot("NVDA", date(2024, 1, 15))
    assert got.pe == 28.0


def test_frozen_provider_rejects_lookahead(tmp_path):
    root = tmp_path / "db"
    (root / "NVDA").mkdir(parents=True)
    bad = _snap(pe=28.0, provenance={"pe": date(2024, 9, 1)})  # future
    (root / "NVDA" / "2024-01-15.json").write_text(
        json.dumps(snapshot_to_dict(bad)), encoding="utf-8")
    prov = FrozenSnapshotProvider(root)
    with pytest.raises(LookaheadError):
        prov.get_snapshot("NVDA", date(2024, 1, 15))


def test_frozen_provider_missing_raises(tmp_path):
    prov = FrozenSnapshotProvider(tmp_path)
    with pytest.raises(FileNotFoundError):
        prov.get_snapshot("AAPL", date(2024, 1, 15))
