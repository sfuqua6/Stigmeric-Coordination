"""Tests for core/symbol_discovery.py — ticker extraction without false positives.

The correctness risk is reading CEO/USA/AI/ETF as tickers, or returning symbols
outside the tradable universe. These lock that down. All offline.
"""

from datetime import date

from core.symbol_discovery import discover_from_news, discover_symbols, Candidate


def _items(*titles, published=None):
    return [{"title": t, "published": published, "source": "test"} for t in titles]


def test_cashtag_extracted():
    cands = discover_from_news(_items("$NVDA jumps 5% on AI demand"))
    assert [c.ticker for c in cands] == ["NVDA"]


def test_exchange_tag_extracted():
    cands = discover_from_news(_items("Apple (NASDAQ: AAPL) climbs to record"))
    assert "AAPL" in [c.ticker for c in cands]


def test_company_name_resolution():
    cands = discover_from_news(
        _items("Walmart raises full-year guidance"),
        name_to_ticker={"walmart": "WMT"})
    assert [c.ticker for c in cands] == ["WMT"]


def test_multiword_company_name():
    cands = discover_from_news(
        _items("Dollar General slumps after weak outlook"),
        name_to_ticker={"dollar general": "DG"})
    assert [c.ticker for c in cands] == ["DG"]


def test_bare_token_requires_universe():
    # No universe -> a bare all-caps token is NOT accepted (too risky).
    assert discover_from_news(_items("NVDA stock keeps climbing")) == []
    # With a universe, the bare token is accepted.
    cands = discover_from_news(_items("NVDA stock keeps climbing"),
                               universe={"NVDA"})
    assert [c.ticker for c in cands] == ["NVDA"]


def test_acronyms_not_treated_as_tickers():
    # CEO / USA / AI / GDP must never be returned, even with a universe.
    cands = discover_from_news(
        _items("CEO says USA GDP and AI will boost the ETF sector"),
        universe={"CEO", "USA", "AI", "GDP", "ETF"})  # deliberately sloppy universe
    assert cands == []


def test_unknown_cashtag_filtered_by_universe():
    cands = discover_from_news(_items("$ZZZZ pumps on hype"), universe={"NVDA"})
    assert cands == []


def test_ranking_by_mention_count():
    items = _items(
        "$NVDA surges", "$NVDA upgraded by analysts", "$NVDA hits record",
        "$AAPL ticks up",
    )
    cands = discover_from_news(items)
    assert cands[0].ticker == "NVDA"
    assert cands[0].mentions == 3
    assert cands[1].ticker == "AAPL"


def test_recency_weight_breaks_ties():
    ref = date(2024, 6, 1)
    items = [
        {"title": "$AAA news", "published": date(2024, 1, 1)},   # old
        {"title": "$BBB news", "published": date(2024, 5, 30)},  # recent
    ]
    cands = discover_from_news(items, recency_ref=ref)
    assert cands[0].ticker == "BBB"  # more recent outranks on equal mentions


def test_discover_symbols_point_in_time_drops_future():
    as_of = date(2024, 6, 1)

    def search_fn(_query):
        return [
            {"title": "$PAST rallies", "published": date(2024, 5, 20)},
            {"title": "$FUTURE leaks", "published": date(2024, 7, 1)},  # after as_of
        ]

    syms = discover_symbols(as_of, search_fn, queries=["movers"])
    assert "PAST" in syms
    assert "FUTURE" not in syms  # never see the future


def test_discover_symbols_aggregates_and_ranks():
    as_of = date(2024, 6, 1)

    def search_fn(query):
        return _items("$NVDA " + query, "$NVDA again", "$AMD once")

    syms = discover_symbols(as_of, search_fn, queries=["a", "b"], top_k=2)
    assert syms[0] == "NVDA"
    assert set(syms) <= {"NVDA", "AMD"}
