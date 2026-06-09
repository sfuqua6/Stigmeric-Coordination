"""Tests for core/stock_compare.py — relative verdict + comparison orchestration."""

import asyncio
from datetime import date

from core.stock_data import Snapshot
from core.stock_compare import build_comparison, run_comparison


# ---- pure verdict logic --------------------------------------------------

def _pred(ticker, direction, ret, conf=0.6):
    return {"schema": "stock_prediction_v1", "ticker": ticker,
            "direction": direction, "predicted_return_pct": ret, "confidence": conf}


def test_build_comparison_prefers_higher_return():
    results = [("WMT", _pred("WMT", "long", 8.0, 0.7)),
               ("DG", _pred("DG", "avoid", -3.0, 0.4))]
    comp = build_comparison(results, 21,
                            relationship={"type": "same_sector", "label": "grocery"})
    assert comp["prefer"] == "WMT"
    assert comp["avoid_relative"] == "DG"
    assert comp["ranking"] == ["WMT", "DG"]
    assert comp["spread_pct"] == 11.0
    assert "grocery" in comp["rationale"]


def test_build_comparison_three_way():
    results = [("A", _pred("A", "long", 3.0)),
               ("B", _pred("B", "long", 9.0)),
               ("C", _pred("C", "short", -5.0))]
    comp = build_comparison(results, 63)
    assert comp["prefer"] == "B"
    assert comp["ranking"][0] == "B"
    assert comp["avoid_relative"] == "C"


def test_build_comparison_confidence_breaks_near_ties():
    results = [("A", _pred("A", "long", 5.0, 0.9)),
               ("B", _pred("B", "long", 5.0, 0.2))]
    comp = build_comparison(results, 21)
    assert comp["prefer"] == "A"  # higher confidence on equal return


# ---- integration: two pipelines, opposite stances ------------------------

class _TickerLLM:
    """bull claims when the prompt is about AAA, bear claims when about BBB.
    All claims cite numbers that match the snapshots so verification fires."""
    name = "tickerllm"

    def __init__(self):
        self._n = {}
        self._bull = [
            "The trailing P/E is 28, undervalued versus peers with clear upside.",
            "Forward P/E of 25 looks cheap given the accelerating growth ahead.",
            "Revenue grew 22% year over year, a strong expanding growth signal.",
            "Gross margin of 45% is strong and expanding, supporting profitability.",
            "Net margin of 25% confirms strong operating leverage and upside.",
        ]
        self._bear = [
            "The trailing P/E is 28, overvalued and expensive versus history.",
            "Forward P/E of 25 looks rich given decelerating growth and downside.",
            "Revenue grew 22% but growth is decelerating, a weakening signal here.",
            "Gross margin of 45% faces compression risk under heavy competition.",
            "Net margin of 25% is under pressure, a clear downside risk ahead.",
        ]

    async def generate(self, prompt, role="agent", max_tokens=120, temperature=0.7, **_):
        bull = "AAA" in prompt
        if role == "hater":
            return "Valuation risk: a P/E of 28 could compress 15% if growth slows."
        if role in ("scout", "developer"):
            pool = self._bull if bull else self._bear
            key = role + ("A" if bull else "B")
            i = self._n.get(key, 0)
            self._n[key] = i + 1
            return pool[i % len(pool)]
        return ""


def _snap(ticker):
    return Snapshot(ticker=ticker, as_of=date(2024, 1, 15), price=150.0, pe=28.0,
                    fwd_pe=25.0, rev_growth_yoy=22.0, gross_margin=45.0,
                    net_margin=25.0, provenance={"pe": date(2024, 1, 10)})


def test_run_comparison_picks_the_bull(tmp_path):
    comp = asyncio.run(run_comparison(
        [_snap("AAA"), _snap("BBB")], lambda _r: _TickerLLM(),
        horizon_days=21, num_rounds=2, iterations_per_round=2,
        relationship={"type": "peer", "label": "test sector"},
        output_dir=str(tmp_path), verbose=False))
    assert comp["schema"] == "stock_comparison_v1"
    assert set(comp["tickers"]) == {"AAA", "BBB"}
    assert comp["prefer"] == "AAA"          # bull field beats bear field
    assert "AAA" in comp["per_ticker"] and "BBB" in comp["per_ticker"]
    assert (tmp_path / "comparison.json").exists()


def test_run_comparison_auto_relationship_from_graph():
    from core.relationships import RelationshipGraph
    g = RelationshipGraph()
    g.add("AAA", "BBB", "same_sector", label="test grocery",
          learned_on=date(2024, 1, 1))  # before the snapshots' as_of
    comp = asyncio.run(run_comparison(
        [_snap("AAA"), _snap("BBB")], lambda _r: _TickerLLM(),
        horizon_days=21, num_rounds=1, iterations_per_round=1, graph=g))
    # the graph auto-supplied the peer relationship (point-in-time visible)
    assert comp["relationship"] == {"type": "same_sector", "label": "test grocery"}
    assert "test grocery" in comp["rationale"]
