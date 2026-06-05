"""End-to-end Stock Swarm pipeline test (in-process, no GPU/network/Groq).

Runs run_stock_pipeline with a scripted LLM whose claims cite numbers that
match the snapshot, so the DataValidator's numeric verification actually fires
(the D4 fix). Proves the full spine: lens scouts -> INITIAL, DataValidator ->
VERIFICATION, developers -> SUPPORT, synthesizer -> prediction.json.
"""

import asyncio
from datetime import date

import pytest

from core.stock_data import Snapshot
from core.stock_pipeline import run_stock_pipeline
from core.signal_types import INITIAL, SUPPORT, VERIFICATION


class _FakeLLM:
    """Async LLM stub returning number-bearing, snapshot-matching claims.

    Per-role pools cycle so deposits are distinct (the store dedups). Numbers
    match the snapshot below so verification strength is high.
    """
    name = "fake"

    def __init__(self):
        self._n: dict[str, int] = {}
        self._pools = {
            "scout": [
                "The trailing P/E is 28, undervalued versus peers with clear upside potential.",
                "Forward P/E of 25 looks cheap given the accelerating growth tailwind ahead.",
                "Revenue grew 22% year over year, a strong and expanding growth signal.",
                "Gross margin of 45% is strong and expanding, supporting durable profitability.",
                "The analyst target of 180 implies meaningful upside from the current price.",
            ],
            "developer": [
                "Free cash flow margin of 30% strengthens the balance sheet and funds buybacks.",
                "Net margin of 25% confirms strong operating leverage and pricing power today.",
                "The 50-day average of 145 sits above trend, a constructive technical setup.",
            ],
            "hater": [
                "Valuation risk: at a P/E of 28 the multiple could compress 15% if growth slows.",
                "Margin compression risk: a 45% gross margin is hard to sustain under competition.",
                "Macro risk: rate sensitivity could pressure the 25 forward multiple near term.",
            ],
        }

    async def generate(self, prompt, role="agent", max_tokens=120, temperature=0.7, **_):
        pool = self._pools.get(role)
        if not pool:
            return ""
        i = self._n.get(role, 0)
        self._n[role] = i + 1
        return pool[i % len(pool)]


def _snapshot():
    return Snapshot(
        ticker="NVDA", as_of=date(2024, 1, 15),
        price=150.0, market_cap=2.4e12, pe=28.0, fwd_pe=25.0, ps=7.0,
        rev_growth_yoy=22.0, gross_margin=45.0, net_margin=25.0, fcf_margin=30.0,
        analyst_target=180.0, sma50=145.0, sma200=140.0,
        week52_high=200.0, week52_low=120.0, debt_to_equity=0.5, div_yield=0.6,
        provenance={"pe": date(2024, 1, 10)},
    )


def test_pipeline_end_to_end(tmp_path):
    snap = _snapshot()
    llm = _FakeLLM()
    result = asyncio.run(run_stock_pipeline(
        snap, lambda _role: llm,
        horizon_days=21, num_rounds=2, iterations_per_round=3,
        output_dir=tmp_path, verbose=False,
    ))
    store = result["store"]

    # scouts produced INITIAL claims
    assert len(store.by_type(INITIAL)) >= 1
    # developers produced SUPPORT
    assert len(store.by_type(SUPPORT)) >= 1
    # the D4 fix: numeric verification actually fired with high strength
    verifs = store.by_type(VERIFICATION)
    assert len(verifs) >= 1
    assert max(v.strength for v in verifs) > 0.5

    # gradable artifact
    pred = result["prediction"]
    assert pred is not None
    assert pred["schema"] == "stock_prediction_v1"
    assert pred["ticker"] == "NVDA"
    assert pred["horizon_days"] == 21
    assert pred["direction"] in ("long", "avoid", "short")
    assert isinstance(pred["predicted_return_pct"], (int, float))
    assert 0.0 <= pred["confidence"] <= 1.0

    # artifacts on disk
    assert (tmp_path / "prediction.json").exists()
    assert (tmp_path / "answer.txt").exists()
    report = (tmp_path / "answer.txt").read_text(encoding="utf-8")
    assert "VERDICT" in report
    # the verified evidence is surfaced (claim vs ground truth), not hidden
    assert "Verified facts" in report
    assert "verified" in report


def test_pipeline_empty_field_is_safe(tmp_path):
    """A silent LLM (no claims pass the number gate) must not crash; it yields an
    'avoid' prediction over an empty field (the MOCK-mode plumbing contract)."""
    class _Silent:
        name = "silent"
        async def generate(self, *a, **k):
            return "the company operates in several business segments worldwide"
    res = asyncio.run(run_stock_pipeline(
        _snapshot(), lambda _r: _Silent(),
        num_rounds=1, iterations_per_round=2, output_dir=tmp_path, verbose=False))
    assert res["prediction"]["direction"] == "avoid"
