"""End-to-end backtest grader test against a synthetic frozen DB.

Exercises the WHOLE Stage-5 loop in-process: frozen snapshot -> swarm pipeline
-> prediction.json -> realized return from frozen prices -> scored Scorecard.
No GPU/network/Groq (injected scripted LLM).
"""

import json
from datetime import date, timedelta

from core.stock_data import Snapshot, snapshot_to_dict
from eval.backtest import run_backtest


class _BullLLM:
    name = "bull"

    def __init__(self):
        self._n = {}
        self._pools = {
            "scout": [
                "The trailing P/E is 28, undervalued versus peers with clear upside.",
                "Revenue grew 22% year over year, a strong and expanding growth signal.",
                "Gross margin of 45% is strong and expanding, supporting profitability.",
            ],
            "developer": [
                "Net margin of 25% confirms strong operating leverage and pricing power.",
                "Free cash flow margin of 30% strengthens the balance sheet materially.",
            ],
            "hater": [
                "Valuation risk: a P/E of 28 could compress 15% if growth decelerates.",
            ],
        }

    async def generate(self, prompt, role="agent", max_tokens=120, temperature=0.7, **_):
        pool = self._pools.get(role)
        if not pool:
            return ""
        i = self._n.get(role, 0)
        self._n[role] = i + 1
        return pool[i % len(pool)]


def _bdays(start: date, n: int):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_backtest_end_to_end(tmp_path):
    as_of = date(2024, 1, 15)
    symbol = "NVDA"

    # frozen snapshot
    snap = Snapshot(
        ticker=symbol, as_of=as_of, price=150.0, market_cap=2.4e12, pe=28.0,
        fwd_pe=25.0, rev_growth_yoy=22.0, gross_margin=45.0, net_margin=25.0,
        fcf_margin=30.0, analyst_target=180.0, sma50=145.0, sma200=140.0,
        provenance={"pe": date(2024, 1, 10)},
    )
    snap_dir = tmp_path / "snapshots" / symbol
    snap_dir.mkdir(parents=True)
    (snap_dir / f"{as_of.isoformat()}.json").write_text(
        json.dumps(snapshot_to_dict(snap)), encoding="utf-8")

    # frozen price series: rising, so a 'long' call realizes a gain
    days = _bdays(as_of, 12)
    closes = [{"date": d.isoformat(), "close": 150.0 + i} for i, d in enumerate(days)]
    price_dir = tmp_path / "prices"
    price_dir.mkdir()
    (price_dir / f"{symbol}.json").write_text(
        json.dumps({"closes": closes}), encoding="utf-8")

    dataset = {
        "snapshot_root": str(tmp_path / "snapshots"),
        "price_root": str(price_dir),
        "cases": [{"symbol": symbol, "as_of": as_of.isoformat(), "horizon_days": 5}],
    }
    ds_path = tmp_path / "dataset.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")

    import os
    os.environ["STOCK_RESULTS_DIR"] = str(tmp_path / "results")
    card = run_backtest(str(ds_path), llm_for=lambda _r: _BullLLM())

    assert card.n == 1
    # realized return over 5 trading days on a rising series is positive
    assert card.baseline_buyhold_pct > 0
    # a results scorecard was persisted
    assert (tmp_path / "results").exists()
    assert list((tmp_path / "results").glob("*.json"))
