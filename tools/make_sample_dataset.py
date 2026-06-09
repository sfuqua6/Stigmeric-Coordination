"""Generate a tiny SYNTHETIC frozen dataset for the Stock Swarm backtest.

Lets you run the whole grader loop end-to-end with NO network and NO real data,
to see the scorecard format and confirm plumbing. It is also a TEMPLATE for the
real historical DB you build (same FrozenSnapshotProvider layout):

    <out>/snapshots/<TICKER>/<as_of>.json     # one point-in-time snapshot per case
    <out>/snapshots/_discovery/<as_of>.json   # {"symbols": [...]} for --symbol-less runs
    <out>/prices/<TICKER>.json                # {"closes": [{"date","close"}, ...]}  (the grader's reality)
    <out>/dataset.json                        # {snapshot_root, price_root, cases:[...]}

NOT real data — synthetic random walk. Numbers here prove the pipeline, not skill.

Usage:
    python tools/make_sample_dataset.py [out_dir]
    python eval/backtest.py <out_dir>/dataset.json     # then grade it
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import date, timedelta

# Allow `python tools/make_sample_dataset.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.stock_data import Snapshot, snapshot_to_dict

SYMBOLS = ["NVDA", "AMD"]
AS_OFS = [date(2024, 1, 15), date(2024, 2, 15), date(2024, 3, 15)]
HORIZON = 5
PRICE_START = date(2024, 1, 2)
PRICE_END = date(2024, 4, 30)


def _bday_series(seed: int) -> list[dict]:
    rng = random.Random(seed)
    series, d, price = [], PRICE_START, 100.0
    while d <= PRICE_END:
        if d.weekday() < 5:
            price *= (1.0 + rng.uniform(-0.010, 0.013))   # slight upward drift
            series.append({"date": d.isoformat(), "close": round(price, 2)})
        d += timedelta(days=1)
    return series


def _close_on(series: list[dict], as_of: date) -> float:
    iso = as_of.isoformat()
    for p in series:
        if p["date"] >= iso:
            return p["close"]
    return series[-1]["close"]


def make(out: str = "eval/datasets/sample") -> str:
    snaps_root = os.path.join(out, "snapshots")
    prices_root = os.path.join(out, "prices")
    os.makedirs(os.path.join(snaps_root, "_discovery"), exist_ok=True)
    os.makedirs(prices_root, exist_ok=True)

    cases = []
    for sym in SYMBOLS:
        series = _bday_series(seed=hash(sym) & 0xFFFF)
        with open(os.path.join(prices_root, f"{sym}.json"), "w", encoding="utf-8") as f:
            json.dump({"closes": series}, f)

        sym_dir = os.path.join(snaps_root, sym)
        os.makedirs(sym_dir, exist_ok=True)
        for asof in AS_OFS:
            px = _close_on(series, asof)
            snap = Snapshot(
                ticker=sym, as_of=asof, price=px, market_cap=px * 1e10,
                pe=28.0, fwd_pe=25.0, ps=12.0, rev_growth_yoy=22.0,
                gross_margin=60.0, net_margin=30.0, fcf_margin=28.0,
                analyst_target=round(px * 1.1, 2), sma50=round(px * 0.98, 2),
                sma200=round(px * 0.92, 2), debt_to_equity=0.4, div_yield=0.1,
                provenance={"pe": asof},   # <= as_of, passes the look-ahead guard
            )
            with open(os.path.join(sym_dir, f"{asof.isoformat()}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(snapshot_to_dict(snap), f)
            cases.append({"symbol": sym, "as_of": asof.isoformat(),
                          "horizon_days": HORIZON})

    for asof in AS_OFS:
        with open(os.path.join(snaps_root, "_discovery", f"{asof.isoformat()}.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"symbols": SYMBOLS}, f)

    dataset = {"snapshot_root": snaps_root, "price_root": prices_root, "cases": cases}
    ds_path = os.path.join(out, "dataset.json")
    with open(ds_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"[sample] wrote {len(cases)} cases to {out}/")
    print(f"[sample] grade it:  python eval/backtest.py {ds_path}")
    print("[sample] NOTE: synthetic data; under MOCK_LLM predictions are 'avoid' "
          "(0 P&L). Set GROQ_API_KEY for real swarm predictions.")
    return ds_path


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "eval/datasets/sample")
