"""Manual entry point for the Stock Swarm (single-ticker prediction).

Examples
--------
    # historical (point-in-time safe) — requires a frozen snapshot DB:
    MOCK_LLM=1 python run_stock.py --symbol NVDA --as-of 2024-01-15 --db eval/datasets/snapshots
    GROQ_API_KEY=... python run_stock.py --symbol NVDA --as-of 2024-01-15 --db eval/datasets/snapshots

    # live (as-of today) — uses yfinance directly (rate-limited; see POC plan Stage 0):
    GROQ_API_KEY=... python run_stock.py --symbol NVDA

This is the round-based pipeline in core/stock_pipeline.py — it does NOT go
through run_swarm.py. Output (answer.txt + prediction.json + signals.json) lands
in outputs/stock_<SYMBOL>_<as_of>/ unless --out is given.

Look-ahead rule: a historical --as-of REQUIRES --db (a frozen snapshot), because
live yfinance fundamentals/news are 'as of now' and would leak the future.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import date


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Stock Swarm on one ticker.")
    ap.add_argument("--symbol", default=None,
                    help="ticker, e.g. NVDA; if omitted, discover from recent news")
    ap.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--horizon", type=int, default=21,
                    help="prediction horizon in trading days (default 21 ~ 1 month)")
    ap.add_argument("--db", default=os.environ.get("STOCK_DB"),
                    help="frozen snapshot root (FrozenSnapshotProvider layout)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=6, help="iterations per round")
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args()

    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    # Resolve the data provider with the look-ahead rule enforced.
    from core.stock_data import FrozenSnapshotProvider, YFinanceProvider
    if as_of != date.today() and not args.db:
        sys.exit("[run_stock] historical --as-of requires --db (frozen snapshot "
                 "root); live yfinance is look-ahead-unsafe for the past.")
    provider = FrozenSnapshotProvider(args.db) if args.db else YFinanceProvider()

    # Symbol: explicit, or discovered from news <= as_of (the Stage-6 front end).
    if args.symbol:
        symbol = args.symbol.upper()
    else:
        print(f"[run_stock] no --symbol given; discovering candidates from news "
              f"<= {as_of.isoformat()} ...")
        candidates = provider.discover_symbols(as_of)
        if not candidates:
            sys.exit("[run_stock] discovery found no candidate tickers — pass "
                     "--symbol explicitly, or populate the _discovery file.")
        print(f"[run_stock] candidates: {candidates}")
        symbol = candidates[0]
        print(f"[run_stock] selected most-mentioned: {symbol}")

    snapshot = provider.get_snapshot(symbol, as_of)   # runs assert_no_lookahead

    from core.stock_pipeline import run_stock_pipeline, make_llm_for
    llm_for, teardown = make_llm_for()

    out_dir = args.out or os.path.join("outputs", f"stock_{symbol}_{as_of.isoformat()}")
    result = asyncio.run(run_stock_pipeline(
        snapshot, llm_for, horizon_days=args.horizon,
        num_rounds=args.rounds, iterations_per_round=args.iters,
        output_dir=out_dir,
    ))
    if callable(teardown):
        try:
            asyncio.run(teardown())
        except Exception:
            pass

    print("\n" + "=" * 64)
    print(result["answer"])
    print("=" * 64)
    print("PREDICTION:", result["prediction"])
    print(f"\n[run_stock] wrote {out_dir}/ (answer.txt, prediction.json, signals.json)")


if __name__ == "__main__":
    main()
