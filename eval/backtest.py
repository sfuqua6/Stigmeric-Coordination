"""Backtest grader — the "make money" scorecard.

Turns a set of predictions (each `prediction.json` the EquityBriefSynthesizer
emits) into a P&L scorecard graded against realized returns, compared to naive
baselines. This is where "does the swarm make money?" becomes answerable.

What's REAL + tested here (tests/test_backtest_scoring.py):
  * score_case()      — directional hit, magnitude error, P&L of acting
  * aggregate()       — hit-rate, mean P&L, equity curve, calibration, baselines

What's SCAFFOLDED (Stage 5):
  * run_backtest()    — loops cases, invokes the swarm per case, reads
                        prediction.json, computes realized return via
                        eval.ground_truth, then calls aggregate(). The swarm
                        invocation (`_run_engine`) is a TODO shell-out.

CLI:  python eval/backtest.py eval/datasets/<set>.json
"""

from __future__ import annotations

import os as _os
import sys as _sys
# Allow `python eval/backtest.py` (script dir would otherwise shadow the repo root).
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Per-case scoring (REAL)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Prediction:
    ticker: str
    as_of_date: str
    horizon_days: int
    direction: str               # "long" | "avoid" | "short"
    predicted_return_pct: float  # percent, e.g. 8.5 means +8.5%
    confidence: float = 0.5

    @staticmethod
    def from_dict(d: dict) -> "Prediction":
        return Prediction(
            ticker=d["ticker"],
            as_of_date=d["as_of_date"],
            horizon_days=int(d["horizon_days"]),
            direction=str(d.get("direction", "long")).lower(),
            predicted_return_pct=float(d["predicted_return_pct"]),
            confidence=float(d.get("confidence", 0.5)),
        )


@dataclass(frozen=True)
class CaseScore:
    ticker: str
    as_of_date: str
    direction: str
    predicted_return_pct: float
    realized_return_pct: float
    confidence: float
    direction_correct: bool
    magnitude_error: float       # |predicted - realized| in pp
    pnl_pct: float               # P&L of acting on the call, in percent


def score_case(pred: Prediction, realized_return: float) -> CaseScore:
    """Grade one prediction against its realized fractional return.

    Acting rule (long-only POC + optional short):
      * long  -> capture realized return
      * avoid -> 0 (sat in cash)
      * short -> capture negated realized return
    Direction is "correct" when the sign of the *call* matches reality:
      * long  correct iff realized > 0
      * avoid correct iff realized <= 0   (you were right to skip)
      * short correct iff realized < 0
    """
    realized_pct = realized_return * 100.0
    d = pred.direction
    if d == "long":
        direction_correct = realized_return > 0
        pnl = realized_pct
    elif d == "short":
        direction_correct = realized_return < 0
        pnl = -realized_pct
    else:  # avoid / neutral
        direction_correct = realized_return <= 0
        pnl = 0.0
    return CaseScore(
        ticker=pred.ticker,
        as_of_date=pred.as_of_date,
        direction=d,
        predicted_return_pct=pred.predicted_return_pct,
        realized_return_pct=realized_pct,
        confidence=pred.confidence,
        direction_correct=direction_correct,
        magnitude_error=abs(pred.predicted_return_pct - realized_pct),
        pnl_pct=pnl,
    )


# ---------------------------------------------------------------------------
# Aggregate scorecard (REAL)
# ---------------------------------------------------------------------------

@dataclass
class Scorecard:
    n: int
    hit_rate: float                  # over actionable (long/short) calls
    mean_pnl_pct: float              # mean P&L per case (acting rule)
    mean_long_return_pct: float      # mean realized on long calls
    equity_multiple: float           # compounded P&L across cases
    sharpe_like: float               # mean/stdev of per-case P&L (unannualised)
    # baselines (the swarm must beat these)
    baseline_buyhold_pct: float      # mean realized across ALL cases (ignore call)
    baseline_spy_pct: Optional[float]
    edge_vs_buyhold_pct: float       # mean_pnl - buyhold
    calibration: list[dict] = field(default_factory=list)
    verified_claim_ratio: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def render(self) -> str:
        lines = [
            "=" * 60,
            "STOCK SWARM BACKTEST SCORECARD",
            "=" * 60,
            f"cases:                 {self.n}",
            f"hit-rate (actionable): {self.hit_rate:.1%}",
            f"mean P&L / case:       {self.mean_pnl_pct:+.2f}%",
            f"mean long return:      {self.mean_long_return_pct:+.2f}%",
            f"equity multiple:       {self.equity_multiple:.3f}x",
            f"sharpe-like:           {self.sharpe_like:.2f}",
            "-" * 60,
            f"baseline buy&hold:     {self.baseline_buyhold_pct:+.2f}%",
            f"baseline SPY:          "
            + (f"{self.baseline_spy_pct:+.2f}%" if self.baseline_spy_pct is not None else "n/a"),
            f"EDGE vs buy&hold:      {self.edge_vs_buyhold_pct:+.2f}%  "
            + ("[BEATS baseline]" if self.edge_vs_buyhold_pct > 0 else "[does NOT beat baseline]"),
        ]
        if self.verified_claim_ratio is not None:
            lines.append(f"verified-claim ratio:  {self.verified_claim_ratio:.1%}")
        if self.calibration:
            lines.append("-" * 60)
            lines.append("calibration (confidence bucket -> hit-rate):")
            for b in self.calibration:
                lines.append(
                    f"  {b['lo']:.1f}-{b['hi']:.1f}: {b['hit_rate']:.0%} "
                    f"(n={b['n']})"
                )
        lines.append("=" * 60)
        return "\n".join(lines)


def aggregate(
    cases: list[CaseScore],
    spy_returns_pct: Optional[list[float]] = None,
    verified_claim_ratio: Optional[float] = None,
) -> Scorecard:
    """Aggregate per-case scores into the scorecard, vs baselines.

    `spy_returns_pct` (optional) is the realized SPY return per case over the
    same windows — the market baseline. Buy&hold baseline = mean realized
    across all cases regardless of the swarm's call.
    """
    n = len(cases)
    if n == 0:
        return Scorecard(0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, None, 0.0)

    actionable = [c for c in cases if c.direction in ("long", "short")]
    hit_rate = (sum(1 for c in actionable if c.direction_correct) / len(actionable)
                if actionable else 0.0)

    pnls = [c.pnl_pct for c in cases]
    mean_pnl = sum(pnls) / n

    longs = [c.realized_return_pct for c in cases if c.direction == "long"]
    mean_long = sum(longs) / len(longs) if longs else 0.0

    # Compounded equity curve from per-case P&L (equal stake per case).
    equity = 1.0
    for p in pnls:
        equity *= (1.0 + p / 100.0)

    # sharpe-like: mean/stdev of per-case P&L (no annualisation — descriptive).
    if n > 1:
        mean = mean_pnl
        var = sum((p - mean) ** 2 for p in pnls) / (n - 1)
        sd = math.sqrt(var)
        sharpe = mean / sd if sd > 1e-9 else 0.0
    else:
        sharpe = 0.0

    buyhold = sum(c.realized_return_pct for c in cases) / n
    spy = (sum(spy_returns_pct) / len(spy_returns_pct)) if spy_returns_pct else None

    return Scorecard(
        n=n,
        hit_rate=hit_rate,
        mean_pnl_pct=mean_pnl,
        mean_long_return_pct=mean_long,
        equity_multiple=equity,
        sharpe_like=sharpe,
        baseline_buyhold_pct=buyhold,
        baseline_spy_pct=spy,
        edge_vs_buyhold_pct=mean_pnl - buyhold,
        calibration=_calibration(cases),
        verified_claim_ratio=verified_claim_ratio,
    )


def _calibration(cases: list[CaseScore], n_buckets: int = 4) -> list[dict]:
    """Bucket actionable calls by confidence and report hit-rate per bucket."""
    actionable = [c for c in cases if c.direction in ("long", "short")]
    out = []
    for b in range(n_buckets):
        lo, hi = b / n_buckets, (b + 1) / n_buckets
        members = [c for c in actionable
                   if (lo <= c.confidence < hi) or (hi == 1.0 and c.confidence == 1.0)]
        if not members:
            continue
        hr = sum(1 for c in members if c.direction_correct) / len(members)
        out.append({"lo": lo, "hi": hi, "n": len(members), "hit_rate": hr})
    return out


# ---------------------------------------------------------------------------
# Run loop (SCAFFOLD — Stage 5)
# ---------------------------------------------------------------------------

def run_backtest(dataset_path: str, llm_for=None) -> Scorecard:
    """Load a dataset of cases, run the swarm per case (in-process), grade, aggregate.

    Dataset JSON shape:
        {"snapshot_root": "eval/datasets/snapshots",   # FrozenSnapshotProvider root
         "price_root":    "eval/datasets/prices",      # frozen close series per symbol
         "cases": [{"symbol": "NVDA", "as_of": "2024-01-15", "horizon_days": 21}, ...]}

    The swarm runs against the FROZEN snapshot (point-in-time safe); realized
    return comes from eval.ground_truth over the frozen price series. Set
    GROQ_API_KEY for a real run (MOCK_LLM=1 only proves plumbing — predictions
    will be 'avoid' because mock claims carry no numbers).
    """
    import os
    import asyncio
    from datetime import datetime
    from core.stock_data import FrozenSnapshotProvider

    data = json.loads(open(dataset_path, encoding="utf-8").read())
    provider = FrozenSnapshotProvider(data.get("snapshot_root", "eval/datasets/snapshots"))

    async def _grade_all(injected_llm_for):
        teardown = None
        lf = injected_llm_for
        if lf is None:
            from core.stock_pipeline import make_llm_for
            lf, teardown = make_llm_for()        # constructed INSIDE this loop
        out: list[CaseScore] = []
        for case in data.get("cases", []):
            try:
                pred = await _run_engine_async(case, provider, lf)
            except FileNotFoundError as exc:
                print(f"[backtest] skip {case.get('symbol')} @ {case.get('as_of')}: {exc}")
                continue
            realized = _realized_for_case(case, data)
            if pred is None or realized is None:
                print(f"[backtest] skip {case.get('symbol')} @ {case.get('as_of')}: "
                      f"pred={pred is not None} realized={realized is not None}")
                continue
            out.append(score_case(pred, realized))
        if callable(teardown):
            try:
                await teardown()
            except Exception:
                pass
        return out

    # ONE event loop for the whole backtest: a shared llm (MockLLM / Groq router)
    # holds loop-bound primitives (semaphores), so a per-case asyncio.run() would
    # bind them to a dead loop on the 2nd case. Run every case on this loop.
    scored = asyncio.run(_grade_all(llm_for))

    card = aggregate(scored)
    print(card.render())

    # Persist the scorecard (plan Stage 5).
    try:
        out_dir = os.environ.get("STOCK_RESULTS_DIR", "eval/results")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(out_dir, f"{ts}.json"), "w", encoding="utf-8") as f:
            json.dump(card.to_dict(), f, indent=2)
    except Exception as exc:
        print(f"[backtest] could not write results: {exc}")
    return card


async def _run_engine_async(case: dict, provider, llm_for) -> Optional[Prediction]:
    """Run the stock pipeline for one case on the CURRENT loop, return its Prediction.

    Point-in-time: the snapshot comes from the FROZEN provider (assert_no_lookahead
    runs on load); the swarm never touches eval.ground_truth (the realized return).
    """
    from datetime import datetime
    from core.stock_pipeline import run_stock_pipeline

    symbol = case["symbol"].upper()
    as_of = datetime.strptime(case["as_of"][:10], "%Y-%m-%d").date()
    horizon = int(case["horizon_days"])

    snap = provider.get_snapshot(symbol, as_of)   # FileNotFoundError if absent
    result = await run_stock_pipeline(
        snap, llm_for, horizon_days=horizon, output_dir=None, verbose=False)
    pred = result.get("prediction")
    if not pred:
        return None
    pred["horizon_days"] = horizon
    return Prediction.from_dict(pred)


def _realized_for_case(case: dict, dataset: dict) -> Optional[float]:
    """Realized return via the grader-only module + frozen price series."""
    from eval.ground_truth import load_price_series_frozen, realized_return
    from datetime import datetime
    price_root = dataset.get("price_root")
    if not price_root:
        return None
    symbol = case["symbol"].upper()
    series_path = f"{price_root}/{symbol}.json"
    try:
        pts = json.loads(open(series_path, encoding="utf-8").read())["closes"]
    except Exception:
        return None
    series = load_price_series_frozen([(p["date"], p["close"]) for p in pts])
    start = datetime.strptime(case["as_of"][:10], "%Y-%m-%d").date()
    return realized_return(series, start, int(case["horizon_days"]))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python eval/backtest.py <dataset.json>")
        raise SystemExit(2)
    run_backtest(sys.argv[1])
