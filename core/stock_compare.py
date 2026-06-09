"""Comparative stock analysis (e.g. WMT vs DG) — relative verdict over 2+ tickers.

Runs the single-ticker pipeline for each ticker (reusing the whole spine), then
ranks them into a relative verdict: which is the better buy over the horizon.
This is the front of the peer-reasoning capability the project is aiming at
("Walmart and Dollar General are both grocery stores"); the dated relationship
graph that makes the peer link explicit is Stage 7.

`build_comparison()` (the ranking/verdict logic) is pure and tested;
`run_comparison()` orchestrates the per-ticker pipelines.
"""

from __future__ import annotations

from typing import Callable, Optional

from core.stock_data import Snapshot
from core.stock_pipeline import run_stock_pipeline


def _buy_score(pred: Optional[dict]) -> float:
    """Relative attractiveness of a prediction. Higher = better buy.

    Predicted return is the spine; confidence scales it; direction breaks ties
    (long > avoid > short for the same magnitude). Deterministic.
    """
    if not pred:
        return -1e9
    ret = float(pred.get("predicted_return_pct", 0.0))
    conf = float(pred.get("confidence", 0.5))
    dir_bonus = {"long": 1.0, "avoid": 0.0, "short": -1.0}.get(
        pred.get("direction", "avoid"), 0.0)
    return ret * (0.5 + 0.5 * conf) + dir_bonus * 0.01


def build_comparison(results: list[tuple[str, dict]], horizon_days: int,
                     relationship: Optional[dict] = None) -> dict:
    """Rank (ticker, prediction) pairs into a relative verdict.

    `relationship` is optional context like {"type": "same_sector",
    "label": "grocery"} — rendered but not required.
    """
    ranked = sorted(results, key=lambda tp: _buy_score(tp[1]), reverse=True)
    best_t, best_p = ranked[0]
    worst_t, worst_p = ranked[-1]
    spread = (float((best_p or {}).get("predicted_return_pct", 0.0))
              - float((worst_p or {}).get("predicted_return_pct", 0.0)))
    rel_txt = ""
    if relationship and relationship.get("label"):
        rel_txt = f" Both are {relationship['label']} names."
    return {
        "schema": "stock_comparison_v1",
        "tickers": [t for t, _ in results],
        "horizon_days": horizon_days,
        "predictions": {t: p for t, p in results},
        "ranking": [t for t, _ in ranked],
        "prefer": best_t,
        "avoid_relative": worst_t,
        "spread_pct": round(spread, 2),
        "relationship": relationship,
        "rationale": (f"{best_t} ranks above {worst_t} by relative buy-score over "
                      f"{horizon_days} trading days.{rel_txt}"),
        "disclaimer": "Not financial advice; backtest artifact.",
    }


async def run_comparison(
    snapshots: list[Snapshot],
    llm_for: Callable[[str], object],
    *,
    horizon_days: int = 21,
    num_rounds: int = 3,
    iterations_per_round: int = 6,
    relationship: Optional[dict] = None,
    graph=None,
    output_dir: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """Run the single-ticker pipeline for each snapshot, then compare.

    Each ticker gets an independent SignalStore (no cross-contamination), so the
    comparison is over two clean analyses. Returns the comparison artifact and
    stashes the per-ticker results under ``["per_ticker"]``.

    If `relationship` is not given but a dated `graph` (core.relationships.
    RelationshipGraph) is, the relationship between the first two tickers visible
    at the snapshot's as_of is auto-supplied (point-in-time).
    """
    if relationship is None and graph is not None and len(snapshots) >= 2:
        relationship = graph.relationship_between(
            snapshots[0].ticker, snapshots[1].ticker, as_of=snapshots[0].as_of)

    results: list[tuple[str, dict]] = []
    per_ticker: dict[str, dict] = {}
    for snap in snapshots:
        sub_out = (f"{output_dir}/{snap.ticker}" if output_dir else None)
        r = await run_stock_pipeline(
            snap, llm_for, horizon_days=horizon_days, num_rounds=num_rounds,
            iterations_per_round=iterations_per_round, output_dir=sub_out,
            verbose=verbose)
        results.append((snap.ticker, r["prediction"]))
        per_ticker[snap.ticker] = {"answer": r["answer"], "prediction": r["prediction"]}

    comparison = build_comparison(results, horizon_days, relationship)
    comparison["per_ticker"] = per_ticker

    if output_dir is not None:
        import json
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "comparison.json").write_text(
            json.dumps({k: v for k, v in comparison.items() if k != "per_ticker"},
                       indent=2), encoding="utf-8")
    return comparison
