"""Ablation sweep driver.

Runs `run_swarm.py` under each combination of the four levers and writes
one CSV row per run. The four levers and their values:

    --mode             ∈ {stigmergic, baseline}
    --corpus           ∈ {real, placeholder}
    --heterogeneous    ∈ {off, on}                  (CLI flag presence)
    --strategy-variant ∈ {diverse, single}

Full grid = 16 conditions. Quick grid = 4 (toggle each lever once
relative to a stigmergic+real+homogeneous+diverse baseline).

Usage:
    python tools/sweep.py debate "Climate action is necessary" --grid=full
    python tools/sweep.py debate "Climate action is necessary" --grid=quick

Output: `sweep_results.csv` in the cwd, appended (one row per condition).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Condition enumeration
# --------------------------------------------------------------------------

def _condition_id(mode: str, corpus: str, het: bool, strat: str) -> str:
    return f"{mode}-{corpus}-{'het' if het else 'hom'}-{strat}"


def full_grid() -> list[dict]:
    grid = []
    for mode in ("stigmergic", "baseline"):
        for corpus in ("real", "placeholder"):
            for het in (False, True):
                for strat in ("diverse", "single"):
                    grid.append({
                        "mode": mode, "corpus": corpus,
                        "heterogeneous": het, "strategy_variant": strat,
                    })
    assert len(grid) == 16
    return grid


def quick_grid() -> list[dict]:
    """Single-toggle ablation around a stigmergic + real + hom + diverse anchor."""
    base = {
        "mode": "stigmergic", "corpus": "real",
        "heterogeneous": False, "strategy_variant": "diverse",
    }
    out = [dict(base)]
    # Toggle one lever at a time
    out.append({**base, "mode": "baseline"})
    out.append({**base, "corpus": "placeholder"})
    out.append({**base, "heterogeneous": True})
    out.append({**base, "strategy_variant": "single"})
    # The 'anchor + 4 toggles' = 5 conditions. The spec says 4; drop the anchor
    # to match — caller compares each toggled run against the most recent
    # anchor run in a separate analysis pass.
    return out[1:]  # 4 conditions, all single-toggle ablations


# --------------------------------------------------------------------------
# Subprocess invocation
# --------------------------------------------------------------------------

def _build_argv(task_type: str, prompt: str, cond: dict) -> list[str]:
    argv = ["python", "run_swarm.py", task_type, prompt,
            f"--mode={cond['mode']}", f"--corpus={cond['corpus']}",
            f"--strategy-variant={cond['strategy_variant']}"]
    if cond["heterogeneous"]:
        argv.append("--heterogeneous")
    return argv


def _newest_run_dir(task_type: str, mode: str, since_ts: float) -> Path | None:
    suffix = "_baseline" if mode == "baseline" else ""
    pat = f"{task_type}_*{suffix}"
    # Check outputs/ first (real); fall back to outputs_mock/ (mock runs).
    for root in ("outputs", "outputs_mock"):
        candidates = list((REPO_ROOT / root).glob(pat))
        candidates = [c for c in candidates if c.stat().st_mtime >= since_ts]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def _parse_summary(run_dir: Path) -> dict:
    """Extract the row fields from summary.json + round_log.json."""
    out: dict = {}
    try:
        s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        s = {}
    out["output_diversity_overall"] = ""
    out["cross_model_delta"] = ""
    try:
        rl = json.loads((run_dir / "round_log.json").read_text(encoding="utf-8"))
        if rl:
            last = rl[-1]
            od = last.get("output_diversity") or {}
            out["output_diversity_overall"] = od.get("centroid_cosine_dist", "")
            div = last.get("diversity") or {}
            cmd = div.get("cross_model_delta") or {}
            out["cross_model_delta"] = (cmd.get("delta") if isinstance(cmd, dict) else "") or ""
    except Exception:
        pass
    out["audit_flag_count"] = s.get("audit_flags", "")
    out["n_surviving_clusters"] = (s.get("n_clusters", {}) or {}).get("surviving", "")
    out["total_elapsed_s"] = s.get("total_elapsed_s", "")
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Ablation sweep over the four levers.")
    p.add_argument("task_type")
    p.add_argument("prompt")
    p.add_argument("--grid", choices=("full", "quick"), default="quick")
    p.add_argument("--results", default=str(REPO_ROOT / "sweep_results.csv"),
                   help="CSV path (appended).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the argv each run would use and exit.")
    args = p.parse_args()

    grid = full_grid() if args.grid == "full" else quick_grid()
    print(f"[sweep] grid={args.grid}: {len(grid)} conditions")

    if args.dry_run:
        for cond in grid:
            print("  ", " ".join(_build_argv(args.task_type, args.prompt, cond)))
        return 0

    fieldnames = [
        "condition_id", "mode", "corpus", "heterogeneous", "strategy_variant",
        "output_diversity_overall", "cross_model_delta",
        "audit_flag_count", "n_surviving_clusters",
        "total_elapsed_s", "run_dir",
    ]
    results_path = Path(args.results)
    is_new = not results_path.exists()
    with results_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()

        for i, cond in enumerate(grid, 1):
            cid = _condition_id(cond["mode"], cond["corpus"],
                                cond["heterogeneous"], cond["strategy_variant"])
            print(f"\n[sweep] [{i}/{len(grid)}] condition: {cid}")
            start_ts = time.time()
            argv = _build_argv(args.task_type, args.prompt, cond)
            proc = subprocess.run(argv, cwd=str(REPO_ROOT))
            if proc.returncode != 0:
                print(f"[sweep] WARNING: {cid} exited with code {proc.returncode}; "
                      f"row will be partial.")

            run_dir = _newest_run_dir(args.task_type, cond["mode"], start_ts)
            run_dir_str = str(run_dir) if run_dir else ""
            row = {
                "condition_id": cid,
                "mode": cond["mode"],
                "corpus": cond["corpus"],
                "heterogeneous": "on" if cond["heterogeneous"] else "off",
                "strategy_variant": cond["strategy_variant"],
                "run_dir": run_dir_str,
            }
            if run_dir is not None:
                row.update(_parse_summary(run_dir))
            writer.writerow(row)
            f.flush()

    print(f"\n[sweep] done. results: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
