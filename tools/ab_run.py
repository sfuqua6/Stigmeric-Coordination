"""Run an A/B pair (stigmergic vs baseline) on one prompt, then compare.

This is the measurement harness for the stigmergic hypothesis (DEFERRED P4.1)
and for latency/quality regression checks: it runs the SAME prompt twice —
once stigmergic (A), once `--mode=baseline` (B) — with matching run-ids, then
prints the grouped QUALITY / LATENCY / PROCESS comparison + verdict from
tools/compare_runs.py.

Usage:
    python tools/ab_run.py debate "Cities should ban private cars"
    MOCK_LLM=1 python tools/ab_run.py analysis "What drives innovation?"

Any extra flags after the prompt are forwarded to BOTH runs (so the only
intended difference is --mode), e.g.:
    python tools/ab_run.py debate "..." --corpus=placeholder --ignore-kb

Pass --judge to additionally run the pairwise LLM answer judge
(tools/judge_answers.py) on the two answer.txt files — the quality comparison
that summary metrics can't give for stigmergic-vs-baseline. Needs a real LLM
(skipped/uninformative under MOCK). --judge is NOT forwarded to the runs.

MOCK runs land in outputs_mock/ and prove only plumbing — never quote MOCK
numbers as evidence (P0.1). For real comparison, run without MOCK_LLM.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _is_mock() -> bool:
    return os.environ.get("MOCK_LLM", "").strip() not in ("", "0", "false", "False")


def _run(task_type: str, prompt: str, run_id: str, extra: list[str],
         baseline: bool) -> Path:
    """Invoke run_swarm.py for one arm; return its output dir."""
    cmd = [sys.executable, str(_REPO / "run_swarm.py"), task_type, prompt,
           f"--run-id={run_id}", *extra]
    if baseline:
        cmd.append("--mode=baseline")
    arm = "baseline (B)" if baseline else "stigmergic (A)"
    print(f"\n[ab] === running {arm}: run-id={run_id} ===", flush=True)
    # Inherit env (MOCK_LLM, GROQ_API_KEY, convergence overrides, ...).
    result = subprocess.run(cmd, cwd=str(_REPO), env=os.environ.copy())
    outputs_root = _REPO / ("outputs_mock" if _is_mock() else "outputs")
    out_dir = outputs_root / run_id
    if result.returncode != 0:
        print(f"[ab] WARNING: {arm} exited {result.returncode}", file=sys.stderr)
    return out_dir


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    task_type, prompt = argv[0], argv[1]
    rest = argv[2:]
    do_judge = "--judge" in rest
    extra = [a for a in rest if a != "--judge"]  # --judge is ours, not run_swarm's
    stamp = time.strftime("%Y%m%d_%H%M%S")
    id_a = f"ab_{stamp}_stig"
    id_b = f"ab_{stamp}_base"

    dir_a = _run(task_type, prompt, id_a, extra, baseline=False)
    dir_b = _run(task_type, prompt, id_b, extra, baseline=True)

    sum_a, sum_b = dir_a / "summary.json", dir_b / "summary.json"
    missing = [str(p) for p in (sum_a, sum_b) if not p.exists()]
    if missing:
        print(f"\n[ab] ERROR: summary.json missing for: {missing}\n"
              f"[ab] (a run likely failed before writing its scorecard)",
              file=sys.stderr)
        return 2

    # Reuse the grouped comparator — A = stigmergic, B = baseline.
    sys.path.insert(0, str(_REPO))
    from tools.compare_runs import compare
    print("\n[ab] === A (stigmergic) vs B (baseline) ===")
    compare(str(sum_a), str(sum_b))

    if do_judge:
        ans_a, ans_b = dir_a / "answer.txt", dir_b / "answer.txt"
        if not (ans_a.exists() and ans_b.exists()):
            print(f"\n[ab] --judge: answer.txt missing "
                  f"({[str(p) for p in (ans_a, ans_b) if not p.exists()]}); skipping judge",
                  file=sys.stderr)
        else:
            import asyncio
            from core.llm import make_llm
            from tools.judge_answers import judge_pair
            print("\n[ab] === answer-text judge (A=stigmergic vs B=baseline) ===")
            llm = make_llm()
            verdict = asyncio.run(judge_pair(
                prompt, ans_a.read_text(encoding="utf-8"),
                ans_b.read_text(encoding="utf-8"), llm))
            winner = {"A": "stigmergic", "B": "baseline", "tie": "tie"}[verdict["winner"]]
            print(f"  judge winner: {winner}  (agreement across orders: {verdict['agreement']})")
            if verdict["rationale"]:
                print(f"  rationale: {verdict['rationale']}")
            if not verdict["agreement"]:
                print("  note: orders disagreed — treat as a tie (position bias or close).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
