"""Phase-isolated execution orchestrator.

Spawns one subprocess per phase-of-a-round, in dependency order, so that
exactly ONE model is resident in memory at any time. This is the canonical
workaround for llama-cpp-python's incomplete model unload on Windows /
16 GB-RAM hardware, where the in-process router cannot reliably swap models
between phases.

Usage:
    python tools/run_isolated.py debate "does god exist"
    python tools/run_isolated.py debate "does god exist" --heterogeneous
    python tools/run_isolated.py debate "does god exist" --run-id=MyRun --keep-going

Flags:
    --run-id=NAME      Override the run directory name (default: timestamp).
    --heterogeneous    Pass --heterogeneous to each subprocess (per-role models).
    --corpus={...}     Forwarded to subprocesses (default: real).
    --ignore-kb        Forwarded.
    --cloud-validator={none,anthropic,gemini}   Forwarded.
    --keep-going       Continue with subsequent phases even if one fails.

Each subprocess is invoked with --isolated --phase=X --round=N --run-id=NAME
--resume=<run_dir>. The orchestrator reads phase_manifest.json after each
subprocess to verify it completed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Phase sequencing
# --------------------------------------------------------------------------

# Same shape as ROLES_FOR_TASK in run_swarm.py, but enumerated as ordered
# phase names per round. Kept in sync with run_swarm._round_phases_for_task.
_TASK_ROLES = {
    "debate":          {"critic", "hater", "validator"},
    "analysis":        {"critic", "hater", "validator"},
    "creative":        {"critic"},
    "problem_solving": {"critic", "hater"},
    "coding":          {"critic", "hater", "validator"},
}


def round_phases(task_type: str) -> list[str]:
    active = _TASK_ROLES.get(task_type, {"critic", "hater", "validator"})
    seq = ["scouts"]
    if "validator" in active:
        seq.append("validators")
    seq.append("foragers")
    if "critic" in active:
        seq.append("critics")
    if "hater" in active:
        seq.append("haters")
    seq.append("decay")
    return seq


# Mirror NUM_ROUNDS from core/config.py at runtime to avoid hardcoding.
def _num_rounds() -> int:
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from core.config import NUM_ROUNDS
        return int(NUM_ROUNDS)
    except Exception:
        return 3


# --------------------------------------------------------------------------
# Subprocess invocation
# --------------------------------------------------------------------------

def _build_argv(task_type: str, prompt: str, phase: str, round_num: int,
                run_id: str, resume: Path | None, extras: list[str]) -> list[str]:
    argv = [sys.executable, "run_swarm.py", task_type, prompt,
            "--isolated", f"--phase={phase}", f"--round={round_num}",
            f"--run-id={run_id}"]
    if resume is not None:
        argv.append(f"--resume={resume}")
    argv.extend(extras)
    return argv


def _phase_already_completed(run_dir: Path, phase: str, round_num: int | None) -> bool:
    """Check phase_manifest.json to see if this phase/round has already finished."""
    mp = run_dir / "phase_manifest.json"
    if not mp.exists():
        return False
    try:
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return False
    for entry in manifest.get("completed", []):
        if entry.get("phase") == phase and entry.get("round") == round_num:
            return True
    return False


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Phase-isolated execution orchestrator.")
    p.add_argument("task_type")
    p.add_argument("prompt")
    p.add_argument("--run-id", default=None)
    p.add_argument("--heterogeneous", action="store_true")
    p.add_argument("--corpus", default="real")
    p.add_argument("--ignore-kb", action="store_true")
    p.add_argument("--cloud-validator", default="none")
    p.add_argument("--keep-going", action="store_true",
                   help="Continue to subsequent phases even if one fails.")
    p.add_argument("--resume-existing", default=None,
                   help="Path to an existing run dir; resume from where the manifest "
                        "says it stopped. Implies --run-id=<dir-name>.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    is_mock = os.environ.get("MOCK_LLM", "").strip() not in ("", "0", "false", "False")
    outputs_root = "outputs_mock" if is_mock else "outputs"

    # Resolve run_id and run_dir
    if args.resume_existing:
        resume_root = Path(args.resume_existing).resolve()
        if not resume_root.exists():
            print(f"[orchestrator] --resume-existing path not found: {resume_root}")
            return 1
        run_id = resume_root.name
        run_dir = resume_root
        outputs_root = resume_root.parent.name  # may be "outputs" or "outputs_mock"
    else:
        run_id = args.run_id or f"{args.task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_iso"
        run_dir = REPO_ROOT / outputs_root / run_id

    print(f"[orchestrator] run_id    = {run_id}")
    print(f"[orchestrator] run_dir   = {run_dir}")
    print(f"[orchestrator] task      = {args.task_type!r} prompt={args.prompt!r}")
    print(f"[orchestrator] mock_llm  = {is_mock}")

    # Build the ordered phase list: (phase, round_num)
    num_rounds = _num_rounds()
    sequence: list[tuple[str, int | None]] = []
    phases = round_phases(args.task_type)
    for rn in range(1, num_rounds + 1):
        for ph in phases:
            sequence.append((ph, rn))
    sequence.append(("synth", None))
    print(f"[orchestrator] sequence: {len(sequence)} subprocesses across {num_rounds} rounds")
    print(f"[orchestrator] phases per round: {phases}")

    # Forwarded flags
    extras: list[str] = [f"--corpus={args.corpus}",
                          f"--cloud-validator={args.cloud_validator}"]
    if args.heterogeneous:
        extras.append("--heterogeneous")
    if args.ignore_kb:
        extras.append("--ignore-kb")

    if args.dry_run:
        for i, (ph, rn) in enumerate(sequence, 1):
            argv = _build_argv(args.task_type, args.prompt, ph, rn or 0, run_id,
                               run_dir if i > 1 else None, extras)
            print(f"  [{i:>2}/{len(sequence)}] {' '.join(argv[1:])}")
        return 0

    # Execute
    failures: list[tuple[str, int | None, int]] = []
    overall_start = time.time()
    for i, (ph, rn) in enumerate(sequence, 1):
        if _phase_already_completed(run_dir, ph, rn):
            print(f"\n[orchestrator] [{i}/{len(sequence)}] SKIP {ph} R{rn} "
                  f"(already completed per manifest)")
            continue
        resume = run_dir if run_dir.exists() else None
        argv = _build_argv(args.task_type, args.prompt, ph, rn or 0, run_id, resume, extras)
        label = f"{ph} R{rn}" if rn is not None else ph
        print(f"\n[orchestrator] [{i}/{len(sequence)}] EXEC {label}")
        print(f"  $ {' '.join(argv)}")
        t0 = time.time()
        proc = subprocess.run(argv, cwd=str(REPO_ROOT))
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"[orchestrator] FAILED {label} (exit={proc.returncode}) in {elapsed:.1f}s")
            failures.append((ph, rn, proc.returncode))
            if not args.keep_going:
                print(f"[orchestrator] stopping (use --keep-going to attempt subsequent phases)")
                break
        else:
            print(f"[orchestrator] OK {label} in {elapsed:.1f}s")

    total = time.time() - overall_start
    print(f"\n[orchestrator] done in {total:.1f}s; {len(failures)} failures")
    for ph, rn, code in failures:
        print(f"  - {ph} R{rn}: exit {code}")

    return 1 if failures and not args.keep_going else 0


if __name__ == "__main__":
    sys.exit(main())
