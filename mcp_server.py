"""MCP server exposing the stigmergic swarm as callable tools.

Runs the pipeline (run_swarm.py) as a subprocess per tool call and returns
the clean reader answer (answer.txt). Field telemetry stays retrievable via
the diagnostics tool. Designed for Odysseus (PewDiePie's self-hosted AI
workspace) and any other MCP client.

Transports:
    python mcp_server.py                # streamable HTTP on 0.0.0.0:8756
                                        #   Odysseus (Docker) connects to
                                        #   http://host.docker.internal:8756/mcp
    python mcp_server.py --port 9000    # custom port
    python mcp_server.py --stdio        # stdio, for local clients like
                                        #   Claude Code / Claude Desktop

The swarm subprocess inherits this process's environment, so GROQ_API_KEY,
SWARM_MODEL, MOCK_LLM, etc. all apply. Never print to stdout from this
module when running under --stdio (stdout is the MCP transport) — use
stderr / logging only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parent

TASK_TYPES = ("debate", "analysis", "creative", "problem_solving", "coding")

# Wall-clock margin past SWARM_MAX_TIME_S before the subprocess is killed:
# model load (GGUF swap can be 30-60 s/phase) + synthesis render happen
# outside the convergence clock.
_KILL_MARGIN_S = 600


def _outputs_root() -> Path:
    """Mirror run_swarm.py's output-dir logic (P0.1 mock/real split included)."""
    root = "outputs_mock" if os.environ.get("MOCK_LLM") == "1" else "outputs"
    base = os.environ.get("SWARM_OUTPUTS_BASE_DIR")
    return (Path(base) / root) if base else (REPO_ROOT / root)


mcp = FastMCP(
    "stigmergic-swarm",
    instructions=(
        "Tools for running a stigmergic multi-agent LLM swarm on a prompt. "
        "run_swarm executes a full pipeline run (minutes, not seconds) and "
        "returns the synthesized answer; use it for questions that benefit "
        "from multi-perspective analysis, not for quick lookups."
    ),
)


@mcp.tool()
async def run_swarm(
    prompt: str,
    task_type: str = "analysis",
    minutes_budget: float = 8.0,
    baseline: bool = False,
) -> str:
    """Run the stigmergic swarm pipeline on a prompt and return its synthesized answer.

    This is a heavyweight call: a pool of partitioned agents explores the
    question concurrently and a synthesizer reads out the surviving claim
    clusters. Expect several minutes of wall-clock time.

    Args:
        prompt: The question, thesis, or task for the swarm.
        task_type: One of debate | analysis | creative | problem_solving |
            coding. Pick debate for a contestable thesis, analysis for an
            open question, problem_solving for "how can we X", coding for
            implementation tasks, creative for writing.
        minutes_budget: Soft cap on the exploration phase in minutes
            (synthesis runs after this). 5-10 is a reasonable interactive
            range; the repo default is 15.
        baseline: If true, run the non-stigmergic independent-agent
            baseline instead (A/B comparison condition).
    """
    if task_type not in TASK_TYPES:
        return f"ERROR: unknown task_type {task_type!r}; use one of {', '.join(TASK_TYPES)}."
    prompt = prompt.strip()
    if not prompt:
        return "ERROR: empty prompt."
    minutes_budget = min(max(minutes_budget, 1.0), 30.0)

    run_id = "mcp_{}_{}".format(task_type, datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = _outputs_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["SWARM_MAX_TIME_S"] = str(minutes_budget * 60)

    cmd = [sys.executable, str(REPO_ROOT / "run_swarm.py"), task_type, prompt,
           f"--run-id={run_id}"]
    if baseline:
        cmd.append("--mode=baseline")

    # Pipe subprocess output to a log file inside the run dir; piping to
    # memory risks pipe-buffer deadlock on chatty runs.
    log_path = run_dir / "mcp_launch.log"
    timeout = minutes_budget * 60 + _KILL_MARGIN_S
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(REPO_ROOT), env=env,
            stdout=log, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            rc = await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return (
                f"ERROR: swarm run {run_id} exceeded the hard kill window "
                f"({timeout:.0f}s) and was terminated. Partial artifacts (if any) "
                f"are in {run_dir}. Try a smaller minutes_budget or check "
                f"mcp_launch.log there."
            )

    answer_path = run_dir / "answer.txt"
    if rc != 0 or not answer_path.exists():
        tail = _log_tail(log_path)
        return (
            f"ERROR: swarm run {run_id} exited with code {rc} and "
            f"{'no answer.txt' if not answer_path.exists() else 'an answer'}. "
            f"Last log lines:\n{tail}"
        )

    answer = answer_path.read_text(encoding="utf-8", errors="replace").strip()
    footer = (
        f"\n\n---\n[swarm run_id: {run_id}"
        f"{' | MODE: baseline' if baseline else ''}"
        f"{' | MOCK RUN — plumbing only, not a real answer' if os.environ.get('MOCK_LLM') == '1' else ''}"
        f" | call get_swarm_diagnostics for field telemetry]"
    )
    return answer + footer


@mcp.tool()
async def get_swarm_diagnostics(run_id: str) -> str:
    """Return field telemetry for a prior run_swarm call: diagnostics.md plus key summary.json health numbers.

    Args:
        run_id: The run_id reported in a run_swarm result footer.
    """
    run_id = run_id.strip()
    if not re.fullmatch(r"[\w.\-]+", run_id):
        return "ERROR: run_id may only contain letters, digits, dot, dash, underscore."
    run_dir = _outputs_root() / run_id
    if not run_dir.is_dir():
        return f"ERROR: no run directory {run_dir}."

    parts: list[str] = []
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            keys = ("halt_reason", "iterations", "wall_clock_s", "surviving_clusters",
                    "avg_verification_score", "self_bleu", "embedder",
                    "scout_gate_skips", "timing")
            picked = {k: s[k] for k in keys if k in s}
            parts.append("## summary.json (health)\n" + json.dumps(picked, indent=2))
        except (json.JSONDecodeError, OSError) as e:
            parts.append(f"## summary.json unreadable: {e}")
    diag_path = run_dir / "diagnostics.md"
    if diag_path.exists():
        parts.append("## diagnostics.md\n" +
                     diag_path.read_text(encoding="utf-8", errors="replace"))
    if not parts:
        return (f"Run {run_id} has no summary.json or diagnostics.md yet "
                f"(still running, or it failed — see {run_dir / 'mcp_launch.log'}).")
    return "\n\n".join(parts)


@mcp.tool()
async def list_swarm_runs(limit: int = 10) -> str:
    """List recent swarm runs (run_id, task, prompt) so past answers/diagnostics can be retrieved.

    Args:
        limit: Maximum number of runs to list, newest first.
    """
    root = _outputs_root()
    if not root.is_dir():
        return f"No runs yet ({root} does not exist)."
    runs = sorted((d for d in root.iterdir() if d.is_dir()),
                  key=lambda d: d.stat().st_mtime, reverse=True)[: max(1, limit)]
    if not runs:
        return f"No runs yet in {root}."
    lines = []
    for d in runs:
        desc = ""
        meta_path = d / "run_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                p = str(meta.get("user_prompt") or meta.get("prompt") or "")
                desc = f" — {meta.get('task_type', '?')}: {p[:80]}"
            except (json.JSONDecodeError, OSError):
                pass
        done = "done" if (d / "answer.txt").exists() else "no answer"
        lines.append(f"- {d.name} [{done}]{desc}")
    return "\n".join(lines)


def _log_tail(log_path: Path, lines: int = 25) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])
    except OSError as e:
        return f"(log unreadable: {e})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stdio", action="store_true",
                    help="serve over stdio instead of HTTP (for local MCP clients)")
    ap.add_argument("--host", default="0.0.0.0",
                    help="HTTP bind address (0.0.0.0 so Docker's "
                         "host.docker.internal can reach it)")
    ap.add_argument("--port", type=int, default=8756, help="HTTP port")
    args = ap.parse_args()
    if args.stdio:
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        print(f"[mcp] serving streamable HTTP on {args.host}:{args.port} "
              f"(MCP endpoint: http://<host>:{args.port}/mcp)", file=sys.stderr)
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
