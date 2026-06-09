"""Enforces the grader-only boundary for eval/ground_truth.py.

Realized forward returns are the future. If any swarm-path module could import
them, the swarm could peek at the answer — look-ahead bias of the worst kind.
This test fails if it ever happens, which is the durable guard (the runtime
stack check in ground_truth.py is only best-effort).
"""

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent

# Modules that are part of the live swarm path and must NOT import the grader.
_SWARM_GLOBS = ["agents/*.py", "run_swarm.py", "core/stock_data.py",
                "core/stock_verify.py"]

_FORBIDDEN = re.compile(r"\b(?:import\s+eval\.ground_truth|from\s+eval(?:\.ground_truth)?\s+import\s+.*(?:ground_truth|realized_return|PriceSeries))")


def _swarm_files():
    out = []
    for g in _SWARM_GLOBS:
        out.extend(_REPO.glob(g))
    return out


def test_no_swarm_module_imports_ground_truth():
    offenders = []
    for f in _swarm_files():
        text = f.read_text(encoding="utf-8", errors="ignore")
        if "ground_truth" in text or "realized_return" in text:
            # allow the words in comments/docstrings, but not in import statements
            for ln in text.splitlines():
                s = ln.strip()
                if s.startswith("#"):
                    continue
                if _FORBIDDEN.search(s):
                    offenders.append(f"{f.name}: {s}")
    assert not offenders, (
        "grader-only module imported from the swarm path (look-ahead leak):\n"
        + "\n".join(offenders)
    )


def test_runtime_guard_blocks_agent_import():
    # Simulate an agent-module import frame; the guard should refuse.
    import importlib
    import sys

    # Ensure a fresh import so _forbid_agent_import() runs again.
    sys.modules.pop("eval.ground_truth", None)

    src = "import eval.ground_truth as gt\n"
    g = {"__name__": "agents.fake_role"}
    with pytest.raises(ImportError):
        exec(compile(src, "<agents.fake_role>", "exec"), g)

    # cleanup so other tests get a clean import
    sys.modules.pop("eval.ground_truth", None)
