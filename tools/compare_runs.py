"""Compare two run summary.json files side-by-side (§3 directive).

Usage:
    python tools/compare_runs.py <path_A/summary.json> <path_B/summary.json>

Typical use: compare a stigmergic run against a baseline run on the same prompt
to check whether stigmergic coordination actually adds value.

    python tools/compare_runs.py \\
        outputs/debate_20240101_120000/summary.json \\
        outputs_mock/debate_20240101_120000_baseline/summary.json

Output: a plain-text diff table printed to stdout. No external deps required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[compare] file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[compare] invalid JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def _fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, dict):
        return json.dumps(val)
    return str(val)


def _delta(a, b) -> str:
    """Return a delta string for two numeric scalars, or '' if not comparable."""
    try:
        fa, fb = float(a), float(b)
        diff = fb - fa
        sign = "+" if diff >= 0 else ""
        return f"  ({sign}{diff:.4f})"
    except (TypeError, ValueError):
        return ""


_COMPARISON_KEYS = [
    ("mode",                   "mode"),
    ("task_type",              "task type"),
    ("total_llm_calls",        "total LLM calls"),
    ("total_elapsed_s",        "elapsed (s)"),
    ("scout_reject_rate",      "scout reject rate"),
    ("non_scout_reject_rate",  "non-scout reject rate"),
    ("n_clusters",             "clusters (surviving/contested/weak/rejected)"),
    ("n_unique_responses",     "unique responses (baseline only)"),
    ("max_verification_score", "max verification score"),
    ("avg_verification_score", "avg verification score"),
    ("audit_flags",            "audit flags"),
    ("corpus_chars",           "corpus chars"),
    # Genome quality metrics (added in v6.0)
    ("genome.avg_composite_fitness", "genome avg composite_fitness"),
    ("genome.max_composite_fitness", "genome max composite_fitness"),
    ("genome.avg_grounding",         "genome avg grounding"),
    ("genome.total_atoms",           "genome total atoms"),
    # Output diversity (P2.2)
    ("output_diversity.centroid_cosine_dist", "output diversity (centroid cosine)"),
    ("output_diversity.self_bleu",            "output diversity (self-BLEU)"),
]


def compare(path_a: str, path_b: str) -> None:
    a = _load(path_a)
    b = _load(path_b)

    label_a = Path(path_a).parent.name or path_a
    label_b = Path(path_b).parent.name or path_b

    col_w = 36
    print(f"\n{'Metric':<30}  {'Run A':<{col_w}}  {'Run B':<{col_w}}  Delta")
    print("-" * (30 + col_w * 2 + 12))

    def _nested_get(d: dict, key: str):
        """Support dotted paths like 'genome.avg_composite_fitness'."""
        parts = key.split(".", 1)
        val = d.get(parts[0], "—")
        if len(parts) == 2 and isinstance(val, dict):
            return val.get(parts[1], "—")
        return val

    for key, label in _COMPARISON_KEYS:
        va = _nested_get(a, key)
        vb = _nested_get(b, key)
        if va == "—" and vb == "—":
            continue
        delta = _delta(va, vb) if va != "—" and vb != "—" else ""
        print(f"{label:<30}  {_fmt(va):<{col_w}}  {_fmt(vb):<{col_w}}{delta}")

    print()
    print(f"  Run A: {path_a}")
    print(f"  Run B: {path_b}")

    # Flag if both runs have the same prompt
    prompt_a = a.get("user_prompt", "")
    prompt_b = b.get("user_prompt", "")
    if prompt_a and prompt_b and prompt_a != prompt_b:
        print(
            "\n  WARNING: prompts differ — this comparison may not be valid.\n"
            f"  A: {prompt_a!r}\n  B: {prompt_b!r}"
        )
    print()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
