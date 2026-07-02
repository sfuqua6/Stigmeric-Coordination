"""Compare two run summary.json files side-by-side (§3 directive).

Usage:
    python tools/compare_runs.py <path_A/summary.json> <path_B/summary.json>

Typical use: A = the established run (e.g. stigmergic, or a pre-change run),
B = the candidate (e.g. --mode=baseline, or a post-optimization run). The output
is grouped into QUALITY / LATENCY / PROCESS so you can answer the optimization
question directly — "did wall-clock drop while quality held?" — and a verdict
line summarizes it.

    python tools/compare_runs.py \\
        outputs/debate_A/summary.json \\
        outputs/debate_B/summary.json

Output: a plain-text grouped diff table + verdict, printed to stdout. No
external deps. Metrics absent from a summary are skipped (forward/backward
compatible across schema versions).
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


_MISSING = "—"


def _nested_get(d: dict, key: str):
    """Resolve a dotted path of arbitrary depth, e.g. 'timing.search.calls'.

    Returns _MISSING if any segment is absent or a non-dict is traversed.
    """
    val = d
    for part in key.split("."):
        if not isinstance(val, dict) or part not in val:
            return _MISSING
        val = val[part]
    return val


def _fmt(val) -> str:
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, dict):
        return json.dumps(val)
    return str(val)


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _delta(a, b) -> str:
    """Delta string for two numeric scalars, or '' if not comparable."""
    fa, fb = _as_float(a), _as_float(b)
    if fa is None or fb is None:
        return ""
    diff = fb - fa
    sign = "+" if diff >= 0 else ""
    return f"  ({sign}{diff:.4f})"


# (key, label, "higher_is_better" | "lower_is_better" | None). The direction is
# used only for the verdict; None means informational (not scored).
_QUALITY = [
    ("max_verification_score",                 "max verification score",          "higher_is_better"),
    ("avg_verification_score",                 "avg verification score",          "higher_is_better"),
    ("genome.avg_composite_fitness",           "genome avg composite_fitness",    "higher_is_better"),
    ("genome.max_composite_fitness",           "genome max composite_fitness",    "higher_is_better"),
    ("genome.avg_grounding",                   "genome avg grounding",            "higher_is_better"),
    ("genome.total_atoms",                     "genome total atoms",              "higher_is_better"),
    ("output_diversity.centroid_cosine_dist",  "output diversity (centroid)",     "higher_is_better"),
    ("output_diversity.self_bleu",             "output diversity (self-BLEU)",    "lower_is_better"),
    ("n_clusters",                             "clusters (surv/cont/weak/rej)",   None),
    ("audit_flags",                            "audit flags",                     "lower_is_better"),
]

_LATENCY = [
    ("wall_clock_s",                           "wall clock (s)",                  "lower_is_better"),
    ("timing.search_fraction_of_wallclock",    "search frac of wallclock",        "lower_is_better"),
    ("timing.search.total_s",                  "search cumulative (s)",           None),
    ("timing.search.calls",                    "search calls",                    None),
    ("timing.search.max_s",                    "slowest search (s)",              None),
    ("timing.search.empty_results",            "empty searches",                  None),
    ("timing.search.fetch_calls",              "page fetches",                    None),
    ("timing.search.fetch_cache_hits",         "page-cache hits",                 None),
    ("timing.search.fetch_total_s",            "page-fetch cumulative (s)",       None),
    ("timing.llm_generate_cumulative_s",       "LLM gen cumulative (s)",          None),
    ("total_llm_calls",                        "total LLM calls",                 None),
    ("total_iterations",                       "iterations",                      None),
]

_PROCESS = [
    ("mode",                                   "mode",                            None),
    ("execution_mode",                         "execution mode",                  None),
    ("task_type",                              "task type",                       None),
    ("convergence_reason",                     "convergence reason",              None),
    ("scout_reject_rate",                      "scout reject rate",               None),
    ("non_scout_reject_rate",                  "non-scout reject rate",           None),
    ("quality_met",                            "quality_met",                     None),
    ("degraded",                               "degraded",                        None),
    ("embedder",                               "embedder",                        None),
]

# Fraction a "higher_is_better" quality metric may drop (or a "lower_is_better"
# one may rise) before the verdict flags a regression. 5% tolerates noise.
_QUALITY_TOLERANCE = 0.05


def _print_section(title: str, rows, a: dict, b: dict, col_w: int) -> list:
    """Print one grouped section; return [(label, direction, va, vb)] for scored rows."""
    printed_any = False
    scored: list = []
    for key, label, direction in rows:
        va = _nested_get(a, key)
        vb = _nested_get(b, key)
        if va is _MISSING and vb is _MISSING:
            continue
        if not printed_any:
            print(f"\n  [{title}]")
            printed_any = True
        delta = _delta(va, vb) if va is not _MISSING and vb is not _MISSING else ""
        print(f"  {label:<30}  {_fmt(va):<{col_w}}  {_fmt(vb):<{col_w}}{delta}")
        if direction and va is not _MISSING and vb is not _MISSING:
            scored.append((label, direction, va, vb))
    return scored


def _verdict(scored: list) -> list[str]:
    """Build verdict lines from scored quality rows. A regression is a scored
    metric that moved the wrong way by more than _QUALITY_TOLERANCE (relative)."""
    regressions: list[str] = []
    for label, direction, va, vb in scored:
        fa, fb = _as_float(va), _as_float(vb)
        if fa is None or fb is None:
            continue
        base = abs(fa) if fa else 1.0
        rel = (fb - fa) / base
        if direction == "higher_is_better" and rel < -_QUALITY_TOLERANCE:
            regressions.append(f"{label}: {fa:.4f} → {fb:.4f} ({rel:+.1%})")
        elif direction == "lower_is_better" and rel > _QUALITY_TOLERANCE:
            regressions.append(f"{label}: {fa:.4f} → {fb:.4f} ({rel:+.1%})")
    return regressions


def compare(path_a: str, path_b: str) -> None:
    a = _load(path_a)
    b = _load(path_b)

    col_w = 30
    label_a = Path(path_a).parent.name or path_a
    label_b = Path(path_b).parent.name or path_b
    print(f"\n{'  Metric':<32}  {'A: ' + label_a:<{col_w}}  {'B: ' + label_b:<{col_w}}  Delta (B-A)")
    print("-" * (32 + col_w * 2 + 14))

    quality_scored = _print_section("QUALITY", _QUALITY, a, b, col_w)
    _print_section("LATENCY", _LATENCY, a, b, col_w)
    _print_section("PROCESS", _PROCESS, a, b, col_w)

    # Verdict: wall-clock change + quality regressions.
    print("\n  [VERDICT]")
    wa = _as_float(_nested_get(a, "wall_clock_s"))
    wb = _as_float(_nested_get(b, "wall_clock_s"))
    if wa and wb:
        rel = (wb - wa) / wa
        faster = "faster" if wb < wa else "slower"
        print(f"  wall-clock: {wa:.1f}s → {wb:.1f}s  ({rel:+.1%}, B is {faster})")
    regressions = _verdict(quality_scored)
    if not quality_scored:
        print("  quality: no scored quality metrics present in both summaries")
    elif regressions:
        print(f"  quality: {len(regressions)} metric(s) REGRESSED beyond "
              f"{_QUALITY_TOLERANCE:.0%}:")
        for r in regressions:
            print(f"    - {r}")
    else:
        print(f"  quality: HELD (no scored metric regressed beyond "
              f"{_QUALITY_TOLERANCE:.0%})")

    print(f"\n  Run A: {path_a}")
    print(f"  Run B: {path_b}")

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
