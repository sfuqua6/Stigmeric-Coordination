"""Pairwise LLM judge for two answers to the same prompt.

The summary-metric comparator (tools/compare_runs.py) can't compare stigmergic
vs --mode=baseline quality, because baseline emits no clusters/verification.
This judges the final ANSWER TEXT instead: an LLM is shown both answers and
asked which better answers the prompt, scored on groundedness / depth /
coherence / coverage.

Position bias is a known LLM-judge failure mode (judges favour the first
answer shown), so each pair is judged in BOTH orders (A-first and B-first) and
the results are aggregated: a winner is declared only if it wins both orders;
disagreement is reported as a tie (position bias detected).

Usage (needs a real LLM — MOCK produces no valid verdict, returns a tie):
    python tools/judge_answers.py "<prompt>" runA/answer.txt runB/answer.txt

Programmatic:
    from tools.judge_answers import judge_pair
    verdict = await judge_pair(prompt, text_a, text_b, llm)
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

_DIMENSIONS = ("groundedness", "depth", "coherence", "coverage")

# Per-answer character budget shown to the judge. The old flat `[:4000]`
# head-truncation was biased, not just lossy: it kept the intro and cut the
# ENDING, and thesis-led answers carry judged content at the end (the
# "conditions under which this holds" close that the rubric rewards). Longer
# answers lost more, non-randomly. Budget raised (a ~70B judge has 128K
# context; two 12K-char answers ≈ 6K tokens total) and clipping preserves
# head AND tail with an explicit marker so the judge knows the middle is
# missing rather than the argument being incomplete.
_JUDGE_ANSWER_MAX_CHARS = 12000
_CLIP_MARKER = "\n[... middle truncated for judging — beginning and end preserved ...]\n"


def _clip_for_judging(text: str, budget: int = _JUDGE_ANSWER_MAX_CHARS) -> str:
    if len(text) <= budget:
        return text
    head = int(budget * 0.6)
    tail = budget - head
    return text[:head] + _CLIP_MARKER + text[-tail:]


def _extract_json(raw: str) -> dict | None:
    """Best-effort: pull the first balanced {...} object out of an LLM reply."""
    if not raw:
        return None
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _build_prompt(prompt: str, first: str, second: str) -> str:
    dims = ", ".join(_DIMENSIONS)
    dim_schema = ", ".join('"%s": <1-5>' % d for d in _DIMENSIONS)
    schema = (
        '{"winner": "1" | "2" | "tie", '
        '"scores": {"1": {' + dim_schema + '}, "2": {' + dim_schema + '}}, '
        '"rationale": "<one sentence>"}'
    )
    return (
        "You are judging two answers to the same question. Decide which answer "
        "better addresses the question, considering: " + dims + ".\n"
        "Do NOT prefer an answer merely for being longer or more elaborate — "
        "length is not quality; judge what each answer actually establishes. "
        "(Overlong answers are clipped to the same budget with their beginning "
        "and end preserved and a marker where the middle was removed; do not "
        "penalize the marker.)\n\n"
        "QUESTION:\n" + prompt[:1500] + "\n\n"
        "ANSWER 1:\n" + _clip_for_judging(first) + "\n\n"
        "ANSWER 2:\n" + _clip_for_judging(second) + "\n\n"
        "Reply with ONLY a JSON object:\n" + schema
    )


async def _judge_once(prompt: str, first: str, second: str, llm) -> dict | None:
    """One judgement; `first`/`second` are the two answers in shown order."""
    try:
        raw = await llm.generate(
            _build_prompt(prompt, first, second),
            # 900, not 300: reasoning-model judges (e.g. Groq openai/gpt-oss-120b,
            # chosen 2026-07-24 as an off-Llama-family judge) spend a chunk of
            # max_tokens on hidden reasoning before the JSON verdict; 300 left
            # zero room for content and produced empty completions (verified:
            # max_tokens=20 -> content='', 26 reasoning tokens alone exceeded it).
            role="synthesizer", max_tokens=900, temperature=0.0,
        )
    except Exception:
        return None
    return _extract_json(raw)


def _winner_for(order_a_first: bool, parsed: dict) -> str | None:
    """Map a round's position-label winner ('1'/'2'/'tie') back to 'A'/'B'/'tie'."""
    w = str(parsed.get("winner", "")).strip().lower()
    if w in ("tie", "draw", "equal"):
        return "tie"
    if w not in ("1", "2"):
        return None
    # Position 1 is A when A was shown first, else B.
    pos1_is_a = order_a_first
    if w == "1":
        return "A" if pos1_is_a else "B"
    return "B" if pos1_is_a else "A"


async def judge_pair(prompt: str, answer_a: str, answer_b: str, llm,
                     rounds: int = 2) -> dict:
    """Judge answer_a vs answer_b, mitigating position bias by swapping order.

    Returns {"winner": "A"|"B"|"tie", "agreement": bool, "rounds": [...],
             "rationale": str}. With rounds>=2 the second round swaps order;
    a decisive winner requires agreement across orders, else "tie".
    """
    results: list = []
    # Round 1: A first. Round 2: B first. (Further rounds alternate.)
    for r in range(max(1, rounds)):
        a_first = (r % 2 == 0)
        first, second = (answer_a, answer_b) if a_first else (answer_b, answer_a)
        parsed = await _judge_once(prompt, first, second, llm)
        winner = _winner_for(a_first, parsed) if parsed else None
        results.append({
            "order": "A-first" if a_first else "B-first",
            "winner": winner,
            "rationale": (parsed or {}).get("rationale", "") if parsed else "",
            "parsed": parsed is not None,
        })

    decided = [r["winner"] for r in results if r["winner"] in ("A", "B")]
    if not decided:
        verdict, agreement = "tie", True  # all ties or all unparseable
    elif len(set(decided)) == 1 and len(decided) == len([r for r in results if r["winner"]]):
        verdict, agreement = decided[0], True
    else:
        # Orders disagree (or mix of decisive + tie) -> position bias / no clear win.
        verdict, agreement = "tie", False

    rationale = next((r["rationale"] for r in results if r["rationale"]), "")
    return {
        "winner": verdict,
        "agreement": agreement,
        "rounds": results,
        "rationale": rationale,
    }


def _read(path_or_text: str) -> str:
    p = Path(path_or_text)
    return p.read_text(encoding="utf-8") if p.exists() else path_or_text


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 1
    prompt, a, b = argv[0], _read(argv[1]), _read(argv[2])
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.llm import make_llm
    llm = make_llm()
    verdict = asyncio.run(judge_pair(prompt, a, b, llm))
    print(json.dumps(verdict, indent=2))
    if not verdict["agreement"]:
        print("\n[judge] note: orders disagreed — treat as a tie (position bias "
              "or genuinely close).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
