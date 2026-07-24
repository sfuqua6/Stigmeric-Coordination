"""Blind pairwise judge + factual scorer + report for the delta experiment.

Reads an eval/results/<exp>/conditions.jsonl produced by eval.ab_harness and
turns it into the deltas that decide the project:

  delta_amp      = A vs B   (does the swarm amplify at all?)
  delta_vs_simple= A vs C   (does it beat a cheap scaffold?)
  delta_vs_strong= A vs D   (small+swarm vs big-direct — the real goal)
  delta_vs_rag   = A vs F   (does orchestration beat single-call RAG over
                            the same retrieved evidence? the practitioner
                            baseline — the decisive attribution rung)
  delta_vs_prompt= A vs E   (does orchestration beat its own synthesis
                             prompt handed to ONE call? the attribution
                             control — if E ties A, the value was the prompt)

Five ways you'll fool yourself, and how this guards each:
  * strawman baseline   -> B uses a genuinely strong prompt (in ab_harness).
  * FORMAT LEAKAGE       -> normalize() strips the swarm's tells (citation tags,
                            EXECUTIVE SUMMARY, PROCESS NOTES, census lines)
                            before the judge ever sees the text. Critical: the
                            judge must grade content, not the "I'm the swarm"
                            format.
  * self-enhancement     -> the judge model must be a DIFFERENT family from
                            every condition. We refuse to judge if --judge-model
                            collides with a condition's model (unless --force).
  * position bias        -> every pair is judged in BOTH orders; disagreement
                            across orders is recorded as a tie.
  * peeking / p-hacking  -> the prompt set is pre-registered (eval/prompts.py);
                            the primary metric is delta_amp on the FULL set.

Factual prompts also get an objective, judge-free score (fraction of
ground-truth checklist items present) — the bias-free anchor.

Stats: per condition-pair we tally wins/ties/losses over prompts x both
orders, and report a win-rate with a Wilson 95% CI. A real win = CI lower
bound clearly above 0.50. Cost-adjust: each pair also reports the cost
multiple (A's llm_calls / opponent's), so a tiny win at 300x cost is called
what it is.

CLI
---
    # score an experiment dir (needs a real judge LLM for pairwise verdicts):
    GROQ_API_KEY=... python -m eval.judge eval/results/delta \
        --judge-model llama-3.3-70b-versatile

    # mock plumbing check (pairwise verdicts will all be ties under mock):
    MOCK_LLM=1 python -m eval.judge eval/results/delta
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.judge_answers import judge_pair  # noqa: E402  (reuse the both-orders judge)


# ---------------------------------------------------------------------------
# Formatting normalizer — strip the swarm's tells so the judge grades CONTENT
# ---------------------------------------------------------------------------

# Citation / signal tags: [INITIAL_00171], [SUPPORT_0012], [VERIFICATION_3],
# [SEARCH_...], [DEVELOP_...], [External context], [UNSOURCED FIGURES ...].
_TAG_RE = re.compile(
    r"\[(?:INITIAL|SUPPORT|VERIFICATION|SEARCH|DEVELOP|CHAIN|REFINE|ATOM)"
    r"[_A-Z0-9]*\]",
    re.IGNORECASE,
)
_BRACKET_NOTE_RE = re.compile(
    r"\[(?:external context|unsourced[^\]]*|atom[^\]]*)\]", re.IGNORECASE)

# Markdown reference links to swarm signal anchors: "[1](#initial_00016)" ->
# keep the "[1]" label (a neutral citation marker), drop the #initial_ anchor
# that gives the swarm away. Survives into kept sections (Section 1 prose).
_ANCHOR_LINK_RE = re.compile(
    r"\[([^\]]*)\]\(#(?:initial|support|verification|search|develop|chain|refine|atom)"
    r"[^)]*\)",
    re.IGNORECASE,
)
# Defensive: parenthetical field-telemetry annotations the swarm appends to
# filtered/rejected claims, e.g. "(rejected: dissent_pressure=1.58 > 1.5)",
# "(held: no verification ...)", "(filtered: support_diversity=2 < 3)". Section 3
# is now cut entirely, but strip these wherever they appear.
_PAREN_TELEMETRY_RE = re.compile(
    r"\((?:rejected|filtered|held|demoted|surviving|contested)\b[^)]*\)",
    re.IGNORECASE,
)

# Telemetry section headers the swarm appends. Everything from one of these to
# the next blank-line-separated header (or EOF) is process noise, not answer.
_TELEMETRY_HEADERS = (
    "PROCESS NOTES",
    "UNEXPLORED ANSWER-SPACE",
    "OUT-OF-BOUNDS CLUSTERS",
    "FIELD TELEMETRY",
    "FIELD SUMMARY",
    "CLUSTER CENSUS",
    "GENOME STATS",
    "CITATIONS",
    "SOURCES REFERENCED",
    "SYNTHESIS PLAN",
    # Section 3 of the swarm read-out lists claims the field FILTERED OUT, each
    # annotated with raw telemetry ("(rejected: dissent_pressure=1.58 > 1.5)").
    # That is process disclosure, not answer content, and it leaked the swarm's
    # identity + padded its length into the judge. Cut from here (it precedes
    # PROCESS NOTES in the answer, so it becomes the effective cut point).
    "CONSIDERED AND FILTERED",
)

# Leading markdown / numbering to strip before matching a header label
# (e.g. "## 4. CITATIONS" / "**Sources referenced above:**" -> "CITATIONS" /
# "SOURCES REFERENCED ABOVE").
_HEADER_DECOR_RE = re.compile(r"^[\s#*>_.\-0-9]+")

# Census / scaffold lines: "51 weakly supported", "3 surviving clusters",
# "Section 5:", "EXECUTIVE SUMMARY" (header label only — keep its prose).
_CENSUS_LINE_RE = re.compile(
    r"^\s*(?:\d+\s+(?:weakly[ -]supported|surviving|contested|rejected|clusters?)\b"
    r"|section\s+\d+\b.*|topology coverage\b.*|verification score\b.*)",
    re.IGNORECASE,
)
_HEADER_LABELS = (
    "EXECUTIVE SUMMARY", "SECTION 1", "SECTION 2", "SECTION 3", "SECTION 4",
    "FINAL ANSWER", "ANSWER:", "THESIS",
)


def _is_telemetry_header(line: str) -> bool:
    s = _HEADER_DECOR_RE.sub("", line.strip()).upper().rstrip(":*_ ")
    return any(s.startswith(h) for h in _TELEMETRY_HEADERS)


def normalize(text: str) -> str:
    """Strip swarm-specific formatting so all conditions read in one neutral
    style. Removes citation/signal tags, bracketed process notes, telemetry
    sections, census lines, and bare section-header labels; collapses
    whitespace. Idempotent; safe on plain direct-model prose (no-op-ish)."""
    if not text:
        return ""
    # Drop everything from the first telemetry header onward (PROCESS NOTES etc.
    # always live at the END of the swarm answer).
    lines = text.splitlines()
    cut = len(lines)
    for i, ln in enumerate(lines):
        if _is_telemetry_header(ln):
            cut = i
            break
    lines = lines[:cut]

    out: list[str] = []
    for ln in lines:
        if _CENSUS_LINE_RE.match(ln):
            continue
        s = _TAG_RE.sub("", ln)
        s = _BRACKET_NOTE_RE.sub("", s)
        s = _ANCHOR_LINK_RE.sub(r"[\1]", s)
        s = _PAREN_TELEMETRY_RE.sub("", s)
        # Strip a bare header label that's alone on its line (keep prose lines
        # that merely start with the word).
        stripped = s.strip().rstrip(":").upper()
        if stripped in _HEADER_LABELS:
            continue
        out.append(s)
    joined = "\n".join(out)
    joined = re.sub(r"[ \t]{2,}", " ", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


# ---------------------------------------------------------------------------
# Objective factual scorer (judge-free anchor)
# ---------------------------------------------------------------------------

def factual_score(answer: str, must_include: list[list[str]]) -> float | None:
    """Fraction of ground-truth checklist items present. Each item is a list of
    acceptable surface forms; the item is hit if ANY form appears (case-insensitive
    substring). Returns None when there's no checklist."""
    if not must_include:
        return None
    low = (answer or "").lower()
    hits = sum(1 for forms in must_include if any(f.lower() in low for f in forms))
    return hits / len(must_include)


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------

def wilson(successes: float, n: float, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. `successes` may be
    fractional (ties counted as half). Returns (low, point, high)."""
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), p, min(1.0, center + half))


# ---------------------------------------------------------------------------
# Load + judge
# ---------------------------------------------------------------------------

def load_rows(exp_dir: Path) -> list[dict]:
    path = exp_dir / "conditions.jsonl"
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return rows


def _index(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """pid -> condition -> row."""
    idx: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        idx[r["pid"]][r["condition"]] = r
    return idx


def _model_of(row: dict) -> str:
    return (row.get("label") or "").lower()


async def judge_pairs(idx, pairs, judge_llm, meta) -> dict:
    """For each (left, right) condition pair, judge every prompt blind in both
    orders and aggregate. `left` is the focus condition (A); a win for left =
    swarm wins."""
    results: dict[str, dict] = {}
    for left, right in pairs:
        per_prompt = []
        cost_ratios = []
        for pid, conds in idx.items():
            if left not in conds or right not in conds:
                continue
            la, ra = conds[left], conds[right]
            a_text, b_text = normalize(la["answer"]), normalize(ra["answer"])
            verdict = await judge_pair(conds[left]["prompt"], a_text, b_text, judge_llm)
            cr = _cost_ratio(la, ra)
            if cr is not None:
                cost_ratios.append(cr)
            per_prompt.append({
                "pid": pid, "winner": verdict["winner"],  # "A"/"B"/"tie"
                "agreement": verdict["agreement"],
                "rationale": verdict["rationale"],
            })
        results[f"{left}_vs_{right}"] = _summarize_pair(
            left, right, per_prompt, cost_ratios)
    return results


def _cost_ratio(left_row: dict, right_row: dict) -> float | None:
    lc = (left_row.get("cost") or {}).get("llm_calls") or 0
    rc = (right_row.get("cost") or {}).get("llm_calls") or 0
    return (lc / rc) if rc else None


def _summarize_pair(left: str, right: str, per_prompt: list[dict],
                    cost_ratios: list[float]) -> dict:
    """Tally a pair's per-prompt verdicts (winner in {left, right, 'tie'}) into
    win-rate + Wilson CI + cost multiple. Shared by the auto-LLM judge and the
    Claude-cowork verdict path so both produce identical report rows."""
    wins = sum(1 for pp in per_prompt if pp["winner"] == left)
    losses = sum(1 for pp in per_prompt if pp["winner"] == right)
    ties = len(per_prompt) - wins - losses
    n = len(per_prompt)
    succ = wins + 0.5 * ties  # ties = half credit
    low, point, high = wilson(succ, n) if n else (0.0, 0.0, 0.0)
    return {
        "left": left, "right": right,
        "n": n, "wins": wins, "ties": ties, "losses": losses,
        "win_rate": round(point, 4),
        "wilson95": [round(low, 4), round(high, 4)],
        "real_win": low > 0.5,
        "median_cost_multiple": (
            round(sorted(cost_ratios)[len(cost_ratios) // 2], 2)
            if cost_ratios else None),
        "per_prompt": per_prompt,
    }


def factual_table(rows: list[dict]) -> dict:
    """Per-condition mean factual score over factual prompts (judge-free)."""
    by_cond: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if not r.get("factual"):
            continue
        sc = factual_score(normalize(r["answer"]), r.get("must_include") or [])
        if sc is not None:
            by_cond[r["condition"]].append(sc)
    return {c: round(sum(v) / len(v), 4) for c, v in sorted(by_cond.items()) if v}


# ---------------------------------------------------------------------------
# Claude-cowork judging packet (no LLM-judge model — you + Claude grade it)
# ---------------------------------------------------------------------------
#
# Instead of an automated judge model, assemble a BLIND packet: every pair is
# presented twice (both orders) as anonymized "Response 1 / Response 2", with
# all swarm formatting stripped and the items shuffled so the grader can't tell
# which response is the swarm or pair up the two orders. You (with Claude as the
# co-working judge) fill verdicts.json; then `--score-verdicts` ingests them +
# the hidden key and assembles report.md. The blindness guards stay intact:
# format-normalized, randomized, both-orders.

_PACKET_INTRO = (
    "# Blind judging packet\n\n"
    "You are judging pairs of answers to the same question. For each item, two "
    "answers are shown as **Response 1** and **Response 2** in a randomized, "
    "anonymized order — you do NOT know which system produced which, and the "
    "two orderings of the same pair are scattered so you can't pair them up. "
    "Judge purely on the text.\n\n"
    "**Rubric** (pick the better answer overall): correctness, grounding, "
    "coherence, completeness, **non-repetition**. If they're genuinely "
    "indistinguishable, say `tie` — don't force a winner.\n\n"
    "**How to record:** edit `verdicts.json` in this directory. For each item id "
    "set the value to `\"1\"`, `\"2\"`, or `\"tie\"` (or an object "
    "`{\"winner\": \"1\", \"note\": \"...\"}`). Then run:\n\n"
    "    python -m eval.judge <this_dir> --score-verdicts verdicts.json\n\n"
    "---\n"
)


def _render_item(n: int, it: dict) -> str:
    return (
        f"\n## Item {n} — `{it['id']}`\n\n"
        f"**Question:** {it['prompt']}\n\n"
        f"**Response 1:**\n\n{it['r1'] or '_(empty)_'}\n\n"
        f"**Response 2:**\n\n{it['r2'] or '_(empty)_'}\n\n"
        f"**Verdict for `{it['id']}` (1 / 2 / tie):** _____\n\n"
        "---"
    )


def build_pack(exp_dir: Path, meta: dict, idx, pairs, seed: int = 1234,
               max_chars: int = 60000, resp_cap: int = 4000) -> list[Path]:
    """Write the blind judging packet, SPLIT into bounded parts so no single
    file/turn is large enough to truncate mid-item (a dropped item = a silently
    dropped judgment). Also writes pack_key.json (hidden) + verdicts.json
    (template) + a packets.zip for one-click download. No LLM call. Returns the
    list of part paths.

    `max_chars` bounds each part (kept well under a comfortable single-turn
    budget); an item is never split across parts. `resp_cap` truncates each
    rendered answer."""
    rng = random.Random(seed)
    items: list[dict] = []
    key_items: dict[str, dict] = {}
    for left, right in pairs:
        for pid, conds in idx.items():
            if left not in conds or right not in conds:
                continue
            prompt = conds[left]["prompt"]
            tl = normalize(conds[left]["answer"])[:resp_cap]
            tr = normalize(conds[right]["answer"])[:resp_cap]
            for order, (c1, c2, r1, r2) in {
                "o1": (left, right, tl, tr),
                "o2": (right, left, tr, tl),
            }.items():
                iid = f"{left}{right}__{pid}__{order}"
                items.append({"id": iid, "prompt": prompt, "r1": r1, "r2": r2})
                key_items[iid] = {"pair": f"{left}{right}", "pid": pid,
                                  "left": left, "right": right,
                                  "resp1": c1, "resp2": c2}
    rng.shuffle(items)

    # Greedy bin-pack items into parts under max_chars (always >=1 item/part).
    parts: list[list[dict]] = [[]]
    sizes: list[int] = [len(_PACKET_INTRO)]
    for it in items:
        body = _render_item(0, it)
        if parts[-1] and sizes[-1] + len(body) > max_chars:
            parts.append([])
            sizes.append(len(_PACKET_INTRO))
        parts[-1].append(it)
        sizes[-1] += len(body)

    # Clean any stale parts from a previous pack with more parts.
    for old in exp_dir.glob("judging_packet_part*.md"):
        old.unlink()

    npart = len(parts)
    part_paths: list[Path] = []
    n = 0
    for pi, part in enumerate(parts, 1):
        lines = [_PACKET_INTRO,
                 f"\n> **Part {pi} of {npart}** — {len(part)} items. Judge every "
                 f"item; verdicts merge across all parts by item id.\n"]
        for it in part:
            n += 1
            lines.append(_render_item(n, it))
        p = exp_dir / (f"judging_packet_part{pi:02d}.md" if npart > 1
                       else "judging_packet.md")
        p.write_text("\n".join(lines), encoding="utf-8")
        part_paths.append(p)

    (exp_dir / "pack_key.json").write_text(
        json.dumps({"seed": seed, "pairs": [list(p) for p in pairs],
                    "n_parts": npart, "items": key_items}, indent=2),
        encoding="utf-8")
    # verdicts template (empty) — only create if absent so re-packing doesn't
    # clobber verdicts already filled in.
    vpath = exp_dir / "verdicts.json"
    if not vpath.exists():
        vpath.write_text(
            json.dumps({iid: "" for iid in key_items}, indent=2),
            encoding="utf-8")
    # Bundle all parts into one zip for a single Colab download.
    import zipfile
    zpath = exp_dir / "judging_packets.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in part_paths:
            zf.write(p, arcname=p.name)
    print(f"[judge] packet: {len(items)} items across {npart} part(s) "
          f"(<= {max_chars} chars each); bundled in {zpath.name}")
    return part_paths


def _resolve_choice(value, resp1_cond: str, resp2_cond: str):
    """Map a recorded verdict ('1'/'2'/'tie' or {winner,note}) to a condition
    or 'tie'. Returns (winner_or_tie_or_None, note)."""
    note = ""
    if isinstance(value, dict):
        note = str(value.get("note", ""))
        value = value.get("winner", "")
    v = str(value).strip().lower()
    if v in ("1", "resp1", "response 1", "response1"):
        return resp1_cond, note
    if v in ("2", "resp2", "response 2", "response2"):
        return resp2_cond, note
    if v in ("tie", "draw", "equal"):
        return "tie", note
    return None, note  # blank / unrecognized -> not yet judged


def score_from_verdicts(exp_dir: Path, idx, verdicts_path: Path) -> dict:
    """Aggregate filled verdicts (both orders per pair) into pair_results,
    mirroring judge_pair's logic: decisive only if both orders agree, else tie."""
    key = json.loads((exp_dir / "pack_key.json").read_text(encoding="utf-8"))
    verdicts = json.loads(verdicts_path.read_text(encoding="utf-8"))
    key_items = key["items"]

    # Group the two order-items back together by (pair, pid).
    grouped: dict[tuple, dict] = defaultdict(dict)
    for iid, info in key_items.items():
        order = iid.rsplit("__", 1)[-1]  # o1 / o2
        grouped[(info["pair"], info["pid"])][order] = (iid, info)

    n_blank = 0
    results: dict[str, dict] = {}
    per_pair: dict[str, list[dict]] = defaultdict(list)
    cost_by_pair: dict[str, list[float]] = defaultdict(list)
    for (pair, pid), orders in grouped.items():
        left, right = pair[0], pair[1]
        winners = []
        notes = []
        for order in ("o1", "o2"):
            if order not in orders:
                continue
            iid, info = orders[order]
            w, note = _resolve_choice(
                verdicts.get(iid, ""), info["resp1"], info["resp2"])
            if w is None:
                n_blank += 1
            else:
                winners.append(w)
            if note:
                notes.append(note)
        decided = [w for w in winners if w in (left, right)]
        if not decided:
            winner, agreement = "tie", True
        elif len(set(decided)) == 1 and len(decided) == len(winners):
            winner, agreement = decided[0], True
        else:
            winner, agreement = "tie", False  # orders disagree -> tie
        per_pair[pair].append({
            "pid": pid, "winner": winner, "agreement": agreement,
            "rationale": "; ".join(notes)[:200],
        })
        la, ra = idx[pid].get(left), idx[pid].get(right)
        if la and ra:
            cr = _cost_ratio(la, ra)
            if cr is not None:
                cost_by_pair[pair].append(cr)

    for pair, pp in per_pair.items():
        left, right = pair[0], pair[1]
        results[f"{left}_vs_{right}"] = _summarize_pair(
            left, right, pp, cost_by_pair[pair])
    if n_blank:
        print(f"[judge] WARNING: {n_blank} order-items still blank in "
              f"{verdicts_path.name} — they count as 'not judged' (treated as a "
              f"missing order). Fill them for a complete result.", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_COND_NAME = {"A": "swarm(M)", "B": "direct(M)", "C": "cheap-scaffold(M)",
              "D": "direct(M+)", "E": "synth-prompt(M)", "F": "rag(M)"}


def write_report(exp_dir: Path, meta: dict, pair_results: dict,
                 facts: dict) -> Path:
    lines: list[str] = []
    lines.append(f"# Amplification-delta report — `{meta.get('name','?')}`\n")
    if meta.get("mock"):
        lines.append("> ⚠️ **MOCK run — plumbing only, NOT empirical (P0.1).** "
                     "Pairwise verdicts are meaningless under MockLLM.\n")
    lines.append(f"- model M (A/B/C): `{meta.get('model_M')}`")
    lines.append(f"- strong model M+ (D): `{meta.get('strong_model')}`")
    lines.append(f"- conditions: {', '.join(meta.get('conditions', []))}")
    if meta.get("judge_model"):
        lines.append(f"- judge: `{meta['judge_model']}`")
    lines.append(f"- prompts: {meta.get('n_prompts')} "
                 f"(pre-registered: {', '.join(meta.get('prompt_ids', []))})")
    lines.append(f"- scaffold (C): {meta.get('scaffold')}\n")

    lines.append("## Deltas (A = swarm is the focus condition)\n")
    lines.append("| Comparison | n | win | tie | loss | win-rate | Wilson 95% | real win? | cost× |")
    lines.append("|---|--:|--:|--:|--:|--:|---|:--:|--:|")
    label_map = {"A_vs_B": "Δ_amp (A vs B)", "A_vs_C": "Δ_vs_simple (A vs C)",
                 "A_vs_D": "Δ_vs_strong (A vs D)",
                 "A_vs_E": "Δ_vs_prompt (A vs E)",
                 "A_vs_F": "Δ_vs_rag (A vs F)"}
    for key in ("A_vs_B", "A_vs_C", "A_vs_D", "A_vs_E", "A_vs_F"):
        r = pair_results.get(key)
        if not r:
            continue
        wl = r["wilson95"]
        lines.append(
            f"| {label_map[key]} | {r['n']} | {r['wins']} | {r['ties']} | "
            f"{r['losses']} | {r['win_rate']:.0%} | [{wl[0]:.2f}, {wl[1]:.2f}] | "
            f"{'**YES**' if r['real_win'] else 'no'} | "
            f"{('%.0f×' % r['median_cost_multiple']) if r['median_cost_multiple'] else '—'} |")
    lines.append("")
    lines.append("> A **real win** = Wilson lower bound clearly above 50%. "
                 "Read the win-rate next to the cost multiple: a small win at a "
                 "large cost multiple is a practical loss.\n")

    if facts:
        lines.append("## Objective factual score (judge-free anchor)\n")
        lines.append("Mean fraction of ground-truth checklist items present, "
                     "over factual prompts.\n")
        lines.append("| Condition | factual score |")
        lines.append("|---|--:|")
        for c, v in facts.items():
            lines.append(f"| {c} — {_COND_NAME.get(c, c)} | {v:.0%} |")
        lines.append("")

    lines.append("## Per-prompt verdicts\n")
    for key in ("A_vs_B", "A_vs_C", "A_vs_D", "A_vs_E", "A_vs_F"):
        r = pair_results.get(key)
        if not r:
            continue
        lines.append(f"### {label_map[key]}\n")
        lines.append("| prompt | winner | agreed orders? | rationale |")
        lines.append("|---|:--:|:--:|---|")
        win_name = {"A": "swarm", "B": "direct", "C": "scaffold",
                    "D": "strong", "E": "synthprompt", "F": "rag",
                    "tie": "tie"}
        for pp in r["per_prompt"]:
            w = pp["winner"]
            mapped = win_name.get(w, w)
            rat = (pp["rationale"] or "").replace("|", "/")[:120]
            lines.append(f"| {pp['pid']} | {mapped} | "
                         f"{'yes' if pp['agreement'] else 'no'} | {rat} |")
        lines.append("")

    report = exp_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    (exp_dir / "scores.json").write_text(
        json.dumps({"pairs": pair_results, "factual": facts,
                    "judge_model": meta.get("judge_model")}, indent=2),
        encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_judge(model: str | None):
    """A judge LLM that is ideally a different family from every condition."""
    if os.environ.get("MOCK_LLM", "").strip() not in ("", "0", "false", "False"):
        from core.llm import MockLLM
        return MockLLM(), "mock"
    if os.environ.get("GROQ_API_KEY") and model:
        from core.llm_groq import GroqBackend
        return GroqBackend(model=model, api_key=os.environ["GROQ_API_KEY"]), f"groq:{model}"
    # No Groq key: if the judge model is given as a Hugging Face id (org/name),
    # load THAT model locally, so the judge can still be a DIFFERENT family from
    # the conditions (e.g. a fully-local Blackwell run). Without this the judge
    # silently fell back to the swarm's own tier model — same family as the
    # conditions, defeating the neutral-judge requirement for any run where the
    # conditions differ (the size sweep, condition D). A bare name (no '/') is a
    # Groq model id, not an HF repo, so fall through to the default engine.
    if model and "/" in model:
        os.environ["SWARM_MODEL"] = model   # set before make_llm() resolves config
        from core.llm import make_llm
        eng = make_llm()
        return eng, getattr(eng, "name", f"local:{model}")
    from core.llm import make_llm
    eng = make_llm()
    return eng, getattr(eng, "name", "local")


_MODEL_FAMILIES = ("llama", "qwen", "mixtral", "mistral", "gemma", "deepseek",
                   "gpt", "claude", "phi", "command")


def _families(label: str) -> set[str]:
    ll = label.lower()
    return {f for f in _MODEL_FAMILIES if f in ll}


def _check_judge_independence(meta: dict, judge_label: str, force: bool) -> bool:
    """Exact-name overlap refuses outright; FAMILY overlap (e.g. a Llama-3.3
    judge scoring Llama-3.1 conditions) warns loudly — same-family judges
    show 10-25% self-preference in the 2026 judge-bias literature, and B/D
    are typically Llama variants here."""
    jl = judge_label.lower()
    condition_models = {
        str(meta.get("model_M") or "").lower(),
        str(meta.get("strong_model") or "").lower(),
    }
    for m in condition_models:
        if m and m in jl:
            print(f"[judge] WARNING: judge model {judge_label!r} overlaps a "
                  f"condition model {m!r} — self-enhancement bias risk.",
                  file=sys.stderr)
            if not force:
                print("[judge] refusing; pass --force to override or pick a "
                      "different --judge-model.", file=sys.stderr)
                return False
    jf = _families(judge_label)
    for m in condition_models:
        shared = jf & _families(m)
        if shared:
            print(f"[judge] WARNING: judge {judge_label!r} shares model "
                  f"family {sorted(shared)} with condition model {m!r} — "
                  f"self-preference bias risk (prefer an off-family judge).",
                  file=sys.stderr)
    return True


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Blind judge + report for delta exp")
    ap.add_argument("exp_dir", help="eval/results/<name> directory")
    ap.add_argument("--judge-model", default=None,
                    help="Groq model name for the judge (different family!)")
    ap.add_argument("--force", action="store_true",
                    help="judge even if model overlaps a condition")
    ap.add_argument("--pack", action="store_true",
                    help="assemble a BLIND judging packet for you+Claude to "
                         "grade by hand (no judge model); writes judging_packet.md"
                         " + pack_key.json + verdicts.json template")
    ap.add_argument("--score-verdicts", default=None, metavar="VERDICTS_JSON",
                    help="ingest a filled verdicts.json (from --pack) and "
                         "assemble report.md — no LLM call")
    ap.add_argument("--seed", type=int, default=1234,
                    help="shuffle seed for --pack (recorded in pack_key.json)")
    ap.add_argument("--max-chars", type=int, default=60000,
                    help="--pack: max chars per packet PART (split to avoid "
                         "truncating a large packet mid-item; default 60000)")
    args = ap.parse_args(argv)

    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_absolute():
        exp_dir = _REPO / exp_dir
    meta = json.loads((exp_dir / "run_meta.json").read_text(encoding="utf-8"))
    rows = load_rows(exp_dir)
    idx = _index(rows)

    pairs = [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("A", "F")]
    pairs = [(l, r) for (l, r) in pairs
             if any(l in c and r in c for c in idx.values())]

    # Mode 1: assemble the packet for hand-judging (Claude-cowork). No LLM.
    if args.pack:
        part_paths = build_pack(exp_dir, meta, idx, pairs, seed=args.seed,
                                max_chars=args.max_chars)
        for p in part_paths:
            print(f"[judge] wrote {p}")
        print(f"[judge] wrote {exp_dir / 'pack_key.json'} (hidden key — don't "
              f"read while judging)")
        print(f"[judge] wrote {exp_dir / 'verdicts.json'} (fill in 1/2/tie, "
              f"then re-run with --score-verdicts verdicts.json)")
        return 0

    # Mode 2: assemble the report from filled verdicts. No LLM.
    if args.score_verdicts:
        vpath = Path(args.score_verdicts)
        if not vpath.is_absolute():
            vpath = exp_dir / vpath.name
        pair_results = score_from_verdicts(exp_dir, idx, vpath)
        facts = factual_table(rows)
        report = write_report(exp_dir, meta, pair_results, facts)
        print(f"[judge] wrote {report}")
        for key, r in pair_results.items():
            wl = r["wilson95"]
            print(f"[judge] {key}: win-rate {r['win_rate']:.0%} "
                  f"Wilson[{wl[0]:.2f},{wl[1]:.2f}] real_win={r['real_win']}")
        return 0

    # Mode 3 (default): automated LLM judge.
    judge_llm, judge_label = _make_judge(args.judge_model)
    if not _check_judge_independence(meta, judge_label, args.force):
        return 2
    print(f"[judge] judge = {judge_label}")
    # Record judge identity in the artifacts — a result whose judge is
    # unknown can't be audited for self-preference later.
    meta["judge_model"] = judge_label

    pair_results = asyncio.run(judge_pairs(idx, pairs, judge_llm, meta))
    facts = factual_table(rows)

    report = write_report(exp_dir, meta, pair_results, facts)
    print(f"[judge] wrote {report}")
    print(f"[judge] wrote {exp_dir / 'scores.json'}")
    for key, r in pair_results.items():
        wl = r["wilson95"]
        print(f"[judge] {key}: win-rate {r['win_rate']:.0%} "
              f"Wilson[{wl[0]:.2f},{wl[1]:.2f}] real_win={r['real_win']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
