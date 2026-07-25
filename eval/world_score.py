"""Mechanical scorer for the Stage-1 synthetic-world eval.

Implements `docs/future/STAGE1_SYNTHETIC_EVAL_SPEC.md` Sec 3: every
checklist item is derived MECHANICALLY from a `FactRegistry` (never
hand-written) and scored with regex/substring matching — no LLM judge in
the headline metric (THEFUTURE.md Sec 3 / Sec 5 rule 3). `eval/judge.py`'s
pairwise LLM judge is untouched and stays the secondary prose comparison;
this module never imports it for scoring, only for the shared Wilson-CI /
pair-summary math (`wilson`, `_summarize_pair`) so both reports use
identical statistics.

Four item types (spec Sec 3.1):
  atomic_fact_present   - entity/event name or alias appears in the answer
  quantity_correct      - a number near the metric's keywords matches the
                           registered value within tolerance_pct
  contradiction_surfaced- both sides of a registered contradiction are
                           anchored in the answer AND a disagreement cue
                           word is present nearby
  citation_grounding     - (condition A only) each footnote in answer.txt's
                           "**Sources**" block actually overlaps some
                           document in the world's rendered corpus

CLI
---
    python -m eval.world_score eval/results/<exp>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval.worlds import (   # noqa: E402
    Contradiction, Entity, Event, FactRegistry, Quantity,
    load_rendered_docs, load_world,
)
from eval.judge import _summarize_pair, wilson  # noqa: E402  (shared stats, no LLM here)


# ---------------------------------------------------------------------------
# Normalization (spec Sec 3.2 — same discipline as eval/judge.py:factual_score)
# ---------------------------------------------------------------------------

def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[,$%]", " ", s)
    s = re.sub(r"-", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


_NUMBER_TOKEN_RE = re.compile(r"\d[\d,]*\.?\d*")
_CUE_WORDS = ("disagree", "disagreement", "conflict", "contradict", "dispute",
             "mixed", "inconsistent", "differ")
_WINDOW_CHARS = 80  # keyword-anchoring window for quantity_correct


# ---------------------------------------------------------------------------
# atomic_fact_present
# ---------------------------------------------------------------------------

def atomic_fact_present(answer: str, entity: Entity) -> bool:
    """Normalized substring match against the entity's name OR any
    registered alias (spec Sec 3.1)."""
    low = _normalize_text(answer)
    forms = [entity.name] + list(entity.aliases)
    return any(_normalize_text(f) in low for f in forms if f)


# ---------------------------------------------------------------------------
# quantity_correct
# ---------------------------------------------------------------------------

def _keyword_windows(answer: str, keywords: list[str], window: int = _WINDOW_CHARS
                     ) -> list[str]:
    low = answer.lower()
    windows: list[str] = []
    for kw in keywords:
        kw = (kw or "").lower().strip()
        if not kw:
            continue
        start = 0
        while True:
            idx = low.find(kw, start)
            if idx == -1:
                break
            lo = max(0, idx - window)
            hi = min(len(answer), idx + len(kw) + window)
            windows.append(answer[lo:hi])
            start = idx + len(kw)
    return windows


def quantity_correct(answer: str, quantity: Quantity,
                     subject_name: Optional[str] = None) -> tuple[bool, str]:
    """A number present but outside `tolerance_pct` is a MISS, not a partial
    hit (spec Sec 3.1 — silently rounding correctness would hide the exact
    fabrication failure mode this eval exists to catch). Tolerance boundary
    is inclusive: a value at exactly `tolerance_pct` away counts as a hit."""
    if not answer:
        return False, "empty answer"
    # Fast path: a registered surface form appears verbatim (or near-verbatim
    # after normalization) — same discipline as verify_render's check.
    for sf in quantity.surface_forms:
        if sf in answer:
            return True, f"surface form {sf!r} present"
    na = _normalize_text(answer)
    for sf in quantity.surface_forms:
        if _normalize_text(sf) in na:
            return True, f"normalized surface form {sf!r} present"

    keywords = [quantity.metric.replace("_", " "), quantity.unit]
    if subject_name:
        keywords.append(subject_name)
    windows = _keyword_windows(answer, keywords) or [answer]
    if quantity.value == 0:
        return False, "quantity value is 0 (cannot compute tolerance ratio)"
    for w in windows:
        for tok in _NUMBER_TOKEN_RE.findall(w):
            try:
                val = float(tok.replace(",", ""))
            except ValueError:
                continue
            pct_diff = abs(val - quantity.value) / abs(quantity.value) * 100.0
            # Tiny epsilon absorbs float rounding noise at the boundary (spec
            # Sec 6.1: "value at exactly tolerance_pct should hit") without
            # loosening the tolerance in any way that matters for scoring.
            if pct_diff <= quantity.tolerance_pct + 1e-9:
                return True, (f"found {tok} (parsed {val}) within "
                             f"{quantity.tolerance_pct}% of {quantity.value}")
    return False, "not found / out of tolerance"


# ---------------------------------------------------------------------------
# contradiction_surfaced / contradiction_resolved
# ---------------------------------------------------------------------------

def _anchor_hit(answer: str, fact) -> bool:
    """Does the answer contain an anchor to this fact's value (Quantity) or
    description (Event)? Used as one half of the dual-anchor requirement."""
    if isinstance(fact, Quantity):
        if any(sf in answer for sf in fact.surface_forms):
            return True
        na = _normalize_text(answer)
        return any(_normalize_text(sf) in na for sf in fact.surface_forms)
    if isinstance(fact, Event):
        return fact.description.lower() in answer.lower()
    return False


def contradiction_surfaced(answer: str, contradiction: Contradiction,
                           registry: FactRegistry) -> bool:
    """Hit requires BOTH: (a) anchors for both fact_id_a and fact_id_b's
    values, AND (b) a disagreement-marking cue word present anywhere in the
    answer (spec Sec 3.1). A cue word alone with no anchor to either fact's
    value must NOT count as a hit — this is the dual-anchor requirement the
    spec's test plan calls out explicitly."""
    fidx = registry.fact_index()
    fa = fidx.get(contradiction.fact_id_a)
    fb = fidx.get(contradiction.fact_id_b)
    if fa is None or fb is None:
        return False
    anchor_a = _anchor_hit(answer, fa)
    anchor_b = _anchor_hit(answer, fb)
    cue = any(w in answer.lower() for w in _CUE_WORDS)
    return bool(anchor_a and anchor_b and cue)


def contradiction_resolved(answer: str, contradiction: Contradiction,
                          registry: FactRegistry) -> Optional[bool]:
    """Stricter, separate item (spec Sec 3.1): did the answer's stated value
    match the GROUND-TRUTH fact rather than the false one? Only emitted
    (non-None) when `ground_truth != "genuinely_ambiguous"`."""
    if contradiction.ground_truth == "genuinely_ambiguous":
        return None
    fidx = registry.fact_index()
    gt = fidx.get(contradiction.ground_truth)
    if gt is None:
        return None
    return _anchor_hit(answer, gt)


# ---------------------------------------------------------------------------
# citation_grounding — condition A only (spec Sec 3.1). Deviation: the spec
# describes resolving a cited signal/chunk id to its source_tag via
# `extracted_from`; citations.json (agents/synthesizer.py:_build_citations)
# does not carry a chunk's source_tag, only chunk_ids, so that join isn't
# available without extra run-time plumbing. Instead this reads the
# reader-facing "**Sources**" block core/clean_answer.py appends to
# answer.txt (raw, pre- eval.judge.normalize() text, as required) and checks
# each cited excerpt for word-level n-gram overlap against ANY document in
# the world's rendered corpus — a coarser-than-specced precision check
# (source-agnostic instead of per-doc), but still a real grounding check:
# an excerpt that overlaps nothing in the corpus is unambiguously fabricated
# or hallucinated support.
# ---------------------------------------------------------------------------

_SOURCES_BLOCK_RE = re.compile(r"\*\*Sources\*\*\s*\n+(.*)", re.DOTALL)
_SOURCES_LINE_RE = re.compile(r"^\[(\d+)\]\s+(.*)$", re.MULTILINE)
_NGRAM_N = 4
_NGRAM_OVERLAP_THRESHOLD = 0.5


def parse_sources_block(answer_text: str) -> dict[int, str]:
    """Extract {footnote_number: excerpt_text} from answer.txt's Sources
    block (core/clean_answer.py:_extract_referenced_sources)."""
    m = _SOURCES_BLOCK_RE.search(answer_text or "")
    if not m:
        return {}
    out: dict[int, str] = {}
    for num_str, content in _SOURCES_LINE_RE.findall(m.group(1)):
        try:
            out[int(num_str)] = content.strip()
        except ValueError:
            continue
    return out


def _word_ngrams(text: str, n: int = _NGRAM_N) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def citation_grounds_claim(excerpt: str, doc_texts: dict[str, str],
                          threshold: float = _NGRAM_OVERLAP_THRESHOLD
                          ) -> tuple[bool, str]:
    """4-gram word-overlap ratio between `excerpt` and each candidate
    document; hit if the best-matching document clears `threshold`."""
    ex_grams = _word_ngrams(excerpt)
    if not ex_grams:
        return False, "empty excerpt"
    best_doc, best_ratio = "", 0.0
    for doc_id, text in doc_texts.items():
        doc_grams = _word_ngrams(text)
        if not doc_grams:
            continue
        overlap = len(ex_grams & doc_grams) / len(ex_grams)
        if overlap > best_ratio:
            best_ratio, best_doc = overlap, doc_id
    return best_ratio >= threshold, f"best overlap {best_ratio:.2f} vs {best_doc or '(none)'}"


def score_citation_grounding(answer_text: str, doc_texts: dict[str, str]) -> list[dict]:
    """One item per footnote referenced in answer_text's Sources block.
    Empty list (not an error) for conditions with no Sources block (B/C/D/E/F
    never emit one) — matches the spec's `citation_grounding: null` for F."""
    sources = parse_sources_block(answer_text)
    rows: list[dict] = []
    for n, excerpt in sources.items():
        hit, detail = citation_grounds_claim(excerpt, doc_texts)
        rows.append({"item_id": f"citation_{n}", "item_type": "citation_grounding",
                    "hit": hit, "detail": detail})
    return rows


# ---------------------------------------------------------------------------
# Checklist assembly + per-item dispatch
# ---------------------------------------------------------------------------

def build_checklist_items(registry: FactRegistry) -> list[dict]:
    """Every item mechanically derived from `registry` — no hand-written
    items (spec Sec 3.1). `quantity_correct` items exclude contradiction
    clones (the deliberately-false side of a Contradiction, `fact_id_b`):
    those aren't "the correct answer" a grounded reader should state, they
    exist only to be surfaced as a conflict via `contradiction_surfaced`."""
    false_clone_ids = {c.fact_id_b for c in registry.contradictions}
    items: list[dict] = []
    for e in registry.entities:
        items.append({"item_id": f"atomic_{e.entity_id}",
                      "item_type": "atomic_fact_present", "entity_id": e.entity_id})
    for q in registry.quantities:
        if q.fact_id in false_clone_ids:
            continue
        items.append({"item_id": f"quantity_{q.fact_id}",
                      "item_type": "quantity_correct", "fact_id": q.fact_id})
    for c in registry.contradictions:
        items.append({"item_id": f"contradiction_{c.contradiction_id}",
                      "item_type": "contradiction_surfaced",
                      "contradiction_id": c.contradiction_id})
        if c.ground_truth != "genuinely_ambiguous":
            items.append({"item_id": f"contradiction_resolved_{c.contradiction_id}",
                         "item_type": "contradiction_resolved",
                         "contradiction_id": c.contradiction_id})
    return items


def score_item(answer: str, item: dict, registry: FactRegistry
              ) -> tuple[Optional[bool], str]:
    """Dispatch a single checklist item to its scorer. Returns (hit, detail);
    hit is None for an item type that's structurally not applicable (e.g.
    `contradiction_resolved` is only ever emitted for applicable items, so
    None here signals an internal lookup failure, not a normal skip)."""
    it = item["item_type"]
    if it == "atomic_fact_present":
        e = registry.entity_index().get(item["entity_id"])
        if e is None:
            return None, "entity not found in registry"
        hit = atomic_fact_present(answer, e)
        return hit, ("present" if hit else "missing")
    if it == "quantity_correct":
        q = registry.fact_index().get(item["fact_id"])
        if q is None:
            return None, "fact not found in registry"
        subj = registry.entity_index().get(q.subject_id)
        return quantity_correct(answer, q, subject_name=subj.name if subj else None)
    if it == "contradiction_surfaced":
        c = next((x for x in registry.contradictions
                 if x.contradiction_id == item["contradiction_id"]), None)
        if c is None:
            return None, "contradiction not found in registry"
        hit = contradiction_surfaced(answer, c, registry)
        return hit, ("dual anchor + cue present" if hit else "missing anchor or cue")
    if it == "contradiction_resolved":
        c = next((x for x in registry.contradictions
                 if x.contradiction_id == item["contradiction_id"]), None)
        if c is None:
            return None, "contradiction not found in registry"
        hit = contradiction_resolved(answer, c, registry)
        return hit, ("matched ground truth" if hit else "did not match ground truth")
    return None, f"unknown item type {it!r}"


def score_answer(answer: str, registry: FactRegistry) -> list[dict]:
    """All non-citation checklist items scored against one answer."""
    rows = []
    for item in build_checklist_items(registry):
        hit, detail = score_item(answer, item, registry)
        rows.append({"item_id": item["item_id"], "item_type": item["item_type"],
                    "hit": hit, "detail": detail})
    return rows


def compute_world_scores(answer: str, registry: FactRegistry,
                         item_types: Optional[tuple[str, ...]] = None) -> float:
    """Mean hit rate over the given item types (default: all). Used by Gate 1
    (`eval/worlds.py:_mechanical_b_score`) restricted to
    `atomic_fact_present` + `quantity_correct` — the two item types a
    memory-only answer could plausibly satisfy by chance (spec Sec 4.1)."""
    rows = score_answer(answer, registry)
    if item_types:
        rows = [r for r in rows if r["item_type"] in item_types]
    scorable = [r for r in rows if r["hit"] is not None]
    if not scorable:
        return 0.0
    return sum(1 for r in scorable if r["hit"]) / len(scorable)


# ---------------------------------------------------------------------------
# CLI: read conditions.jsonl + run_meta.json, write world_scores.jsonl +
# world_report.md (spec Sec 3.3, Sec 5.2).
# ---------------------------------------------------------------------------

def _overall_score(rows: list[dict]) -> Optional[float]:
    scorable = [r for r in rows if r["hit"] is not None]
    if not scorable:
        return None
    return sum(1 for r in scorable if r["hit"]) / len(scorable)


def _per_item_type(rows: list[dict]) -> dict:
    by_type: dict[str, list[bool]] = {}
    for r in rows:
        if r["hit"] is None:
            continue
        by_type.setdefault(r["item_type"], []).append(bool(r["hit"]))
    out = {t: round(sum(v) / len(v), 4) for t, v in by_type.items()}
    out["n_items"] = sum(len(v) for v in by_type.values())
    return out


def score_experiment(exp_dir: Path) -> tuple[list[dict], dict]:
    """Reads conditions.jsonl + run_meta.json from `exp_dir`, scores every
    row against its world's FactRegistry, and returns (world_scores_rows,
    world_report_dict) without writing anything (writing is the CLI's job,
    so tests can call this directly)."""
    meta = json.loads((exp_dir / "run_meta.json").read_text(encoding="utf-8"))
    world_seed = meta.get("world_seed")
    if world_seed is None:
        raise ValueError(
            f"{exp_dir}/run_meta.json has no 'world_seed' - this experiment "
            f"wasn't run with --prompt-set synthetic, so eval.world_score "
            f"has no ground truth to score against.")
    registry = load_world(world_seed)
    doc_texts: dict[str, str] = {}
    try:
        doc_texts = {d.doc_id: d.text for d in load_rendered_docs(world_seed)}
    except FileNotFoundError:
        pass  # citation_grounding items simply score empty for every row

    rows = [json.loads(ln) for ln in
           (exp_dir / "conditions.jsonl").read_text(encoding="utf-8").splitlines()
           if ln.strip()]

    world_scores: list[dict] = []
    per_condition_rows: dict[str, list[dict]] = {}
    per_prompt_overall: dict[str, dict[str, float]] = {}   # pid -> condition -> score
    for r in rows:
        pid, cond, answer = r["pid"], r["condition"], r.get("answer", "")
        item_rows = score_answer(answer, registry)
        if cond == "A":
            item_rows = item_rows + score_citation_grounding(answer, doc_texts)
        for ir in item_rows:
            world_scores.append({"pid": pid, "condition": cond, **ir})
        per_condition_rows.setdefault(cond, []).extend(item_rows)
        overall = _overall_score(item_rows)
        if overall is not None:
            per_prompt_overall.setdefault(pid, {})[cond] = overall

    per_condition = {c: _per_item_type(rs) for c, rs in per_condition_rows.items()}

    paired: dict[str, dict] = {}
    conds_present = set(per_condition_rows)
    for right in ("B", "C", "D", "E", "F"):
        if "A" not in conds_present or right not in conds_present:
            continue
        per_prompt = []
        for pid, by_cond in per_prompt_overall.items():
            if "A" not in by_cond or right not in by_cond:
                continue
            a_score, r_score = by_cond["A"], by_cond[right]
            if a_score > r_score:
                winner = "A"
            elif r_score > a_score:
                winner = right
            else:
                winner = "tie"
            per_prompt.append({"pid": pid, "winner": winner, "a_score": round(a_score, 4),
                              "opponent_score": round(r_score, 4)})
        if per_prompt:
            paired[f"A_vs_{right}"] = _summarize_pair("A", right, per_prompt, [])

    report = {
        "world_seed": world_seed,
        "template": meta.get("world_template"),
        "scale": meta.get("pack_scale"),
        "per_condition": per_condition,
        "paired_win_rate": paired,
        "mock": bool(meta.get("mock")),
    }
    return world_scores, report


def write_world_report(exp_dir: Path, report: dict) -> Path:
    lines = [f"# Synthetic-world mechanical score report — `{exp_dir.name}`\n"]
    if report.get("mock"):
        lines.append("> WARNING: MOCK run - plumbing only, NOT empirical (P0.1). "
                     "Scores below prove plumbing, not fact-grounding behavior.\n")
    lines.append(f"- world seed: `{report.get('world_seed')}`")
    lines.append(f"- template: `{report.get('template')}`")
    lines.append(f"- scale: `{report.get('scale')}`\n")

    lines.append("## Per-condition mechanical scores\n")
    lines.append("| condition | atomic_fact_present | quantity_correct | "
                 "contradiction_surfaced | contradiction_resolved | "
                 "citation_grounding | n_items |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for cond, scores in sorted(report.get("per_condition", {}).items()):
        def _fmt(k):
            v = scores.get(k)
            return f"{v:.0%}" if v is not None else "-"
        lines.append(f"| {cond} | {_fmt('atomic_fact_present')} | "
                     f"{_fmt('quantity_correct')} | {_fmt('contradiction_surfaced')} | "
                     f"{_fmt('contradiction_resolved')} | {_fmt('citation_grounding')} | "
                     f"{scores.get('n_items', 0)} |")
    lines.append("")

    lines.append("## Paired mechanical win rate (headline metric)\n")
    lines.append("| pair | n | win | tie | loss | win-rate | Wilson 95% | real win? |")
    lines.append("|---|--:|--:|--:|--:|--:|---|:--:|")
    for key, r in sorted(report.get("paired_win_rate", {}).items()):
        wl = r["wilson95"]
        lines.append(f"| {key} | {r['n']} | {r['wins']} | {r['ties']} | {r['losses']} | "
                     f"{r['win_rate']:.0%} | [{wl[0]:.2f}, {wl[1]:.2f}] | "
                     f"{'**YES**' if r['real_win'] else 'no'} |")
    lines.append("\n> A real win = Wilson lower bound clearly above 50% "
                 "(THEFUTURE.md Sec 4 Stage 1 decision gate).\n")

    path = exp_dir / "world_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Mechanical scorer for synthetic-world experiments")
    ap.add_argument("exp_dir", help="eval/results/<name> directory")
    args = ap.parse_args(argv)
    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_absolute():
        exp_dir = _REPO / exp_dir

    world_scores, report = score_experiment(exp_dir)
    (exp_dir / "world_scores.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in world_scores) + "\n",
        encoding="utf-8")
    report_path = write_world_report(exp_dir, report)
    print(f"[world_score] wrote {exp_dir / 'world_scores.jsonl'} "
         f"({len(world_scores)} rows)")
    print(f"[world_score] wrote {report_path}")
    for key, r in report.get("paired_win_rate", {}).items():
        wl = r["wilson95"]
        print(f"[world_score] {key}: win-rate {r['win_rate']:.0%} "
             f"Wilson[{wl[0]:.2f},{wl[1]:.2f}] real_win={r['real_win']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
