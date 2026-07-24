# Claude Code Brief — Ship the Answer, Not the Debug Log

**Repo:** root pipeline (`run_swarm.py`, `agents/synthesizer.py`, `eval/`) — the one `eval/ab_harness.py` actually runs. (Apply the same change to `Attempt At Cleaning/` only if you want parity; the eval targets root.)

**One-line mission:** make the swarm's `answer.txt` contain only the reader-facing argument (Section 1 POSITION SYNTHESIS, plus Section 2 OPEN QUESTIONS if present). Route all field telemetry — Section 3 CONSIDERED AND FILTERED, PROCESS NOTES, truncated "Sources referenced above", Section 4 CITATIONS, any leading EXECUTIVE SUMMARY — to a separate `diagnostics.md`. Strip leaked `Brief N` / `[1]` render scaffolding from the prose. Keep the old behavior behind a flag.

---

## 1. Why (evidence)

A blind, neutral judge (different model family) read the captured eval answers in `eval/results/.../conditions.jsonl` and found:

- The swarm's actual argument (Section 1) is **competitive** with a single direct call of the same 7B, and on "ban_cars" brought the sharpest framing of all four conditions (context-dependence; displacement to ride-hailing).
- But as **delivered**, each swarm answer is ~400 words of essay followed by **~1,000–1,400 words of machine exhaust**: `(rejected: dissent_pressure=19.99 > 1.5)`, `PROCESS NOTES … composite_fitness=0.498`, `CLAIM [INITIAL_00012] … Supports: SUPPORT_00192 … worker_023 … support_diversity=19`, plus truncated source fragments.
- Two visible defects in the prose: a **leaked scaffolding label** ("as highlighted by **Brief 1** ([1])", "(Brief 6)") and an **off-topic contamination** — a claim about **AAAI** (an AI org) rendered into a *"does God exist"* answer (item `[12]`, in Section 3).
- Net: the swarm is the **worst artifact** of the four every time, purely because of delivery — not reasoning. The single-call answers (direct 7B, best-of-5 revise, Llama-70B) all read better.

**Therefore:** the highest-leverage fix is not the orchestration — it's the synthesizer's final output. If `answer.txt` is just Section 1 (+2), the swarm goes from "obviously worst" to "competitive on quality," and the contamination disappears (it lives in Section 3, which we drop). The AAAI item and all `support_diversity=` noise are already in the dropped region — this fix removes them for free.

---

## 2. Design decision

- **Reader answer** (`answer.txt`): Section 1 `POSITION SYNTHESIS` and, if present, Section 2 `OPEN QUESTIONS AND DISSENT`. Inline `Brief N` and `[N]` citation scaffolding stripped (the source list they point to is being removed and is low quality anyway).
- **Diagnostics** (`diagnostics.md`): everything from the first telemetry marker onward, verbatim, plus any leading EXECUTIVE SUMMARY. Nothing is lost — it moves.
- **Escape hatch:** `SWARM_SYNTH_VERBOSE=1` (or `--synth-verbose`) restores the old combined output in `answer.txt`.
- **Low risk:** this is a pure post-processor on the assembled answer string. It does **not** touch how the synthesizer builds sections, the faithfulness audit, or `citations.json`. Fully reversible.

---

## 3. Task A — the splitter module (full code)

Create `core/clean_answer.py`:

```python
"""Reader-facing answer extraction.

The synthesizer emits one combined string: a reader-facing argument
(Section 1 POSITION SYNTHESIS, optionally Section 2 OPEN QUESTIONS) followed by
internal telemetry (Section 3 CONSIDERED AND FILTERED, PROCESS NOTES, truncated
"Sources referenced above", Section 4 CITATIONS, and any leading EXECUTIVE
SUMMARY). Judges and humans only want the argument; the telemetry makes the
deliverable 3-4x longer and reads as machine exhaust.

split_answer() returns (reader, diagnostics):
    reader      -> answer.txt      (Sections 1 [+2]; Brief/[N] scaffolding stripped)
    diagnostics -> diagnostics.md  (everything else, verbatim)

Pure post-processor: does not change how sections are built, so it is low-risk
and reversible (SWARM_SYNTH_VERBOSE=1 restores the combined output).
"""
from __future__ import annotations
import re

# Telemetry region begins at the FIRST of these markers that appears after
# Section 1. Keep these in sync with agents/synthesizer.py section headers.
_DIAG_MARKERS = (
    "## 3. CONSIDERED AND FILTERED",
    "## PROCESS NOTES",
    "**Sources referenced above:**",
    "## 4. CITATIONS",
)
# Section 1 start. A leading EXECUTIVE SUMMARY (if any) is telemetry -> diagnostics.
_SEC1 = ("## 1.", "POSITION SYNTHESIS")

# Leaked render scaffolding to strip from reader prose:
#   "(Brief 6)" / "Brief 1"   — per-cluster brief labels the composer echoed
#   "([1])" / "[1][2]"        — numeric citation tags whose source list we drop
_SCAFFOLD = re.compile(
    r"""\s*(?:
            \(?\bBrief\s+\d+\b\)?
          | \(\s*\[\d+\]\s*\)
          | \[\d+\]
        )""",
    re.IGNORECASE | re.VERBOSE,
)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:)])")
_MULTI_WS = re.compile(r"[ \t]{2,}")

_MIN_READER_CHARS = 200  # never ship an emptier answer than this


def split_answer(full: str) -> tuple[str, str]:
    """Return (reader_answer, diagnostics)."""
    if not full or not full.strip():
        return "", ""

    # Locate Section 1.
    start = -1
    for mk in _SEC1:
        i = full.find(mk)
        if i != -1:
            start = i if start == -1 else min(start, i)
    if start == -1:
        # No recognizable Section 1 header — don't risk mangling; ship as-is.
        return full.strip(), ""

    head = full[:start]            # leading EXEC SUMMARY etc. -> diagnostics
    body = full[start:]

    cut = len(body)
    for mk in _DIAG_MARKERS:
        i = body.find(mk)
        if i != -1:
            cut = min(cut, i)

    reader = _SCAFFOLD.sub("", body[:cut])
    reader = _SPACE_BEFORE_PUNCT.sub(r"\1", reader)
    reader = _MULTI_WS.sub(" ", reader).strip()

    diagnostics = (head + "\n\n" + body[cut:]).strip()

    if len(reader) < _MIN_READER_CHARS:
        # Degenerate Section 1 — better to ship the full text than an empty
        # answer. (Surface this in logs at the call site.)
        return full.strip(), diagnostics
    return reader, diagnostics
```

---

## 4. Task B — wire it where `answer.txt` is written

Find the write site (root): `grep -n "answer.txt" run_swarm.py`. There will be a line like
`(output_dir / "answer.txt").write_text(final_answer, encoding="utf-8")` (in `run_continuous_pipeline`, and likely also in the legacy `run_pipeline`). Replace each with:

```python
import os
from core.clean_answer import split_answer

_synth_verbose = os.environ.get("SWARM_SYNTH_VERBOSE", "").strip() not in ("", "0", "false", "False")
reader, diagnostics = split_answer(final_answer)

if _synth_verbose:
    (output_dir / "answer.txt").write_text(final_answer, encoding="utf-8")
else:
    if len(reader) < 200:
        print("[synth] WARNING: cleaned reader answer is very short — "
              "shipping full text; check Section 1 rendering.")
    (output_dir / "answer.txt").write_text(reader, encoding="utf-8")
    if diagnostics:
        (output_dir / "diagnostics.md").write_text(
            "# Synthesis diagnostics — NOT part of the answer\n\n"
            "The reader-facing answer is in `answer.txt`. Everything below is the "
            "swarm's internal field telemetry (filtered clusters, process notes, "
            "the citation graph). Useful for debugging, not for a reader.\n\n---\n\n"
            + diagnostics, encoding="utf-8")
```

**Do not change** the faithfulness audit: it must keep running on `final_answer` (the full text) so citation-tag coverage is unchanged. Confirm the audit call in `agents/synthesizer.py` precedes this write and operates on the full answer — leave it alone.

Add a `--synth-verbose` CLI flag in `main()` that sets `os.environ["SWARM_SYNTH_VERBOSE"]="1"` (mirror how other flags are parsed), so it works without an env var too.

---

## 5. Task C — (defense in depth) keep contamination out of Section 1

The AAAI item was already in Section 3 (so Task A drops it from the reader). But contamination *could* reach Section 1 on another run. Add a cheap relevance gate so off-topic clusters can't be rendered into Sections 1–2.

Locate where the synthesizer selects clusters to render into Section 1 (`grep -n "POSITION SYNTHESIS\|build_plan\|render_full\|to_render" agents/synthesizer.py`). Before a cluster is rendered, gate it on topical relevance to the task prompt:

```python
def _on_topic(cluster_text: str, task_prompt: str, store, *, min_cos=0.18, min_overlap=0.10) -> bool:
    """True if the cluster is plausibly about the task. Embedding cosine if the
    store embedder is available; else a keyword-overlap fallback. Permissive on
    purpose — only meant to catch gross contamination (e.g. an AAAI/ML claim in
    a theology answer), not to second-guess borderline-relevant claims."""
    emb = getattr(store, "_encode", None)
    if emb is not None:
        a, b = store._encode(cluster_text[:400]), store._encode(task_prompt[:400])
        if a and b:
            cos = sum(x * y for x, y in zip(a, b))  # both L2-normalized in _encode
            return cos >= min_cos
    # fallback: content-word Jaccard against the prompt
    import re
    toks = lambda s: {w for w in re.findall(r"[a-z]{4,}", s.lower())}
    ct, pt = toks(cluster_text), toks(task_prompt)
    if not ct or not pt:
        return True
    return len(ct & pt) / max(1, len(pt)) >= min_overlap
```

Filter the render set: skip clusters where `not _on_topic(rep_content, self.task_prompt, store)`, and log each drop (`print(f"[synth] dropped off-topic cluster {cid}: {rep[:60]}")`). Tune `min_cos` down if it drops legitimate clusters in your test prompts — start permissive.

---

## 6. Tests

Create `tests/test_clean_answer.py`:

```python
from core.clean_answer import split_answer

FULL = """## 1. POSITION SYNTHESIS

Cities should ban private cars, as highlighted by Brief 1 ([1]). It works in small communities ([2]).

## 2. OPEN QUESTIONS AND DISSENT

The contested claim [5] posits parking bans push traffic outward.

## 3. CONSIDERED AND FILTERED

- [12] The membership base of organizations like AAAI ... (held: support_diversity=3 < 4)

---

## PROCESS NOTES

Of 48 claim clusters: 11 survived ... composite_fitness=0.498

**Sources referenced above:**

[1] However, such a measure ...

## 4. CITATIONS
============================================================
CLAIM  [INITIAL_00012]: ...
  support_diversity=19  dissent_pressure=0.15
"""

def test_keeps_sections_1_and_2():
    reader, _ = split_answer(FULL)
    assert "POSITION SYNTHESIS" in reader
    assert "OPEN QUESTIONS" in reader
    assert "ban private cars" in reader

def test_drops_all_telemetry():
    reader, diag = split_answer(FULL)
    for bad in ("CONSIDERED AND FILTERED", "PROCESS NOTES", "CITATIONS",
                "INITIAL_00012", "support_diversity=", "composite_fitness",
                "AAAI", "Sources referenced above"):
        assert bad not in reader, f"leaked into reader: {bad}"
    assert "PROCESS NOTES" in diag and "INITIAL_00012" in diag  # preserved

def test_strips_brief_and_citation_scaffolding():
    reader, _ = split_answer(FULL)
    assert "Brief 1" not in reader and "Brief" not in reader
    assert "[1]" not in reader and "[2]" not in reader and "[5]" not in reader

def test_leading_exec_summary_goes_to_diagnostics():
    txt = "## EXECUTIVE SUMMARY\n\nOf 39 clusters, 10 survived...\n\n" + FULL
    reader, diag = split_answer(txt)
    assert "EXECUTIVE SUMMARY" not in reader
    assert "EXECUTIVE SUMMARY" in diag

def test_fallback_when_no_section1():
    txt = "A plain answer with no section headers."
    reader, diag = split_answer(txt)
    assert reader == txt and diag == ""
```

Run: `MOCK_LLM=1 pytest tests/test_clean_answer.py -q`.

---

## 7. Fast validation WITHOUT a GPU (do this first)

You already have real captured answers in the two `conditions.jsonl` files (uploaded; also under `eval/results/`). Re-clean condition A and re-judge — no swarm re-run needed. Create `tools/reclean_eval.py`:

```python
"""Re-clean the swarm (condition A) answers in a conditions.jsonl and report the
word-count collapse. Optionally re-judge A_clean vs B/C/D with the existing
both-orders judge. No GPU / no swarm run required."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.clean_answer import split_answer

def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = []
    for r in rows:
        if r.get("condition") == "A":
            before = len(r["answer"].split())
            reader, _diag = split_answer(r["answer"])
            r["answer"] = reader
            after = len(reader.split())
            print(f"  {r['pid']:12s} A: {before} -> {after} words")
        out.append(r)
    dst = Path(path).with_name(Path(path).stem + "_cleaned.jsonl")
    dst.write_text("\n".join(json.dumps(r) for r in out), encoding="utf-8")
    print(f"wrote {dst}")

if __name__ == "__main__":
    main(sys.argv[1])
```

Run it on both captured sets, then re-point `eval/judge.py` at the `_cleaned.jsonl` files (it consumes the same schema via `tools.judge_answers.judge_pair`). This tells you **today**, with no GPU, whether stripping the exhaust moves the A-vs-B / A-vs-D verdicts. Expect the cleaned A word counts to drop ~1,681→~600 (ban_cars) and ~1,828→~450 (god_exists).

> Judge caveat from the last run still applies: the judge had 100% order-disagreement (pure position bias) and rewarded length. Before trusting any re-judge, also apply the judge fixes (length-normalize, neutral non-Qwen judge, n≥20). But the *word-count collapse* and the *disappearance of `support_diversity=`/`AAAI`/`PROCESS NOTES`* from the reader answer are objective wins you can confirm immediately.

---

## 8. Full re-run + expected result

```bash
# clean answer is now the default
python run_swarm.py debate "Cities should ban private cars to fight climate change."
cat outputs/<run>/answer.txt        # Sections 1(+2) only, no telemetry, no [INITIAL_...]
cat outputs/<run>/diagnostics.md    # all the census/citations live here now

# verbose escape hatch still produces the old combined output
SWARM_SYNTH_VERBOSE=1 python run_swarm.py debate "..."
```

Then re-run the delta harness (it reads `answer.txt`, so it now compares the clean answer):
`python eval/ab_harness.py --set smoke --conditions A,B,C,D` → judge.

**Expected:** A's answer length drops to parity with B/C/D; `answer.txt` contains no `support_diversity=`, `INITIAL_`, `PROCESS NOTES`, `Brief `, or off-topic AAAI text. Quality verdict should move from "obviously worst artifact" toward "competitive." The cost multiple (127–382×) is unchanged — that remains the real verdict, and is the next thing to attack.

---

## 9. Acceptance criteria / Definition of done

1. `core/clean_answer.py` exists; `tests/test_clean_answer.py` passes.
2. Default `answer.txt` for a debate run contains Section 1 (+2 if present) and **none** of: `## 3. CONSIDERED AND FILTERED`, `## PROCESS NOTES`, `## 4. CITATIONS`, `Sources referenced above`, `[INITIAL_`, `support_diversity=`, `composite_fitness`, `Brief `.
3. `diagnostics.md` exists and contains that moved telemetry (nothing lost).
4. `SWARM_SYNTH_VERBOSE=1` / `--synth-verbose` reproduces the old combined `answer.txt`.
5. `citations.json`, `renderer_audit.json`, and the faithfulness audit are unchanged (audit still runs on the full text).
6. `tools/reclean_eval.py` runs on the captured `conditions.jsonl` and reports the word-count collapse.
7. Task C: a deliberately off-topic cluster is logged-and-skipped from Section 1 (add a small test or a logged run).

## 10. Risk / rollback

- Pure additive module + a guarded write-site change; revert by deleting `core/clean_answer.py` and restoring the one `write_text` line, or just set `SWARM_SYNTH_VERBOSE=1`.
- Main risk: a future synthesizer change renames a section header → `_DIAG_MARKERS` / `_SEC1` drift. Mitigation: the markers are centralized in one module with the sync note, and `split_answer` falls back to shipping the full text (never an empty answer) if Section 1 isn't found.
