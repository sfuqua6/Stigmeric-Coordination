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

write_answer_files() is the wiring helper used at every answer.txt write site in
run_swarm.py: it honors SWARM_SYNTH_VERBOSE (restore the old combined output) and
writes diagnostics.md alongside the clean answer.

Pure post-processor: does not change how sections are built, so it is low-risk
and reversible (SWARM_SYNTH_VERBOSE=1 restores the combined output).

Section headers are kept in sync with agents/synthesizer.py (verified against the
literal headers it emits: "## 1. POSITION SYNTHESIS", "## 2. OPEN QUESTIONS AND
DISSENT", "## 3. CONSIDERED AND FILTERED", "## PROCESS NOTES",
"**Sources referenced above:**", "## 4. CITATIONS").
"""
from __future__ import annotations

import os
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
# Trailing horizontal rule the cohesive path leaves before the telemetry block.
_TRAIL_RULE = re.compile(r"\s*-{3,}\s*$")

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
        # No "## 1. POSITION SYNTHESIS" header. Two cases:
        #   (a) a COHESIVE-strategy artifact (cohesive_exploration/optimization):
        #       reader prose FIRST, then telemetry (---/PROCESS NOTES/CITATIONS).
        #       The reader portion is everything before the first telemetry marker.
        #   (b) a genuinely plain answer with no telemetry — ship it untouched.
        head = ""
        body = full
    else:
        head = full[:start]        # leading EXEC SUMMARY etc. -> diagnostics
        body = full[start:]

    cut = len(body)
    for mk in _DIAG_MARKERS:
        i = body.find(mk)
        if i != -1:
            cut = min(cut, i)

    if start == -1 and cut == len(body):
        # Case (b): headerless and no telemetry marker — don't risk mangling.
        return full.strip(), ""

    reader = _SCAFFOLD.sub("", body[:cut])
    reader = _SPACE_BEFORE_PUNCT.sub(r"\1", reader)
    reader = _MULTI_WS.sub(" ", reader)
    reader = _TRAIL_RULE.sub("", reader).strip()

    diagnostics = (head + "\n\n" + body[cut:]).strip()

    if len(reader) < _MIN_READER_CHARS:
        # Degenerate Section 1 — better to ship the full text than an empty
        # answer. (Surface this in logs at the call site.)
        return full.strip(), diagnostics
    return reader, diagnostics


def synth_verbose() -> bool:
    """True when SWARM_SYNTH_VERBOSE asks for the old combined answer.txt."""
    return os.environ.get("SWARM_SYNTH_VERBOSE", "").strip() not in (
        "", "0", "false", "False")


_DIAG_HEADER = (
    "# Synthesis diagnostics — NOT part of the answer\n\n"
    "The reader-facing answer is in `answer.txt`. Everything below is the "
    "swarm's internal field telemetry (filtered clusters, process notes, "
    "the citation graph). Useful for debugging, not for a reader.\n\n---\n\n"
)


def write_answer_files(out_dir, final_answer: str) -> None:
    """Write answer.txt (+ diagnostics.md) for a run.

    Default: answer.txt = reader-facing Section 1 (+2); telemetry -> diagnostics.md.
    SWARM_SYNTH_VERBOSE=1: restore the old combined answer.txt, no diagnostics.md.

    Pure I/O wrapper around split_answer — centralized so every write site in
    run_swarm.py shares one implementation and the section markers don't drift.
    """
    if synth_verbose():
        (out_dir / "answer.txt").write_text(final_answer, encoding="utf-8")
        return

    reader, diagnostics = split_answer(final_answer)
    if len(reader) < _MIN_READER_CHARS:
        print("[synth] WARNING: cleaned reader answer is very short — "
              "shipping full text; check Section 1 rendering.")
    (out_dir / "answer.txt").write_text(reader, encoding="utf-8")
    if diagnostics:
        (out_dir / "diagnostics.md").write_text(
            _DIAG_HEADER + diagnostics, encoding="utf-8")
