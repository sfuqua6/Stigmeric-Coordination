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


def test_strips_brief_scaffolding_but_keeps_citation_markers():
    # "Brief N" is leaked render scaffolding (a per-cluster brief label) and
    # is stripped. [N]-style citation markers are kept in place — the reader
    # answer now preserves provenance instead of stripping all attribution.
    reader, _ = split_answer(FULL)
    assert "Brief 1" not in reader and "Brief" not in reader
    assert "[1]" in reader and "[2]" in reader and "[5]" in reader


def test_appends_sources_block_for_referenced_footnotes():
    # The reader answer gets a compact "Sources" block built from the
    # synthesizer's "**Sources referenced above:**" appendix, but only for
    # footnote numbers actually kept in the Section 1/2 prose.
    reader, _ = split_answer(FULL)
    assert "**Sources**" in reader
    assert "[1] However, such a measure" in reader
    # Sections 3/4 telemetry (support_diversity, composite_fitness, etc.)
    # must still not leak into the reader answer via the Sources block.
    assert "support_diversity=" not in reader
    assert "composite_fitness" not in reader


def test_leading_exec_summary_goes_to_diagnostics():
    txt = "## EXECUTIVE SUMMARY\n\nOf 39 clusters, 10 survived...\n\n" + FULL
    reader, diag = split_answer(txt)
    assert "EXECUTIVE SUMMARY" not in reader
    assert "EXECUTIVE SUMMARY" in diag


def test_fallback_when_no_section1():
    txt = "A plain answer with no section headers."
    reader, diag = split_answer(txt)
    assert reader == txt and diag == ""


# Cohesive-strategy artifacts have NO "## 1." header: reader prose first, then
# telemetry. The reader portion must still be split out from PROCESS NOTES/etc.
COHESIVE = (
    "The case for banning private cars is strong but context-dependent: it "
    "works in dense cities with transit alternatives, and risks displacing "
    "trips to ride-hailing where it does not. The honest position is conditional "
    "adoption with measured rollout and explicit equity safeguards for those "
    "without alternatives.\n\n"
    "---\n\n"
    "## PROCESS NOTES\n\n"
    "Of 48 claim clusters: 11 survived ... composite_fitness=0.498\n\n"
    "## 4. CITATIONS\n"
    "============================================================\n"
    "CLAIM  [INITIAL_00012]: ...  support_diversity=19\n"
)


def test_cohesive_headerless_split():
    reader, diag = split_answer(COHESIVE)
    assert "banning private cars" in reader
    for bad in ("PROCESS NOTES", "CITATIONS", "INITIAL_00012",
                "support_diversity=", "composite_fitness"):
        assert bad not in reader, f"leaked into reader: {bad}"
    assert not reader.rstrip().endswith("---")  # trailing rule stripped
    assert "PROCESS NOTES" in diag and "INITIAL_00012" in diag
