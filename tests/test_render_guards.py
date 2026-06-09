"""Fix R — Renderer guard tests.

Tests _apply_word_budget() and _strip_hallucinated_citations() in isolation,
plus config-value assertions and the module-level sampling assertion.
All tests run without GPU or LLM.

Run with:
    pytest tests/test_render_guards.py -v
"""

import sys
import os
import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agents.synthesizer import _apply_word_budget, _strip_hallucinated_citations
from core.config import (
    RENDER_POSITION_MAX_WORDS,
    RENDER_DISSENT_MAX_WORDS,
    SAMPLING_PER_ENGINE,
)


# ---------------------------------------------------------------------------
# _apply_word_budget
# ---------------------------------------------------------------------------

def test_word_budget_under_limit_unchanged():
    text = "This is a short sentence."
    result = _apply_word_budget(text, 20, "INITIAL_00001", "position")
    assert result == text


def test_word_budget_at_limit_unchanged():
    words = ["word"] * 10
    text = " ".join(words)
    result = _apply_word_budget(text, 10, "INITIAL_00001", "position")
    assert result == text


def test_word_budget_over_limit_truncates_to_max():
    words = ["word"] * 15
    text = " ".join(words)
    result = _apply_word_budget(text, 10, "INITIAL_00001", "position")
    # Result is 10 words + optional period; strip trailing period before counting.
    result_words = result.rstrip(".").split()
    assert len(result_words) == 10, (
        f"Expected 10 words after truncation, got {len(result_words)}: {result!r}"
    )


def test_word_budget_adds_period_when_missing():
    words = ["word"] * 15
    text = " ".join(words)  # no terminal punctuation
    result = _apply_word_budget(text, 10, "INITIAL_00001", "position")
    assert result.endswith("."), (
        f"Expected terminal period after truncation, got: {result!r}"
    )


def test_word_budget_does_not_double_punctuate():
    """If the 10th word already ends with period the guard must not add another."""
    words = ["word"] * 9 + ["end."] + ["overflow"] * 5
    text = " ".join(words)
    result = _apply_word_budget(text, 10, "INITIAL_00001", "position")
    assert not result.endswith(".."), (
        f"Double period detected: {result!r}"
    )


def test_word_budget_empty_text_returns_empty():
    result = _apply_word_budget("", 10, "INITIAL_00001", "position")
    assert result == ""


def test_word_budget_logs_render_guard(capsys):
    words = ["word"] * 20
    text = " ".join(words)
    _apply_word_budget(text, 5, "INITIAL_00001", "position")
    captured = capsys.readouterr()
    assert "[RENDER GUARD]" in captured.out, (
        f"Expected [RENDER GUARD] in stdout, got: {captured.out!r}"
    )
    assert "INITIAL_00001" in captured.out


def test_word_budget_no_log_when_under_limit(capsys):
    text = "Short sentence."
    _apply_word_budget(text, 100, "INITIAL_00001", "position")
    captured = capsys.readouterr()
    assert "[RENDER GUARD]" not in captured.out


def test_word_budget_dissent_label_in_log(capsys):
    words = ["word"] * 20
    _apply_word_budget(" ".join(words), 5, "INITIAL_00002", "dissent")
    captured = capsys.readouterr()
    assert "dissent" in captured.out
    assert "INITIAL_00002" in captured.out


# ---------------------------------------------------------------------------
# _strip_hallucinated_citations
# ---------------------------------------------------------------------------

def test_strip_no_citations_unchanged():
    text = "A paragraph with no citation tags at all."
    valid = {"INITIAL_00001"}
    result = _strip_hallucinated_citations(text, valid, "INITIAL_00001")
    assert result == text


def test_strip_valid_citation_kept():
    text = "The claim [INITIAL_00001] is well supported."
    valid = {"INITIAL_00001"}
    result = _strip_hallucinated_citations(text, valid, "INITIAL_00001")
    assert "[INITIAL_00001]" in result


def test_strip_invalid_citation_removed():
    text = "The claim [INITIAL_99999] is hallucinated."
    valid = {"INITIAL_00001"}
    result = _strip_hallucinated_citations(text, valid, "INITIAL_00001")
    assert "[INITIAL_99999]" not in result


def test_strip_mixed_keeps_valid_removes_invalid():
    text = "Good [INITIAL_00001] and bad [INITIAL_99999] citations."
    valid = {"INITIAL_00001"}
    result = _strip_hallucinated_citations(text, valid, "INITIAL_00001")
    assert "[INITIAL_00001]" in result
    assert "[INITIAL_99999]" not in result


def test_strip_logs_render_guard(capsys):
    text = "The claim [INITIAL_99999] is hallucinated."
    valid = {"INITIAL_00001"}
    _strip_hallucinated_citations(text, valid, "INITIAL_00001")
    captured = capsys.readouterr()
    assert "[RENDER GUARD]" in captured.out, (
        f"Expected [RENDER GUARD] in stdout, got: {captured.out!r}"
    )
    assert "INITIAL_99999" in captured.out


def test_strip_no_log_when_all_valid(capsys):
    text = "The claim [INITIAL_00001] is valid."
    valid = {"INITIAL_00001"}
    _strip_hallucinated_citations(text, valid, "INITIAL_00001")
    captured = capsys.readouterr()
    assert "[RENDER GUARD]" not in captured.out


def test_strip_empty_text():
    result = _strip_hallucinated_citations("", {"INITIAL_00001"}, "INITIAL_00001")
    assert result == ""


def test_strip_support_and_dissent_ids_are_valid():
    """Support/dissent IDs passed to the renderer are valid citations."""
    text = "Supported by [SUPPORT_00042] and challenged by [OBJECTION_00099]."
    valid = {"INITIAL_00001", "SUPPORT_00042", "OBJECTION_00099"}
    result = _strip_hallucinated_citations(text, valid, "INITIAL_00001")
    assert "[SUPPORT_00042]" in result
    assert "[OBJECTION_00099]" in result


def test_strip_multiple_invalid_all_removed():
    text = "[INITIAL_11111] and [INITIAL_22222] and [INITIAL_33333] are all bad."
    valid = {"INITIAL_00001"}
    result = _strip_hallucinated_citations(text, valid, "INITIAL_00001")
    assert "[INITIAL_11111]" not in result
    assert "[INITIAL_22222]" not in result
    assert "[INITIAL_33333]" not in result


# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------

def test_render_position_max_words_is_reasonable():
    assert RENDER_POSITION_MAX_WORDS >= 20, (
        f"RENDER_POSITION_MAX_WORDS={RENDER_POSITION_MAX_WORDS} too small for a coherent paragraph"
    )


def test_render_dissent_max_words_is_reasonable():
    assert RENDER_DISSENT_MAX_WORDS >= 20, (
        f"RENDER_DISSENT_MAX_WORDS={RENDER_DISSENT_MAX_WORDS} too small for a coherent paragraph"
    )


def test_render_dissent_budget_geq_position_budget():
    """Dissent paragraphs allow a slightly wider budget than position paragraphs."""
    assert RENDER_DISSENT_MAX_WORDS >= RENDER_POSITION_MAX_WORDS, (
        f"Expected RENDER_DISSENT_MAX_WORDS ({RENDER_DISSENT_MAX_WORDS}) >= "
        f"RENDER_POSITION_MAX_WORDS ({RENDER_POSITION_MAX_WORDS})"
    )


def test_sampling_primary_repetition_penalty():
    assert SAMPLING_PER_ENGINE["primary"]["repetition_penalty"] == 1.15, (
        f"primary engine repetition_penalty should be 1.15; "
        f"got {SAMPLING_PER_ENGINE['primary']['repetition_penalty']}"
    )


def test_sampling_primary_key_exists():
    assert "primary" in SAMPLING_PER_ENGINE, (
        "SAMPLING_PER_ENGINE must have a 'primary' key (used by the synthesizer role)"
    )
