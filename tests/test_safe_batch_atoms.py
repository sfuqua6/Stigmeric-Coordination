"""SAFE atom batching (verification coverage on serial API backends).

The per-atom validate loop fires ~3 LLM calls/atom; on a serial API backend
(Groq) that forced a cap to 1 atom. The batched path collapses the loop into
two calls (decompose+plan, then score-all) so the API path keeps full atom
coverage. These tests pin the parse/alignment/fallback contract of the two
batched helpers — the LLM is faked so they run without a model.
"""

import asyncio

from core.worker_pool import (
    _safe_decompose_and_plan,
    _safe_score_atoms_batch,
    _safe_score_atom,
    _is_no_result_snippet,
    _format_safe_external,
)


class _FakeLLM:
    """Returns a fixed completion; records prompts for inspection."""

    def __init__(self, out):
        self.out = out
        self.prompts = []

    async def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.out


def _run(coro):
    return asyncio.run(coro)


def test_decompose_and_plan_parses_text_weight_query():
    llm = _FakeLLM(
        "ATOM: Solar PV cost fell about 80 percent | WEIGHT: 0.9 | QUERY: solar pv cost decline\n"
        "ATOM: Wind capacity grew worldwide | WEIGHT: 0.6 | QUERY: global wind capacity growth\n"
        "ATOM: Grids require storage to firm supply | WEIGHT: 0.4 | QUERY: grid energy storage"
    )
    atoms = _run(_safe_decompose_and_plan("Renewables are cheap now", llm, max_atoms=3))
    assert len(atoms) == 3
    assert atoms[0] == {
        "text": "Solar PV cost fell about 80 percent",
        "weight": 0.9,
        "query": "solar pv cost decline",
    }
    # single LLM call — that is the whole point of batching
    assert len(llm.prompts) == 1


def test_decompose_respects_max_atoms():
    lines = "\n".join(
        f"ATOM: proposition number {i} here | WEIGHT: 0.5 | QUERY: topic {i}"
        for i in range(6)
    )
    atoms = _run(_safe_decompose_and_plan("x", _FakeLLM(lines), max_atoms=3))
    assert len(atoms) == 3


def test_decompose_fallback_on_unparseable():
    atoms = _run(_safe_decompose_and_plan("Some claim text here", _FakeLLM("I cannot do that"), max_atoms=3))
    assert len(atoms) == 1
    assert atoms[0]["weight"] == 1.0
    assert atoms[0]["query"] == ""


def test_score_batch_positional_alignment_with_abstain():
    items = [{"text": "a", "snippet": "s"}, {"text": "b", "snippet": "s"}, {"text": "c", "snippet": "s"}]
    # atom 2's line is malformed: it must abstain (0.5) without sinking 1 and 3.
    llm = _FakeLLM("1: SCORE: 0.9\n2: garbage line\n3: SCORE: 0.1")
    scores = _run(_safe_score_atoms_batch(items, llm, task_type="analysis"))
    assert scores == [0.9, 0.5, 0.1]
    assert len(llm.prompts) == 1


def test_score_batch_all_abstain_on_total_failure():
    items = [{"text": "a", "snippet": "s"}, {"text": "b", "snippet": "s"}]
    scores = _run(_safe_score_atoms_batch(items, _FakeLLM("no scores here"), task_type="debate"))
    assert scores == [0.5, 0.5]


def test_score_batch_clamps_and_abstains_on_invalid():
    items = [{"text": "a", "snippet": "s"}, {"text": "b", "snippet": "s"}]
    # 1.7 clamps to 1.0; a negative is out-of-spec (scale is 0-1) and abstains to 0.5
    # — matching the per-atom scorer's [0-9.]+ regex, which never captures a minus.
    scores = _run(_safe_score_atoms_batch(items, _FakeLLM("1: SCORE: 1.7\n2: SCORE: -0.3"), task_type="analysis"))
    assert scores == [1.0, 0.5]


def test_score_batch_bracketed_index_format():
    items = [{"text": "a", "snippet": "s"}, {"text": "b", "snippet": "s"}]
    scores = _run(_safe_score_atoms_batch(items, _FakeLLM("[1] SCORE: 0.8\n[2] SCORE: 0.2"), task_type="analysis"))
    assert scores == [0.8, 0.2]


def test_score_batch_single_atom_uses_focused_scorer():
    # n==1 should delegate to the per-atom scorer (SCORE: X.X freeform).
    items = [{"text": "only one", "snippet": "snip"}]
    scores = _run(_safe_score_atoms_batch(items, _FakeLLM("SCORE: 0.7"), task_type="analysis"))
    assert scores == [0.7]


# --- no-result abstain + coverage (verification deflation fix) -------------

def test_is_no_result_snippet_detection():
    assert _is_no_result_snippet("")
    assert _is_no_result_snippet("   ")
    assert _is_no_result_snippet("(no result for: 'solar costs')")
    assert _is_no_result_snippet("(no result)")
    assert _is_no_result_snippet("(no snippet for 'x')")
    assert not _is_no_result_snippet("[Wikipedia] Solar PV costs fell 80%.")


def test_score_atom_abstains_on_no_result_without_llm_call():
    # A no-result snippet must NOT reach the LLM (which would score it ~0.0);
    # it abstains at 0.5. Asserted by the fake LLM recording zero prompts.
    llm = _FakeLLM("SCORE: 0.0")
    score = _run(_safe_score_atom("some claim", "(no result for: 'q')", llm, task_type="debate"))
    assert score == 0.5
    assert len(llm.prompts) == 0


def test_score_batch_excludes_no_result_atoms_from_llm():
    # Atom 2 has no evidence: it abstains (0.5) and is dropped from the prompt,
    # so the scorable atoms (1 and 3) are renumbered 1,2 inside the call.
    items = [
        {"text": "a", "snippet": "real snippet a"},
        {"text": "b", "snippet": "(no result for: 'b')"},
        {"text": "c", "snippet": "real snippet c"},
    ]
    llm = _FakeLLM("1: SCORE: 0.9\n2: SCORE: 0.1")
    scores = _run(_safe_score_atoms_batch(items, llm, task_type="analysis"))
    assert scores == [0.9, 0.5, 0.1]
    # The no-result atom's text must not appear in the scoring prompt.
    assert "real snippet a" in llm.prompts[0]
    assert "(no result" not in llm.prompts[0]


def test_score_batch_all_no_result_abstains_without_llm():
    items = [
        {"text": "a", "snippet": "(no result)"},
        {"text": "b", "snippet": ""},
    ]
    llm = _FakeLLM("1: SCORE: 0.9\n2: SCORE: 0.8")
    scores = _run(_safe_score_atoms_batch(items, llm, task_type="analysis"))
    assert scores == [0.5, 0.5]
    assert len(llm.prompts) == 0


def test_format_external_aggregates_evidenced_only_and_reports_coverage():
    # One confirmed atom (1.0) + one unevidenced abstain (0.5). The aggregate
    # must reflect ONLY the evidenced atom, not be dragged to 0.75.
    atom_results = [
        {"text": "a", "weight": 1.0, "score": 1.0, "snippet_tag": "[Wikipedia]", "query": "q1"},
        {"text": "b", "weight": 1.0, "score": 0.5, "snippet_tag": "(no result)", "query": "q2"},
    ]
    out = _format_safe_external(atom_results, coverage=0.5)
    assert "1.00" in out                       # aggregate over evidenced only
    assert "EVIDENCE COVERAGE: 50%" in out
    assert "(no evidence — abstained)" in out
