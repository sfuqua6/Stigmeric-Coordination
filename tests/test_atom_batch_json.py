"""JSON-first atom decompose/score batch parsing (index-keyed, not positional)."""
import asyncio

from core.worker_pool import _safe_decompose_and_plan, _safe_score_atoms_batch


class _FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply

    async def generate(self, prompt, **kw):
        return self.reply


def test_decompose_json():
    llm = _FakeLLM('{"atoms": [{"text": "Transport is 23 percent of emissions", '
                   '"weight": 0.9, "query": "transport share global emissions"}]}')
    atoms = asyncio.run(_safe_decompose_and_plan("claim", llm, max_atoms=3))
    assert len(atoms) == 1
    assert atoms[0]["weight"] == 0.9
    assert atoms[0]["query"].startswith("transport")


def test_decompose_legacy_line_fallback():
    llm = _FakeLLM("ATOM: Cars cause most urban smog today | WEIGHT: 0.7 | QUERY: urban smog sources")
    atoms = asyncio.run(_safe_decompose_and_plan("claim", llm, max_atoms=3))
    assert len(atoms) == 1 and atoms[0]["weight"] == 0.7


def test_decompose_total_failure_single_atom():
    llm = _FakeLLM("no structure at all")
    atoms = asyncio.run(_safe_decompose_and_plan("some claim text", llm))
    assert len(atoms) == 1 and atoms[0]["weight"] == 1.0


def _items():
    return [
        {"text": "a", "snippet": "evidence one is long enough"},
        {"text": "b", "snippet": "(no result for: q)"},   # abstains at 0.5
        {"text": "c", "snippet": "evidence three is long enough"},
    ]


def test_score_batch_json_index_keyed_out_of_order():
    # Entries arrive reversed; index keys must still map correctly to the
    # scorable subset (items 0 and 2; item 1 abstained without an LLM slot).
    llm = _FakeLLM('{"scores": [{"i": 2, "score": 0.2}, {"i": 1, "score": 0.9}]}')
    scores = asyncio.run(_safe_score_atoms_batch(_items(), llm, task_type="coding"))
    assert scores == [0.9, 0.5, 0.2]


def test_score_batch_legacy_positional_fallback():
    llm = _FakeLLM("1: SCORE: 0.8\n2: SCORE: 0.3")
    scores = asyncio.run(_safe_score_atoms_batch(_items(), llm, task_type="coding"))
    assert scores == [0.8, 0.5, 0.3]
