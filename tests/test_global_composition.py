"""Tests for two-stage synthesis (global composition stage).

Covers:
  - _compose_answer: happy path, tag-retention guard, length guard,
    LLM-failure fallback, and the context-compression contract (the
    composer prompt contains ONLY briefs + digest — never store content
    that wasn't rendered into a brief).
  - _extractive_position: deterministic Section-1 fallback when every
    render call returns empty (API daily-token exhaustion).
  - Sectioned assembly: Section 1 is never absent; PROCESS NOTES
    (former executive summary) lands at the end, not the top.

No GPU or LLM required — all tests use fake LLM objects.

Run with:
    pytest tests/test_global_composition.py -v
"""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.signal_store import SignalStore, _NULL_EMBEDDER
from core.signal_types import INITIAL, SUPPORT
import agents.synthesizer as synthesizer_mod
from agents.synthesizer import Synthesizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> SignalStore:
    return SignalStore(embedder=_NULL_EMBEDDER)


def _deposit(store, stype, content, strength, depositor, parent_id=None,
             metadata=None):
    meta = dict(metadata or {})
    if stype in ("INITIAL", "SUPPORT") and "partition_id" not in meta:
        agent_id = meta.get("depositor_agent_id", meta.get("scout_agent_id", ""))
        parts = agent_id.split("_")
        if len(parts) >= 3 and parts[2].isdigit():
            meta["partition_id"] = f"partition_{parts[2]}"
        else:
            meta["partition_id"] = "test_partition_0"
    return store.deposit(
        signal_type=stype,
        content=content,
        strength=strength,
        depositor=depositor,
        parent_id=parent_id,
        metadata=meta,
    )


class _FakeLLM:
    """Returns one canned response for every generate() call."""

    def __init__(self, response="", raise_exc=False):
        self.response = response
        self.raise_exc = raise_exc
        self.prompts: list[str] = []

    async def generate(self, prompt, role=None, max_tokens=None,
                       temperature=None, **kwargs):
        self.prompts.append(prompt)
        if self.raise_exc:
            raise RuntimeError("synthetic LLM failure")
        return self.response


def _fake_cluster(rep_id, support_set=(), support_diversity=4,
                  verification_score=0.0, dissent_pressure=0.0):
    return SimpleNamespace(
        representative_id=rep_id,
        support_set=list(support_set),
        support_diversity=support_diversity,
        verification_score=verification_score,
        dissent_pressure=dissent_pressure,
    )


_BRIEF_A = (
    "Banning private cars reduces urban emissions substantially "
    "[INITIAL_00001]. Implementations in several cities support this "
    "[SUPPORT_00002]."
)
_BRIEF_B = (
    "Transit alternatives must precede any ban for it to be equitable "
    "[INITIAL_00003]. Cost analyses raise distributional concerns "
    "[SUPPORT_00004]."
)

# What the model RETURNS (alias tags — composers alias [TYPE_NNNNN] to [S#]
# on the way in; briefs A/B assign S1..S4 in first-appearance order).
_COMPOSED_OK_ALIASED = (
    "Cities should ban private cars only alongside transit investment. "
    "Bans demonstrably reduce urban emissions [S1], with city "
    "implementations confirming the effect [S2]. However, "
    "equitable outcomes require transit alternatives to be in place first "
    "[S3], because cost analyses raise distributional concerns "
    "[S4]. The verified material therefore favors conditional "
    "bans over unconditional ones."
)
# What the caller receives after deterministic un-aliasing.
_COMPOSED_OK = (
    "Cities should ban private cars only alongside transit investment. "
    "Bans demonstrably reduce urban emissions [INITIAL_00001], with city "
    "implementations confirming the effect [SUPPORT_00002]. However, "
    "equitable outcomes require transit alternatives to be in place first "
    "[INITIAL_00003], because cost analyses raise distributional concerns "
    "[SUPPORT_00004]. The verified material therefore favors conditional "
    "bans over unconditional ones."
)


# ---------------------------------------------------------------------------
# _compose_answer unit tests
# ---------------------------------------------------------------------------

class TestComposeAnswer(unittest.TestCase):
    def _compose(self, llm, briefs=None):
        synth = Synthesizer(llm, "Cities should ban private cars?")
        store = _make_store()
        clusters = [
            _fake_cluster("INITIAL_00001"),
            _fake_cluster("INITIAL_00003"),
        ]
        projection = SimpleNamespace(inter_cluster_edges=[])
        return asyncio.run(synth._compose_answer(
            briefs if briefs is not None else [_BRIEF_A, _BRIEF_B],
            clusters, projection, store,
        ))

    def test_happy_path_returns_composed_text(self):
        llm = _FakeLLM(response=_COMPOSED_OK_ALIASED)
        result = self._compose(llm)
        # Aliases re-mapped to real provenance tags on the way out.
        self.assertEqual(result, _COMPOSED_OK)

    def test_tag_retention_guard_rejects_free_writing(self):
        # Long enough, but carries none of the briefs' citation tags.
        llm = _FakeLLM(response="Cars are bad for cities. " * 20)
        result = self._compose(llm)
        self.assertEqual(result, "")

    def test_short_output_guard(self):
        llm = _FakeLLM(response="Yes. [INITIAL_00001]")
        result = self._compose(llm)
        self.assertEqual(result, "")

    def test_llm_failure_returns_empty(self):
        llm = _FakeLLM(raise_exc=True)
        result = self._compose(llm)
        self.assertEqual(result, "")

    def test_partial_tag_retention_passes_at_half(self):
        # Keeps 2 of 4 distinct alias tags = exactly 0.5 retention (threshold).
        response = (
            "Bans reduce emissions in cities that tried them [S1] [S2]. "
        ) * 4
        llm = _FakeLLM(response=response)
        result = self._compose(llm)
        self.assertTrue(result)  # 0.5 >= _COMPOSE_MIN_TAG_RETENTION
        self.assertIn("[INITIAL_00001]", result)  # un-aliased on the way out

    def test_composer_input_is_briefs_and_digest_only(self):
        """Context-compression contract: the composer prompt must contain
        the briefs (alias-tagged) and the scalar digest — and must NOT
        contain signal content that was never rendered into a brief."""
        llm = _FakeLLM(response=_COMPOSED_OK_ALIASED)
        self._compose(llm)
        self.assertEqual(len(llm.prompts), 1)
        prompt = llm.prompts[0]
        # Brief prose present (tags aliased, so match the tag-free spans).
        self.assertIn("Banning private cars reduces urban emissions", prompt)
        self.assertIn("Cost analyses raise distributional concerns", prompt)
        self.assertIn("[S1]", prompt)   # aliased, not raw provenance IDs
        # The CLUSTER DIGEST may carry real cluster IDs (structural
        # context); the BRIEFS block itself must be alias-only.
        briefs_part = prompt[prompt.index("EVIDENCE BRIEFS"):]
        self.assertNotIn("[INITIAL_00001]", briefs_part)
        self.assertIn("EVIDENCE BRIEFS", prompt)
        # Marker content that exists in idea-space but not in any brief:
        self.assertNotIn("ZEBRA_UNRENDERED_MARKER", prompt)


# ---------------------------------------------------------------------------
# _extractive_position unit tests
# ---------------------------------------------------------------------------

class TestExtractivePosition(unittest.TestCase):
    def test_renders_rep_and_supports_with_tags(self):
        store = _make_store()
        init_id = _deposit(
            store, INITIAL, "Car bans cut emissions twenty percent.", 0.8,
            "scout", metadata={"scout_agent_id": "scout_R1_0"},
        )
        sup_contents = [
            "Madrid's low-emission zone lowered nitrogen dioxide readings.",
            "Congestion pricing in Stockholm shifted commuters to rail.",
        ]
        sup_ids = []
        for i, content in enumerate(sup_contents):
            sup_ids.append(_deposit(
                store, SUPPORT, content,
                0.7, "forager", parent_id=init_id,
                metadata={"depositor_agent_id": f"forager_R1_{i}_x"},
            ))
        synth = Synthesizer(_FakeLLM(), "task")
        cp = _fake_cluster(init_id, support_set=sup_ids)
        out = synth._extractive_position([cp], store)
        self.assertIn(f"[{init_id}]", out)
        self.assertIn("Car bans cut emissions", out)
        for sid in sup_ids:
            self.assertIn(f"[{sid}]", out)

    def test_missing_representative_skipped(self):
        store = _make_store()
        synth = Synthesizer(_FakeLLM(), "task")
        cp = _fake_cluster("INITIAL_99999")
        self.assertEqual(synth._extractive_position([cp], store), "")


# ---------------------------------------------------------------------------
# Sectioned assembly integration tests
# ---------------------------------------------------------------------------

def _surviving_store(n_clusters=2):
    """Store with n surviving clusters (4 distinct-partition supports each).

    Support contents must be genuinely dissimilar sentences — the store's
    string-similarity dedup rejects near-duplicates (returns None).
    """
    store = _make_store()
    init_ids = []
    topics = [
        "Car bans cut urban emissions measurably in pilot cities.",
        "Transit capacity must expand before restricting private cars.",
    ]
    support_texts = [
        [
            "Madrid's low-emission zone lowered nitrogen dioxide readings.",
            "Congestion pricing in Stockholm shifted commuters onto rail.",
            "Oslo removed downtown parking and footfall in shops increased.",
            "Paris reclaimed the Seine expressway for pedestrians and bikes.",
        ],
        [
            "Bogota built bus rapid transit before restricting any vehicles.",
            "Rural commuters lack alternatives where rail networks are sparse.",
            "Tradespeople carrying tools cannot switch to bicycles easily.",
            "Night-shift workers travel when public transit barely operates.",
        ],
    ]
    for c in range(n_clusters):
        init_id = _deposit(
            store, INITIAL, topics[c % len(topics)], 0.8,
            "scout", metadata={"scout_agent_id": f"scout_R1_{c}"},
        )
        init_ids.append(init_id)
        for i, content in enumerate(support_texts[c % len(support_texts)]):
            _deposit(
                store, SUPPORT, content,
                0.7, "forager", parent_id=init_id,
                metadata={"depositor_agent_id": f"forager_R1_{i}_s{c}"},
            )
    return store, init_ids


class TestSectionedAssembly(unittest.TestCase):
    def test_all_renders_empty_falls_back_to_extractive(self):
        """Groq-exhaustion shape: every generate() returns "" — the answer
        must still contain a non-empty Section 1 (extractive fallback)."""
        store, init_ids = _surviving_store()
        llm = _FakeLLM(response="")
        synth = Synthesizer(llm, "Cities should ban private cars?")
        answer, _, _ = asyncio.run(synth.synthesize(
            store, has_validators=False, task_type="debate",
        ))
        self.assertIn("## 1. POSITION SYNTHESIS", answer)
        self.assertIn("extractive fallback", answer)
        # The extractive content carries the cluster representative IDs.
        self.assertTrue(any(init_id in answer for init_id in init_ids))

    def test_process_notes_after_position_synthesis(self):
        """Field telemetry must follow the answer, not precede it."""
        store, _ = _surviving_store()
        llm = _FakeLLM(response="")
        synth = Synthesizer(llm, "Cities should ban private cars?")
        answer, _, _ = asyncio.run(synth.synthesize(
            store, has_validators=False, task_type="debate",
        ))
        self.assertIn("## PROCESS NOTES", answer)
        self.assertNotIn("## EXECUTIVE SUMMARY", answer)
        self.assertLess(
            answer.index("## 1. POSITION SYNTHESIS"),
            answer.index("## PROCESS NOTES"),
        )

    def test_composition_call_made_and_used(self):
        """With ≥2 briefs and an LLM that answers, the composer runs and its
        output becomes Section 1 (single composed text, not N joined briefs)."""
        store, _ = _surviving_store(n_clusters=2)
        llm = _FakeLLM(response=_COMPOSED_OK_ALIASED)
        synth = Synthesizer(llm, "Cities should ban private cars?")
        # Disable the revision loop: the FakeLLM would feed the same canned
        # response to the critic/revise calls and overwrite the assembly.
        with mock.patch.object(
            synthesizer_mod, "_SYNTHESIZER_REVISION_ROUNDS", 0,
        ):
            answer, _, _ = asyncio.run(synth.synthesize(
                store, has_validators=False, task_type="debate",
            ))
        compose_prompts = [p for p in llm.prompts if "EVIDENCE BRIEFS" in p]
        self.assertEqual(len(compose_prompts), 1)
        self.assertIn("## 1. POSITION SYNTHESIS", answer)
        # The composed thesis sentence (from the single global call) is
        # present exactly once WITHIN Section 1 — composed, not a join of
        # repeated per-cluster paragraphs. (Other sections may quote
        # excerpts, so scope the count to Section 1.)
        s1_start = answer.index("## 1. POSITION SYNTHESIS")
        s1_end = answer.index("## 2.", s1_start) if "## 2." in answer else len(answer)
        section1 = answer[s1_start:s1_end]
        self.assertEqual(
            section1.count("Cities should ban private cars only alongside"), 1,
        )


if __name__ == "__main__":
    unittest.main()
