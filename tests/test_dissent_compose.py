"""Tests for Section-2 cap-and-compose and the embedder fallback chain.

Covers:
  - _compose_dissent: happy path, length guard, tag-retention guard,
    LLM-failure fallback.
  - Dissent render cap: with more dissent-bearing clusters than
    _DISSENT_RENDER_CAP, only the cap is LLM-rendered and the overflow
    appears as deterministic one-liners.
  - _try_load_embedder: falls back to the transformers AutoModel path when
    sentence_transformers fails (the Colab torchcodec/FFmpeg failure), and
    reports UNAVAILABLE only when both loaders fail.

No GPU, LLM, or network required.

Run with:
    pytest tests/test_dissent_compose.py -v
"""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

import core.signal_store as signal_store_mod
from core.signal_store import SignalStore, _NULL_EMBEDDER
from core.signal_types import INITIAL, SUPPORT, OBJECTION
import agents.synthesizer as synthesizer_mod
from agents.synthesizer import Synthesizer


def _make_store() -> SignalStore:
    return SignalStore(embedder=_NULL_EMBEDDER)


class _FakeLLM:
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


_NOTE_A = (
    "Singapore's vehicle-control model needs political will that varies by "
    "country [INITIAL_00005]. Comparative studies would settle whether it "
    "transfers."
)
_NOTE_B = (
    "Car bans may overload public transit without capacity investment "
    "[INITIAL_00007]. Case studies from Oslo would settle this."
)

# Model output uses [S#] aliases; the caller receives real tags back.
_COMPOSED_DISSENT_ALIASED = (
    "The central unresolved question is transferability: Singapore's "
    "vehicle-control model requires political will that varies by country "
    "[S1], and bans may overload transit systems without prior "
    "capacity investment [S2]. Comparative studies across cities "
    "with different transit baselines, including Oslo's restriction "
    "experience, would settle both disputes."
)
_COMPOSED_DISSENT_OK = (
    "The central unresolved question is transferability: Singapore's "
    "vehicle-control model requires political will that varies by country "
    "[INITIAL_00005], and bans may overload transit systems without prior "
    "capacity investment [INITIAL_00007]. Comparative studies across cities "
    "with different transit baselines, including Oslo's restriction "
    "experience, would settle both disputes."
)


# ---------------------------------------------------------------------------
# _compose_dissent unit tests
# ---------------------------------------------------------------------------

class TestComposeDissent(unittest.TestCase):
    def _compose(self, llm, fragments=None):
        synth = Synthesizer(llm, "Cities should ban private cars?")
        store = _make_store()
        clusters = [
            SimpleNamespace(representative_id="INITIAL_00005",
                            dissent_pressure=0.5),
            SimpleNamespace(representative_id="INITIAL_00007",
                            dissent_pressure=0.3),
        ]
        return asyncio.run(synth._compose_dissent(
            fragments if fragments is not None else [_NOTE_A, _NOTE_B],
            clusters, store,
        ))

    def test_happy_path(self):
        llm = _FakeLLM(response=_COMPOSED_DISSENT_ALIASED)
        # Aliases re-mapped to real provenance tags on the way out.
        self.assertEqual(self._compose(llm), _COMPOSED_DISSENT_OK)

    def test_short_output_falls_back(self):
        llm = _FakeLLM(response="Open question. [INITIAL_00005]")
        self.assertEqual(self._compose(llm), "")

    def test_tag_retention_guard(self):
        llm = _FakeLLM(response="A long passage about open questions in "
                                "urban policy that cites nothing at all. " * 8)
        self.assertEqual(self._compose(llm), "")

    def test_llm_failure_returns_empty(self):
        llm = _FakeLLM(raise_exc=True)
        self.assertEqual(self._compose(llm), "")

    def test_prompt_contains_notes_only(self):
        llm = _FakeLLM(response=_COMPOSED_DISSENT_ALIASED)
        self._compose(llm)
        prompt = llm.prompts[0]
        # Notes present with provenance tags aliased to [S#].
        self.assertIn("Singapore's vehicle-control model needs political",
                      prompt)
        self.assertIn("Car bans may overload public transit", prompt)
        self.assertIn("[S1]", prompt)
        self.assertNotIn("[INITIAL_00005]", prompt)
        self.assertIn("DISSENT NOTES", prompt)

    def test_long_fragments_budgeted(self):
        """Six full paragraphs overflowed the AWQ context — each note must
        be truncated to the per-fragment budget before prompting."""
        llm = _FakeLLM(response=_COMPOSED_DISSENT_OK)
        long_note = ("A very long dissent paragraph about transit capacity. "
                     * 60) + "[INITIAL_00005]"
        self._compose(llm, fragments=[long_note, _NOTE_B])
        prompt = llm.prompts[0]
        budget = synthesizer_mod._DISSENT_COMPOSE_FRAGMENT_CHARS
        # The long note appears truncated, not verbatim.
        self.assertNotIn(long_note, prompt)
        start = prompt.index("DISSENT NOTE 1:")
        end = prompt.index("DISSENT NOTE 2:")
        self.assertLessEqual(end - start, budget + 50)


# ---------------------------------------------------------------------------
# Dissent render cap integration test
# ---------------------------------------------------------------------------

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
        signal_type=stype, content=content, strength=strength,
        depositor=depositor, parent_id=parent_id, metadata=meta,
    )


_TOPICS = [
    "Madrid's low-emission zone lowered nitrogen dioxide measurably.",
    "Freight delivery costs rise sharply under blanket vehicle bans.",
    "Suburban commuters lose mobility where rail networks are sparse.",
    "Congestion pricing shifts behavior without full prohibition.",
    "Transit-adjacent housing costs rise after car restrictions.",
    "Ride-sharing rebound can increase total vehicle traffic.",
    "Small businesses depend on personal-vehicle deliveries downtown.",
    "Automotive industry lobbying obstructs municipal restrictions.",
    "Night-shift workers travel when public transit barely operates.",
]

_SUPPORT_TEXTS = [
    "Sensor readings confirmed the change across monitoring stations",
    "Municipal budget reports document the cost shifts in detail",
    "Survey data from affected districts quantifies the burden",
    "Longitudinal studies tracked the behavioral response over years",
]

_OBJECTION_TEXTS = [
    "This understates compliance costs for low-income drivers entirely.",
    "Delivery windows shrink and spoilage rises for perishable goods.",
    "Park-and-ride capacity cannot absorb the displaced commuter volume.",
    "Pricing elasticity estimates here rely on outdated fuel-cost data.",
    "Construction timelines for transit expansion span multiple decades.",
    "Curbside access rules already accommodate most commercial loading.",
    "Modal-shift surveys oversample central districts and miss exurbs.",
    "Charging infrastructure gaps stall the electric-vehicle alternative.",
    "Enforcement costs erode the projected municipal revenue gains.",
]


def _dissent_heavy_store(n_clusters):
    """Store with n surviving clusters that each carry one OBJECTION.

    All contents must be genuinely dissimilar sentences — the store's
    string-similarity pre-screen rejects near-duplicates against the 3
    most recent same-type signals (returns None).
    """
    store = _make_store()
    for c in range(n_clusters):
        topic = _TOPICS[c % len(_TOPICS)]
        init_id = _deposit(
            store, INITIAL, topic, 0.8, "scout",
            metadata={"scout_agent_id": f"scout_R1_{c}"},
        )
        for i in range(4):
            _deposit(
                store, SUPPORT,
                f"{_SUPPORT_TEXTS[i]} regarding {topic.lower()}",
                0.7, "forager", parent_id=init_id,
                metadata={"depositor_agent_id": f"forager_R1_{i}_c{c}"},
            )
        store.deposit(
            signal_type=OBJECTION,
            content=_OBJECTION_TEXTS[c % len(_OBJECTION_TEXTS)],
            strength=0.4, depositor="hater", parent_id=init_id,
            metadata={"depositor_agent_id": f"hater_R1_{c}"},
        )
    return store


class TestDissentRenderCap(unittest.TestCase):
    def test_overflow_dissent_listed_not_rendered(self):
        n = synthesizer_mod._DISSENT_RENDER_CAP + 3
        store = _dissent_heavy_store(n)
        llm = _FakeLLM(response="")  # all renders empty -> extractive paths
        synth = Synthesizer(llm, "Cities should ban private cars?")
        with mock.patch.object(
            synthesizer_mod, "_SYNTHESIZER_REVISION_ROUNDS", 0,
        ):
            answer, _, _ = asyncio.run(synth.synthesize(
                store, has_validators=False, task_type="debate",
            ))
        self.assertIn("## 2. OPEN QUESTIONS AND DISSENT", answer)
        self.assertIn("Further contested positions, not expanded here:", answer)
        # Overflow lines carry dissent_pressure annotations.
        self.assertIn("dissent_pressure=", answer)


# ---------------------------------------------------------------------------
# Contradiction pair dedupe + cap
# ---------------------------------------------------------------------------

class TestContradictionDedupe(unittest.TestCase):
    def _projection_with_tension_edges(self, pairs):
        cps = {}
        for a, b in pairs:
            for cid in (a, b):
                if cid not in cps:
                    cps[cid] = SimpleNamespace(
                        representative_id=cid, dissent_set=[],
                        dissent_pressure=0.0,
                    )
        edges = [SimpleNamespace(relation="tension", source=a, target=b)
                 for a, b in pairs]
        return SimpleNamespace(
            surviving=list(cps.values()), contested=[],
            inter_cluster_edges=edges,
        )

    def test_directed_duplicates_collapse(self):
        proj = self._projection_with_tension_edges([
            ("INITIAL_00001", "INITIAL_00002"),
            ("INITIAL_00002", "INITIAL_00001"),   # reverse duplicate
            ("INITIAL_00001", "INITIAL_00003"),
        ])
        pairs = synthesizer_mod._contradictions_from_projection(
            proj, _make_store())
        self.assertEqual(len(pairs), 2)

    def test_capped(self):
        edge_pairs = [(f"INITIAL_{i:05d}", f"INITIAL_{i + 100:05d}")
                      for i in range(20)]
        proj = self._projection_with_tension_edges(edge_pairs)
        pairs = synthesizer_mod._contradictions_from_projection(
            proj, _make_store())
        self.assertEqual(len(pairs), synthesizer_mod._CONTRADICTION_CAP)

    def test_self_edge_ignored(self):
        proj = self._projection_with_tension_edges([
            ("INITIAL_00001", "INITIAL_00001"),
        ])
        pairs = synthesizer_mod._contradictions_from_projection(
            proj, _make_store())
        self.assertEqual(len(pairs), 0)


# ---------------------------------------------------------------------------
# Embedder fallback chain
# ---------------------------------------------------------------------------

class TestEmbedderFallback(unittest.TestCase):
    def setUp(self):
        # Reset the module-level cache around each test.
        self._cache = signal_store_mod._EMBEDDER_CACHE
        self._status = signal_store_mod._EMBEDDER_STATUS
        signal_store_mod._EMBEDDER_CACHE = signal_store_mod._EMBEDDER_SENTINEL
        signal_store_mod._EMBEDDER_STATUS = "unset"

    def tearDown(self):
        signal_store_mod._EMBEDDER_CACHE = self._cache
        signal_store_mod._EMBEDDER_STATUS = self._status

    def test_fallback_used_when_sentence_transformers_fails(self):
        fake = SimpleNamespace(encode=lambda text: [0.0] * 8)
        with mock.patch.object(
            signal_store_mod, "_load_sentence_transformer",
            side_effect=RuntimeError("Could not load libtorchcodec"),
        ), mock.patch.object(
            signal_store_mod, "_load_transformers_fallback",
            return_value=fake,
        ):
            emb = signal_store_mod._try_load_embedder()
        self.assertIs(emb, fake)
        self.assertIn("transformers fallback",
                      signal_store_mod._EMBEDDER_STATUS)

    def test_unavailable_when_both_fail(self):
        with mock.patch.object(
            signal_store_mod, "_load_sentence_transformer",
            side_effect=RuntimeError("Could not load libtorchcodec"),
        ), mock.patch.object(
            signal_store_mod, "_load_transformers_fallback",
            side_effect=ImportError("no transformers"),
        ):
            emb = signal_store_mod._try_load_embedder()
        self.assertIsNone(emb)
        self.assertIn("UNAVAILABLE", signal_store_mod._EMBEDDER_STATUS)
        self.assertIn("libtorchcodec", signal_store_mod._EMBEDDER_STATUS)

    def test_primary_path_unchanged(self):
        fake_st = SimpleNamespace(encode=lambda text: [0.0] * 8)
        with mock.patch.object(
            signal_store_mod, "_load_sentence_transformer",
            return_value=fake_st,
        ):
            emb = signal_store_mod._try_load_embedder()
        self.assertIs(emb, fake_st)
        self.assertEqual(signal_store_mod._EMBEDDER_STATUS,
                         "all-MiniLM-L6-v2")


if __name__ == "__main__":
    unittest.main()
