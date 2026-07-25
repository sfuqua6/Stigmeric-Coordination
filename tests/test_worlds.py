"""Tests for the Stage-1 synthetic-world generator (eval/worlds.py).

Covers docs/future/STAGE1_SYNTHETIC_EVAL_SPEC.md Sec 6.1:
  - determinism of build_fact_registry + the reuse-if-exists render contract
  - referential integrity of the generated FactRegistry
  - verify_render catching a dropped fact / a fabricated number
  - Gate 1 (B-from-memory ceiling) firing and rerolling with seed+1
  - Gate 2 (name collision) firing and rerolling before any rendering
  - build_pack_from_world schema compatibility with eval.packs.load_pack()

No GPU or live LLM required — the template renderer is deterministic pure
Python, and the two gates are exercised against small stub LLM objects (an
async .generate(...) is all either gate needs).

Run with:
    python -m unittest tests.test_worlds -v
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.worlds import (
    Contradiction, CausalRelation, DocumentSpec, Entity, Event, FactRegistry,
    Quantity, RenderedDoc, available_templates, build_fact_registry,
    generate_world, load_rendered_docs, load_world, name_collision_check_async,
    render_documents_template, run_gate1_async, save_world,
    synthetic_prompts_for_world, verify_render,
)


class _StubLLM:
    """Minimal async .generate(...) stub — satisfies the contract both gates
    need without pulling in MockLLM's own (intentionally non-deterministic)
    sampling."""

    def __init__(self, response: str = "none"):
        self.response = response
        self.prompts: list[str] = []

    async def generate(self, prompt, role="agent", max_tokens=100, temperature=0.7):
        self.prompts.append(prompt)
        return self.response


class TestWorldDeterminism(unittest.TestCase):
    def test_world_determinism(self):
        r1 = build_fact_registry(42, "invented_company")
        r2 = build_fact_registry(42, "invented_company")
        self.assertEqual(r1, r2)
        docs1 = render_documents_template(r1)
        docs2 = render_documents_template(r2)
        self.assertEqual([d.text for d in docs1], [d.text for d in docs2])

    def test_different_seed_differs(self):
        r1 = build_fact_registry(1, "invented_company")
        r2 = build_fact_registry(2, "invented_company")
        self.assertNotEqual(r1, r2)

    def test_render_reuse_survives_registry_deletion(self):
        """generate_world called a THIRD time after deleting only the
        registry (not the rendered docs) must still reuse the existing
        rendered docs unchanged — proves the reuse-if-exists contract, not
        just full-pipeline determinism."""
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "worlds"
            gate_log = Path(td) / "gate_log.jsonl"
            r1, docs1 = generate_world(
                7, "invented_company", renderer="template",
                skip_gate1=True, skip_gate2=True,
                out_dir=out_dir / "7", gate_log_path=gate_log)
            rendered_dir = out_dir / "7" / "rendered"
            self.assertTrue(rendered_dir.exists())
            saved_texts = {p.name: p.read_text(encoding="utf-8")
                          for p in rendered_dir.glob("*.txt")}

            # Delete only the registry + gate log entry (force regeneration
            # of the FactRegistry object), leave rendered/ untouched.
            (out_dir / "7" / "registry.json").unlink()
            gate_log.unlink()

            r2, docs2 = generate_world(
                7, "invented_company", renderer="template",
                skip_gate1=True, skip_gate2=True,
                out_dir=out_dir / "7", gate_log_path=gate_log)
            reloaded_texts = {p.name: p.read_text(encoding="utf-8")
                             for p in rendered_dir.glob("*.txt")}
            self.assertEqual(saved_texts, reloaded_texts)
            self.assertEqual(r1, r2)


class TestReferentialIntegrity(unittest.TestCase):
    def _check(self, registry: FactRegistry):
        entity_ids = {e.entity_id for e in registry.entities}
        fact_ids = set(registry.fact_index())
        doc_ids = {d.doc_id for d in registry.documents}
        for rel in registry.relations:
            self.assertIn(rel.cause_id, fact_ids)
            self.assertIn(rel.effect_id, fact_ids)
        for c in registry.contradictions:
            self.assertIn(c.fact_id_a, fact_ids)
            self.assertIn(c.fact_id_b, fact_ids)
            self.assertIn(c.doc_id_a, doc_ids)
            self.assertIn(c.doc_id_b, doc_ids)
            self.assertTrue(c.ground_truth == "genuinely_ambiguous"
                           or c.ground_truth in fact_ids)
        for ev in registry.events:
            for eid in ev.entity_ids:
                self.assertIn(eid, entity_ids)
        for d in registry.documents:
            for fid in d.fact_ids:
                self.assertIn(fid, fact_ids)
            for eid in d.entity_ids:
                self.assertIn(eid, entity_ids)

    def test_hand_built_fixture(self):
        """A small hand-built registry (3 entities, a couple of facts, 1
        contradiction) — the base case the fuzz pass below generalizes."""
        entities = [
            Entity("ent_0000", "company", "Fixture Co", ("Fixture",)),
            Entity("ent_0001", "person", "Ann Otter", ("Ann",)),
            Entity("ent_0002", "agency", "Bureau of Fixtures", ()),
        ]
        q1 = Quantity("q_0000", "ent_0000", "revenue", 100.0, "USD million",
                     "FY2024Q1", 2.0, ("$100.0M", "100.0 million"))
        q2 = Quantity("q_0001", "ent_0000", "revenue", 150.0, "USD million",
                     "FY2024Q1_revised", 2.0, ("$150.0M", "150.0 million"))
        docs = [DocumentSpec("doc_a", "filing", ["ent_0000"], ["q_0000"]),
               DocumentSpec("doc_b", "press", ["ent_0000"], ["q_0001"])]
        contradiction = Contradiction("c_0000", "q_0000", "q_0001", "doc_a",
                                      "doc_b", "q_0000")
        registry = FactRegistry(
            world_seed=0, template_name="invented_company", entities=entities,
            quantities=[q1, q2], relations=[], events=[],
            contradictions=[contradiction], filler_seed_material=["Filler."],
            documents=docs)
        self._check(registry)

    def test_fuzz_across_seeds_and_templates(self):
        for seed in (1, 2, 42, 99, 12345):
            for template in available_templates():
                self._check(build_fact_registry(seed, template))


class TestVerifyRender(unittest.TestCase):
    def _minimal_registry(self) -> FactRegistry:
        entities = [Entity("ent_0000", "company", "Fixture Co", ("Fixture",))]
        q1 = Quantity("q_0000", "ent_0000", "revenue", 340.2, "USD million",
                     "FY2024Q1", 2.0, ("$340.2M", "340.2 million"))
        docs = [DocumentSpec("doc_a", "filing", ["ent_0000"], ["q_0000"])]
        return FactRegistry(world_seed=0, template_name="invented_company",
                            entities=entities, quantities=[q1], relations=[],
                            events=[], contradictions=[],
                            filler_seed_material=["Filler sentence."],
                            documents=docs)

    def test_catches_dropped_fact(self):
        registry = self._minimal_registry()
        # extracted_from claims q_0000 but the rendered text never mentions it.
        bad_doc = RenderedDoc(doc_id="doc_a", doc_type="filing",
                              text="Nothing of note happened this quarter.",
                              extracted_from=["q_0000"], renderer="template")
        result = verify_render(registry, [bad_doc])[0]
        self.assertFalse(result.ok)
        self.assertIn("q_0000", result.missing_facts)

    def test_passes_when_fact_present(self):
        registry = self._minimal_registry()
        good_doc = RenderedDoc(doc_id="doc_a", doc_type="filing",
                               text="Fixture Co's revenue as of FY2024Q1 was "
                                    "$340.2M, a strong quarter.",
                               extracted_from=["q_0000"], renderer="template")
        result = verify_render(registry, [good_doc])[0]
        self.assertTrue(result.ok, result)

    def test_catches_number_leakage(self):
        registry = self._minimal_registry()
        # Contains the registered fact AND a fabricated figure (72.5) that
        # isn't in the registry, isn't a date, isn't on the connective
        # whitelist.
        bad_doc = RenderedDoc(
            doc_id="doc_a", doc_type="filing",
            text=("Fixture Co's revenue as of FY2024Q1 was $340.2M. Analysts "
                 "had separately estimated 72.5 percent upside potential."),
            extracted_from=["q_0000"], renderer="template")
        result = verify_render(registry, [bad_doc])[0]
        self.assertFalse(result.ok)
        self.assertIn("72.5", result.fabricated_numbers)

    def test_dates_and_entity_name_numbers_not_flagged(self):
        """Dates and digits embedded in entity names (e.g. a location pool
        entry like "Route 14 overpass") must not misfire as fabrications."""
        entities = [Entity("ent_0000", "location", "Route 14 overpass", ())]
        registry = FactRegistry(world_seed=0, template_name="case_file",
                                entities=entities, quantities=[], relations=[],
                                events=[], contradictions=[],
                                filler_seed_material=[], documents=[])
        doc = RenderedDoc(
            doc_id="doc_a", doc_type="witness_statement",
            text="On 2021-03-14, a witness passed Route 14 overpass.",
            extracted_from=[], renderer="template")
        result = verify_render(registry, [doc])[0]
        self.assertTrue(result.ok, result)


class TestGates(unittest.TestCase):
    def test_gate2_name_collision_fires(self):
        import asyncio
        entities = [Entity("ent_0000", "company", "Tesla", ("Tesla",)),
                   Entity("ent_0001", "product", "Widget One", ())]
        llm = _StubLLM(response="yes, Tesla rings a bell")
        flagged = asyncio.run(name_collision_check_async(entities, llm))
        self.assertIn("ent_0000", flagged)
        self.assertNotIn("ent_0001", flagged)

    def test_gate2_clears_on_none(self):
        import asyncio
        entities = [Entity("ent_0000", "company", "Meridian Analytics", ())]
        llm = _StubLLM(response="none")
        flagged = asyncio.run(name_collision_check_async(entities, llm))
        self.assertEqual(flagged, [])

    def test_gate1_b_ceiling_fires(self):
        """A stubbed condition-B answer scoring high (the full corpus, so it
        trivially matches every checklist item) must cause generate_world to
        log a rejection and regenerate with seed+1 — NOT the same seed."""
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "worlds"
            gate_log = Path(td) / "gate_log.jsonl"

            call_count = {"n": 0}
            seed_used = {}

            class _FirstFailsLLM:
                async def generate(self, prompt, role="agent", max_tokens=100,
                                   temperature=0.7):
                    call_count["n"] += 1
                    if call_count["n"] == 1:
                        # Reconstruct world 200's own corpus text and hand it
                        # back verbatim as the "B answer" -> guaranteed to
                        # score >= the 0.20 ceiling.
                        reg = build_fact_registry(200, "invented_company")
                        docs = render_documents_template(reg)
                        seed_used[1] = 200
                        return "\n".join(d.text for d in docs)
                    seed_used[call_count["n"]] = 200 + call_count["n"] - 1
                    return "no information available"

            registry, docs = generate_world(
                200, "invented_company", renderer="template",
                gate1_llm=_FirstFailsLLM(), skip_gate2=True,
                out_dir=out_dir, gate_log_path=gate_log, max_world_regen=5)

            log_lines = gate_log.read_text(encoding="utf-8").splitlines()
            entries = [line for line in log_lines if line.strip()]
            self.assertGreaterEqual(len(entries), 2)
            import json
            first = json.loads(entries[0])
            self.assertEqual(first["seed"], 200)
            self.assertFalse(first["accepted"])
            self.assertGreaterEqual(first["b_score"], 0.20)
            # The accepted world must NOT be the rejected seed.
            self.assertNotEqual(registry.world_seed, 200)
            self.assertEqual(registry.world_seed, 201)

    def test_gate1_passes_with_generic_answer(self):
        """A generic hedge with no invented facts should score near-zero and
        pass Gate 1 on the first try (no reroll)."""
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "worlds"
            gate_log = Path(td) / "gate_log.jsonl"
            llm = _StubLLM(response="I do not have specific information "
                                    "about this particular entity.")
            registry, docs = generate_world(
                55, "invented_company", renderer="template",
                gate1_llm=llm, skip_gate2=True,
                out_dir=out_dir, gate_log_path=gate_log)
            self.assertEqual(registry.world_seed, 55)


class TestSyntheticPrompts(unittest.TestCase):
    def test_synthetic_prompts_for_world_shape(self):
        registry = build_fact_registry(9, "invented_company")
        prompts = synthetic_prompts_for_world(registry)
        self.assertTrue(prompts)
        p = prompts[0]
        self.assertTrue(p.factual)
        self.assertTrue(p.must_include)
        self.assertIn(str(registry.world_seed), p.pid)


class TestPackFromWorldSchemaCompat(unittest.TestCase):
    def test_pack_round_trips_through_load_pack(self):
        from eval.packs import build_pack_from_world, load_pack

        registry = build_fact_registry(3, "invented_company")
        docs = render_documents_template(registry)
        with tempfile.TemporaryDirectory() as td:
            packs_dir = Path(td) / "packs"
            path = build_pack_from_world(registry, docs, "1x", packs_dir, pid="t3")
            chunks = load_pack(path)
            self.assertTrue(chunks)
            for i, ch in enumerate(chunks):
                self.assertTrue(ch.text)
                self.assertTrue(ch.source_tag)
                self.assertEqual(ch.chunk_id, f"pack_{i:05d}")

    def test_pack_build_is_reused_not_rebuilt(self):
        from eval.packs import build_pack_from_world

        registry = build_fact_registry(3, "invented_company")
        docs = render_documents_template(registry)
        with tempfile.TemporaryDirectory() as td:
            packs_dir = Path(td) / "packs"
            p1 = build_pack_from_world(registry, docs, "1x", packs_dir, pid="t3")
            text1 = p1.read_text(encoding="utf-8")
            p2 = build_pack_from_world(registry, docs, "1x", packs_dir, pid="t3")
            text2 = p2.read_text(encoding="utf-8")
            self.assertEqual(text1, text2)


if __name__ == "__main__":
    unittest.main()
