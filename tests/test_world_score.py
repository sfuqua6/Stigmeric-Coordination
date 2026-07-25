"""Tests for the Stage-1 mechanical scorer (eval/world_score.py).

Covers docs/future/STAGE1_SYNTHETIC_EVAL_SPEC.md Sec 3 + Sec 6.1: hand-built
fixture answers (correct / wrong-value-out-of-tolerance / missing entirely /
hedged-but-not-anchored) run through each item-type scorer in isolation,
including the tolerance boundary (exactly at tolerance_pct hits, just past
misses) and the dual-anchor requirement for contradiction_surfaced (a cue
word alone, with no anchor to either fact's value, must NOT count as a hit).

No LLM anywhere in this module — that's the point (THEFUTURE.md Sec 3).

Run with:
    python -m unittest tests.test_world_score -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.worlds import Contradiction, DocumentSpec, Entity, Event, FactRegistry, Quantity
from eval.world_score import (
    atomic_fact_present, build_checklist_items, citation_grounds_claim,
    compute_world_scores, contradiction_resolved, contradiction_surfaced,
    parse_sources_block, quantity_correct, score_answer, score_citation_grounding,
)


def _registry() -> FactRegistry:
    entities = [
        Entity("ent_0000", "company", "Meridian Analytics", ("Meridian", "MDA")),
        Entity("ent_0001", "person", "Elena Voss", ("Elena",)),
    ]
    q_revenue = Quantity("q_0000", "ent_0000", "quarterly_revenue_usd_m", 340.2,
                        "USD million", "FY2024Q1", 2.0,
                        ("$340.2M", "340.2 million", "$340.2 million"))
    q_revenue_false = Quantity("q_0001", "ent_0000", "quarterly_revenue_usd_m", 410.0,
                              "USD million", "FY2024Q1_revised", 2.0,
                              ("$410.0M", "410.0 million"))
    events = [Event("e_0000", "Meridian Analytics opened a new facility near Elena Voss",
                    "2024-03-01", ("ent_0000", "ent_0001"))]
    contradiction = Contradiction("c_0000", "q_0000", "q_0001", "doc_a", "doc_b", "q_0000")
    docs = [DocumentSpec("doc_a", "10k_filing", ["ent_0000"], ["q_0000", "e_0000"]),
           DocumentSpec("doc_b", "trade_press", ["ent_0000"], ["q_0001"])]
    return FactRegistry(world_seed=0, template_name="invented_company",
                       entities=entities, quantities=[q_revenue, q_revenue_false],
                       relations=[], events=events, contradictions=[contradiction],
                       filler_seed_material=["Filler sentence."], documents=docs)


class TestAtomicFactPresent(unittest.TestCase):
    def test_present_by_full_name(self):
        r = _registry()
        e = r.entity_index()["ent_0000"]
        self.assertTrue(atomic_fact_present("Meridian Analytics reported strong "
                                           "quarterly results.", e))

    def test_present_by_alias(self):
        r = _registry()
        e = r.entity_index()["ent_0000"]
        self.assertTrue(atomic_fact_present("MDA reported strong results.", e))

    def test_missing_entirely(self):
        r = _registry()
        e = r.entity_index()["ent_0000"]
        self.assertFalse(atomic_fact_present("The company reported strong results.", e))


class TestQuantityTolerance(unittest.TestCase):
    def test_exact_surface_form_hits(self):
        r = _registry()
        q = r.fact_index()["q_0000"]
        hit, _ = quantity_correct("Revenue for the quarter was $340.2M, up "
                                  "from the prior period.", q)
        self.assertTrue(hit)

    def test_value_exactly_at_tolerance_boundary_hits(self):
        r = _registry()
        q = r.fact_index()["q_0000"]  # value=340.2, tolerance_pct=2.0
        # 2.0% of 340.2 = 6.804 -> boundary value = 340.2 + 6.804 = 347.004
        boundary_value = 340.2 * 1.02
        answer = (f"Meridian Analytics reported quarterly revenue usd m of "
                 f"{boundary_value:.3f} for the period.")
        hit, detail = quantity_correct(answer, q, subject_name="Meridian Analytics")
        self.assertTrue(hit, detail)

    def test_value_just_past_tolerance_misses(self):
        r = _registry()
        q = r.fact_index()["q_0000"]
        just_past = 340.2 * 1.021  # 2.1%, just past the 2.0% tolerance
        answer = (f"Meridian Analytics reported quarterly revenue usd m of "
                 f"{just_past:.3f} for the period.")
        hit, detail = quantity_correct(answer, q, subject_name="Meridian Analytics")
        self.assertFalse(hit, detail)

    def test_missing_entirely_misses(self):
        r = _registry()
        q = r.fact_index()["q_0000"]
        hit, _ = quantity_correct("The company did not disclose specific figures.", q)
        self.assertFalse(hit)

    def test_wrong_value_out_of_tolerance_misses(self):
        r = _registry()
        q = r.fact_index()["q_0000"]
        hit, _ = quantity_correct(
            "Meridian Analytics reported quarterly revenue usd m of 999.9 for the period.",
            q, subject_name="Meridian Analytics")
        self.assertFalse(hit)


class TestContradictionSurfaced(unittest.TestCase):
    def test_dual_anchor_plus_cue_hits(self):
        r = _registry()
        c = r.contradictions[0]
        answer = ("Sources disagree on Meridian Analytics' quarterly revenue: "
                 "one filing states $340.2M while a later trade-press piece "
                 "claims $410.0M.")
        self.assertTrue(contradiction_surfaced(answer, c, r))

    def test_cue_word_alone_without_anchors_misses(self):
        """A cue word with NO anchor to either fact's value must NOT count as
        a hit — the dual-anchor requirement (spec Sec 3.1, Sec 6.1)."""
        r = _registry()
        c = r.contradictions[0]
        answer = "The sources sometimes disagree on financial matters generally."
        self.assertFalse(contradiction_surfaced(answer, c, r))

    def test_single_anchor_without_cue_misses(self):
        r = _registry()
        c = r.contradictions[0]
        answer = "Meridian Analytics reported quarterly revenue of $340.2M."
        self.assertFalse(contradiction_surfaced(answer, c, r))

    def test_both_anchors_without_cue_word_misses(self):
        r = _registry()
        c = r.contradictions[0]
        answer = "One document says $340.2M and another says $410.0M."
        self.assertFalse(contradiction_surfaced(answer, c, r))

    def test_contradiction_resolved_matches_ground_truth(self):
        r = _registry()
        c = r.contradictions[0]  # ground_truth = q_0000 ($340.2M)
        answer = "The correct figure, per the original filing, was $340.2M."
        self.assertTrue(contradiction_resolved(answer, c, r))

    def test_contradiction_resolved_none_when_ambiguous(self):
        r = _registry()
        c = Contradiction("c_0001", "q_0000", "q_0001", "doc_a", "doc_b",
                          "genuinely_ambiguous")
        self.assertIsNone(contradiction_resolved("anything", c, r))


class TestCitationGrounding(unittest.TestCase):
    def test_grounded_excerpt_hits(self):
        doc_texts = {"doc_a": "Meridian Analytics's quarterly revenue as of "
                             "FY2024Q1 was $340.2 million, a strong result."}
        hit, detail = citation_grounds_claim(
            "Meridian Analytics's quarterly revenue as of FY2024Q1 was $340.2 million",
            doc_texts)
        self.assertTrue(hit, detail)

    def test_hallucinated_excerpt_misses(self):
        doc_texts = {"doc_a": "Meridian Analytics's quarterly revenue as of "
                             "FY2024Q1 was $340.2 million, a strong result."}
        hit, detail = citation_grounds_claim(
            "The board unanimously approved a merger with an unrelated firm "
            "in a completely different industry.", doc_texts)
        self.assertFalse(hit, detail)

    def test_parse_and_score_sources_block(self):
        answer = (
            "Meridian Analytics reported strong results [1].\n\n"
            "**Sources**\n\n"
            "[1] Meridian Analytics's quarterly revenue as of FY2024Q1 was "
            "$340.2 million\n"
        )
        doc_texts = {"doc_a": "Meridian Analytics's quarterly revenue as of "
                             "FY2024Q1 was $340.2 million, a strong result."}
        parsed = parse_sources_block(answer)
        self.assertIn(1, parsed)
        rows = score_citation_grounding(answer, doc_texts)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["hit"])

    def test_no_sources_block_scores_empty(self):
        """Conditions with no Sources block (B/C/D/E/F) score an empty list
        (not an error) — matches the spec's citation_grounding: null for F."""
        rows = score_citation_grounding("A plain direct answer with no citations.", {})
        self.assertEqual(rows, [])


class TestChecklistAndGate1Score(unittest.TestCase):
    def test_build_checklist_items_excludes_false_clone(self):
        r = _registry()
        items = build_checklist_items(r)
        quantity_fact_ids = {it["fact_id"] for it in items
                            if it["item_type"] == "quantity_correct"}
        self.assertIn("q_0000", quantity_fact_ids)
        self.assertNotIn("q_0001", quantity_fact_ids)  # the false clone

    def test_full_corpus_answer_scores_perfectly(self):
        r = _registry()
        full_answer = (
            "Meridian Analytics (MDA) reported quarterly revenue usd m of "
            "$340.2M as of FY2024Q1. On 2024-03-01, Meridian Analytics opened "
            "a new facility near Elena Voss. Sources disagree on the figure: "
            "one filing states $340.2M while another claims $410.0M; the "
            "correct figure was $340.2M."
        )
        rows = score_answer(full_answer, r)
        scorable = [row for row in rows if row["hit"] is not None]
        self.assertTrue(all(row["hit"] for row in scorable), scorable)

    def test_empty_answer_scores_zero(self):
        r = _registry()
        self.assertEqual(compute_world_scores("", r), 0.0)

    def test_gate1_item_type_restriction(self):
        r = _registry()
        # Only atomic_fact_present + quantity_correct count toward Gate 1
        # (spec Sec 4.1) — a hedge that only ever hits contradiction items
        # (which it doesn't here) should not inflate the gate score.
        score = compute_world_scores(
            "I have no specific information about this.", r,
            item_types=("atomic_fact_present", "quantity_correct"))
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
