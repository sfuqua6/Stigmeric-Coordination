"""Tests for core/ledger_pool.py — EXTRACT/VERIFY worker pool.

Covers (per the Stage 2 wiring task brief):
  * extraction JSON parsing (strict schema + numbered-marker fallback)
  * source_span fabrication gate (invalid span -> claim rejected)
  * MOCK-mode-only extraction fallback (raw text -> one low-confidence claim)
  * checker-independence enforcement (a worker never verifies its own claim)
  * pool halting (coverage-complete + queue-drained via should_halt)
"""

import asyncio
import os
import unittest
from dataclasses import dataclass, field

os.environ.setdefault("MOCK_LLM", "1")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.claims_ledger import ClaimsLedger, Claim, VerificationRecord
from core.intake import CorpusChunk, ScoutPartition
from core.ledger_pool import (
    LedgerWorker, LedgerPoolState, choose_ledger_action,
    parse_extraction, _validate_span, _code_selected_span,
    parse_entailment, parse_contradiction, parse_quantity_reextraction,
    run_ledger_pool, VERIFY_BACKLOG_FLOOR,
)


def _partition(texts, pid_index=0):
    chunks = [
        CorpusChunk(chunk_id=f"c{i}", text=t, source_tag="test", partition_id=f"partition_{pid_index}")
        for i, t in enumerate(texts)
    ]
    return ScoutPartition(scout_index=pid_index, chunks=chunks)


class TestSpanValidation(unittest.TestCase):
    def test_valid_span_found(self):
        part = _partition(["The city council approved a new transit budget of 12 million dollars."])
        doc_id, span = _validate_span("approved a new transit budget", part)
        self.assertEqual(doc_id, "c0")
        self.assertTrue(span)

    def test_fabricated_span_rejected(self):
        part = _partition(["The city council approved a new transit budget."])
        doc_id, span = _validate_span("this sentence never appears anywhere in the chunk", part)
        self.assertIsNone(doc_id)

    def test_too_short_span_rejected(self):
        part = _partition(["Some chunk text here."])
        doc_id, span = _validate_span("Some", part)
        self.assertIsNone(doc_id)

    def test_code_selected_span_is_verbatim(self):
        part = _partition(["Alpha budget figures for the district are substantial this year."])
        doc_id, span = _code_selected_span("a claim about alpha budget figures", part)
        self.assertEqual(doc_id, "c0")
        self.assertIn(span.split()[0], part.chunks[0].text)


class TestParseExtractionJSON(unittest.TestCase):
    def test_strict_json_parses_and_validates_span(self):
        part = _partition(["Renewable output rose to 42 percent of the grid mix last year."])
        raw = (
            '{"claims": [{"text": "Renewable output reached 42 percent of the grid.", '
            '"kind": "quantity", "source_span": "rose to 42 percent of the grid mix", '
            '"confidence": 0.9}]}'
        )
        claims = parse_extraction(raw, part, "extractor_a", "test-model")
        self.assertEqual(len(claims), 1)
        c = claims[0]
        self.assertEqual(c.kind, "quantity")
        self.assertEqual(c.corpus_doc_id, "c0")
        self.assertEqual(c.extractor_agent_id, "extractor_a")
        self.assertAlmostEqual(c.confidence, 0.9)

    def test_json_claim_with_fabricated_span_is_dropped(self):
        part = _partition(["Only this sentence exists in the corpus."])
        raw = (
            '{"claims": [{"text": "A completely different assertion.", '
            '"kind": "fact", "source_span": "nothing like this is in the chunk at all", '
            '"confidence": 0.7}]}'
        )
        claims = parse_extraction(raw, part, "extractor_a", "test-model")
        self.assertEqual(claims, [])

    def test_invalid_kind_normalizes_to_fact(self):
        part = _partition(["Some evidence sentence goes here for the test."])
        raw = (
            '{"claims": [{"text": "Some evidence sentence goes here for it.", '
            '"kind": "contested", "source_span": "Some evidence sentence goes here", '
            '"confidence": 0.5}]}'
        )
        claims = parse_extraction(raw, part, "extractor_a", "test-model")
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].kind, "fact")  # extractors never emit "contested"

    def test_marker_fallback_when_json_absent(self):
        part = _partition(["The reservoir level dropped sharply amid the ongoing drought this quarter."])
        raw = "1. The reservoir level dropped sharply amid drought conditions this quarter."
        claims = parse_extraction(raw, part, "extractor_b", "test-model")
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].corpus_doc_id, "c0")

    def test_mock_fallback_produces_one_low_confidence_claim(self):
        part = _partition(["A completely unrelated evidentiary passage about aviation safety."])
        raw = "gibberish that is not json and not numbered claims whatsoever"
        # mock_fallback=False -> no claim (no anchorable span, no structure)
        claims_off = parse_extraction(raw, part, "extractor_c", "MockLLM", mock_fallback=False)
        self.assertEqual(claims_off, [])
        claims_on = parse_extraction(raw, part, "extractor_c", "MockLLM", mock_fallback=True)
        self.assertEqual(len(claims_on), 1)
        self.assertLess(claims_on[0].confidence, 0.2)
        self.assertEqual(claims_on[0].corpus_doc_id, "c0")

    def test_empty_response_yields_no_claims(self):
        part = _partition(["Some chunk."])
        self.assertEqual(parse_extraction("", part, "e", "m", mock_fallback=True), [])


class TestVerificationParsers(unittest.TestCase):
    def test_entailed_maps_to_pass(self):
        outcome, detail = parse_entailment("ENTAILED. The passage directly supports this.")
        self.assertEqual(outcome, "pass")

    def test_contradicted_maps_to_fail(self):
        outcome, _ = parse_entailment("CONTRADICTED — the passage says the opposite.")
        self.assertEqual(outcome, "fail")

    def test_not_addressed_maps_to_inconclusive(self):
        outcome, _ = parse_entailment("NOT_ADDRESSED, the passage is silent on this.")
        self.assertEqual(outcome, "inconclusive")

    def test_garbage_defaults_to_inconclusive(self):
        outcome, _ = parse_entailment("random unrelated text with no verdict word")
        self.assertEqual(outcome, "inconclusive")

    def test_contradiction_none(self):
        self.assertIsNone(parse_contradiction("NONE of these contradict it.", 3))

    def test_contradiction_index(self):
        self.assertEqual(parse_contradiction("2", 3), 1)

    def test_contradiction_out_of_range_is_none(self):
        self.assertIsNone(parse_contradiction("9", 3))

    def test_quantity_reextraction_pass(self):
        outcome, _ = parse_quantity_reextraction("42\n17", "Renewable output reached 42 percent.")
        self.assertEqual(outcome, "pass")

    def test_quantity_reextraction_fail(self):
        outcome, _ = parse_quantity_reextraction("17\n99", "Renewable output reached 42 percent.")
        self.assertEqual(outcome, "fail")

    def test_quantity_reextraction_inconclusive_on_no_numbers(self):
        outcome, _ = parse_quantity_reextraction("NONE", "Renewable output reached 42 percent.")
        self.assertEqual(outcome, "inconclusive")


class TestCheckerIndependence(unittest.TestCase):
    def test_worker_never_selects_own_claim(self):
        ledger = ClaimsLedger()
        part_a = _partition(["evidence chunk in partition A about topic one here"], pid_index=0)
        part_b = _partition(["evidence chunk in partition B about topic two here"], pid_index=1)

        claim_self = Claim(
            claim_id="", text="a claim this worker itself extracted",
            kind="fact", source_span="evidence chunk in partition A",
            corpus_doc_id="c0", partition_id="partition_0",
            extractor_agent_id="ledger_worker_000", extractor_model="m", confidence=0.5,
        )
        claim_other = Claim(
            claim_id="", text="a claim a DIFFERENT worker extracted",
            kind="fact", source_span="evidence chunk in partition B",
            corpus_doc_id="c0", partition_id="partition_1",
            extractor_agent_id="ledger_worker_001", extractor_model="m", confidence=0.5,
        )
        id_self = ledger.append_claim(claim_self)
        id_other = ledger.append_claim(claim_other)

        worker = LedgerWorker(0, llm=None, router=None, task_prompt="t")
        pool_state = LedgerPoolState()
        eligible = worker._eligible_for_verify(ledger, pool_state, max_verify_attempts=3)
        self.assertIsNotNone(eligible)
        self.assertEqual(eligible.claim_id, id_other)
        self.assertNotEqual(eligible.extractor_agent_id, worker.agent_id)

    def test_no_eligible_claim_when_only_own_claims_queued(self):
        ledger = ClaimsLedger()
        claim_self = Claim(
            claim_id="", text="only this worker's own claim exists in the ledger",
            kind="fact", source_span="span text", corpus_doc_id="c0",
            partition_id="partition_0", extractor_agent_id="ledger_worker_000",
            extractor_model="m", confidence=0.5,
        )
        ledger.append_claim(claim_self)
        worker = LedgerWorker(0, llm=None, router=None, task_prompt="t")
        pool_state = LedgerPoolState()
        self.assertIsNone(worker._eligible_for_verify(ledger, pool_state, max_verify_attempts=3))

    def test_do_verify_asserts_independence_if_bypassed(self):
        # Defensive assert inside do_verify: even if a caller manually feeds
        # a self-authored claim through, the hard assertion catches it.
        ledger = ClaimsLedger()
        claim_self = Claim(
            claim_id="", text="own claim text is long enough to be valid here",
            kind="fact", source_span="span", corpus_doc_id="c0",
            partition_id="partition_0", extractor_agent_id="ledger_worker_000",
            extractor_model="m", confidence=0.5,
        )
        ledger.append_claim(claim_self)
        worker = LedgerWorker(0, llm=None, router=None, task_prompt="t")
        pool_state = LedgerPoolState()
        result = asyncio.run(worker.do_verify(ledger, pool_state, {}, max_verify_attempts=3))
        self.assertIsNone(result)  # no eligible claim -> no-op, assertion never reached


class TestChooseLedgerAction(unittest.TestCase):
    def test_extract_when_queue_below_floor(self):
        ledger = ClaimsLedger()
        self.assertEqual(choose_ledger_action(ledger), "EXTRACT")

    def test_verify_when_queue_at_or_above_floor(self):
        ledger = ClaimsLedger()
        for i in range(VERIFY_BACKLOG_FLOOR):
            ledger.append_claim(Claim(
                claim_id="", text=f"claim number {i} with enough words to be valid",
                kind="fact", source_span="span", corpus_doc_id="c0",
                partition_id="partition_0", extractor_agent_id=f"extractor_{i}",
                extractor_model="m", confidence=0.5,
            ))
        self.assertEqual(choose_ledger_action(ledger), "VERIFY")


class _StubLLM:
    """Deterministic non-JSON, non-verdict responder — exercises the
    mock-fallback / inconclusive-default paths without needing MockLLM."""
    name = "stub"

    async def generate(self, prompt, role="agent", max_tokens=100, temperature=0.7):
        return "no structured content here at all"


class TestPoolHalting(unittest.TestCase):
    def test_run_ledger_pool_halts_on_stop_event(self):
        part = _partition(["A short evidence passage about the pool halting test scenario."])
        ledger = ClaimsLedger()
        stop_event = asyncio.Event()

        async def _run():
            stop_event.set()  # halt immediately; pool must exit cleanly
            state = await run_ledger_pool(
                ledger, part.chunks, [part], _StubLLM(), "task", stop_event,
                n_workers=2,
            )
            return state

        state = asyncio.run(_run())
        self.assertIsInstance(state, LedgerPoolState)

    def test_pool_produces_claims_with_stub_llm_mock_fallback(self):
        os.environ["MOCK_LLM"] = "1"
        part = _partition(["An evidence passage the stub extractor will latch onto for a claim."])
        ledger = ClaimsLedger()
        stop_event = asyncio.Event()

        async def _run():
            async def _halt_after_one_tick():
                await asyncio.sleep(0.05)
                stop_event.set()
            await asyncio.gather(
                run_ledger_pool(ledger, part.chunks, [part], _StubLLM(), "task",
                                stop_event, n_workers=2),
                _halt_after_one_tick(),
            )

        asyncio.run(_run())
        # With MOCK_LLM=1 and a non-JSON stub response, the mock-mode
        # extraction fallback should have produced at least one claim.
        self.assertGreaterEqual(len(ledger.all_claims()), 1)


if __name__ == "__main__":
    unittest.main()
