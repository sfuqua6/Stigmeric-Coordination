"""Tests for core/claims_ledger.py — Stage 2 fixture-buildable core.

Covers: schema validation, append-only/frozen invariant, the status state
machine (legal + illegal transitions), confidence-independence, thread-safety
under concurrent appends, JSON persistence round-trip, and the halting
predicate. See docs/future/STAGE2_CLAIMS_LEDGER_SPEC.md sec7.1 for the test
plan this file implements (`tests/test_claims_ledger.py` bullet).
"""
import asyncio
import dataclasses
import json
import threading

import pytest

from core.claims_ledger import (
    Claim,
    ClaimsLedger,
    VerificationRecord,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    STATUS_CONTESTED,
    STATUS_REFUTED,
    compute_status,
    coverage_ratio,
    queue_drained,
    should_halt,
)


def _make_claim(**overrides) -> Claim:
    defaults = dict(
        claim_id="",  # ledger reassigns regardless
        text="Solar capacity grew 30% in 2024.",
        kind="fact",
        source_span="Solar capacity grew 30% in 2024.",
        corpus_doc_id="doc_1",
        partition_id="partition_0",
        extractor_agent_id="extract_0",
        extractor_model="mock-model",
        confidence=0.8,
    )
    defaults.update(overrides)
    return Claim(**defaults)


def _rec(outcome, check_type="span_entailment", checker="checker_0", **kw) -> VerificationRecord:
    return VerificationRecord(
        checker_agent_id=checker,
        checker_model="mock-model",
        check_type=check_type,
        outcome=outcome,
        detail="test",
        **kw,
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_claim_rejects_invalid_kind():
    with pytest.raises(ValueError):
        _make_claim(kind="opinion")


def test_claim_rejects_empty_partition_id():
    with pytest.raises(ValueError):
        _make_claim(partition_id="")


def test_claim_rejects_empty_text():
    with pytest.raises(ValueError):
        _make_claim(text="")


def test_verification_record_rejects_invalid_check_type():
    with pytest.raises(ValueError):
        _rec("pass", check_type="vibes_check")


def test_verification_record_rejects_invalid_outcome():
    with pytest.raises(ValueError):
        _rec("maybe")


# ---------------------------------------------------------------------------
# Frozen / append-only invariant
# ---------------------------------------------------------------------------

def test_claim_is_frozen():
    claim = _make_claim()
    with pytest.raises(dataclasses.FrozenInstanceError):
        claim.text = "mutated"


def test_verification_record_is_frozen():
    rec = _rec("pass")
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.outcome = "fail"


def test_append_claim_assigns_monotonic_ids_and_ignores_caller_id():
    ledger = ClaimsLedger()
    c1 = _make_claim(claim_id="CLAIM_99999")
    cid1 = ledger.append_claim(c1)
    cid2 = ledger.append_claim(_make_claim())
    assert cid1 != "CLAIM_99999"  # caller-supplied id ignored (ledger-assigned)
    assert cid1 == "CLAIM_00001"
    assert cid2 == "CLAIM_00002"


def test_mutating_a_returned_claim_object_does_not_affect_stored_state():
    ledger = ClaimsLedger()
    cid = ledger.append_claim(_make_claim(text="original"))
    fetched = ledger.get(cid)
    # frozen dataclass: cannot mutate in place at all
    with pytest.raises(dataclasses.FrozenInstanceError):
        fetched.text = "tampered"
    assert ledger.get(cid).text == "original"


# ---------------------------------------------------------------------------
# Status state machine — compute_status pure function, survival table (sec3d)
# ---------------------------------------------------------------------------

def test_status_unverified_with_no_records():
    assert compute_status([]) == STATUS_UNVERIFIED


def test_status_unverified_with_only_inconclusive():
    recs = [_rec("inconclusive"), _rec("inconclusive", checker="checker_1")]
    assert compute_status(recs) == STATUS_UNVERIFIED


def test_status_unverified_with_single_pass():
    assert compute_status([_rec("pass")]) == STATUS_UNVERIFIED


def test_status_verified_at_boundary_two_distinct_passes():
    recs = [_rec("pass", checker="checker_0"), _rec("pass", checker="checker_1")]
    assert compute_status(recs) == STATUS_VERIFIED


def test_status_not_verified_when_two_passes_from_same_checker():
    recs = [_rec("pass", checker="checker_0"), _rec("pass", checker="checker_0")]
    assert compute_status(recs) == STATUS_UNVERIFIED


def test_status_contested_with_one_pass_and_one_fail():
    recs = [_rec("pass", checker="checker_0"), _rec("fail", checker="checker_1")]
    assert compute_status(recs) == STATUS_CONTESTED


def test_status_contested_via_cross_partition_with_no_passes():
    recs = [_rec("fail", check_type="cross_partition", checker="checker_0",
                  contradicting_claim_id="CLAIM_00042")]
    assert compute_status(recs) == STATUS_CONTESTED


def test_status_refuted_at_boundary_two_distinct_fails():
    recs = [_rec("fail", checker="checker_0"), _rec("fail", checker="checker_1")]
    assert compute_status(recs) == STATUS_REFUTED


def test_status_not_refuted_when_two_fails_from_same_checker():
    recs = [_rec("fail", checker="checker_0"), _rec("fail", checker="checker_0")]
    assert compute_status(recs) == STATUS_UNVERIFIED


def test_status_refuted_via_single_quantity_reextraction_fail_no_pass():
    recs = [_rec("fail", check_type="quantity_reextraction", checker="checker_0")]
    assert compute_status(recs) == STATUS_REFUTED


def test_status_verified_survives_extra_inconclusive_records():
    recs = [
        _rec("pass", checker="checker_0"),
        _rec("pass", checker="checker_1"),
        _rec("inconclusive", checker="checker_2"),
    ]
    assert compute_status(recs) == STATUS_VERIFIED


def test_ledger_verified_can_move_to_contested_on_later_contradiction():
    ledger = ClaimsLedger()
    cid = ledger.append_claim(_make_claim())
    ledger.append_verification(cid, _rec("pass", checker="checker_0"))
    ledger.append_verification(cid, _rec("pass", checker="checker_1"))
    assert ledger.status(cid) == STATUS_VERIFIED
    ledger.append_verification(
        cid, _rec("fail", check_type="cross_partition", checker="checker_2",
                   contradicting_claim_id="CLAIM_00099"))
    assert ledger.status(cid) == STATUS_CONTESTED


def test_ledger_cross_partition_fail_reclassifies_both_claims_kind():
    ledger = ClaimsLedger()
    cid_a = ledger.append_claim(_make_claim(text="A", kind="fact"))
    cid_b = ledger.append_claim(_make_claim(text="B", kind="causal"))
    ledger.append_verification(
        cid_a, _rec("fail", check_type="cross_partition", checker="checker_0",
                     contradicting_claim_id=cid_b))
    assert ledger.get(cid_a).kind == "contested"
    assert ledger.get(cid_b).kind == "contested"


def test_illegal_backward_transition_is_rejected():
    """Once a claim has left 'unverified', no sequence of new records may
    push compute_status() back to 'unverified'. We can't drive this through
    the public API with legal inputs (append-only + the rule set makes it
    naturally monotone), so this test exercises the assertion directly to
    prove it is load-bearing, not just documentation."""
    from core.claims_ledger import _is_illegal_backward_transition
    assert _is_illegal_backward_transition(STATUS_VERIFIED, STATUS_UNVERIFIED)
    assert _is_illegal_backward_transition(STATUS_CONTESTED, STATUS_UNVERIFIED)
    assert _is_illegal_backward_transition(STATUS_REFUTED, STATUS_UNVERIFIED)
    assert not _is_illegal_backward_transition(STATUS_UNVERIFIED, STATUS_VERIFIED)
    assert not _is_illegal_backward_transition(STATUS_VERIFIED, STATUS_CONTESTED)


# ---------------------------------------------------------------------------
# confidence must NEVER influence status or survival
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("confidence", [0.0, 0.01, 0.5, 0.99, 1.0])
def test_confidence_never_affects_survival(confidence):
    ledger = ClaimsLedger()
    cid = ledger.append_claim(_make_claim(confidence=confidence))
    ledger.append_verification(cid, _rec("pass", checker="checker_0"))
    ledger.append_verification(cid, _rec("pass", checker="checker_1"))
    assert ledger.status(cid) == STATUS_VERIFIED


def test_confidence_does_not_affect_refuted_either():
    for confidence in (0.0, 0.5, 1.0):
        ledger = ClaimsLedger()
        cid = ledger.append_claim(_make_claim(confidence=confidence))
        ledger.append_verification(cid, _rec("fail", checker="checker_0"))
        ledger.append_verification(cid, _rec("fail", checker="checker_1"))
        assert ledger.status(cid) == STATUS_REFUTED


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def test_claims_for_partition_and_by_status():
    ledger = ClaimsLedger()
    cid_a = ledger.append_claim(_make_claim(partition_id="p0"))
    cid_b = ledger.append_claim(_make_claim(partition_id="p1"))
    ledger.append_verification(cid_a, _rec("pass", checker="checker_0"))
    ledger.append_verification(cid_a, _rec("pass", checker="checker_1"))

    assert [c.claim_id for c in ledger.claims_for_partition("p0")] == [cid_a]
    assert [c.claim_id for c in ledger.claims_for_partition("p1")] == [cid_b]
    assert [c.claim_id for c in ledger.claims_by_status(STATUS_VERIFIED)] == [cid_a]
    assert [c.claim_id for c in ledger.claims_by_status(STATUS_UNVERIFIED)] == [cid_b]


def test_unverified_queue_respects_limit_and_order():
    ledger = ClaimsLedger()
    ids = [ledger.append_claim(_make_claim()) for _ in range(3)]
    q = ledger.unverified_queue(limit=2)
    assert [c.claim_id for c in q] == ids[:2]


def test_contradiction_candidates_excludes_same_partition_and_self():
    ledger = ClaimsLedger()
    cid_self = ledger.append_claim(_make_claim(text="renewables grew fast", partition_id="p0"))
    ledger.append_claim(_make_claim(text="renewables grew quickly", partition_id="p0"))  # same partition
    cid_other = ledger.append_claim(_make_claim(text="renewables grew rapidly", partition_id="p1"))
    claim_self = ledger.get(cid_self)
    candidates = ledger.contradiction_candidates(claim_self, k=5)
    assert [c.claim_id for c in candidates] == [cid_other]


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------

def test_concurrent_append_claim_produces_no_duplicate_ids():
    ledger = ClaimsLedger()
    n_workers = 20
    ids_per_thread = 10
    collected = []
    collected_lock = threading.Lock()

    def worker():
        local_ids = []
        for _ in range(ids_per_thread):
            local_ids.append(ledger.append_claim(_make_claim()))
        with collected_lock:
            collected.extend(local_ids)

    threads = [threading.Thread(target=worker) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(collected) == n_workers * ids_per_thread
    assert len(set(collected)) == len(collected)  # no duplicates


def test_concurrent_append_claim_under_asyncio_tasks():
    """Direct analogue of the worker-pool's async-task concurrency shape
    (spec sec1.4 thread-safety note). Uses asyncio.run (matches this repo's
    existing sync-wrapper test convention, e.g. tests/test_baseline_mode.py)
    rather than pytest.mark.asyncio, which is not a configured plugin here.
    """
    ledger = ClaimsLedger()

    async def worker(n):
        ids = []
        for _ in range(n):
            ids.append(await asyncio.to_thread(ledger.append_claim, _make_claim()))
        return ids

    async def run_all():
        return await asyncio.gather(*[worker(15) for _ in range(10)])

    results = asyncio.run(run_all())
    all_ids = [cid for ids in results for cid in ids]
    assert len(all_ids) == 150
    assert len(set(all_ids)) == 150


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

def test_json_round_trip_preserves_claims_and_status(tmp_path):
    ledger = ClaimsLedger()
    cid_v = ledger.append_claim(_make_claim(text="verified one"))
    cid_c = ledger.append_claim(_make_claim(text="contested one"))
    cid_u = ledger.append_claim(_make_claim(text="unverified one"))
    ledger.append_verification(cid_v, _rec("pass", checker="checker_0"))
    ledger.append_verification(cid_v, _rec("pass", checker="checker_1"))
    ledger.append_verification(cid_c, _rec("pass", checker="checker_0"))
    ledger.append_verification(cid_c, _rec("fail", checker="checker_1"))

    path = tmp_path / "ledger.json"
    ledger.to_json(path)
    assert path.exists()

    reloaded = ClaimsLedger.from_json(path)
    assert reloaded.status(cid_v) == STATUS_VERIFIED
    assert reloaded.status(cid_c) == STATUS_CONTESTED
    assert reloaded.status(cid_u) == STATUS_UNVERIFIED
    assert reloaded.get(cid_v).text == "verified one"
    assert [c.claim_id for c in reloaded.all_claims()] == [cid_v, cid_c, cid_u]

    # A claim appended after reload must not collide with restored ids.
    new_cid = reloaded.append_claim(_make_claim())
    assert new_cid not in (cid_v, cid_c, cid_u)


def test_json_document_is_valid_plain_json(tmp_path):
    ledger = ClaimsLedger()
    cid = ledger.append_claim(_make_claim())
    ledger.append_verification(cid, _rec("pass"))
    path = tmp_path / "ledger.json"
    ledger.to_json(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(doc["claims"], list)
    assert isinstance(doc["verifications"], list)
    assert doc["verifications"][0]["claim_id"] == cid


# ---------------------------------------------------------------------------
# Halting predicate (spec sec5.1)
# ---------------------------------------------------------------------------

def test_coverage_ratio_basic():
    assert coverage_ratio({"c1", "c2"}, {"c1", "c2", "c3", "c4"}) == 0.5
    assert coverage_ratio(set(), set()) == 1.0
    assert coverage_ratio({"c1"}, set()) == 1.0


def test_queue_drained_true_when_empty():
    ledger = ClaimsLedger()
    assert queue_drained(ledger, {}, max_verify_attempts=3) is True


def test_queue_drained_false_when_unverified_under_retry_cap():
    ledger = ClaimsLedger()
    cid = ledger.append_claim(_make_claim())
    assert queue_drained(ledger, {cid: 0}, max_verify_attempts=3) is False


def test_queue_drained_true_when_unverified_exhausted_retry_cap():
    ledger = ClaimsLedger()
    cid = ledger.append_claim(_make_claim())
    assert queue_drained(ledger, {cid: 3}, max_verify_attempts=3) is True


def test_should_halt_false_when_coverage_incomplete():
    ledger = ClaimsLedger()
    halt, reason = should_halt(ledger, covered_chunk_ids={"a"}, total_chunk_ids={"a", "b"})
    assert halt is False
    assert "coverage" in reason


def test_should_halt_true_when_coverage_and_queue_both_satisfied():
    ledger = ClaimsLedger()
    cid = ledger.append_claim(_make_claim())
    ledger.append_verification(cid, _rec("pass", checker="checker_0"))
    ledger.append_verification(cid, _rec("pass", checker="checker_1"))
    halt, reason = should_halt(
        ledger, covered_chunk_ids={"a", "b"}, total_chunk_ids={"a", "b"},
        verify_attempts={cid: 0},
    )
    assert halt is True


def test_should_halt_false_when_queue_not_drained_despite_full_coverage():
    ledger = ClaimsLedger()
    cid = ledger.append_claim(_make_claim())
    halt, reason = should_halt(
        ledger, covered_chunk_ids={"a"}, total_chunk_ids={"a"},
        verify_attempts={cid: 0}, max_verify_attempts=3,
    )
    assert halt is False
    assert "queue" in reason


def test_should_halt_true_when_unresolvable_claim_exhausts_retry_cap():
    ledger = ClaimsLedger()
    cid = ledger.append_claim(_make_claim())
    ledger.append_verification(cid, _rec("inconclusive"))  # never conclusive
    halt, reason = should_halt(
        ledger, covered_chunk_ids={"a"}, total_chunk_ids={"a"},
        verify_attempts={cid: 3}, max_verify_attempts=3,
    )
    assert halt is True
