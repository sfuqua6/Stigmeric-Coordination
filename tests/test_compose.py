"""Tests for core/compose.py — deterministic composition + polish audits.

See docs/future/STAGE2_CLAIMS_LEDGER_SPEC.md sec7.1
(`tests/test_compose.py` bullet): claim-coverage audit (100% retention,
stricter than the cluster path's >=0.5), no-new-numbers audit (reusing
`ungrounded_numbers` directly), hard-gate-discards-to-deterministic-draft
path, and refuted/unverified claims never appearing in composed output.
"""
from core.claims_ledger import ClaimsLedger, Claim, VerificationRecord
from core.compose import (
    compose_draft,
    claim_ids_in_draft,
    audit_claim_coverage,
    audit_no_new_numbers,
    audit_faithfulness,
    run_polish_audits,
    gate_polish,
)


def _claim(**overrides) -> Claim:
    defaults = dict(
        claim_id="",
        text="Solar capacity grew 30 percent in 2024.",
        kind="fact",
        source_span="Solar capacity grew 30 percent in 2024.",
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


def _ledger_with_verified_contested_refuted_unverified():
    ledger = ClaimsLedger()
    cid_v = ledger.append_claim(_claim(text="Verified claim about solar.", corpus_doc_id="doc_1"))
    cid_c = ledger.append_claim(_claim(text="Contested claim about wind.", corpus_doc_id="doc_2"))
    cid_r = ledger.append_claim(_claim(text="Refuted claim about coal.", corpus_doc_id="doc_3"))
    cid_u = ledger.append_claim(_claim(text="Unverified claim about hydro.", corpus_doc_id="doc_4"))
    ledger.append_verification(cid_v, _rec("pass", checker="checker_0"))
    ledger.append_verification(cid_v, _rec("pass", checker="checker_1"))
    ledger.append_verification(cid_c, _rec("pass", checker="checker_0"))
    ledger.append_verification(cid_c, _rec("fail", checker="checker_1"))
    ledger.append_verification(cid_r, _rec("fail", checker="checker_0"))
    ledger.append_verification(cid_r, _rec("fail", checker="checker_1"))
    return ledger, cid_v, cid_c, cid_r, cid_u


# ---------------------------------------------------------------------------
# Deterministic composition
# ---------------------------------------------------------------------------

def test_compose_draft_includes_verified_and_contested_only():
    ledger, cid_v, cid_c, cid_r, cid_u = _ledger_with_verified_contested_refuted_unverified()
    draft = compose_draft(ledger)
    ids_present = claim_ids_in_draft(draft)
    assert cid_v in ids_present
    assert cid_c in ids_present
    assert cid_r not in ids_present
    assert cid_u not in ids_present


def test_compose_draft_is_byte_identical_for_same_ledger_state():
    ledger, *_ = _ledger_with_verified_contested_refuted_unverified()
    draft1 = compose_draft(ledger)
    draft2 = compose_draft(ledger)
    assert draft1 == draft2


def test_compose_draft_reflects_added_verification_deterministically():
    ledger = ClaimsLedger()
    cid = ledger.append_claim(_claim())
    draft_before = compose_draft(ledger)
    assert cid not in claim_ids_in_draft(draft_before)
    ledger.append_verification(cid, _rec("pass", checker="checker_0"))
    ledger.append_verification(cid, _rec("pass", checker="checker_1"))
    draft_after = compose_draft(ledger)
    assert cid in claim_ids_in_draft(draft_after)


def test_compose_draft_contested_section_cross_links_contradiction():
    ledger = ClaimsLedger()
    cid_a = ledger.append_claim(_claim(text="A says X.", corpus_doc_id="doc_a"))
    cid_b = ledger.append_claim(_claim(text="B says not X.", corpus_doc_id="doc_b"))
    ledger.append_verification(
        cid_a, _rec("fail", check_type="cross_partition", checker="checker_0",
                     contradicting_claim_id=cid_b))
    draft = compose_draft(ledger)
    assert cid_a in draft
    assert cid_b in draft  # cross-linked, even though cid_b has no records of its own


def test_compose_draft_empty_ledger_has_no_claim_sections_populated():
    ledger = ClaimsLedger()
    draft = compose_draft(ledger)
    assert "(none)" in draft
    assert claim_ids_in_draft(draft) == set()


# ---------------------------------------------------------------------------
# Claim-coverage audit (100% retention)
# ---------------------------------------------------------------------------

def test_audit_claim_coverage_passes_when_all_ids_retained():
    draft = "para one [CLAIM_00001] more text.\n\npara two [CLAIM_00002] more."
    polished = "Para ONE, reworded. [CLAIM_00001] Para two, reworded. [CLAIM_00002]"
    result = audit_claim_coverage(draft, polished)
    assert result["passed"] is True
    assert result["missing"] == set()


def test_audit_claim_coverage_fails_when_one_id_dropped():
    draft = "para one [CLAIM_00001] more.\n\npara two [CLAIM_00002] more."
    polished = "Para one, reworded. [CLAIM_00001] Para two with no tag now."
    result = audit_claim_coverage(draft, polished)
    assert result["passed"] is False
    assert result["missing"] == {"CLAIM_00002"}


def test_audit_claim_coverage_stricter_than_half_retention():
    """The cluster-path guard accepts >=0.5 tag retention; the ledger's gate
    must NOT accept a 50% drop the way that guard would."""
    draft = "[CLAIM_00001] [CLAIM_00002] [CLAIM_00003] [CLAIM_00004]"
    polished = "[CLAIM_00001] [CLAIM_00002]"  # 50% retained
    result = audit_claim_coverage(draft, polished)
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# No-new-numbers audit (reuses core.actions.ungrounded_numbers verbatim)
# ---------------------------------------------------------------------------

def test_audit_no_new_numbers_passes_when_number_present_in_draft():
    draft = "Solar capacity grew 30 percent in 2024."
    polished = "In 2024, solar capacity grew by 30 percent, a notable jump."
    result = audit_no_new_numbers(draft, polished)
    assert result["passed"] is True


def test_audit_no_new_numbers_fails_on_fabricated_figure():
    draft = "Solar capacity grew last year."
    polished = "Solar capacity grew 47 percent last year."
    result = audit_no_new_numbers(draft, polished)
    assert result["passed"] is False
    assert any("47" in tok for tok in result["ungrounded_numbers"])


# ---------------------------------------------------------------------------
# Faithfulness audit
# ---------------------------------------------------------------------------

def test_audit_faithfulness_passes_on_overlapping_paragraph():
    ledger = ClaimsLedger()
    cid = ledger.append_claim(_claim(text="Solar capacity grew 30 percent in 2024."))
    polished = f"Solar capacity grew 30 percent in 2024, a significant jump. [{cid}]"
    result = audit_faithfulness(polished, ledger)
    assert result["passed"] is True


def test_audit_faithfulness_fails_on_unrelated_paragraph():
    ledger = ClaimsLedger()
    cid = ledger.append_claim(_claim(text="Solar capacity grew 30 percent in 2024."))
    polished = f"Coal usage declined sharply due to regulation changes. [{cid}]"
    result = audit_faithfulness(polished, ledger)
    assert result["passed"] is False
    assert result["flags"][0]["claim_id"] == cid


def test_audit_faithfulness_fails_on_fabricated_claim_id():
    ledger = ClaimsLedger()
    polished = "Some prose citing a claim that was never extracted. [CLAIM_09999]"
    result = audit_faithfulness(polished, ledger)
    assert result["passed"] is False
    assert result["flags"][0]["issue"] == "fabricated citation"


# ---------------------------------------------------------------------------
# Hard gate: discard-to-deterministic-draft path
# ---------------------------------------------------------------------------

def test_gate_polish_ships_polished_text_when_all_audits_pass():
    ledger, cid_v, cid_c, *_ = _ledger_with_verified_contested_refuted_unverified()
    draft = compose_draft(ledger)
    claim_v = ledger.get(cid_v)
    claim_c = ledger.get(cid_c)
    # Reworded but faithful, full retention, no new numbers.
    polished = (
        f"{claim_v.text} [{cid_v}]\n\n{claim_c.text} [{cid_c}]"
    )
    result = gate_polish(draft, polished, ledger)
    assert result["gated"] is False
    assert result["answer"] == polished


def test_gate_polish_discards_to_draft_when_coverage_audit_fails():
    ledger, cid_v, cid_c, *_ = _ledger_with_verified_contested_refuted_unverified()
    draft = compose_draft(ledger)
    polished = "A polished paragraph that drops every citation tag entirely."
    result = gate_polish(draft, polished, ledger)
    assert result["gated"] is True
    assert result["answer"] == draft


def test_gate_polish_discards_to_draft_when_new_number_fabricated():
    ledger, cid_v, cid_c, *_ = _ledger_with_verified_contested_refuted_unverified()
    draft = compose_draft(ledger)
    claim_v = ledger.get(cid_v)
    claim_c = ledger.get(cid_c)
    polished = (
        f"{claim_v.text} up a stunning 999 percent. [{cid_v}]\n\n"
        f"{claim_c.text} [{cid_c}]"
    )
    result = gate_polish(draft, polished, ledger)
    assert result["gated"] is True
    assert result["answer"] == draft


def test_gate_polish_discards_to_draft_when_polish_output_is_empty():
    ledger, *_ = _ledger_with_verified_contested_refuted_unverified()
    draft = compose_draft(ledger)
    result = gate_polish(draft, "", ledger)
    assert result["gated"] is True
    assert result["answer"] == draft
    assert result["audits"] is None


def test_run_polish_audits_reports_all_three_subaudits():
    ledger, cid_v, cid_c, *_ = _ledger_with_verified_contested_refuted_unverified()
    draft = compose_draft(ledger)
    result = run_polish_audits(draft, draft, ledger)  # unpolished == draft: must pass trivially
    assert result["passed"] is True
    assert set(result.keys()) == {"coverage", "no_new_numbers", "faithfulness", "passed"}
