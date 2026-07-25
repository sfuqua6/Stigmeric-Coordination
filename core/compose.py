"""Deterministic composition + polish-audit functions for the claims ledger.

See docs/future/STAGE2_CLAIMS_LEDGER_SPEC.md sec4. This module has no LLM
call in it at all — `compose_draft()` is a pure function of a
`ClaimsLedger`, and the three audit functions are pure functions of
(draft_text, polished_text[, ledger]). The one constrained polish LLM call
the spec describes (sec4.2) is NOT implemented this round; this module only
provides what that call must be audited against, so it can be wired in a
later round without touching this file's contract.

Adapted-from-existing-code notes (spec sec4, "keep/adapt/retire" table):

  * Section ordering/grouping/citation-rendering below is a new, simpler
    implementation of the JOB `agents/synthesizer.py:_extractive_position`
    does for clusters (verbatim content, deterministically ordered,
    citation-tagged) — re-derived here for CLAIMS instead of clusters
    because clusters/embeddings have no ledger equivalent (spec sec4.1: "a
    ledger with no synthesis LLM doing the heavy lifting doesn't need
    topically coherent paragraphs, it needs correct, ordered ones"). No code
    was copied from that function; the shape (verbatim content + citation
    tag, deterministic order) is the thing being adapted, not the text.
  * `audit_no_new_numbers()` below imports and calls
    `core.actions.ungrounded_numbers` VERBATIM (no copy, no modification) —
    spec sec4.2 item 2 explicitly asks for this reuse.
  * `audit_faithfulness()` below is a NEW, deliberately simplified
    implementation, not a copy of `agents/synthesizer.py:_build_faithfulness_audit`.
    That function does per-sentence isolation with a negation screen across
    cluster content; this one checks 4-gram paragraph-level overlap between
    a polished paragraph and the `Claim.text` of every `[CLAIM_id]` it cites.
    Recorded as a deviation: the spec's sec4.2 item 3 asks for the
    sentence-isolated version to be "adapted"; paragraph-level overlap is a
    strictly weaker (more permissive) check, sufficient to prove the audit
    contract exists and is enforced, but should be tightened to sentence
    granularity in the wiring round before this gate is trusted in
    production the way the cluster-path audit is.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Optional

from core.actions import ungrounded_numbers
from core.claims_ledger import ClaimsLedger, STATUS_CONTESTED, STATUS_VERIFIED

_CLAIM_TAG_RE = re.compile(r"\[(CLAIM_\d+)\]")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _ngrams(text: str, n: int = 4) -> set:
    words = _WORD_RE.findall((text or "").lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


# ---------------------------------------------------------------------------
# Deterministic composition (spec sec4.1)
# ---------------------------------------------------------------------------

def _render_claim_line(claim) -> str:
    return f"{claim.text} [{claim.claim_id}] (source: {claim.corpus_doc_id})"


def _group_by_doc(claims: list) -> "OrderedDict[str, list]":
    """Group by corpus_doc_id, preserving each doc's first-seen order and
    the claims' own extraction order within a doc. Spec sec4.1: "group by
    corpus_doc_id, then by kind, is enough for a first cut" — kind-level
    sub-grouping is deferred (flagged below) since a first cut only needs
    doc-level grouping to be materially more useful than flat ledger order.
    """
    groups: "OrderedDict[str, list]" = OrderedDict()
    for c in claims:
        groups.setdefault(c.corpus_doc_id, []).append(c)
    return groups


def compose_draft(ledger: ClaimsLedger) -> str:
    """Deterministic Section-shaped draft over a ledger's claims.

    Sections (spec sec4.1):
      1. VERIFIED CLAIMS — grouped by corpus_doc_id, verbatim text + citation.
      2. CONTESTED CLAIMS — every contested claim, cross-linked to its
         contradicting claim id if one is on record. No demotion/cap: the
         spec is explicit that "EVERY contested claim appears" here since
         there is no LLM-context budget forcing a cut at this stage.
      3. Refuted/unverified claims are NOT rendered (spec sec4.1 item 3) —
         they remain visible only via the ledger dump / audit trail.

    Same input twice -> byte-identical output: iteration order is the
    ledger's own append order (`ClaimsLedger.all_claims()`), which is
    deterministic, and no wall-clock or RNG value is rendered into the text.
    """
    verified = [c for c in ledger.all_claims() if ledger.status(c.claim_id) == STATUS_VERIFIED]
    contested = [c for c in ledger.all_claims() if ledger.status(c.claim_id) == STATUS_CONTESTED]

    lines: list = []
    lines.append("## VERIFIED CLAIMS")
    if not verified:
        lines.append("(none)")
    else:
        for doc_id, claims in _group_by_doc(verified).items():
            lines.append(f"### {doc_id}")
            for c in claims:
                lines.append(_render_claim_line(c))
    lines.append("")
    lines.append("## CONTESTED CLAIMS")
    if not contested:
        lines.append("(none)")
    else:
        for c in contested:
            counterparts = [
                r.contradicting_claim_id
                for r in ledger.verifications_for(c.claim_id)
                if r.contradicting_claim_id
            ]
            tag = f"[{c.claim_id}] (source: {c.corpus_doc_id})"
            if counterparts:
                tag += f" (contradicts: {', '.join(sorted(set(counterparts)))})"
            lines.append(f"{c.text} {tag}")

    return "\n".join(lines).rstrip() + "\n"


def claim_ids_in_draft(draft: str) -> set:
    """The set of [CLAIM_id] tags actually present in a composed draft —
    the reference set the coverage audit checks a polish pass against."""
    return set(_CLAIM_TAG_RE.findall(draft or ""))


# ---------------------------------------------------------------------------
# Polish-audit functions (spec sec4.2) — pure, no LLM call in this module.
# ---------------------------------------------------------------------------

def audit_claim_coverage(draft: str, polished: str) -> dict:
    """100%-retention audit: every [CLAIM_id] present in `draft` must still
    be present in `polished`. Stricter than the cluster-path's existing
    ">= 0.5 tag retention" guard (spec sec4.2 item 1: "the polish pass has
    no license to drop content... it's polishing an already-complete draft,
    not authoring one").

    Returns {"passed": bool, "required": set, "present": set, "missing": set}.
    """
    required = claim_ids_in_draft(draft)
    present = claim_ids_in_draft(polished)
    missing = required - present
    return {
        "passed": not missing,
        "required": required,
        "present": present,
        "missing": missing,
    }


def audit_no_new_numbers(draft: str, polished: str) -> dict:
    """No-new-numbers audit: any number in `polished` absent from `draft`
    is a fabrication introduced by the polish pass. Calls
    `core.actions.ungrounded_numbers` verbatim (spec sec4.2 item 2) with the
    deterministic draft — not corpus chunks — as the source list.
    """
    bad = ungrounded_numbers(polished, sources=[draft])
    return {"passed": not bad, "ungrounded_numbers": bad}


def audit_faithfulness(polished: str, ledger: ClaimsLedger) -> dict:
    """Paragraph-level 4-gram overlap between `polished` and the `Claim.text`
    of every [CLAIM_id] a paragraph cites. See module docstring for the
    deviation vs the cluster-path's per-sentence + negation-screen version.

    Returns {"passed": bool, "flags": [ {claim_id, paragraph_excerpt}, ... ]}.
    A claim tag whose claim_id doesn't exist in the ledger at all is always
    flagged (fabricated citation, the direct analogue of the cluster path's
    "existence" check).
    """
    flags: list = []
    paragraphs = [p for p in (polished or "").split("\n\n") if p.strip()]
    for para in paragraphs:
        for cid in _CLAIM_TAG_RE.findall(para):
            claim = ledger.get(cid)
            if claim is None:
                flags.append({"claim_id": cid, "issue": "fabricated citation",
                               "paragraph_excerpt": para[:200]})
                continue
            claim_grams = _ngrams(claim.text)
            para_grams = _ngrams(para)
            if claim_grams and not (claim_grams & para_grams):
                flags.append({"claim_id": cid, "issue": "no n-gram overlap",
                               "paragraph_excerpt": para[:200]})
    return {"passed": not flags, "flags": flags}


def run_polish_audits(draft: str, polished: str, ledger: ClaimsLedger) -> dict:
    """Run all three audits and report per-audit + overall pass/fail. Does
    NOT decide fallback — see `gate_polish()` for the hard-gate behavior."""
    coverage = audit_claim_coverage(draft, polished)
    numbers = audit_no_new_numbers(draft, polished)
    faithfulness = audit_faithfulness(polished, ledger)
    return {
        "coverage": coverage,
        "no_new_numbers": numbers,
        "faithfulness": faithfulness,
        "passed": coverage["passed"] and numbers["passed"] and faithfulness["passed"],
    }


def gate_polish(draft: str, polished: Optional[str], ledger: ClaimsLedger) -> dict:
    """Hard gate (spec sec4.2 item 4): if any audit fails, or `polished` is
    falsy (no polish attempted / API failure), discard it and ship the
    deterministic `draft` as-is — the direct structural analogue of
    `agents/synthesizer.py`'s `_AUDIT_HARD_GATE` discarding composed prose in
    favor of `_extractive_position()`. Unlike the cluster path's multi-rung
    fallback chain, this gate can never fail open: the deterministic draft
    is provably claim-complete and number-accurate by construction (it's
    built directly from ledger records, not generated), so it is ALWAYS a
    safe fallback (spec sec4.2, closing paragraph).

    Returns {"answer": str, "gated": bool, "audits": dict|None}.
    """
    if not polished:
        return {"answer": draft, "gated": True, "audits": None}
    audits = run_polish_audits(draft, polished, ledger)
    if audits["passed"]:
        return {"answer": polished, "gated": False, "audits": audits}
    return {"answer": draft, "gated": True, "audits": audits}
