"""Stage 2 claims ledger — see docs/future/STAGE2_CLAIMS_LEDGER_SPEC.md.

Design contract (mirrors core/signal_store.py's no-leak docstring, adapted):

    Claim.text              (str)   — the extracted proposition
    Claim.claim_id          (str)   — ledger-assigned, monotonic
    Claim.status            (str)   — NOT a field on Claim; computed by
                                       ClaimsLedger.status(claim_id) from the
                                       accumulated VerificationRecords. There
                                       is no strength scalar anywhere in this
                                       module — see THEFUTURE.md sec6 ("no
                                       strength scalars, no decay, no
                                       amplification, no pheromones").

`confidence` on Claim is diagnostic-only. Nothing in `compute_status()` or
the survival/halting logic below reads it. If a future patch makes any
transition or survival decision depend on `confidence`, that patch has
reintroduced a strength scalar under a new name and should be rejected on
sight (spec sec1.1).

Status state machine (spec sec1.3):

    unverified --(verification record added)--> verified | contested | refuted

`verified` can later move to `contested` (a new contradiction surfaces), but
nothing ever moves backward to `unverified` once it has left that state —
there is no clock/decay term in this module. `append_verification()` asserts
this invariant on every call (see the assertion below `compute_status`).

Append-only. `append_claim`/`append_verification` only ever add records;
`status()` is a pure function of the accumulated VerificationRecords,
recomputed (and cached) on every append — there is no analogue of
`_apply_dedup_amplify` mutating strength in place.

Thread-safety: one `RLock` guards claim-id allocation and both append paths,
matching `core/signal_store.py`'s `deposit()` lock discipline. Plain reads
(`status`, `claims_by_status`, `claims_for_partition`) are lock-free, reading
from dicts that are only ever appended-to under the lock — safe under
CPython's GIL for the simple dict/list reads used here, matching the pattern
the spec cites for the store's existing lock-free getters (sec1.4).

Deviations from the spec, recorded here so a reviewer doesn't go looking for
something that was deliberately not built this round:

  * `append_claim(claim)` ALWAYS assigns `claim_id` itself (ignoring whatever
    the caller populated on the passed-in Claim) rather than trusting a
    caller-supplied id — the spec's own comment on the field says
    "ledger-assigned", so a caller-supplied id would contradict the
    docstring. Implemented via `dataclasses.replace` since Claim is frozen.
  * `contradiction_candidates()` ranks candidates by `difflib.SequenceMatcher`
    text-ratio rather than the store's embedding-cosine primitive
    (`SignalStore._encode`). The spec suggests reusing that primitive
    (sec3b); this module intentionally has zero dependency on an embedding
    model so it stays a pure-data module with fast, deterministic tests.
    Wiring the real embedding search through is a later (wiring-round) task,
    not a schema change.
  * Corpus-coverage tracking (spec sec5.1) is NOT owned by this module — it
    depends on which chunks an (unbuilt-this-round) EXTRACT worker has
    visited. `should_halt()` below takes `covered_chunk_ids`/`total_chunk_ids`
    as plain arguments so it is testable now against hand-built fixtures and
    can be wired to a real EXTRACT worker's bookkeeping later without any
    change to this function's contract.
  * EXTRACT/VERIFY worker classes and their prompts (spec sec2, sec3) are
    OUT OF SCOPE this round per the task brief ("fixture-buildable core");
    this module only provides the ledger + pure survival/halting functions
    those workers will call once built.
"""
from __future__ import annotations

import dataclasses
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

VALID_KINDS = ("fact", "quantity", "causal", "contested")
VALID_CHECK_TYPES = ("span_entailment", "cross_partition", "quantity_reextraction")
VALID_OUTCOMES = ("pass", "fail", "inconclusive")

STATUS_UNVERIFIED = "unverified"
STATUS_VERIFIED = "verified"
STATUS_CONTESTED = "contested"
STATUS_REFUTED = "refuted"
VALID_STATUSES = (STATUS_UNVERIFIED, STATUS_VERIFIED, STATUS_CONTESTED, STATUS_REFUTED)


@dataclass(frozen=True)
class Claim:
    """A single extracted proposition. See spec sec1.1.

    `confidence` is diagnostic-only (Stage 3 calibration study) and MUST
    NEVER influence status transitions or survival — see module docstring
    and `tests/test_claims_ledger.py::test_confidence_never_affects_survival`.
    """
    claim_id: str
    text: str
    kind: str
    source_span: str
    corpus_doc_id: str
    partition_id: str
    extractor_agent_id: str
    extractor_model: str
    confidence: float
    depends_on: tuple = ()
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"invalid Claim.kind: {self.kind!r} (expected one of {VALID_KINDS})")
        if not self.text:
            raise ValueError("Claim.text must be non-empty")
        if not self.partition_id:
            raise ValueError("Claim.partition_id must be non-empty (partition invariant)")


@dataclass(frozen=True)
class VerificationRecord:
    """A single checker's verdict on one claim. See spec sec1.2."""
    checker_agent_id: str
    checker_model: str
    check_type: str
    outcome: str
    detail: str
    contradicting_claim_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.check_type not in VALID_CHECK_TYPES:
            raise ValueError(
                f"invalid VerificationRecord.check_type: {self.check_type!r} "
                f"(expected one of {VALID_CHECK_TYPES})"
            )
        if self.outcome not in VALID_OUTCOMES:
            raise ValueError(
                f"invalid VerificationRecord.outcome: {self.outcome!r} "
                f"(expected one of {VALID_OUTCOMES})"
            )


# ---------------------------------------------------------------------------
# Survival rule (spec sec3d) — pure function, no ledger state, no LLM.
# ---------------------------------------------------------------------------

def compute_status(records: list) -> str:
    """Pure function: a claim's accumulated VerificationRecords -> status.

    Recomputed from the FULL record set every time (no incremental nudging,
    no order-dependence to debug — spec sec3d). `inconclusive` records are
    excluded from both the pass and fail tallies entirely; they never move
    status in either direction.

    Rule precedence (deliberate, spec table read top-to-bottom with one
    explicit tie-break noted below):

      1. >=1 pass AND >=1 fail (any check type)              -> contested
      2. a cross_partition fail with zero passes              -> contested
      3. a quantity_reextraction fail with zero passes         -> refuted
         (checked only if rule 2 didn't already fire — see the
         tie-break note below)
      4. >=2 distinct-checker fails, zero passes               -> refuted
      5. >=2 distinct-checker passes, zero fails                -> verified
      6. otherwise                                              -> unverified

    Tie-break note (not explicit in the spec table): if a claim has BOTH a
    cross_partition fail and a quantity_reextraction fail with no passes at
    all, rule 2 (contested) fires before rule 3 (refuted) — contested is the
    more conservative label (it keeps the claim visible with its
    contradiction rather than dropping it from the composed answer outright)
    and cross-partition contradiction is, by construction, evidence from
    elsewhere in the SAME corpus rather than a re-extraction mismatch that
    could itself be a re-extraction artifact. Flagged here for a future
    reviewer rather than left implicit.
    """
    passes = [r for r in records if r.outcome == "pass"]
    fails = [r for r in records if r.outcome == "fail"]

    if passes and fails:
        return STATUS_CONTESTED

    cross_partition_fail_no_pass = (
        not passes and any(r.check_type == "cross_partition" for r in fails)
    )
    if cross_partition_fail_no_pass:
        return STATUS_CONTESTED

    quantity_fail_no_pass = (
        not passes and any(r.check_type == "quantity_reextraction" for r in fails)
    )
    if quantity_fail_no_pass:
        return STATUS_REFUTED

    distinct_fail_checkers = {r.checker_agent_id for r in fails}
    if not passes and len(distinct_fail_checkers) >= 2:
        return STATUS_REFUTED

    distinct_pass_checkers = {r.checker_agent_id for r in passes}
    if not fails and len(distinct_pass_checkers) >= 2:
        return STATUS_VERIFIED

    return STATUS_UNVERIFIED


_TERMINAL_RANK = {
    STATUS_UNVERIFIED: 0,
    STATUS_VERIFIED: 1,
    STATUS_CONTESTED: 1,
    STATUS_REFUTED: 1,
}


def _is_illegal_backward_transition(old_status: str, new_status: str) -> bool:
    """True iff `old_status` had already left `unverified` and `new_status`
    would move it back. This is the one hard rule spec sec1.3 states: no
    status ever decays back toward `unverified`."""
    return _TERMINAL_RANK[old_status] > 0 and new_status == STATUS_UNVERIFIED


# ---------------------------------------------------------------------------
# Ledger store
# ---------------------------------------------------------------------------

class ClaimsLedger:
    """Append-only claim + verification store. See spec sec1.4."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._claims: dict = {}
        self._claim_order: list = []
        self._verifications: dict = {}
        self._status_cache: dict = {}
        self._next_claim_num = 1

    # -- writes (locked) ---------------------------------------------------

    def append_claim(self, claim: Claim) -> str:
        """Assign a monotonic claim_id and store `claim`. Returns the id.

        `claim.claim_id` as passed in is IGNORED and overwritten — see the
        module-level "Deviations from the spec" note. `dataclasses.replace`
        is used because Claim is frozen.
        """
        with self._lock:
            cid = f"CLAIM_{self._next_claim_num:05d}"
            self._next_claim_num += 1
            final = dataclasses.replace(claim, claim_id=cid)
            self._claims[cid] = final
            self._claim_order.append(cid)
            self._verifications[cid] = []
            self._status_cache[cid] = STATUS_UNVERIFIED
            return cid

    def append_verification(self, claim_id: str, record: VerificationRecord) -> None:
        with self._lock:
            if claim_id not in self._claims:
                raise KeyError(f"unknown claim_id: {claim_id!r}")
            old_status = self._status_cache.get(claim_id, STATUS_UNVERIFIED)
            self._verifications[claim_id].append(record)
            new_status = compute_status(self._verifications[claim_id])
            assert not _is_illegal_backward_transition(old_status, new_status), (
                f"illegal backward status transition for {claim_id}: "
                f"{old_status!r} -> {new_status!r} (a status may never "
                f"revert to 'unverified' once it has left that state)"
            )
            self._status_cache[claim_id] = new_status

            # spec sec3b: a cross-partition contradiction hit retroactively
            # reclassifies BOTH sides' `kind` to "contested" if they were
            # fact/causal — a claim doesn't get to stay quietly "verified"
            # while something the ledger knows contradicts it exists
            # elsewhere.
            if (record.check_type == "cross_partition"
                    and record.outcome == "fail"
                    and record.contradicting_claim_id):
                self._reclassify_contested(claim_id)
                self._reclassify_contested(record.contradicting_claim_id)

    def _reclassify_contested(self, claim_id: str) -> None:
        """Must be called with `self._lock` already held."""
        claim = self._claims.get(claim_id)
        if claim is None:
            return
        if claim.kind in ("fact", "causal"):
            self._claims[claim_id] = dataclasses.replace(claim, kind="contested")

    # -- reads (lock-free; see class docstring) -----------------------------

    def get(self, claim_id: str) -> Optional[Claim]:
        return self._claims.get(claim_id)

    def status(self, claim_id: str) -> str:
        if claim_id not in self._claims:
            raise KeyError(f"unknown claim_id: {claim_id!r}")
        return self._status_cache.get(claim_id, STATUS_UNVERIFIED)

    def verifications_for(self, claim_id: str) -> list:
        return list(self._verifications.get(claim_id, []))

    def all_claims(self) -> list:
        return [self._claims[cid] for cid in self._claim_order]

    def claims_for_partition(self, partition_id: str) -> list:
        return [c for c in self.all_claims() if c.partition_id == partition_id]

    def claims_by_status(self, status: str) -> list:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status!r}")
        return [c for c in self.all_claims() if self._status_cache.get(c.claim_id) == status]

    def unverified_queue(self, limit: int = 1) -> list:
        """Claims still awaiting a conclusive check, in extraction order —
        the dispatch source for VERIFY workers (spec sec2.1)."""
        out = []
        for cid in self._claim_order:
            if self._status_cache.get(cid) == STATUS_UNVERIFIED:
                out.append(self._claims[cid])
                if len(out) >= limit:
                    break
        return out

    def contradiction_candidates(self, claim: Claim, k: int) -> list:
        """OTHER-partition claims ranked by text-similarity to `claim`.

        See the module "Deviations from the spec" note: this uses
        difflib.SequenceMatcher rather than the store's embedding-cosine
        primitive, to keep this module dependency-free and deterministic.
        """
        import difflib
        others = [
            c for c in self.all_claims()
            if c.partition_id != claim.partition_id and c.claim_id != claim.claim_id
        ]
        scored = [
            (difflib.SequenceMatcher(None, claim.text.lower(), c.text.lower()).ratio(), c)
            for c in others
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in scored[:max(0, k)]]

    # -- persistence ---------------------------------------------------------

    def to_json(self, path) -> None:
        """Atomic write: build the full document, then rename into place —
        no partially-written `path` is ever visible to a concurrent reader.
        Both top-level lists are append-only-shaped (claim/verification
        records in extraction/append order), per spec sec1.4."""
        with self._lock:
            claims_out = [dataclasses.asdict(self._claims[cid]) for cid in self._claim_order]
            verifications_out = []
            for cid in self._claim_order:
                for rec in self._verifications.get(cid, []):
                    d = dataclasses.asdict(rec)
                    d["claim_id"] = cid
                    verifications_out.append(d)
            doc = {
                "claims": claims_out,
                "verifications": verifications_out,
                "next_claim_num": self._next_claim_num,
            }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def from_json(cls, path) -> "ClaimsLedger":
        path = Path(path)
        doc = json.loads(path.read_text(encoding="utf-8"))
        ledger = cls()
        for cdict in doc.get("claims", []):
            cdict = dict(cdict)
            cdict["depends_on"] = tuple(cdict.get("depends_on") or ())
            claim = Claim(**cdict)
            cid = claim.claim_id
            ledger._claims[cid] = claim
            ledger._claim_order.append(cid)
            ledger._verifications[cid] = []
            ledger._status_cache[cid] = STATUS_UNVERIFIED
        for vdict in doc.get("verifications", []):
            vdict = dict(vdict)
            cid = vdict.pop("claim_id")
            record = VerificationRecord(**vdict)
            ledger._verifications[cid].append(record)
            ledger._status_cache[cid] = compute_status(ledger._verifications[cid])
        ledger._next_claim_num = doc.get("next_claim_num", ledger._next_claim_num)
        return ledger


# ---------------------------------------------------------------------------
# Halting logic (spec sec5.1) — pure functions over ledger state + coverage.
# ---------------------------------------------------------------------------

def coverage_ratio(covered_chunk_ids, total_chunk_ids) -> float:
    """Fraction of the assigned corpus that has been read by >=1 EXTRACT
    pass. `total_chunk_ids` empty is defined as fully covered (1.0) so an
    empty-corpus fixture doesn't divide by zero or hang a test forever."""
    total = set(total_chunk_ids)
    if not total:
        return 1.0
    covered = set(covered_chunk_ids) & total
    return len(covered) / len(total)


def queue_drained(ledger: ClaimsLedger, verify_attempts: dict, max_verify_attempts: int) -> bool:
    """True when no claim is both unverified AND under its retry cap.

    `verify_attempts` maps claim_id -> number of VERIFY passes already spent
    on it (a MAX_VERIFY_ATTEMPTS retry cap, spec sec5.1, so an unresolvable
    claim — e.g. all checks come back inconclusive forever — can't hang the
    run; it's logged as a coverage gap by the caller, not silently dropped).
    """
    for claim in ledger.unverified_queue(limit=10 ** 9):
        attempts = verify_attempts.get(claim.claim_id, 0)
        if attempts < max_verify_attempts:
            return False
    return True


def should_halt(
    ledger: ClaimsLedger,
    covered_chunk_ids,
    total_chunk_ids,
    verify_attempts: Optional[dict] = None,
    coverage_floor: float = 0.95,
    max_verify_attempts: int = 3,
):
    """Halt when (a) corpus coverage >= coverage_floor AND (b) the
    verification queue is drained (empty, or every remaining unverified
    claim has exhausted `max_verify_attempts`). Returns (halt: bool,
    reason: str). No novelty tracking, no saturation window, no render-set
    stability polling — spec sec5.1 deliberately keeps this simpler than
    core/convergence.py because there is no strength/cluster-churn metric
    here to defend against.
    """
    verify_attempts = verify_attempts or {}
    cov = coverage_ratio(covered_chunk_ids, total_chunk_ids)
    if cov < coverage_floor:
        return False, f"coverage {cov:.2f} < floor {coverage_floor:.2f}"
    if not queue_drained(ledger, verify_attempts, max_verify_attempts):
        return False, "verification queue not drained"
    return True, f"coverage {cov:.2f} >= floor and queue drained"
