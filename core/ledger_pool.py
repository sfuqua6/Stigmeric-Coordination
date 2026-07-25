"""Stage 2 EXTRACT/VERIFY worker pool — see docs/future/STAGE2_CLAIMS_LEDGER_SPEC.md sec2/sec3.

Design choice (spec sec2.1, "recommendation" heading): the spec recommends
"keep run_pool's async worker-loop shape, replace the action set." This
module follows that recommendation's SHAPE (asyncio.gather over N worker
coroutines against one shared mutable store under a lock, blocking calls
wrapped so they never stall the event loop) but is a NEW, small module
rather than a subclass of core/worker_pool.py's `Worker`. Reason: `Worker` is
built around `choose_action`'s share-based balancing over the 7-action
registry, `ClusterRegistry`/genome-cache plumbing, and `SignalStore`'s
partition-inheritance/dedup machinery — none of which the ledger has an
equivalent for (spec sec2.1: "That is simple enough it may not need a
general `choose_action` abstraction at all"). Subclassing `Worker` and
stripping all of that out would leave more dead surface than writing the
~2-action dispatcher fresh. What IS reused, verbatim or near-verbatim, per
the spec's own keep/adapt table: `core.actions.extract_json_object` and
`split_scout_claims` (JSON-first / marker-fallback parsing contract),
`core.actions.ungrounded_numbers` / `_normalize_number` (number grounding),
`agents.base.strip_reasoning` (scratchpad stripping), and the
asyncio.to_thread-free "just await llm.generate()" concurrency shape (no
blocking I/O happens in this module — no search calls — so no to_thread
wrapping is needed here, unlike worker_pool.py's SCOUT/VALIDATE actions).

No-leak boundary enforced here (spec sec2.4):
  * EXTRACT sees ONLY its assigned partition's chunk text. No other
    worker's claim text, no VerificationRecord.detail.
  * VERIFY sees the single claim under check, the cited chunk's text (for
    span_entailment), other claims' `.text` from OTHER partitions (for
    cross_partition — the spec's one deliberate exception: claim text is a
    finished falsifiable proposition, not another agent's reasoning chain),
    and never which worker wrote a claim or that worker's model family.

Checker-independence (spec sec2.1): a claim's verifier must not be its
extractor. Enforced in `_eligible_for_verify` by filtering the candidate
queue on `claim.extractor_agent_id != self.agent_id` before a claim is ever
handed to `_run_check`.
"""
from __future__ import annotations

import asyncio
import difflib
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from agents.base import strip_reasoning
from core.actions import (
    extract_json_object,
    split_scout_claims,
    ungrounded_numbers,
    _normalize_number,
    _NUMBER_TOKEN_RE,
    _CLAIM_MARKER_RE,
)
from core.claims_ledger import Claim, ClaimsLedger, VerificationRecord, should_halt
from core.filters import is_junk_output

EXTRACT = "EXTRACT"
VERIFY = "VERIFY"

# ---------------------------------------------------------------------------
# Tunables (self-contained env vars — this module owns them rather than
# touching core/config.py, per the "keep diffs to existing modules minimal"
# instruction; nothing here is read by the strength-store pipeline).
# ---------------------------------------------------------------------------

VERIFY_BACKLOG_FLOOR = int(os.environ.get("SWARM_LEDGER_VERIFY_FLOOR", "3"))
EXTRACT_CLAIMS_PER_CALL = int(os.environ.get("SWARM_LEDGER_EXTRACT_K", "3"))
MAX_TOKENS_EXTRACT = int(os.environ.get("SWARM_LEDGER_MAX_TOKENS_EXTRACT", "500"))
MAX_VERIFY_ATTEMPTS = int(os.environ.get("SWARM_MAX_VERIFY_ATTEMPTS", "3"))
LEDGER_COVERAGE_FLOOR = float(os.environ.get("SWARM_LEDGER_COVERAGE_FLOOR", "0.95"))
LEDGER_MAX_TIME_S = float(os.environ.get("SWARM_LEDGER_MAX_TIME_S", "600"))
_MIN_SPAN_CHARS = 8


def _is_mock_mode() -> bool:
    return os.environ.get("MOCK_LLM", "").strip() not in ("", "0", "false", "False")


def choose_ledger_action(ledger: ClaimsLedger, floor: int = VERIFY_BACKLOG_FLOOR) -> str:
    """Floor policy (spec sec2.1): VERIFY whenever the unverified backlog is
    >= floor, else EXTRACT. No share-based balancing, no cold-start
    weighting, no cluster-local biases — a ledger has no strength dynamics
    for any of that machinery to balance against."""
    backlog = len(ledger.unverified_queue(limit=floor + 1))
    return VERIFY if backlog >= floor else EXTRACT


@dataclass
class LedgerPoolState:
    iterations: int = 0
    extract_count: int = 0
    extract_rejected: int = 0
    verify_count: int = 0
    verify_attempts: dict = field(default_factory=dict)     # claim_id -> attempts
    covered_chunk_ids: set = field(default_factory=set)
    action_log: list = field(default_factory=list)          # (worker_id, action)


# ---------------------------------------------------------------------------
# Span validation — the extraction-time analogue of ungrounded_numbers().
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _norm_ws(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())


def _validate_span(span: str, partition) -> tuple:
    """Return (corpus_doc_id, verbatim_span) if `span` is a (whitespace-
    normalized) substring of some chunk in `partition`, else (None, None).

    This is the extraction-time fabrication gate (spec sec2.3): a
    model-supplied source_span that doesn't appear in the partition's chunks
    is rejected outright, stricter than `ungrounded_numbers()`'s tag-but-keep
    behavior for numbers with no evidence shown at all — the ledger has no
    "no evidence shown" case, so there's no legitimate reason to keep an
    unanchored claim.
    """
    norm_span = _norm_ws(span).lower()
    if len(norm_span) < _MIN_SPAN_CHARS:
        return None, None
    for chunk in getattr(partition, "chunks", []) or []:
        norm_chunk = _norm_ws(chunk.text).lower()
        if norm_span in norm_chunk:
            return chunk.chunk_id, span.strip()
    return None, None


def _code_selected_span(claim_text: str, partition) -> tuple:
    """Fallback span selection for claims that arrived with no (or an
    invalid) model-supplied source_span — e.g. the marker-fallback parse
    path, which has no per-claim span at all. The span is CODE-selected (the
    best-matching chunk's own text, sliced verbatim) rather than model-
    supplied, so it is a substring of the chunk by construction and needs no
    separate validation. This is a lower-confidence path; callers should
    treat it as diagnostic (it proves plumbing, not that the claim is
    actually about that chunk)."""
    chunks = getattr(partition, "chunks", []) or []
    if not chunks:
        return None, None
    best = max(
        chunks,
        key=lambda c: difflib.SequenceMatcher(None, claim_text.lower(), c.text.lower()).ratio(),
    )
    excerpt = _norm_ws(best.text)[:200]
    if len(excerpt) < _MIN_SPAN_CHARS:
        return None, None
    return best.chunk_id, excerpt


# ---------------------------------------------------------------------------
# Extraction prompt + parsing (spec sec2.3 — guided-JSON precedent, commit 08c65f4)
# ---------------------------------------------------------------------------

def extraction_prompt(task_prompt: str, partition, chunk_offset: int = 0,
                       k: int = EXTRACT_CLAIMS_PER_CALL) -> str:
    evidence = partition.render(offset=chunk_offset) if partition is not None else ""
    if not evidence:
        evidence = "(no corpus partition assigned)"
    return (
        f"TASK: {task_prompt}\n\n"
        f"Extract up to {k} DISTINCT, checkable claims from the evidence "
        f"below. Each claim must be directly supported by a verbatim excerpt "
        f"you quote from the evidence (source_span) — do not paraphrase the "
        f"excerpt, copy it exactly. Classify each claim's kind as "
        f'"fact" (a plain assertion), "quantity" (contains a number), or '
        f'"causal" (an X causes/enables/prevents Y claim).\n\n'
        f"---EVIDENCE---\n{evidence}\n---END EVIDENCE---\n\n"
        f"Reply with ONLY this JSON object:\n"
        f'{{"claims": ['
        f'{{"text": "...", "kind": "fact", "source_span": "<verbatim excerpt>", '
        f'"confidence": 0.8}}, "..."]}}\n'
        f"JSON:"
    )


def parse_extraction(raw: str, partition, extractor_agent_id: str,
                      extractor_model: str, mock_fallback: bool = False) -> list:
    """Parse an EXTRACT worker's raw completion into validated `Claim`s
    (unassigned claim_id — the ledger assigns it on append).

    JSON-first (guided-JSON precedent, commit 08c65f4's `split_scout_claims`)
    with the same numbered-marker fallback `split_scout_claims` already uses
    when JSON parsing fails — no third parser invented (spec sec2.3).
    Every candidate's `source_span` must validate as a real substring of the
    partition (`_validate_span`); marker-fallback candidates carry no
    model-supplied span, so they get a code-selected one (`_code_selected_span`).

    `mock_fallback`: MOCK-MODE-ONLY plumbing path (spec task item 3). MockLLM
    emits SHA1-seeded phrases, never JSON, so with `mock_fallback=True` a
    completely unparseable response becomes ONE low-confidence claim built
    from the raw text — this exists purely so `MOCK_LLM=1` runs exercise the
    full pipeline end-to-end. It must never fire outside MOCK_LLM=1 (the
    caller gates this) and the confidence is deliberately low (0.15) so a
    reviewer skimming ledger.json for real-run artifacts is not misled.
    """
    text = strip_reasoning((raw or "").strip())
    if not text:
        return []

    raw_candidates = []
    obj = extract_json_object(text)
    if obj is not None and isinstance(obj.get("claims"), list):
        for item in obj["claims"]:
            if isinstance(item, dict) and item.get("text"):
                raw_candidates.append(dict(item))

    # Numbered-marker fallback (split_scout_claims), but ONLY when the text
    # actually contains numbered-list structure — split_scout_claims returns
    # `[text]` unchanged for genuinely unstructured text too (its own no-op
    # fallback), and treating THAT as a valid extraction would let any plain
    # completion through with a code-selected span, defeating the
    # fabrication gate. `_CLAIM_MARKER_RE` (imported from core.actions, the
    # same regex `split_scout_claims` itself uses to find markers) tells the
    # two cases apart.
    used_marker_fallback = False
    if not raw_candidates and _CLAIM_MARKER_RE.search(text):
        pieces = split_scout_claims(text)
        used_marker_fallback = True
        for p in pieces:
            if p and len(p.split()) >= 4 and not is_junk_output(p):
                raw_candidates.append({"text": p, "kind": "fact", "source_span": "",
                                       "confidence": 0.5})

    claims: list = []
    for item in raw_candidates:
        txt = str(item.get("text", "")).strip()
        if not txt or is_junk_output(txt):
            continue
        kind = str(item.get("kind", "fact")).strip().lower()
        if kind not in ("fact", "quantity", "causal"):
            kind = "fact"   # extractors never emit "contested" (spec sec1.1)
        try:
            conf = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        span = str(item.get("source_span", "")).strip()
        doc_id, verbatim_span = (None, None)
        if span:
            doc_id, verbatim_span = _validate_span(span, partition)
        if doc_id is None:
            # Either no span was supplied (marker-fallback path) or the
            # model's span didn't validate — try the code-selected fallback
            # ONLY for the marker-fallback path (JSON-path fabricated spans
            # are rejected outright, not silently re-anchored, per spec).
            if used_marker_fallback or not span:
                doc_id, verbatim_span = _code_selected_span(txt, partition)
            if doc_id is None:
                continue  # fabricated / unanchored citation — reject
        claims.append(Claim(
            claim_id="", text=txt, kind=kind, source_span=verbatim_span,
            corpus_doc_id=doc_id, partition_id=partition.partition_id,
            extractor_agent_id=extractor_agent_id, extractor_model=extractor_model,
            confidence=conf,
        ))

    if not claims and mock_fallback:
        # --- MOCK-MODE-ONLY plumbing fallback (see docstring). Never fires
        # when MOCK_LLM is unset; the caller enforces this. ---
        doc_id, span = _code_selected_span(text, partition)
        if doc_id is not None:
            claims.append(Claim(
                claim_id="", text=text[:300], kind="fact", source_span=span,
                corpus_doc_id=doc_id, partition_id=partition.partition_id,
                extractor_agent_id=extractor_agent_id, extractor_model=extractor_model,
                confidence=0.15,
            ))
    return claims


# ---------------------------------------------------------------------------
# Verification prompts + parsing (spec sec3a/3b/3c)
# ---------------------------------------------------------------------------

_ENTAIL_RE = re.compile(r"\b(ENTAILED|CONTRADICTED|NOT_ADDRESSED)\b", re.IGNORECASE)
_CONTRA_IDX_RE = re.compile(r"\b(\d+)\b")
_NONE_RE = re.compile(r"\bNONE\b", re.IGNORECASE)


def span_entailment_prompt(claim: Claim, chunk_text: str) -> str:
    return (
        f"CLAIM: {claim.text}\n\n"
        f"PASSAGE:\n{(chunk_text or claim.source_span)[:1500]}\n\n"
        f"Does the passage support the claim? Answer with exactly one word "
        f"first — ENTAILED, CONTRADICTED, or NOT_ADDRESSED — then one short "
        f"sentence explaining why."
    )


def parse_entailment(raw: str) -> tuple:
    m = _ENTAIL_RE.search(raw or "")
    detail = (raw or "").strip()[:200]
    if not m:
        return "inconclusive", detail or "no clear entailment verdict"
    word = m.group(1).upper()
    if word == "ENTAILED":
        return "pass", detail
    if word == "CONTRADICTED":
        return "fail", detail
    return "inconclusive", detail


def cross_partition_prompt(claim: Claim, candidates: list) -> str:
    blocks = "\n".join(f"{i + 1}. {c.text}" for i, c in enumerate(candidates))
    return (
        f"CLAIM UNDER CHECK: {claim.text}\n\n"
        f"OTHER CLAIMS FROM DIFFERENT PARTITIONS OF THE SAME CORPUS:\n{blocks}\n\n"
        f"Does any of the numbered claims CONTRADICT the claim under check? "
        f"Answer with exactly the word NONE, or the number of the "
        f"contradicting claim."
    )


def parse_contradiction(raw: str, n_candidates: int) -> Optional[int]:
    text = (raw or "").strip()
    if _NONE_RE.search(text):
        return None
    m = _CONTRA_IDX_RE.search(text)
    if not m:
        return None
    idx = int(m.group(1)) - 1
    return idx if 0 <= idx < n_candidates else None


def quantity_reextraction_prompt(chunk_text: str) -> str:
    return (
        f"PASSAGE:\n{(chunk_text or '')[:1500]}\n\n"
        f"List every numeric figure present in the passage, one per line, "
        f"as plain numbers only. If the passage contains no numbers, write "
        f"NONE."
    )


def parse_quantity_reextraction(raw: str, claim_text: str) -> tuple:
    found = {_normalize_number(t) for t in _NUMBER_TOKEN_RE.findall(raw or "")}
    claim_numbers = {
        _normalize_number(t) for t in _NUMBER_TOKEN_RE.findall(claim_text or "")
        if sum(ch.isdigit() for ch in _normalize_number(t)) >= 2
    }
    if not found:
        return "inconclusive", "independent re-extraction found no numbers at all"
    if claim_numbers & found:
        return "pass", "number reappeared in independent re-extraction"
    return "fail", "original number absent from independent re-extraction"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class LedgerWorker:
    """One EXTRACT/VERIFY worker. Dynamically picks its action per iteration
    via `choose_ledger_action` — workers are not pinned to one role, mirroring
    the spec's floor-policy recommendation over a fixed role split."""

    def __init__(self, worker_id: int, llm, router, task_prompt: str,
                 rng_seed: Optional[int] = None):
        self.worker_id = worker_id
        self.agent_id = f"ledger_worker_{worker_id:03d}"
        self.llm = llm
        self.router = router
        self.task_prompt = task_prompt
        self.partition = None
        self._rng = random.Random(rng_seed if rng_seed is not None else worker_id)

    def _llm_for(self, role: str):
        """Model-family split (spec sec3a): extractor gets the router's
        "scout" family, checker gets "validator" — built from day one even
        though *measuring* whether heterogeneity matters is Stage 3, per the
        spec's explicit "costs nothing extra" note. Falls back to the single
        `self.llm` when no router is wired (MOCK_LLM / single-engine runs)."""
        if self.router is None:
            return self.llm
        try:
            return self.router.engine_for(role, worker_id=self.worker_id)
        except TypeError:
            return self.router.engine_for(role)

    # -- EXTRACT ------------------------------------------------------------

    async def do_extract(self, ledger: ClaimsLedger, pool_state: LedgerPoolState) -> list:
        if self.partition is None:
            return []
        llm = self._llm_for("scout")
        prompt = extraction_prompt(self.task_prompt, self.partition,
                                   chunk_offset=pool_state.iterations)
        raw = await llm.generate(prompt, role="scout", max_tokens=MAX_TOKENS_EXTRACT,
                                 temperature=0.7)
        claims = parse_extraction(raw, self.partition, self.agent_id,
                                  getattr(llm, "name", "unknown"),
                                  mock_fallback=_is_mock_mode())
        for claim in claims:
            ledger.append_claim(claim)
        pool_state.covered_chunk_ids.update(self.partition.chunk_ids)
        return claims

    # -- VERIFY ---------------------------------------------------------------

    def _eligible_for_verify(self, ledger: ClaimsLedger, pool_state: LedgerPoolState,
                             max_verify_attempts: int) -> Optional[Claim]:
        """Checker-independence rule (spec sec2.1): a claim's verifier must
        not be its extractor. Filters `unverified_queue()` on
        `extractor_agent_id != self.agent_id`, so this worker can NEVER be
        assigned to check its own claim."""
        for claim in ledger.unverified_queue(limit=max(50, VERIFY_BACKLOG_FLOOR * 4)):
            if claim.extractor_agent_id == self.agent_id:
                continue
            if pool_state.verify_attempts.get(claim.claim_id, 0) >= max_verify_attempts:
                continue
            return claim
        return None

    async def do_verify(self, ledger: ClaimsLedger, pool_state: LedgerPoolState,
                        chunks_by_id: dict, max_verify_attempts: int) -> Optional[str]:
        claim = self._eligible_for_verify(ledger, pool_state, max_verify_attempts)
        if claim is None:
            return None
        assert claim.extractor_agent_id != self.agent_id, (
            "checker-independence violated: a worker may not verify its own claim"
        )
        attempt = pool_state.verify_attempts.get(claim.claim_id, 0) + 1
        pool_state.verify_attempts[claim.claim_id] = attempt
        llm = self._llm_for("validator")
        checker_model = getattr(llm, "name", "unknown")

        if claim.kind == "quantity":
            check_type = "quantity_reextraction" if attempt % 2 == 1 else "span_entailment"
        else:
            check_type = "span_entailment" if attempt % 2 == 1 else "cross_partition"

        if check_type == "span_entailment":
            chunk_text = chunks_by_id.get(claim.corpus_doc_id, claim.source_span)
            raw = await llm.generate(span_entailment_prompt(claim, chunk_text),
                                     role="validator", max_tokens=80, temperature=0.2)
            outcome, detail = parse_entailment(raw)
            record = VerificationRecord(
                checker_agent_id=self.agent_id, checker_model=checker_model,
                check_type="span_entailment", outcome=outcome, detail=detail,
            )
        elif check_type == "cross_partition":
            candidates = ledger.contradiction_candidates(claim, k=3)
            if not candidates:
                record = VerificationRecord(
                    checker_agent_id=self.agent_id, checker_model=checker_model,
                    check_type="cross_partition", outcome="inconclusive",
                    detail="no cross-partition candidates available",
                )
            else:
                raw = await llm.generate(cross_partition_prompt(claim, candidates),
                                         role="validator", max_tokens=40, temperature=0.2)
                idx = parse_contradiction(raw, len(candidates))
                if idx is None:
                    record = VerificationRecord(
                        checker_agent_id=self.agent_id, checker_model=checker_model,
                        check_type="cross_partition", outcome="inconclusive",
                        detail="no contradiction signal",
                    )
                else:
                    record = VerificationRecord(
                        checker_agent_id=self.agent_id, checker_model=checker_model,
                        check_type="cross_partition", outcome="fail",
                        detail=f"contradicted by candidate {idx + 1}",
                        contradicting_claim_id=candidates[idx].claim_id,
                    )
        else:  # quantity_reextraction
            chunk_text = chunks_by_id.get(claim.corpus_doc_id, "")
            raw = await llm.generate(quantity_reextraction_prompt(chunk_text),
                                     role="validator", max_tokens=60, temperature=0.2)
            outcome, detail = parse_quantity_reextraction(raw, claim.text)
            record = VerificationRecord(
                checker_agent_id=self.agent_id, checker_model=checker_model,
                check_type="quantity_reextraction", outcome=outcome, detail=detail,
            )

        ledger.append_verification(claim.claim_id, record)
        return record.check_type

    async def iterate(self, ledger: ClaimsLedger, pool_state: LedgerPoolState,
                      chunks_by_id: dict, max_verify_attempts: int) -> Optional[str]:
        action = choose_ledger_action(ledger)
        if action == VERIFY:
            did = await self.do_verify(ledger, pool_state, chunks_by_id, max_verify_attempts)
            if did:
                pool_state.verify_count += 1
                pool_state.action_log.append((self.worker_id, VERIFY))
                return VERIFY
            # No eligible unverified claim for THIS worker (e.g. it authored
            # everything currently queued) — fall through to EXTRACT so the
            # worker isn't wasted on a no-op iteration.
        claims = await self.do_extract(ledger, pool_state)
        if claims:
            pool_state.extract_count += len(claims)
            pool_state.action_log.append((self.worker_id, EXTRACT))
            return EXTRACT
        pool_state.extract_rejected += 1
        return None


async def worker_loop(worker: LedgerWorker, ledger: ClaimsLedger,
                      pool_state: LedgerPoolState, chunks_by_id: dict,
                      stop_event: asyncio.Event, max_verify_attempts: int) -> None:
    while not stop_event.is_set():
        try:
            await worker.iterate(ledger, pool_state, chunks_by_id, max_verify_attempts)
        except Exception as exc:
            print(f"[ledger] worker {worker.agent_id} error: {type(exc).__name__}: {exc}")
        pool_state.iterations += 1
        await asyncio.sleep(0)   # yield to the event loop (no blocking I/O here)


async def run_ledger_pool(ledger: ClaimsLedger, chunks: list, partitions: list,
                          llm, task_prompt: str, stop_event: asyncio.Event,
                          n_workers: int = 8, router=None,
                          max_verify_attempts: int = MAX_VERIFY_ATTEMPTS,
                          pool_state: Optional[LedgerPoolState] = None) -> LedgerPoolState:
    """Spin up `n_workers` EXTRACT/VERIFY workers against the shared ledger
    until `stop_event` fires. Mirrors `core.worker_pool.run_pool`'s shape
    (see module docstring).

    `pool_state`: pass a pre-built `LedgerPoolState` when a concurrent task
    (e.g. `halting_loop`) needs to observe coverage/attempts WHILE the pool
    is still running, rather than only after `asyncio.gather` returns —
    `run_swarm.py:run_ledger_pipeline` does exactly this so the halting loop
    can evaluate `should_halt` against live state.
    """
    if pool_state is None:
        pool_state = LedgerPoolState()
    chunks_by_id = {c.chunk_id: c.text for c in chunks}
    workers = [
        LedgerWorker(i, llm, router, task_prompt, rng_seed=i)
        for i in range(max(1, n_workers))
    ]
    if partitions:
        for i, w in enumerate(workers):
            w.partition = partitions[i % len(partitions)]
    tasks = [
        asyncio.create_task(worker_loop(w, ledger, pool_state, chunks_by_id,
                                        stop_event, max_verify_attempts))
        for w in workers
    ]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    return pool_state


async def halting_loop(ledger: ClaimsLedger, pool_state: LedgerPoolState,
                       total_chunk_ids, stop_event: asyncio.Event,
                       max_time_s: float = LEDGER_MAX_TIME_S,
                       coverage_floor: float = LEDGER_COVERAGE_FLOOR,
                       max_verify_attempts: int = MAX_VERIFY_ATTEMPTS,
                       poll_interval: float = 0.2) -> str:
    """Poll `should_halt` (spec sec5.1: coverage-complete + queue-drained)
    and set `stop_event` when satisfied, or when the wall-clock safety valve
    fires. Returns the halt reason string."""
    start = time.time()
    while not stop_event.is_set():
        halted, reason = should_halt(
            ledger, pool_state.covered_chunk_ids, total_chunk_ids,
            verify_attempts=pool_state.verify_attempts,
            coverage_floor=coverage_floor, max_verify_attempts=max_verify_attempts,
        )
        if halted:
            print(f"[ledger] halt: {reason}")
            stop_event.set()
            return reason
        if time.time() - start > max_time_s:
            print(f"[ledger] halt: wall-clock cap {max_time_s:.0f}s reached")
            stop_event.set()
            return f"max_time_s ({max_time_s:.0f}s)"
        await asyncio.sleep(poll_interval)
    return "external_stop"
