# Stage 2 — Claims Ledger Spec

**Status:** design spec, not implemented. Implements `THEFUTURE.md` §4 Stage 2
("The claims ledger — replaces pheromones with procedure"), which fires under
the pre-registered gate in `THEFUTURE.md` §4 Stage 1: if condition F ≈ A on
the synthetic-world eval, "proceed to Stage 2 with the claims-ledger design
replacing the signal store outright, and the old pipeline becomes the
ablation baseline." This document assumes that gate has been pulled (or is
about to be) and specifies the replacement in implementation-grade detail —
schemas and signatures, not code. Nothing here is built yet.

Read alongside: `THEFUTURE.md` (authoritative roadmap), `CLAUDE.md`
(architecture as currently built), `docs/CRITIQUE_LOOP_2026-07-06.md` §8
Thesis 5 (the blackboard null-model challenge this design answers by
construction).

---

## 0. What problem this solves, stated precisely

The current store's strength arithmetic (`core/signal_store.py`) has one
directional bug that three independent code-audit iterations converged on
(`docs/CRITIQUE_LOOP_2026-07-06.md` Iter 2b, Iter 4, Iter 5): **no code path
reduces a signal's strength in response to CRITIQUE, OBJECTION, or a failed
VERIFICATION.** Dedup, trail-amplification, and provenance boost are all
positive-only; survival classification happens at projection time as a label,
not a force (`core/projection.py` — `rejected_by_field` is a status string
assigned after the fact, not a consequence of any deposit-time strength
change). `THEFUTURE.md` §3 names the underlying gap as having **no mechanical
vote**: aggregation happened via strength-weighted popularity among agents
sharing one prior, not via independent, falsifiable checks.

The claims ledger replaces "strength that decays/amplifies" with "status that
transitions on verification outcomes, once, by a checker that did not write
the claim." There is no scalar to tune, no decay rate, no dedup-amplify delta.
A claim is `verified`, `contested`, `refuted`, or stuck at `unverified`
because nobody has checked it yet. That's the entire dynamics.

---

## 1. Claim schema and ledger store

### 1.1 `Claim` (typed record)

```python
@dataclass(frozen=True)
class Claim:
    claim_id: str                  # "CLAIM_00001", monotonic, ledger-assigned
    text: str                      # the proposition, as extracted
    kind: str                      # "fact" | "quantity" | "causal" | "contested"
    source_span: str               # verbatim excerpt of the corpus chunk cited
    corpus_doc_id: str             # stable id of the source document
    partition_id: str              # which scout partition produced this claim
    extractor_agent_id: str        # which worker deposited it
    extractor_model: str           # model family/name that generated it (Stage 3 needs this)
    confidence: float              # extractor's self-reported confidence, 0-1, DIAGNOSTIC ONLY
    depends_on: tuple[str, ...] = ()   # claim_ids this claim's causal/contested kind references
    timestamp: float = field(default_factory=time.time)
```

`confidence` is carried but is explicitly **not load-bearing** — nothing in
survival logic reads it. It exists so extractor calibration can be studied
later (Stage 3 error-decorrelation measurement) without a schema migration.
This is the one field a reviewer should be suspicious of by default: if any
future patch makes survival depend on `confidence`, that patch has
reintroduced a strength scalar under a new name and should be rejected on
sight.

`kind` classification purposes:
- `fact` — a plain assertion checkable against its span (entailment).
- `quantity` — contains a number; additionally subject to re-extraction
  (§3c) since numbers are where `core/actions.py:478 ungrounded_numbers()`
  already proved fabrication is the dominant failure mode on this codebase.
- `causal` — an "X causes/enables/prevents Y" claim; entailment against a
  single span is necessary but not sufficient (span may only support
  correlation), so causal claims get an extra scrutiny flag in the checker
  prompt (§3a) rather than a new pipeline.
- `contested` — assigned by the ledger (not the extractor) when a
  cross-partition check (§3b) finds a claim elsewhere in the corpus that
  contradicts this one. Extractors never emit `kind="contested"` directly.

### 1.2 `VerificationRecord`

```python
@dataclass(frozen=True)
class VerificationRecord:
    checker_agent_id: str
    checker_model: str
    check_type: str            # "span_entailment" | "cross_partition" | "quantity_reextraction"
    outcome: str                # "pass" | "fail" | "inconclusive"
    detail: str                 # short reason string, for audit trail — never re-fed to a prompt
    contradicting_claim_id: Optional[str] = None   # set for cross_partition fails
    timestamp: float = field(default_factory=time.time)
```

`inconclusive` exists so a checker that can't get a confident read doesn't
silently count as a pass or corrupt the fail tally — it is excluded from both
counts in the survival rule (§3d) and logged for coverage reporting
("N claims stuck with only inconclusive checks" is a health metric the old
pipeline never had a peer to: the current store has nothing between "boosted"
and "not yet boosted").

### 1.3 Claim status (state machine, not a scalar)

```
unverified --(checker record added)--> verified | contested | refuted
verified, contested, refuted are terminal for THIS run;
a later record can move verified -> contested (new contradiction found)
but never verified/contested/refuted -> unverified.
```

No status ever decays back toward `unverified` on its own — there is no
clock in this state machine. That is the entire point: `THEFUTURE.md` §4
retires "no decay, no amplification, no pheromones" and this state machine
is what "no decay" cashes out to concretely. Compare
`core/signal_store.py:37` ("Decay: per round, every signal s_v ← s_v *
(1-DECAY_RATE)") — there is no analogous line here because there is nothing
that degrades with time; a claim's status is a monotone function of the
verification records attached to it, full stop.

### 1.4 Ledger store — `core/claims_ledger.py` (proposed, new file)

```python
class ClaimsLedger:
    def append_claim(self, claim: Claim) -> str: ...
    def append_verification(self, claim_id: str, record: VerificationRecord) -> None: ...
    def status(self, claim_id: str) -> str: ...
    def claims_for_partition(self, partition_id: str) -> list[Claim]: ...
    def claims_by_status(self, status: str) -> list[Claim]: ...
    def unverified_queue(self, limit: int = 1) -> list[Claim]: ...   # for checker dispatch
    def contradiction_candidates(self, claim: Claim, k: int) -> list[Claim]: ...  # cross-partition sampling pool
    def to_json(self, path: Path) -> None: ...
    @classmethod
    def from_json(cls, path: Path) -> "ClaimsLedger": ...
```

**Append-only.** `append_claim` and `append_verification` only ever add
records; nothing is mutated or deleted in place (no analogue of
`_apply_dedup_amplify` mutating `existing.strength`,
`core/signal_store.py:359`). `status(claim_id)` is a pure function computed
from the accumulated `VerificationRecord`s each time it's called (or cached
and invalidated on new-record-append — an implementation detail, not a
design commitment). This makes the ledger trivially replayable: JSON-dump the
claim list + verification list, and `from_json` reconstructs identical state.
No RNG, no time-dependent decay term to reproduce.

**Thread-safety.** The existing pool (`core/worker_pool.py`) runs
`n_workers` async tasks against one event loop, and `SignalStore.deposit()`
takes an `RLock` around the whole critical section
(`core/signal_store.py:338 with self._lock:`) because dedup-scan +
encode + id-assignment must be atomic. The ledger needs the same shape for
the same reason (`append_claim` assigns a monotonic `claim_id` and must not
race with the extractor's own dedup check if one exists — see §2). Recommend
one `RLock` guarding `append_claim`/`append_verification`/`claim_id`
allocation; reads (`status`, `claims_by_status`) can run lock-free once the
append path is atomic, matching the store's existing pattern of read methods
that don't touch `self._lock` (e.g. `_avg_verification_strength`,
`core/signal_store.py:388`, is called *inside* the lock already held by
`deposit`, so it isn't a counter-example — but plain getters like the
provenance-shape walkers described in the docstring at line 29 are safe to
leave lock-free). Given `LLM_CONCURRENCY=1` is asserted intentional for the
default GPU path (`CLAUDE.md` under Configuration), lock contention on the
ledger is not expected to be a bottleneck even with a naive single-lock
design; don't over-engineer sharding here.

**Persistence.** JSON to the run dir, replacing `signals.json`: propose
`claims.json` (claim records) and `verifications.json` (verification
records), or one `ledger.json` with both lists — either is fine, but keep
them append-only-shaped on disk too (a list of records, not a dict keyed by
mutable state) so a partial/crashed write is still parseable up to the last
complete record. This mirrors the incremental-crash-safety fix already
applied to `eval/partition_probe.py` per CLAUDE.md item 3 ("incremental
crash-safe report writing") — reuse that pattern, don't re-derive it.

---

## 2. Extraction workers

### 2.1 Recommendation: reuse the pool shape, not the action set

`core/worker_pool.py`'s `Worker` class (`worker_pool.py:880`) and
`run_pool()` (`worker_pool.py:2035`) already solve every infrastructure
problem an extraction pool needs: async concurrency against a shared mutable
store under a lock, per-worker cooldown/history tracking
(`self.recent_actions`), graceful handling of a `choose_action` precondition
failing at sample time (`worker_pool.py:1014-1059`, "re-picks a fresh action
with a new snapshot"), and the existing `set_validator_raw_log`-style
best-effort debug logging hook. None of that is specific to strength
dynamics — it's dispatch plumbing. Rebuilding it would re-derive
`asyncio.to_thread` wrapping for blocking search calls (CLAUDE.md's
"Retrieval is the wall-clock bottleneck" invariant applies unchanged: a
ledger pipeline still calls an LLM and possibly a search backend per
extraction, and still must not block the event loop).

**Recommendation: keep `run_pool`'s async worker-loop skeleton, replace the
action set.** Two roles, not seven:

- `EXTRACT` — reads one partition slice, emits 1..K `Claim` candidates
  (batched like the existing multi-claim scout, §2.3).
- `VERIFY` — pulls one `unverified` claim NOT authored by itself
  (`unverified_queue`, with the "did not write it" constraint enforced by
  filtering out `claim.extractor_agent_id == self.agent_id`) and runs one of
  the three check types (§3), depositing a `VerificationRecord`.

Everything else — `ACTIONS = (SCOUT, DEVELOP, CHAIN, CRITIQUE, OBJECT,
VALIDATE, REFINE)` (`core/actions.py:44`), `choose_action`'s share-based
balancing (`worker_pool.py:260`), cold-start weighting
(`worker_pool.py:86-100`), cluster-aware sampling, trail amplification — is
signal-store machinery with no ledger equivalent, because none of it exists
to serve verification-as-survival; it exists to serve strength-as-survival.
Don't port it. A ledger pool's `choose_action` reduces to a floor policy:
run `VERIFY` whenever the unverified queue is non-empty and above some
`VERIFY_BACKLOG_FLOOR`, else `EXTRACT`. That is simple enough it may not need
a general `choose_action` abstraction at all — a priority queue plus a
worker-count split (e.g. 60% EXTRACT / 40% VERIFY, tunable) is defensible as
a first cut, with the explicit understanding that load-balancing here is an
implementation nicety, not a research question the way strength-based action
selection was claimed to be.

### 2.2 Partition assignment

Reuse `assemble_partitions()` (`run_swarm.py:351`) unchanged. It already
returns `(chunks, partitions: list[ScoutPartition])` with `partition_id`
resolution handled (`core/intake.py:32-46`, `custom_partition_id` for web
partitions, `partition_{i}` for corpus slices) and already degrades safely
(`run_swarm.py:1131-1145`: any `assemble_partitions` exception logs a warning
and continues with `partitions = []` rather than crashing the run). The
ledger pipeline's `EXTRACT` workers are assigned one `ScoutPartition` each,
identically to how `agents/scout.py`'s `ScoutConfig.partition` is wired
today (`agents/scout.py:48-50`). This is exactly the "partitioning remains as
coverage logistics only" claim in `THEFUTURE.md` §2b / §6: the partition
assembly code is retained wholesale; only what consumes it changes.

`Claim.partition_id` is stamped from `self.config.partition.partition_id` at
extraction time, the direct analogue of
`agents/scout.py:137/208 "partition_id": self.config.partition.partition_id`.
There is no inheritance-through-parent logic to replicate
(`core/signal_store.py:402-409`) because claims don't chain off other
claims the way SUPPORT chains off INITIAL — each claim is extracted straight
from a corpus span, so `partition_id` is always assigned at creation, never
inherited. This removes an entire class of bug: the "PARTITION LEAK"
assertion (`core/signal_store.py:427-431`) and its defensive carry-forward in
`agents/base.py` (documented in CLAUDE.md's "base.py deposit_meta
partition_id carry-forward" section) exist only because SUPPORT's
inheritance path can go stale between sample and deposit. The ledger has no
inheritance path, so it has no equivalent failure mode to guard — this is a
simplification, not a gap; call it out in the PR description when built so a
reviewer doesn't go looking for the guard and wonder why it's missing.

### 2.3 Prompt shape — schema-constrained JSON, following the guided-JSON precedent

Commit `08c65f4` ("Guided-JSON scout claims: exact K-way portfolio split")
already solved "make the model emit N distinct claims in a parseable
structure" for the current scout: `core/actions.py:377-420
split_scout_claims()` — JSON-first (`extract_json_object(text)`,
`obj.get("claims")` as `list[str]`), falling back to a numbered-marker regex
split only when guided decoding isn't available or fails. The extraction
worker's prompt should follow the same contract, extended with structure:

```json
{
  "claims": [
    {"text": "...", "kind": "fact", "source_span": "...", "confidence": 0.8},
    {"text": "...", "kind": "quantity", "source_span": "...", "confidence": 0.6}
  ]
}
```

Where guided JSON decoding is available (vLLM / Groq structured outputs),
constrain the schema server-side; where it isn't, reuse
`split_scout_claims`'s exact fallback shape (JSON-first, numbered-marker
regex second) rather than inventing a third parser. `corpus_doc_id` and
`partition_id` are NOT requested from the model — they are stamped by the
worker from `self.config.partition` after parsing, the same division of
labor the current scout already uses for `partition_id`
(model never sees or emits it; code stamps it). This keeps the extractor's
JSON surface small and keeps provenance fields tamper-proof (a hallucinated
`corpus_doc_id` in the model's own output would be a leak vector; not
asking for it closes that off by construction).

`source_span` is model-selected but **must be validated as a verbatim
substring (or near-verbatim, e.g. whitespace-normalized) of the chunk it was
extracted from** before the claim is accepted into the ledger — this is the
extraction-time analogue of `ungrounded_numbers()`
(`core/actions.py:478-494`) and should probably call the same substring/token
matching primitive rather than a new one. A `source_span` that doesn't
appear in the partition's chunks is a fabricated citation and the claim
should be rejected at extraction time, not merely flagged — this is stricter
than the current number-grounding gate, which tags-but-keeps ungrounded
figures when no evidence was shown (`core/actions.py` comment block above
line 464); the ledger has no "no evidence shown" case, since every claim is
partition-anchored by construction, so there is no legitimate reason to keep
an unanchored claim.

### 2.4 No-leak boundary, stated precisely

Extraction workers may render into their prompt:
1. Their assigned `ScoutPartition.chunks` (corpus text only — identical
   input class to today's scout, `agents/scout.py`).
2. Ledger artifacts: `Claim.text`, `Claim.kind`, `Claim.claim_id`,
   `Claim.status` (scalar), and **aggregate counts** (e.g. "14 claims
   extracted so far from other partitions" — a number, not content) if the
   design wants extractors to avoid duplicating known claims. This is the
   direct analogue of the scout's novelty-reference mechanism
   (`select_novel_claim` / `store.max_similarity_to_recent`,
   `core/actions.py:497-521`) which already establishes the pattern: **the
   selection is code-side scalars only; no other agent's content enters the
   prompt** (comment at CLAUDE.md's "Multi-claim scout sampling" section).
   The ledger keeps this pattern; extractors may be told "your candidate is
   90% similar to an existing claim" as a number, never shown the existing
   claim's text as a nudge (that would be leaking another worker's output
   into this worker's reasoning input, the same leak class the no-leak rule
   exists to prevent).

Extraction workers may NOT render: another worker's raw extraction reasoning,
any `VerificationRecord.detail` string from a checker (that field is audit
trail for humans/the render step, not extractor input — feeding a checker's
critique back to future extractors is exactly the "conditioning on another
agent's reasoning concentrates the posterior" failure `THEFUTURE.md` §1
identifies as the one part of the old theory that's right and load-bearing),
or contested-claim contradiction text beyond the scalar "contested" status.

Verifier (checker) workers may render:
1. The single claim under check (`Claim.text`, `Claim.source_span`,
   `Claim.kind`).
2. For span-entailment: the cited `corpus_doc_id`'s chunk text (to check the
   span is actually IN the chunk and the chunk actually supports the claim).
3. For cross-partition contradiction: `contradiction_candidates()` returns
   OTHER claims' `.text` + `.source_span` from different partitions — this
   is claim-content-to-claim-content, not reasoning-to-reasoning, and is the
   ledger's one deliberate exception to "workers never see each other's
   output": checkers must see claim text to check it, the same way a human
   fact-checker reads the claim being checked. This is qualitatively
   different from an extractor reading another extractor's draft reasoning
   — a claim is a finished, falsifiable proposition, not a chain-of-thought.
   State this distinction explicitly in the implementation docstring the way
   `core/signal_store.py:29-33` states the shapes-not-text rule, because a
   future contributor could otherwise "leak-scope-creep" this into checkers
   reading extractor scratch notes.

Verifier workers may NOT render: which worker wrote the claim, that worker's
model family (blind check — Stage 3 heterogeneity measurement needs the
*ledger* to record `extractor_model` vs `checker_model` for later analysis,
but the checker prompt itself should not condition on it, to avoid a
checker anchoring "trust this because model X said it").

---

## 3. Verification protocol

Each claim gets checked by 0+ independent `VerificationRecord`s from
checkers that did not author it. Three check types:

### 3a. Span check (entailment)

Prompt: given `claim.text` and the full chunk text of `claim.corpus_doc_id`
(not just `source_span` — the checker should see the surrounding context to
catch quote-mining), ask an entailment-style question: "Does this passage
support this claim? Answer ENTAILED / CONTRADICTED / NOT_ADDRESSED, one
sentence why." Map ENTAILED→`pass`, CONTRADICTED→`fail`,
NOT_ADDRESSED→`inconclusive`. Prefer an off-family model when the run has
one available (`core/llm_groq.py GroqRouter` / `core/llm_hybrid.py
HybridRouter` already give model-family heterogeneity without new
infrastructure — this is the "Extractor and verifier from different model
families" item `THEFUTURE.md` §Stage 3 names explicitly; wire it as "if the
router has ≥2 families, extractor gets family A, checker gets family B" from
day one even though the *measurement* of whether it matters is Stage 3 —
building the plumbing now costs nothing extra).

### 3b. Cross-partition contradiction sampling

For a sample of `k` claims from OTHER partitions on similar topics
(`contradiction_candidates(claim, k)` — implement via the same embedding
similarity primitive the current store already has,
`SignalStore._encode` / cosine sim, no new embedding infra needed), ask the
checker: "Do any of these contradict the claim under check?" A hit sets
`outcome="fail"`, `contradicting_claim_id=<the other claim>`, and (per §1.1)
retroactively reclassifies BOTH claims' `kind` to `"contested"` if they were
previously `fact`/`causal` — a claim doesn't get to stay quietly "verified"
while something the ledger knows contradicts it exists elsewhere; contested
status must be visible on both sides of a contradiction, otherwise the
composer (§4) could cite one side as settled fact while burying the
contradiction only under the other claim's record.

### 3c. Quantity re-extraction

For `kind="quantity"` claims only: give a *different* worker the same
`corpus_doc_id` chunk (not the claim) and ask it to independently extract
any numeric claims present. `pass` if the same number (normalized the same
way `core/actions.py:474 _normalize_number()` already does — reuse that
function verbatim, don't reinvent number normalization) appears in the
independent extraction; `fail` if the original number doesn't reappear;
`inconclusive` if the re-extraction finds no numbers at all (chunk may be
long enough that the independent pass missed it without being wrong about
the number's absence). This is the ledger's direct answer to the
"$54M Oslo, 90K tons Copenhagen" fabrication problem CLAUDE.md documents for
the current pipeline (`core/actions.py` comment above line 464) — instead of
one worker checking its own figure against sources it was shown (which is
what `ungrounded_numbers` does today, a self-check), this is an independent
SECOND extraction, a stronger guarantee.

### 3d. Survival rules from verification outcomes

Proposed thresholds (tune during Stage 2 build, but this is the right
shape):

| Status | Condition |
|---|---|
| `verified` | ≥2 independent `pass` records (from ≥2 distinct `checker_agent_id`s), 0 `fail` records |
| `contested` | ≥1 `pass` AND ≥1 `fail` (regardless of count), OR flagged by §3b cross-partition contradiction |
| `refuted` | ≥2 independent `fail` records, 0 `pass` records — OR 1 `fail` from a `quantity_reextraction` check with no corroborating `pass` (a wrong number is disqualifying faster than a wrong fact-check, since re-extraction is close to ground truth) |
| `unverified` | fewer than 2 total non-`inconclusive` records — still in the queue |

Note what is deliberately absent: no count of `inconclusive` records changes
status in either direction, no time-based decay pushes a `verified` claim
back to `unverified`, and no single `pass` is ever sufficient alone
(requiring ≥2 independent passes is the mechanical-vote `THEFUTURE.md` §2c
asks for — "aggregation is mechanical," not LLM-synthesizer-mediated
popularity). A claim can accumulate arbitrarily many verification records
over a long run; status is recomputed from the full record set each time,
not incrementally nudged, so there's no order-dependence to debug (contrast
`core/signal_store.py`'s documented order-dependence bug between decay and
amplify, fixed via logit-space additivity — the ledger has no analogous bug
class because status is a pure function of the record set, not an
accumulator that different orderings can drive to different endpoints).

---

## 4. Deterministic composition (`core/compose.py`, proposed new file)

### 4.1 Ordering/grouping — reuse the promoted extractive path

`THEFUTURE.md` §1 item 4 already promotes "the deterministic composition
path... the extractive fallback out-shipped the LLM composer 4/6 on the last
run" to the design. The current implementation lives in
`agents/synthesizer.py:2675 _extractive_position()` — read it as the starting
point, not something to redesign from scratch. Its job today is "verbatim
cluster content, deterministically ordered" over surviving *clusters*; the
ledger version does the same job over surviving *claims*, grouped instead of
clustered (grouping is a much simpler operation than the current
embedding-cosine cluster registry, `core/cluster_registry.py`, and needs no
runtime clustering machinery at all — group by `corpus_doc_id`, then by
`kind`, is enough for a first cut; topic grouping via embedding similarity
is a nice-to-have, not a requirement, since a ledger with no synthesis LLM
doing the heavy lifting doesn't need topically coherent paragraphs, it needs
correct, ordered ones).

Proposed section shape:
1. **Verified claims**, grouped by `corpus_doc_id` (or topic, if a grouping
   pass is added later), each rendered as `text [claim_id → doc_id]`.
2. **Contested claims** section — both sides of every contradiction shown
   side by side with their respective `contradicting_claim_id` cross-link,
   the direct structural analogue of the current Section 2's dissent-cluster
   handling (`agents/synthesizer.py`, "Section 2 consumes
   `plan.dissent_clusters`") but mechanical instead of planner-selected —
   EVERY contested claim appears, there is no `_MINORITY_ONELINER_CAP`-style
   demotion, because there is no LLM-context budget forcing a cut (composer
   input here is O(verified + contested claims), not O(K×brief) — actually
   look at whether this needs its own size cap once real corpora are run
   through it; flag as an open question, not a design commitment either
   way).
3. **Refuted / unverified** are NOT rendered into the answer at all
   (refuted = failed check, shouldn't ship; unverified = not yet checked,
   shouldn't ship as if it were). Both are visible in the ledger dump for
   audit but excluded from the composed answer — this is a much simpler cut
   rule than today's `weakly_supported`/`rejected_by_field` classification
   dance in `core/projection.py`, because there's no partial-credit status
   to render around; a claim either survived checking or it didn't.
4. **Citation rendering**: `[claim_id]` inline, resolved to a footnote block
   listing `corpus_doc_id` + `source_span` excerpt — the direct structural
   analogue of `resolve_inline_citations()` (`agents/synthesizer.py:3641`)
   and the Sources-block append in `core/clean_answer.py` (the "Sources"
   section CLAUDE.md documents as fixing "previously ALL citations were
   stripped"). Reuse `resolve_inline_citations`'s regex/footnote-numbering
   approach directly if the tag format stays similar (`[CLAIM_00001]` in
   place of `[INITIAL_00001]` is a near-zero-effort adaptation); don't write
   a second footnote renderer.

### 4.2 The one constrained LLM polish pass

Exactly one LLM call, at the end, over the deterministic composition from
§4.1. Its ONLY job is prose smoothing (transitions, merging adjacent
sentences about the same doc, fixing grammar across grouped claims) —
**it may not add or remove claims, invent numbers, or change which
`[claim_id]` tags appear.** This is `THEFUTURE.md` §4's "one constrained LLM
polish pass, audited against the ledger, with the existing hard gate."

Constraint mechanism, concretely:

1. **Claim-coverage audit (reuse, adapted).** Before polish: record the set
   of `claim_id`s present in the deterministic draft. After polish: check
   every one of those ids still appears in the polished text. This is a
   *stronger, simpler* version of
   `agents/synthesizer.py`'s existing "minimum length + citation-tag
   retention ≥ 0.5" guard (mentioned in CLAUDE.md's Synthesizer section) —
   the ledger's gate should be retention == 1.0 (every claim_id present, not
   just half), because the polish pass has no license to drop content the
   way a from-scratch LLM composition might reasonably compress; it's
   polishing an already-complete draft, not authoring one.
2. **No-new-numbers audit (reuse, near-verbatim).** Run
   `core/actions.py:478 ungrounded_numbers(polished_text,
   sources=[deterministic_draft_text])` — literally the existing function,
   with the deterministic draft (not corpus chunks) as the "sources" list.
   Any number in the polished text absent from the pre-polish draft is a
   fabrication introduced by the polish pass itself and fails the gate. This
   reuses the exact function signature already in the codebase; no new
   number-matching logic needed.
3. **Faithfulness audit (reuse the fixed-ordering version, adapted).**
   `agents/synthesizer.py`'s `_build_faithfulness_audit` (currently run
   BEFORE citation resolution per the 2026-07-06 ordering fix,
   `agents/synthesizer.py:1379-1397`) checks n-gram overlap between prose
   and cited content per citation tag. Adapt it to check polished-paragraph
   n-gram overlap against `Claim.text` (not cluster content) for every
   `[claim_id]` the paragraph contains. Run it on the pre-resolution
   (tag-still-inline) text, for the same reason the current fix exists:
   auditing after footnote-resolution audits the appendix, not the prose.
4. **Hard gate (reuse the mechanism, not the threshold).** If any of 1-3
   fail, discard the polish output and ship the deterministic draft from
   §4.1 as-is, exactly as `agents/synthesizer.py:1413-1460`'s
   `_AUDIT_HARD_GATE` discards composed prose in favor of
   `_extractive_position()` output today. Since the deterministic draft here
   is provably claim-complete and number-accurate (it's built directly from
   ledger records, not generated), it is ALWAYS a safe fallback — this gate
   can never fail open the way a from-scratch LLM composer's fallback chain
   has more failure surface (`agents/synthesizer.py`'s documented fallback
   chain: "global composition → edge composition → plain join →
   deterministic extractive rendering" — the ledger collapses that whole
   chain to two rungs: polished or deterministic, because there's only one
   LLM call in the whole composition path to begin with).

What's genuinely new vs reused: the reuse list is
`ungrounded_numbers()` (verbatim), the resolve/footnote pattern (near-
verbatim), the audit-then-gate control flow shape (adapted), and the
gate-discards-to-a-provably-safe-fallback pattern (structurally identical).
What's new: the claim-coverage-==1.0 check (simpler than what it replaces),
and re-pointing the faithfulness audit's "source content" input from cluster
text to `Claim.text` (a rename plus a schema change, not new logic).

---

## 5. Keep / adapt / retire table

| Module | Fate | Why |
|---|---|---|
| `core/signal_store.py` | **Ablation-baseline only** | This IS the thing being ablated against (`THEFUTURE.md` §6: "re-entry only via ablation victory over the ledger"). Keep running, unmodified, as the `--legacy` / baseline path. Do not delete — a negative ablation result needs it to still run. |
| `core/projection.py` | **Ablation-baseline only, mostly** | `ClusterProjection`, survival-status classification (`weakly_supported`/`rejected_by_field`/`contested`), `support_diversity`, `dissent_pressure` are all strength-store-specific concepts with no ledger equivalent (the ledger has no clusters, no dissent_pressure — it has claim status). `build_plan()`'s MMR-over-embeddings selection logic could theoretically be reused for grouping claims by topic in §4.1, but that's a "steal the algorithm, not the module" move, not a keep. |
| `core/convergence.py` | **Retire for the ledger path; keep for baseline** | Its whole vocabulary (`quality`/`saturation`/`render_set_stable`/`novelty_saturation`) is strength-and-cluster-shaped. Replacement halting condition (§5.1 below) is structurally simpler and doesn't need this module's machinery. Keep it running unmodified for the baseline ablation arm. |
| `core/fitness.py` | **Retire for the ledger path; keep for baseline** | `ClusterGenome`/`composite_fitness` has no referent once there are no clusters. `docs/CRITIQUE_LOOP_2026-07-06.md` Iter 5 already found this module internally contradictory (rewards monotone growth 1.0, contestation 0.2, while the survival gate counts dissent as credibility) — the ledger's verification-count survival rule (§3d) is the direct fix Iter 5's "re-sign consensus to reward survived contestation" fix was gesturing at, done properly instead of patched. |
| `core/cluster_registry.py` | **Retire for the ledger path; keep for baseline** | No clustering step exists in claim extraction; grouping in §4.1 is a groupby, not incremental centroid clustering. |
| `core/worker_pool.py` | **Adapt (skeleton reused, action set replaced)** | See §2.1. The `Worker`/`run_pool` async-loop shape, lock discipline, and `asyncio.to_thread` search-blocking-avoidance are infrastructure worth keeping; `choose_action`'s share-balancing and the 7-action registry are strength-store-specific and are replaced by the 2-action EXTRACT/VERIFY split. |
| `agents/scout.py` | **Adapt into `EXTRACT`** | Partition wiring (`ScoutConfig.partition`), prompt-construction pattern, and the multi-claim JSON-first parsing precedent (`split_scout_claims`) transfer near-directly (§2.2, §2.3). Deposit target changes from `SignalStore.deposit(INITIAL, ...)` to `ClaimsLedger.append_claim(...)`. |
| `agents/developer.py` (`agents/forager.py` alias) | **Retire for the ledger path; keep for baseline** | DEVELOP's whole job — sample an under-supported INITIAL, add a SUPPORT — is strength-accumulation logic. The ledger has no "add support to strengthen a claim" move; a claim either passes independent checks or it doesn't. |
| Critic/Hater roles (`CRITIQUE`/`OBJECT` actions) | **Retire for the ledger path; keep for baseline** | Superseded by the VERIFY role's span-check/cross-partition-contradiction checks, which are procedural rather than rhetorical (a checker runs a fixed protocol; a Hater/Critic in the current pipeline free-writes an objection whose only effect is a label, per Iter 2b's central finding). |
| Validator role (`VALIDATE` action) | **Direct conceptual ancestor of VERIFY, but not reused as code** | Validator already does "check a claim against external search" — closest existing analogue to the checker. Its prompt-construction pattern (query formation, evidence-vs-claim comparison) is worth reading before writing `VERIFY`'s prompt, but its deposit target (`VERIFICATION` signal feeding `_avg_verification_strength` boosts) has no ledger equivalent — VERIFY writes a `VerificationRecord`, not a boost. |
| `agents/synthesizer.py` | **Retire wholesale for the ledger path; keep for baseline; salvage 3 functions** | See §4: `_extractive_position` (starting point for the deterministic composer), `resolve_inline_citations` (near-verbatim reuse), `_build_faithfulness_audit` (adapted reuse). The two-stage cluster-brief LLM composition (`_compose_answer`, `_plan_synthesis`), topology-aware rendering (Sections 5/6), and sensitivity annotation have no referent without clusters/genomes. The LLM planner (`_plan_synthesis`, gated `USE_LLM_PLANNER=False` already, per CLAUDE.md — "retired to opt-in") stays retired; don't resurrect it for the ledger. |
| `core/clean_answer.py` | **Adapt** | `split_answer()`'s reader/diagnostics split and the Sources-block-append pattern are format-agnostic — apply unchanged to the ledger composer's output. This is the one module that needs essentially no change. |
| `core/topology.py` | **Retire for the ledger path; keep for baseline** | Bounds-first exploration scaffolding assumes scouts staking out conceptual territory for later coverage-tracking against clusters. A claims ledger's coverage question is answered directly and mechanically: which `corpus_doc_id`s have ≥1 extracted claim, which chunks within a doc are unclaimed. That's a much simpler coverage check computable straight from the ledger (`claims_for_partition` grouped by `corpus_doc_id` against the known chunk list) — no topology-generation LLM call needed. |
| `core/knowledge_base.py` | **Redesign target for Stage 4, not reused as-is now** | `THEFUTURE.md` §Stage 4 explicitly flags this: "Revisit the knowledge base as organizational memory: verified claims (not clusters) persist across runs." The schema (`genome_hash`, `genome_atoms`, contradiction-detection via embedding cosine + atom-text Jaccard) is cluster/genome-shaped and needs a `Claim`-shaped v4 schema — persist `verified` claims with `corpus_doc_id` + provenance, age out `refuted`/stale `unverified` ones. Out of scope for the Stage 2 build itself; flag as the Stage 4 follow-on. |
| `core/query_planner.py`, `core/search_tool.py`, `core/retrieval.py` | **Keep unmodified, used by both paths** | `THEFUTURE.md` §1 item 3 keeps retrieval + packs wholesale. `assemble_partitions()` and everything under it (facet planning, composite retrieval, pack-mode) is corpus/partition assembly, orthogonal to what consumes the partitions. No changes needed for the ledger path. |
| `core/output_diversity.py`, `core/diversity.py` | **Retire for the ledger path (no claim to make); keep for baseline** | These measure output/input diversity across agent text — a diversity metric with no diversity thesis behind it (`THEFUTURE.md` §6 retires "partitioning-as-diversity" and "same-model populations for perspective diversity" outright). The ledger's success metric is grounded accuracy + verification coverage, not diversity. |
| `core/baseline.py` (`--mode=baseline`) | **Keep unmodified** | Already the "no store, no partitioning" A/B condition; stays as one arm of the eventual 3-way comparison (baseline / signal-store / ledger) once Stage 2 lands. |

### 5.1 Convergence/halting replacement for the ledger pipeline

Proposed replacement, deliberately simple (per §0's "no scalar to tune"
ethos):

**Halt when:** (a) corpus coverage is complete — every chunk in every
assigned partition has been read by at least one `EXTRACT` pass (or a
configurable coverage floor, e.g. ≥95%, to tolerate a few genuinely
un-extractable chunks like boilerplate), AND (b) the verification queue is
drained — `unverified_queue()` returns empty, or every remaining
`unverified` claim has hit a `MAX_VERIFY_ATTEMPTS` retry cap without a
conclusive record (logged as a coverage gap, not silently dropped).

No novelty tracking, no saturation window, no render-set stability polling
(`core/convergence.py`'s `RENDER_STABLE_ITERS` mechanism exists to defend
against near-duplicate clusters resetting counters — a problem that doesn't
exist here because there's no cluster-churn metric to fool). A time/iteration
cap (`MAX_TIME_S`/`MAX_ITERATIONS`-equivalent) should still exist as a safety
valve, but it should essentially never fire in a healthy run, because
coverage-complete + queue-drained is a well-defined terminal condition on a
finite corpus, unlike "has the field stopped changing," which the current
pipeline's `docs/CRITIQUE_LOOP_2026-07-06.md` Iter 4 documents as gameable by
churn.

---

## 6. Ablation protocol

Fair-fight requirements, directly inheriting the discipline
`THEFUTURE.md` §5 and `CLAUDE.md`'s "Rules of evidence" already established
for the A/B/E/F harness — apply the same rigor here, not a looser bar just
because this is an internal architecture comparison rather than a
vs-single-model comparison:

1. **Same partitions.** Run `assemble_partitions()` once per prompt, feed the
   IDENTICAL `(chunks, partitions)` tuple to both the signal-store pipeline
   and the ledger pipeline. This is the same discipline `run_swarm.py`'s
   `--corpus=pack:<path>` mode already enforces for the A-vs-F comparison
   (`run_swarm.py:376-394`, "so condition A (swarm) and eval.ab_harness
   condition F ... see the IDENTICAL evidence — the whole point of the
   A-vs-F comparison"). Reuse pack-mode wholesale for this: build the ledger
   ablation on top of `eval/packs.py`, don't build a second fixture system.
2. **Same model(s).** Both pipelines' extractor/scout and checker/validator
   roles get the same backend/model assignment per run (same `GroqRouter`
   manifest or same local model). Model-family heterogeneity (Stage 3) is a
   separate, later variable — don't conflate "does the ledger beat the
   store" with "does heterogeneous verification help," or a positive result
   won't tell you which change carried it.
3. **Same budget.** Cap both pipelines by the same wall-clock or token
   budget, not by "run until each pipeline's own halting condition fires" —
   the whole point of §5.1 is that the ledger's halting condition is
   structurally different (coverage-complete vs quality-gate), so letting
   each run to its own natural halt would confound "the ledger is more
   efficient" with "the ledger was given a different stopping rule." Use
   the synthetic-world eval's existing per-run budget knobs
   (`eval/ab_harness.py --mini`, token/time caps) as the shared ceiling.
4. **Same corpus scale sweep.** Run the comparison at 1x/4x/16x pack scale
   (the existing `--pack-scale` axis, `THEFUTURE.md` §4 Stage 1) since the
   ledger's coverage-based halting is expected to behave very differently
   at scale than the signal store's quality-gate halting — this axis is
   where a real difference, if any, should show up most clearly.
5. **Headline metric: grounded accuracy + citation precision/recall,
   mechanically scored** — per `THEFUTURE.md` §5 rule 3 ("Headline metrics
   are mechanical... LLM judges are for secondary prose comparisons only").
   For the ledger specifically, add: **verification coverage** (% of
   extracted claims that received ≥1 non-`inconclusive` check by run's end)
   and **contradiction catch rate** (synthetic corpora can seed a known
   number of planted contradictions per Stage 1's corpus-generation design;
   measure what fraction the checker layer actually caught) as secondary,
   mechanical, ledger-specific metrics — these have no signal-store
   equivalent and are exactly the "does verification-as-survival do what it
   claims" measurement the store never had a way to make.
6. **Pre-registered decision gate** (mirroring `THEFUTURE.md` §4's own gate
   structure): the ledger only earns default status if it beats the
   signal-store baseline on grounded accuracy at Wilson lower bound > 0.5,
   at the SAME corpus scale where the store was given its best shot (16x,
   per the existing over-context framing). If the ledger loses or ties, that
   is a publishable negative result exactly as `THEFUTURE.md` §5 rule
   demands for every other comparison in this project — do not quietly keep
   the store as default because the ledger "should" win on priors; the
   entire point of this project's last six months is that priors about
   which architecture should win have been wrong every time they were
   checked.

---

## 7. Test plan, effort estimate, build sequence

### 7.1 Test plan (mirrors the existing test-file-per-module convention)

- `tests/test_claims_ledger.py` — schema validation, append-only invariant
  (mutating a returned `Claim` object should not affect stored state — use
  `@dataclass(frozen=True)` and test the TypeError on mutation attempt),
  status-transition state machine (§1.3) including the "verified can move to
  contested, never back to unverified" rule, thread-safety under concurrent
  `append_claim` calls (spawn N async tasks depositing concurrently, assert
  no duplicate `claim_id`s — direct analogue of whatever concurrency test
  exists for `SignalStore.deposit`'s lock, if one does; check
  `tests/test_no_leak_real_patterns.py` and neighbors for the pattern to
  follow), JSON round-trip (`to_json`/`from_json` produces an identical
  ledger).
- `tests/test_claims_extraction.py` — `EXTRACT` worker prompt construction
  (no-leak assertion per §2.4, reusing `_assert_no_leak`'s forbidden-token
  pattern from `agents/base.py:386-403` extended with ledger-specific
  forbidden tokens like "checker reasoning", "verification detail"), JSON
  parsing with the guided-JSON-first / marker-fallback contract (direct port
  of whatever test currently covers `split_scout_claims`), `source_span`
  verbatim-substring validation rejecting fabricated spans.
- `tests/test_claims_verification.py` — the three check types (§3a/b/c) each
  independently: span check against a known-good/known-bad span pair,
  cross-partition contradiction detection with a planted contradiction pair,
  quantity re-extraction with a planted correct/incorrect number. Survival
  rule table (§3d) as a pure-function unit test — no LLM calls needed, just
  feed synthetic `VerificationRecord` lists and assert the resulting status.
- `tests/test_compose.py` — claim-coverage audit (100% retention required,
  stricter than the existing ≥0.5 threshold — assert the stricter bound is
  actually enforced), no-new-numbers audit (reusing `ungrounded_numbers`
  directly — a regression here is really a regression in the shared
  function, so this test doubles as extra coverage on existing code),
  hard-gate-discards-to-deterministic-draft path, refuted/unverified claims
  never appearing in composed output.
- `tests/test_ledger_convergence.py` — coverage-complete + queue-drained halt
  fires correctly; `MAX_VERIFY_ATTEMPTS` retry cap prevents an unresolvable
  claim from hanging the run forever.
- Reuse existing subprocess-test env-var discipline
  (`SWARM_MIN_TIME_S=0` etc., per CLAUDE.md's Convergence section) for any
  ledger pipeline test that spawns a real run — the ledger needs its own
  equivalent fast-test env vars (e.g. `SWARM_MAX_VERIFY_ATTEMPTS`,
  `SWARM_LEDGER_MAX_TIME_S`) from day one so tests don't wait on real
  coverage completion against a large corpus.

### 7.2 Effort estimate (rough, in engineering-days, single implementer)

| Component | Estimate | Gates on Stage 1? |
|---|---|---|
| `Claim`/`VerificationRecord` schema + `ClaimsLedger` store (§1) | 1.5 days | No |
| `EXTRACT` worker + prompt (§2) | 1.5 days | No (needs `assemble_partitions`, already built) |
| `VERIFY` worker + 3 check types (§3) | 2.5 days | No |
| Worker-pool skeleton adaptation (§2.1) | 1 day | No |
| Deterministic composer (§4.1) | 1.5 days | No (can build against a hand-written fixture ledger) |
| Constrained polish pass + audits (§4.2) | 1.5 days | No, but should reuse §4.1's fixtures |
| Halting replacement (§5.1) | 0.5 day | No |
| Ledger-vs-store ablation harness (§6) | 2 days | **Yes — needs Stage 1's synthetic corpora + B-validity-checked prompt set to be a fair fight at all** |
| Test suite (§7.1) | 2 days | Mostly no; ablation-harness tests gate on Stage 1 |
| **Total** | **~14 engineer-days** | Everything except the ablation harness itself can be built and unit-tested now |

### 7.3 Ordered build sequence

**Can start before Stage 1 lands** (build against synthetic/hand-written
fixtures, not live Stage-1 corpora):
1. `Claim`/`VerificationRecord` schema + `ClaimsLedger` (§1) — pure data
   structure, testable in isolation.
2. Deterministic composer (§4.1) against a hand-written fixture ledger
   (a handful of `verified`/`contested`/`refuted` claims written directly
   into a test file) — proves the ordering/grouping/citation logic without
   needing a real extraction run.
3. `EXTRACT`/`VERIFY` worker classes and prompts (§2, §3) — can be written
   and unit-tested against `MOCK_LLM=1` the same way every other agent role
   in this codebase is developed and tested per CLAUDE.md's documented
   workflow ("Develop without a GPU / model download").
4. Constrained polish pass + its three audits (§4.2) — the audits are pure
   functions over text; testable without any real LLM call for the audit
   logic itself (only the polish LLM call needs a real/mock backend).
5. Halting logic (§5.1) — pure function over ledger state + coverage set.

**Gates on Stage 1 landing:**
6. The ablation protocol (§6) itself cannot produce a meaningful result
   until Stage 1's synthetic, pretraining-unknowable corpora and
   B-validity-checked prompt set exist — running the ledger-vs-store
   comparison on the CURRENT prompt set (ban_cars/god_exists/climate_action)
   would inherit the exact "θ dominates, x is a rounding error" problem
   `THEFUTURE.md` §2a diagnosed for every prior swarm-vs-baseline
   comparison, and would prove nothing about the ledger either. Do not run
   the ablation early "just to get a signal" — a signal from an invalid
   fixture is worse than no signal, per this project's own accumulated
   experience (`THEFUTURE.md` §0: "every time measurement got more honest,
   results got worse... that is what a wrong hypothesis looks like from
   inside" — the inverse failure, an easy fixture producing a flattering
   false positive, is the same trap from the other side).
7. Full three-way comparison (baseline / signal-store / ledger) on Stage 1
   corpora at 1x/4x/16x, judged mechanically per §6.

Items 1-5 total roughly 8 of the 14 estimated engineer-days and have zero
dependency on Stage 1 — a team could build the entire ledger mechanism
now and have it sitting ready the moment Stage 1's gate resolves, rather
than starting the ledger build only after the gate fires.
