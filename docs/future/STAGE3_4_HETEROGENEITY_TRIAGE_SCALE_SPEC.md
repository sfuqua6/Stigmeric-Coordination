# Stage 3/4 Spec — Heterogeneity, Triage, Scale, Organizational Memory

**Status:** sketch, not implementation. Implements `THEFUTURE.md` §4 Stage 3
("Heterogeneity where it pays") and Stage 4 ("Scale and memory"). Both stages
are gated: **nothing here executes until Stage 1's decision gate produces a
verdict** (`THEFUTURE.md` §4 Stage 1: F≈A on synthetic corpora → claims-ledger
replaces the signal store outright and this spec's extractor/verifier roles
are Stage-2 roles by another name; A>F at 16x → Stage 2 hardens the existing
coverage mechanism instead). This document does not re-derive Stage 2's role
names or claim schema — it refers to "the extractor," "the verifier," and
"the composer" as roles a concurrent spec defines, and only specs what
Stage 3/4 do *to* those roles (route them to different model families, decide
whether they run at all, put them on faster iron, remember what they found).

Read alongside: `THEFUTURE.md` (authoritative, supersedes the old diversity
claim), `CLAUDE.md` §"LLM backends" / §"Knowledge base" / §"Constraints to
respect" / §"Empirical status", `core/llm_groq.py`, `core/llm_hybrid.py`,
`core/llm_router.py`, `core/knowledge_base.py`, `DEFERRED.md` ("Cloud
validator implementation").

---

## Stage 3a — Heterogeneous verification

### The claim being tested

`THEFUTURE.md` §2c: the old design had no mechanical vote — an LLM synthesizer
aggregated via strength dynamics and failed its own faithfulness audit 4/6.
Stage 2 replaces that with per-claim verification by a checker that "did not
write it" (§4 Stage 2). Stage 3a asks the next question: does the checker
also need to be a **different model family** than the writer, or does
same-family-different-instance verification already catch most errors? This
is an empirical question with a cheap, mechanical answer available from
Stage 1's synthetic corpora (ground truth exists, so "the extractor was
wrong" is not a judgment call — see below).

### Router contract mapping

The existing router contract (`GroqRouter.engine_for(role)` at
`core/llm_groq.py:478-484`, `HybridRouter.engine_for(role)` at
`core/llm_hybrid.py:128-131`, `MultiEngineRouter`/`LoRAHeterogeneousRouter` in
`core/llm_router.py`) is role-keyed: a role string in, a backend out. Nothing
about it assumes today's role set (scout/developer/critic/hater/validator/
synthesizer) — `_role_models` is just a dict (`core/llm_groq.py:373-396`) and
`manifest()` (`core/llm_groq.py:497-498`) reports whatever keys it's given.
Stage 2's role names (extractor, verifier, composer — coordinate with the
concurrent Stage 2 spec, do not assume these are final) slot into the same
dict with zero router changes required:

```python
_DEFAULT_GROQ_ROLE_MODELS = {
    "extractor": "meta-llama/llama-4-scout-17b-16e-instruct",  # was "scout"/"forager"
    "verifier":  "llama-3.1-8b-instant",                        # was "validator"
    "composer":  "llama-3.3-70b-versatile",                     # was "synthesizer"
}
```

The one real change needed: `core/config.py:ACTION_TO_ROLE` (consumed by
`role_disabled`/`action_disabled` in both routers, e.g.
`core/llm_groq.py:489-495`) must gain entries for whatever actions Stage 2's
worker loop emits. This is a config-table edit, not a router redesign — flag
it to the Stage 2 author so the mapping table lands once, not twice.

`HybridRouter` needs one more thing: its `_DEFAULT_GROQ_ROLES = {"hater"}`
(`core/llm_hybrid.py:44`) is a *low-volume* role list — the hater fires ~1-2
calls/round. The verifier in Stage 2's design fires **once per claim**, i.e.
O(claims), not O(1) per round. If the verifier is Groq-routed under Hybrid,
it is a high-volume role and belongs on the local engine by the same logic
that keeps the synthesizer local today (`core/llm_hybrid.py:12-19`: local has
no TPD ceiling, a model that *completes* beats one that 413s mid-run). Default
recommendation: verifier stays local under Hybrid; only go off-family via
GroqRouter (full-cloud) or a second local adapter
(`LoRAHeterogeneousRouter`), not via Hybrid's Groq-overflow path, unless the
claim volume for the eval in question is proven to fit the TPD budget (see
budget math below).

### Concrete family pairings against current infra

Everything routes through Groq's OpenAI-compatible endpoint
(`core/llm_groq.py:185-203`) or a local model. Confirmed-working non-Llama
Groq model already in the codebase's own defaults: `gemma2` is *mentioned* in
comments (`core/llm_groq.py:6`, "Gemma2 for haters") but the actual
`_DEFAULT_GROQ_ROLE_MODELS` (`core/llm_groq.py:373-396`) and
`_RPM_LIMITS`/`_SEM_LIMITS` tables (`core/llm_groq.py:44-53`, `105-114`) list
**zero** Gemma or Mixtral entries — every configured model in the current
table is a Llama variant (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`,
`llama3-8b-8192`, `llama-4-scout`, `llama-4-maverick`) plus two non-Llama
entries: `deepseek-r1-distill-llama-70b` (Llama-derived despite the name —
distilled from a Llama base, so it is *not* a clean off-family pairing) and
`qwen-qwq-32b` (genuinely different lineage — Alibaba, not Meta). The
module's own docstring (`core/llm_groq.py:6`, "Mixtral for foragers") is
stale relative to the table beneath it — a prior model in this role list was
deprecated by Groq and the table was patched without the docstring. **Do not
trust the docstring for what's actually configured; trust the dict.**

That leaves exactly one confirmed off-family pairing available today without
adding a new provider: **Llama (writer) / Qwen (checker)** —
`qwen-qwq-32b` as verifier against any Llama extractor. Two more pairings are
proposable but need a preflight check before relying on them (Groq's free-tier
catalog rotates — `_is_unavailable_error` at `core/llm_groq.py:224-243` exists
precisely because models get decommissioned or gated without notice):

1. **Llama extractor / Qwen verifier** (confirmed available). `qwen-qwq-32b`
   is already in `_SEM_LIMITS`/`_RPM_LIMITS`, so it has a rate-limit profile
   the code already respects. Lowest-effort pairing to stand up.
2. **Llama extractor / local non-Llama verifier** (HybridRouter, own iron).
   Run the verifier as a second local model (Qwen2.5, Phi, or whatever
   `SWARM_MODEL` isn't already loaded as extractor/composer) via
   `core/llm_router.py`'s multi-model path. No Groq quota burn at all;
   architectural diversity bought with VRAM instead of TPM. Best fit if
   claim volume is high (see budget math — this is the only pairing that
   doesn't hit a TPM wall at scale).
3. **Llama extractor / Gemma or Mixtral verifier** — needs a live preflight
   (`router.preflight()`, `core/llm_groq.py:520-526`) against the current
   Groq catalog before it can be called "available"; do not hardcode a model
   name into `_DEFAULT_GROQ_ROLE_MODELS` without running preflight first,
   given the deprecation history logged in the module's own comments
   (`core/llm_groq.py:388-391`: two prior hater defaults 404'd and silently
   zeroed the role for a full run before `_is_unavailable_error` existed).

**Recommendation:** build against pairing 1 (confirmed, zero new preflight
risk) for the decorrelation experiment below; treat pairing 2 as the
scale-safe fallback (Stage 4a); treat pairing 3 as a stretch goal gated on a
preflight ping, not a design assumption.

### Token-budget math against free-tier limits

Groq free tier (per the task context, `llama-3.1-8b-instant`): **6,000 TPM /
500,000 TPD** per model. `qwen-qwq-32b` is not the same model as the budget
figure given, so treat the 6k/500k numbers as the conservative planning
baseline (the 8B-instant model in the code's own comment block claims 20k TPM
at `core/llm_groq.py:22`, but the module's `_SAFE_REQUEST_TOKENS` default of
5500, set 2026-07-24, `core/llm_groq.py:130-145`, documents a directly
observed 6000-TPM ceiling that contradicts the stale comment — **use the
observed 6000 figure, not the docstring's 20000**, the same lesson as the
Gemma/Mixtral docstring drift above: comments in this file rot, the runtime
guard is ground truth).

Per-claim verification call: assume ~400 tokens prompt (claim text + cited
span + instruction) + ~100 tokens completion (verdict + short rationale) =
~500 tokens/call. At 6,000 TPM that's **~12 verifier calls/minute** sustained,
ceilinged by TPD at 500,000/500 = **1,000 verifier calls/day** before the
daily cap bites. For a Stage-1-scale synthetic-corpus eval (n≥20 prompts,
Stage 1 target), if each prompt yields ~15-30 extracted claims (a plausible
range for a document-grounded task), that is 300-600 claims — **one day's
Groq free-tier budget for the entire eval, with no headroom for retries or a
second checker pass.** This is why pairing 2 (local second model) is the
scale-safe default: it has no TPD ceiling at all, only wall-clock and VRAM.
If pairing 1 (Qwen on Groq) is used for the decorrelation experiment
specifically (a smaller, one-time measurement, not the recurring eval), the
budget is comfortable; do not default the *production* verifier path to a
free-tier Groq model without re-deriving this math against the corpus size
actually in play.

### The decorrelation measurement

**Question:** does an off-family checker catch a materially different set of
extractor errors than a same-family checker, on the same claims?

**Corpus:** Stage 1's synthetic-world corpora (`THEFUTURE.md` §4 Stage 1) —
ground truth is generation-time, so "the extractor's claim is wrong" is a
mechanical check (string/atom match against the known-true fact list), not a
judgment call. This is the property that makes the measurement honest; do not
substitute a real-world corpus here, for the same reason Stage 1 exists at
all (§0: "every time measurement got more honest, results got worse" — don't
reintroduce a soft metric into the one experiment designed to avoid that).

**Design:**
1. Run the extractor (single family, e.g. Llama) over the synthetic corpus,
   producing N claims with known ground truth (correct/incorrect per the
   mechanical checklist).
2. Deliberately do **not** filter to only "hard" claims — include the full
   natural error rate; a checker that only sees pre-selected errors isn't
   measuring what it will see in production.
3. Run every claim through **two** verifiers: (a) same-family (another Llama
   instance/size), (b) off-family (Qwen, or the local second model per
   pairing 2).
4. For claims the extractor got **wrong**, record: did checker (a) flag it?
   did checker (b) flag it? Compute catch-rate for each, and — the number
   that actually answers the decorrelation question — the **rate at which
   (b) catches an error (a) missed**, and vice versa.
5. Repeat with roles reversed (Qwen extracts, Llama + Qwen-instance check) to
   separate "off-family is just a better model" from "off-family is
   decorrelated" — if the *same* family pairing wins regardless of which
   side is the writer, that's a capability gap, not a decorrelation effect,
   and the experiment should say so.

**Pre-registered success criterion:** off-family catch-rate on
same-family-missed errors ≥ 15 percentage points above same-family
re-check-rate on same-family-missed errors (i.e., running the *same* family
twice on its own misses, as the null comparison — a same-family re-check will
catch some errors too, by chance/sampling variance; the bar is that off-family
must beat that null, not beat zero). If off-family clears this bar, family
heterogeneity in verification is justified going forward and Stage 2's
verifier role should default to off-family per the pairings above. If it does
not clear the bar, per `THEFUTURE.md` §5 rule 1 ("no mechanism without a
pre-registered eval that could kill it") — **do not build the off-family
verifier path into the production pipeline**; same-family verification (much
cheaper, no cross-provider preflight risk) is the default until re-tested.

### Test plan

- Unit: `engine_for("extractor")`/`engine_for("verifier")` return distinct
  backend instances when configured for different models (extend
  `tests/test_heterogeneous_routing.py` pattern, MOCK_LLM subprocess harness
  per `CLAUDE.md`'s convergence env-var table).
- Integration: run the decorrelation experiment end-to-end on a small (n=5
  synthetic docs) Stage-1 corpus slice with `MOCK_LLM=1` first to prove
  plumbing (mock verdicts are meaningless but the harness — claim routing,
  ground-truth lookup, catch-rate tabulation — must run clean before spending
  real tokens).
- Real run: n≥20 synthetic docs (piggyback Stage 1's already-generated
  corpora — do not build a second synthetic corpus for this).

### Effort estimate

Router/config wiring: ~0.5 day (mostly `ACTION_TO_ROLE` table entries + one
preflight call for pairing 3 if pursued). Decorrelation harness (claim
extraction → dual verification → ground-truth scoring → report): ~1.5-2 days,
mostly because "ground-truth lookup for a claim" needs a stable claim→fact
mapping that Stage 1's corpus generator must expose (dependency below).
Total: **~2.5 days**, contingent on Stage 1/2 being done first.

### Dependencies on Stages 1-2

- **Hard dependency on Stage 1**: needs the synthetic corpora + their
  generation-time ground truth. Cannot run against real-world prompts (no
  mechanical ground truth) or the pre-2026-07-24 prompt sets (θ-dominance
  contaminates the signal — `THEFUTURE.md` §2a).
- **Hard dependency on Stage 2**: needs the extractor/claim schema
  (`{text, source_span, corpus_doc_id, confidence}` per §4 Stage 2) to exist
  before there's anything to verify. This spec assumes that schema; if Stage
  2 changes field names, update the harness, not this design.

---

## Stage 3b — Triage front door

### The decision

`THEFUTURE.md`'s corollary (§3): "any task a single call can do, a single
call does. Orchestration activates only on proof of need." Stage 3b is the
router that enforces this *before* any organization work happens — a cheap
gate upstream of Stage 2's extraction pipeline, not a post-hoc comparison.

**Default: single call.** The organization (extraction → verification →
composition) is the exception that must be earned per-task, not the default
path that occasionally gets bypassed. This inverts the current
`run_swarm.py` posture (continuous pool is *the* default,
`--corpus=placeholder` etc. are the opt-outs) — Stage 3b's triage node is a
new front door upstream of that entry point, not a flag alongside it.

### Decision inputs

1. **Corpus size vs. model context.** The single most defensible trigger per
   `THEFUTURE.md` §3's "honest caveat": "the defensible claims are (i) corpus
   scale beyond even frontier contexts... Claim whichever the data supports
   and no more." Concretely: `estimated_corpus_tokens > single_call_context *
   safety_margin` (margin because a single call also needs room for the
   question, instructions, and completion — not just raw corpus). This reuses
   `eval/packs.py`'s pack-size accounting (the pack builder already knows its
   own token budget per `--pack-scale {1x,4x,16x}`) — don't build a second
   token counter, call the one `eval/packs.py` uses.
2. **Task type.** Some task types are definitionally single-document /
   single-question (the existing `creative`, most `problem_solving` prompts)
   — no corpus to organize over, triage should short-circuit on task type
   alone before even measuring corpus size. Cross-reference
   `ROLES_FOR_TASK` in `run_swarm.py` (per CLAUDE.md's role-activation
   section) — task types that already suppress most roles are candidates for
   an even harder single-call default.
3. **Verification demanded.** A task that explicitly asks for audited/
   sourced claims (vs. a prose/opinion task) is a case where organization's
   verification advantage (§3 point 2) might justify the cost even at
   in-context corpus size. This is a weaker signal than corpus-size overflow
   and should not alone flip the default — it raises the bar the corpus-size
   check has to clear, it doesn't replace it. Sketch: a boolean/enum flag on
   the task request (`verification_required: bool` or similar), not an LLM
   judgment call — keep triage itself mechanical and cheap, per its own
   justification for existing.

**None of these should be an LLM call.** A triage gate that itself costs an
LLM call to decide "should we make more LLM calls" undermines its own
economics for the majority of tasks where the answer is an obvious no
(prompt fits in context, no corpus, no verification ask). Token/prompt-length
counting and a task-type lookup table are suf1ficient; reserve an LLM
classifier only if mechanical signals prove insufficient in practice (and
even then, route it through the cheapest available model, not the
extractor's model).

### Implementation shape

A new module, e.g. `core/triage.py`, called from `run_swarm.py`'s `main()`
**before** any router/engine construction (before the `--corpus=` parsing
block around `run_swarm.py:2184-2210`, since triage's verdict determines
whether the rest of that setup even runs). Sketch:

```python
def triage(task_type: str, corpus_tokens: int, context_window: int,
           verification_required: bool) -> TriageVerdict:
    """Mechanical gate. Returns SINGLE_CALL or ORGANIZE + a reason string
    that gets logged into summary.json / condition-F-style output for
    auditability — the whole point is that the "why we organized" reasoning
    is inspectable, not a black box."""
```

`TriageVerdict.SINGLE_CALL` is the default return for any input that doesn't
clear the corpus-size-vs-context test (weighted by task type and the
verification flag as described above). On `SINGLE_CALL`, `run_swarm.py`
**does not enter the worker pool at all** — it degrades to something
structurally identical to `eval/ab_harness.py`'s condition F path (direct
call given the retrieved/packed evidence, `eval/ab_harness.py` around
lines 254-334 for the evidence-assembly logic already used there) and then
runs the output through `core/clean_answer.py:split_answer()` (the existing
reader/diagnostics split) so the shape of the output is indistinguishable
from an organized run's `answer.txt` + `diagnostics.md` — callers and the
judge harness should not need to know which path fired except via the
provenance field below.

### Condition E/F are this path, wearing a different hat

State this plainly because it saves building something twice: **triage's
single-call path IS condition F.** `THEFUTURE.md` names F "the practitioner
baseline" and the decisive comparison (§0, §4 Stage 1 decision gate). Stage
3b does not invent a new "cheap mode" — it takes the F implementation that
already exists in `eval/ab_harness.py` (direct call + same retrieved
evidence, capped per `_MAX_EVIDENCE_CHARS`-style logic around
`eval/ab_harness.py:254`) and promotes it from "an eval condition" to "the
literal production code path triage degrades into." Condition E (direct call
+ swarm's synthesis prompt, the attribution control per CLAUDE.md's rules of
evidence) is *not* what triage degrades into — E exists to isolate whether
the swarm's value was the prompt, and that question is orthogonal to
triage's job (decide whether to organize at all). Reuse plan: extract the
evidence-assembly + direct-call-with-evidence logic in `eval/ab_harness.py`
condition F's implementation into a shared function both the harness and
`core/triage.py`'s degrade path import — do not fork it into two copies that
drift (the docstring-drift lesson from Stage 3a's Gemma/Mixtral example is
exactly what happens when two copies of "the same thing" are maintained
separately).

### Cost accounting

`THEFUTURE.md` §5 rule 4: "Cost multiple reported next to every win. A win
at 129x must say so." Triage's output — regardless of which path fired —
must carry a cost record: LLM call count, approx total tokens, and (when
organize path fired) the **cost multiple vs. what the single-call path would
have cost**, i.e. triage should still compute the single-call cost estimate
even when it decides to organize, so the eventual output can honestly state
"this answer cost Nx a single call because [reason from the triage verdict]."
This is a new field on `summary.json` (e.g. `triage: {verdict, reason,
single_call_cost_estimate, actual_cost, cost_multiple}`) — additive, doesn't
touch existing summary.json fields.

### Test plan

- Unit: triage verdicts are deterministic given fixed inputs (corpus_tokens,
  context_window, task_type, verification_required) — table-driven tests
  covering each trigger independently and combined.
- Unit: `SINGLE_CALL` degrade path produces output shape-compatible with
  `split_answer()`'s two-file contract (answer.txt / diagnostics.md), same
  assertion style as existing clean_answer tests
  (`tests/test_clean_answer.py`).
- Integration: MOCK_LLM run where a small placeholder corpus triggers
  `SINGLE_CALL` and a `--corpus=pack:<large>` run triggers `ORGANIZE`,
  asserting the router/worker-pool construction path is/isn't entered
  (patch `run_pool` and assert call count, don't need a real model for this).
- Regression: assert the triage-degrade F-equivalent path and
  `eval/ab_harness.py`'s condition F, given the same inputs, produce
  byte-identical evidence blocks (proves the "reuse, don't fork" requirement
  above actually holds, not just in prose).

### Effort estimate

`core/triage.py` mechanical gate + tests: ~1 day. Extracting condition F's
evidence-assembly into a shared function + wiring both callers: ~1 day
(mostly care not to regress `eval/ab_harness.py`'s existing behavior — it has
production eval history riding on it, don't refactor it carelessly). Cost
accounting fields in `summary.json`: ~0.5 day. Total: **~2.5 days**.

### Dependencies on Stages 1-2

- **Soft dependency on Stage 2**: triage's `ORGANIZE` path hands off to
  whatever Stage 2 builds (claims-ledger pipeline) instead of today's
  worker pool, once Stage 2 lands. Until then, triage can be built and
  tested against the *current* pipeline as the organize path — it is a
  routing decision, not a rewrite of what's on the other side of the gate,
  so it does not strictly block on Stage 2's completion. Recommend building
  Stage 3b in parallel with Stage 2, wired to the current pipeline first,
  then re-pointing the `ORGANIZE` branch at Stage 2's ledger pipeline when
  ready.
- **No dependency on Stage 1** beyond wanting Stage 1's corpora available as
  test fixtures for the size-trigger tests (nice-to-have, not required —
  synthetic docs of known token length work fine for triage's own tests
  without needing Stage 1's ground-truth checklists, which triage never
  consults).

---

## Stage 4a — Scale runs

### Execution plan

`THEFUTURE.md` §4 Stage 4: "Scale runs on Colab A100/H100 ... but only after
Stage 1's gate passes something." This section specs *how*, gated on that
*if*.

**Which conditions run where:**

| Condition | Engine | Rationale |
|---|---|---|
| A (organization / swarm-successor) | Colab local vLLM (A100/H100) | High call volume (extraction + verification per claim); local avoids Groq TPD ceiling entirely (Stage 3a budget math above: ~500-600 claims already eats a full day's free-tier budget at n=20; n≥20 real scale runs will exceed it fast). |
| B (direct, no evidence) | Groq (cheap, single call/prompt) | Low volume — n prompts, 1 call each. Free tier easily covers this. |
| E (direct + synth prompt) | Groq | Same as B — single call/prompt. |
| F (direct + same evidence, single-call RAG) | Groq, with `_fit_request`'s token-budget clamp (`core/llm_groq.py:155-182`) actively in play at 16x pack scale | This is the condition most likely to hit Groq's per-request 5500-token safe ceiling at 16x — expect prompt truncation to fire (logged per `core/llm_groq.py:175-181`); if it fires routinely at 16x, that's a finding to report, not a bug to silently absorb (an F that got truncated isn't a fair F). |
| D (stronger model M+) | Groq (if a suitable strong model is on the free/paid tier) or Colab if it needs to be a local strong model not available via Groq | Case-by-case; depends what "stronger" means for the specific eval. |
| verifier (off-family, Stage 3a pairing 2) | Colab local, second model resident alongside the extractor | Per Stage 3a's budget conclusion: local is the scale-safe default for the high-volume verifier role. |

**Colab CLI workflow** (per the task context: `colab.exe: new/exec/upload/
download/stop`, A100/H100 on paid tier). Sketch, scripted, sessions always
stopped when idle (cost discipline — a forgotten running A100 session is
pure waste):

```
colab.exe new --gpu a100                     # provision
colab.exe upload <repo tarball / git pull>   # sync code
colab.exe exec "pip install -r requirements.txt"
colab.exe exec "python -m eval.ab_harness --mini 20 --conditions A --backend local --pack-scale 16x"
colab.exe download eval/results/<run>/       # pull artifacts back
colab.exe stop                               # ALWAYS — no exceptions, script this as the last
                                              # line of every invocation, not a manual follow-up
```

The "sessions always stopped" instruction from the task context should be
enforced structurally, not by discipline: wrap the whole sequence in a
driver script (`scripts/colab_scale_run.sh` or similar, out of scope for this
doc to write but flagged as a build item) with a `trap ... EXIT` / `finally`
that calls `colab.exe stop` even if the exec step fails or the harness
crashes mid-run. A scale-run harness that leaves a paid A100 running because
the eval script threw is a real cost bug, not a hypothetical.

**Token/compute budgets:** n≥20 prompts × (up to) 16x pack scale is the
Stage-1-inherited scale target. Per-prompt organize-path cost at 16x is the
open unknown this whole document exists to measure — do not pre-commit a
budget number here beyond "expect it to be large; that's the cost multiple
Stage 3b's accounting must report per THEFUTURE.md §5 rule 4." Groq-routed
conditions (B/E, and D/F where applicable) are bounded by the per-prompt math
in Stage 3a (single call each, well inside free-tier TPD).

### Provenance recording

`summary.json` must record, for every scale run, enough to keep parity checks
honest — the existing `run_meta.json`/`summary.json` provenance fields
(model bundle, backend) are a start but need three additions for scale-run
integrity:
- `engine`: `vllm` / `groq` / `hybrid` per condition (reuse the router
  `manifest()` output already produced by every router — `core/llm_groq.py:
  497-498`, `core/llm_hybrid.py:144-158` — don't invent a new provenance
  format, just persist the existing `manifest()` dict into `summary.json`
  if it isn't already there for the continuous-pool path).
- `weights`: exact model identifier as reported by the backend post-preflight
  (i.e. the *actual* model used after any decommissioned-model fallback swap,
  not the requested name — `core/llm_groq.py:351-353` already tracks this
  distinction for exactly this reason; surface it, don't re-derive it).
- `serving_stack`: vLLM version / Groq API version-equivalent (whatever's
  inspectable), plus GPU type (A100/H100/T4) for the local-engine conditions
  — a parity check between "condition A on A100 vLLM" and "condition D on a
  different serving stack" is not a fair comparison if the stacks quantize
  or batch differently, and CLAUDE.md's vLLM section already flags this kind
  of load-cascade variability as something to log.

### Test plan

- Dry-run the Colab CLI script sequence against a `MOCK_LLM=1` / cheapest
  possible config first (cost discipline — don't debug the driver script on
  billed A100 time).
- Assert `summary.json` contains all three new provenance fields on at least
  one real (non-mock) scale-run condition before calling this stage done.
- Verify the `trap`/`finally` stop-on-failure behavior by deliberately
  making the exec step fail and confirming `colab.exe stop` still runs.

### Effort estimate

Driver script + provenance field wiring: ~1.5 days. First real dry run +
debugging Colab CLI quirks (session limits, upload size, etc. — unknowns
until tried): ~0.5-1 day contingency. Total: **~2-2.5 days**, excluding the
actual multi-hour/multi-day compute time of the scale runs themselves (which
is compute cost, not engineering effort).

### Dependencies on Stages 1-2

- **Hard dependency on Stage 1's gate outcome** — per `THEFUTURE.md` §4,
  this stage does not run at all if Stage 1 doesn't "pass something." Do not
  provision Colab compute before that gate has a verdict.
- **Hard dependency on Stage 2** for what "condition A" even means at scale
  (the claims-ledger pipeline, not the retired signal-store swarm, assuming
  Stage 1's likely-per-§0-evidence outcome that F≈A holds on synthetic
  corpora too and Stage 2 replaces the old design per the decision gate).

---

## Stage 4b — Organizational memory

### The reimagining

`THEFUTURE.md` §4 Stage 4: "Revisit the knowledge base as organizational
memory: verified claims (not clusters) persist across runs with provenance
and age-out. This is the third organizational advantage and currently unbuilt
in the new design." The existing `core/knowledge_base.py` persists **cluster**
entries (`_build_genomes`-derived, schema v3, `core/knowledge_base.py:60`).
Stage 4b's ledger persists **claims** (Stage 2's atomic unit) instead. This is
a schema change in kind, not a version bump — clusters are an artifact of the
retired signal-store design (`THEFUTURE.md` §6: "strength/pheromone dynamics
... never beat its null model"); claims are Stage 2's native unit and survive
the transition cleanly.

### Adapting the two-channel contradiction detection

`core/knowledge_base.py:_detect_contradictions` (lines 390-473) already does
almost exactly what a claims ledger needs, at the cluster level:
- **Channel 1** (embedding cosine ≥ `_KB_CONTRADICTION_THRESHOLD` = 0.75,
  `core/knowledge_base.py:65`, checked at line 454): representative-embedding
  similarity between a new and prior entry.
- **Channel 2** (atom-level word-Jaccard ≥ `_ATOM_CONTRADICTION_JACCARD` =
  0.50, `core/knowledge_base.py:445`, checked at lines 458-469): already
  operates on **atom text**, extracted via `_atom_texts()`
  (`core/knowledge_base.py:429-435`) from `entry.get("genome_atoms")`. This
  is, functionally, already claim-level matching wearing a cluster-shaped
  wrapper — the genome's `atoms` field (per CLAUDE.md's Cluster Genome
  section) is the closest existing analogue to Stage 2's claim schema.

**The adaptation is mostly a rename + flattening, not new logic:** instead of
"cluster A's genome_atoms vs cluster B's genome_atoms," Stage 4b's ledger
runs the same `_word_jaccard` check (`core/knowledge_base.py:437-443`)
directly between a new claim's text and every prior claim's text — the
`_check_and_flag` double-loop (`core/knowledge_base.py:447-469`) already has
the right shape (all-pairs comparison between a new group and a prior group);
it just needs to iterate over `claims` instead of `entries-carrying-atoms`.
Channel 1 (embedding cosine) carries over unchanged — a claim, like a
cluster, has a representative text that can be embedded. Reuse
`_cosine_sim` (`core/knowledge_base.py:72-73`) as-is.

**One real design decision**, not present at the cluster level: cross-run
contradiction at the claim level needs to know *which prior run* a
contradicting claim came from, and whether that prior claim was itself
verified or since superseded — the ledger's `contradicts` field (mirroring
`core/knowledge_base.py:419-426`'s `ea["contradicts"]`/`eb["contradicts"]`
pattern) should carry a **verification-status chain**, not just a hash
pointer, so a query against the ledger can say "this claim contradicts a
claim verified in run R, still active" vs. "...a claim that was itself later
retracted" — the existing cluster-KB has no notion of retraction because
clusters don't get individually retracted, only decayed/archived wholesale.

### Age-out / TTL

Reuse, don't reinvent: `core/knowledge_base.py` already has two age
mechanisms that map directly:
- **Decay-on-load**: `_KB_DECAY_PER_LOAD = 0.95` (`core/knowledge_base.py:69`)
  applied every load, entries below `_KB_ARCHIVE_THRESHOLD = 0.3`
  (`core/knowledge_base.py:67`) moved to an archived list (per the module
  docstring, `core/knowledge_base.py:18-20`). For a claims ledger, "decay
  strength" doesn't map as cleanly (a verified fact about an invented
  company's Q3 filing doesn't get less true over time) — but a **confidence
  decay tied to source recency** does make sense for claims whose truth is
  time-sensitive (vs. claims that are permanently true given the frozen
  corpus). Stage 4b should make this decay conditional on a claim-type flag
  (time-sensitive vs. static), not apply the flat 0.95 unconditionally —
  applying cluster-era decay logic to a permanently-true claim would
  eventually age out a fact that never stopped being true, which is worse
  than no decay at all.
- **Explicit TTL / prune-before**: `KnowledgeBase.prune_before(date_str)`
  (`core/knowledge_base.py:261`, already implemented per DEFERRED.md's
  resolved-tasks list, "KB `prune_before(date_str)`") is a direct fit —
  reuse verbatim against `originating_run`/timestamp fields on claim entries,
  same as it already works for cluster entries.

### The eval that proves memory helps

Design (sequential, same synthetic world, per the task instructions):

1. Build one Stage-1 synthetic world (fixed corpus, fixed ground truth).
2. Run a **sequence** of related tasks against it (e.g., "summarize Q1
   filings" → "compare Q1 to Q2" → "what changed between the filings" — tasks
   that share underlying facts but ask different questions), across separate
   pipeline invocations, WITH the claims ledger active (`--use-kb`-equivalent
   opt-in, see default-off note below) vs. a control run with the ledger
   disabled (`--reset-kb`/no-memory baseline, same corpus, same task
   sequence, from-scratch extraction+verification every time).
3. **Metrics**: (a) re-verification cost saved — count of claims in task N+1
   that match (channel 1 or 2) an already-verified claim from task N and can
   be reused without a fresh verifier call, vs. the control's full
   re-verification cost; report as a token/call-count delta, not prose.
   (b) accuracy delta — grounded-accuracy score (Stage 1's mechanical
   checklist scoring, per `THEFUTURE.md` §4 Stage 1) on the later tasks in
   the sequence, memory-on vs. memory-off. A real "memory helps" result needs
   **both** a cost saving and a non-negative accuracy delta — a ledger that
   saves cost by skipping re-verification but degrades accuracy (stale/wrong
   claims getting reused unchallenged) is not a win, it's the KB's own
   documented risk (contradiction detection exists precisely because prior
   entries can go stale — see `core/knowledge_base.py`'s own module
   docstring on this).

**Pre-registered success criterion:** re-verification cost saved ≥ 20% of
control-run verifier calls on tasks 2+ in the sequence, AND grounded-accuracy
delta ≥ 0 (memory must not cost accuracy to save cost) at n≥10 sequential
task-pairs. Either condition failing means memory is not yet proven — per
`THEFUTURE.md` §5 rule 1, do not ship the ledger as default-on without this
passing.

**Default OFF**, matching the current KB (`CLAUDE.md`: "Default is OFF —
pass `--use-kb` to opt in") — until this eval passes, Stage 4b's ledger is an
opt-in ablation arm, not a pipeline default. State this explicitly in
whatever flag gates it (e.g. `--use-ledger`, mirroring `--use-kb`'s naming
so the "off by default, proven before promotion" convention is visually
obvious to a future reader).

### Test plan

- Unit: claim-level `_word_jaccard`/`_cosine_sim` contradiction check against
  a hand-built fixture of known-contradicting and known-consistent claim
  pairs (mirror the existing `tests/test_knowledge_base.py` /
  `tests/test_kb_migration.py` pattern cited in CLAUDE.md's
  support_diversity section for the "correct test pattern" convention).
- Unit: `prune_before` against claim entries with `originating_run`
  timestamps (same as existing cluster-level test, retargeted).
- Unit: decay-conditional-on-claim-type — a static claim survives N decay
  cycles unarchived where a time-sensitive claim at the same starting
  confidence does not.
- Integration: the sequential-task eval itself, first at small n (n=2
  task-pairs, MOCK_LLM plumbing check) then real (n≥10 per the pre-registered
  criterion).

### Effort estimate

Schema adaptation (claim-shaped entries, contradiction channel retargeting,
conditional decay): ~1.5 days, mostly careful reuse of existing functions
rather than new logic. Sequential eval harness (task-sequence runner +
cost/accuracy scoring against Stage 1's mechanical checklist infra): ~2 days.
Total: **~3.5 days**, plus the real-run compute time for the n≥10 eval
(depends on Stage 4a's scale-run infra being available, since this eval
benefits from the same Colab/local-engine setup for the verifier-heavy
control run).

### Dependencies on Stages 1-2

- **Hard dependency on Stage 1**: needs the mechanical grounded-accuracy
  scorer for the accuracy-delta metric, and ideally a synthetic world
  structured to support a *sequence* of related tasks (single-snapshot Stage
  1 corpora may need a small extension — multiple related questions against
  one world — flag this to the Stage 1 owner as a possible shared need
  rather than building a second corpus generator).
- **Hard dependency on Stage 2**: the claim schema being persisted IS Stage
  2's claim schema; this stage cannot be built before that schema is fixed
  (rename risk noted in Stage 3a applies here too).
- **Soft dependency on Stage 4a**: benefits from, but does not strictly
  require, the scale-run infra — the sequential eval can be run at small
  scale locally/on Groq before a full Colab scale-run pass is justified.

---

## Cross-stage build order (recommended)

1. Stage 3b (triage) — least coupled to Stage 2's internals, can start
   against the *current* pipeline immediately, re-point later. Also the
   cheapest win: it stops any future wasted-compute run before it starts.
2. Stage 3a (heterogeneous verification decorrelation experiment) — needs
   Stage 1 + Stage 2's claim schema; run the experiment once, get a verdict,
   then either wire the off-family default (pairing 1 or 2) or don't.
3. Stage 4a (scale infra) — only after Stage 1's gate passes something, per
   `THEFUTURE.md` explicitly. Build the driver script early (it's cheap and
   useful for Stage 3a's real-scale decorrelation run too), but don't spend
   real Colab compute on organize-path scale runs until gated.
4. Stage 4b (organizational memory) — last, both because it depends on
   Stage 4a's infra for its real-scale control run and because it is the
   least urgent of the three organizational advantages per `THEFUTURE.md`
   §3's own ordering (coverage, verification, memory — memory is listed
   third for a reason; it also has the highest design risk of quietly
   reintroducing the "prior consensus feeds back into new extraction and
   collapses diversity" failure mode the existing KB docstring already
   warns against, `core/knowledge_base.py:9-11`: "prior_consensus does NOT
   re-deposit as scout INITIALs... injecting cached consensus into their
   input space defeats the divergence layer" — the claims-ledger version of
   this warning is: cached claims must not get fed back into the *extractor's*
   prompt, only used post-hoc for contradiction-checking and cost-saving on
   the *verifier* side, or memory quietly becomes a second copy of the
   partitioning-as-diversity mistake `THEFUTURE.md` §6 already retired).
