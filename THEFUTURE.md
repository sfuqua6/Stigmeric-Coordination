# THEFUTURE.md — The Organization Program

**Status:** Active roadmap as of 2026-07-24. Supersedes the empirical program in
`docs/research/STIGMERGIC_INFORMATION_PARTITIONING.md` §8 and the "decision
procedure" in CLAUDE.md item 7. The theory doc remains valid as the record of
the anti-deliberation argument; its diversity claim is retired below.

**The goal, restated:** a proper *organization* of weaker models outcompetes a
stronger model. Not "a swarm out-thinks a genius" — that version died on
2026-07-24 and this document exists so we never rebuild it by accident.

---

## 0. The verdict that forced this document

First fully-valid run of the harness (real retrieval, real 16x packs, model
parity enforced, strong off-family judge `openai/gpt-oss-120b`, both orders,
6/6 order-agreement — `eval/results/overctx_16x/`):

| Comparison | Result | Meaning |
|---|---|---|
| A vs B (swarm vs one call of same model, *from memory*) | **0W / 0T / 6L** | The orchestration subtracts value |
| A vs E (swarm vs one call + the swarm's own synthesis prompt) | **1W / 0T / 5L** | The prompt was most of the value |
| A vs F (swarm vs single-call RAG, 16x pack) | blocked (free-tier infra; `F_BLOCKED.md`) | Kill criterion untested, not failed |
| B factual checklist, zero evidence shown | **100%** | The prompt set never required the corpus |

Plus a judge bug (`eval/judge.py` position-label mis-tally, fixed 2026-07-24)
that had converted right-side losses into "ties" in every historical
A-vs-{C,D,E,F} report. The pattern across the project's whole history: **every
time measurement got more honest, results got worse.** That is what a wrong
hypothesis looks like from inside.

## 1. What survives (do not rebuild these; do not delete them)

1. **The anti-deliberation argument and the no-leak rule.** Conditioning on
   another agent's reasoning chain concentrates the posterior (conformity
   cascade). This part of the theory doc is right, unrefuted, and stays a hard
   constraint.
2. **The measurement harness.** Pre-registered kill criteria, attribution
   controls (E, F), blind both-order judging, Wilson bounds, parity checks.
   It caught three generations of self-deception including its own judge bug.
   It is the most valuable artifact this project has produced.
3. **Retrieval + packs.** Tavily-backed `core/search_tool.py`, the
   keyword-query pack builder (`eval/packs.py`), `--corpus=pack:` mode.
4. **The deterministic composition path.** The extractive fallback out-shipped
   the LLM composer 4/6 on the last run. It is the de facto renderer; §4
   promotes it to the design.
5. Infra: vLLM backends, GroqRouter, Colab CLI workflow, MCP surface.

## 2. The diagnosis (why the old bet lost — three structural facts)

**2a. Prior dominance.** The coin metaphor claimed P(y | x_i, θ) diversifies
when x_i does. That holds only where **x matters more than θ**. On every prompt
ever tested (public debate/analysis topics), θ dominates: the model already
holds a settled posterior and the partition is a rounding error. B scoring 100%
from memory is the proof. The theory doc's own §8 named this failure mode
("pretraining leakage... we have no answer") — it was the whole ballgame.
Bagging retrains members on different data (different mappings); partitioning
feeds different inputs to the *same* mapping. Not an analog.

**2b. Partitioning is coverage, not diversity.** Partitioned reading is
genuinely valuable — as division of labor over a corpus no single context can
hold. It never was a diversity engine. Diversity lives in different weights
(model families) and different jobs, and must be bought there.

**2c. No mechanical vote.** Ensembles work because aggregation is mechanical
(majority vote over independent errors). The swarm aggregated via strength
dynamics (popularity among agents sharing one prior) and then an LLM
synthesizer that failed its own faithfulness audit 4/6. The committee had no
honest vote-counter.

## 3. The new thesis

Human organizations do not beat smarter individuals by aggregating opinions.
They win through three things a lone genius structurally cannot do:

1. **Coverage** — hold and process more information than one head (or one
   context window) can.
2. **Verification** — independent procedural checking of every load-bearing
   claim.
3. **Memory** — retained, queryable knowledge across engagements.

**Thesis:** an organization of weak models beats a stronger single model on
tasks that require coverage + verification — i.e., where *knowing more, checked
better* beats *being smarter* — and nowhere else. Corollary: the system must
**triage**: any task a single call can do, a single call does. Orchestration
activates only on proof of need (corpus > context, or verification demanded).

**The fair fight (pre-registered):** the stronger model gets the same retrieval
and its full context window; the organization reads everything in partitions.
Scoring is **grounded accuracy against the corpus, mechanically checked** — not
prose quality judged by an LLM (prose is the strong model's home turf and not
the product). Win condition: Wilson lower bound > 0.5 on grounded accuracy at
corpus ≥ 4x the strong model's context, n ≥ 20.

**Honest caveat carried forward:** frontier context windows keep growing. The
defensible claims are (i) corpus scale beyond even frontier contexts,
(ii) verification quality (auditability), (iii) equal grounded quality at a
fraction of the cost. Claim whichever the data supports and no more.

## 4. The program

> **Implementation specs (2026-07-24, `docs/future/`):**
> - Stage 1 → `docs/future/STAGE1_SYNTHETIC_EVAL_SPEC.md` — fact-registry-first world generator (pure Python, seeded), LLM rendering with cache-and-reuse determinism and a mechanical fact-fidelity gate, exact pack-JSONL reuse (`--corpus=pack:` unchanged), no-LLM scorer (4 item types incl. citation-grounding), Gate 1: B-from-memory < 20% (target ≤ 5%). ~6.5–9 days.
> - Stage 2 → `docs/future/STAGE2_CLAIMS_LEDGER_SPEC.md` — typed Claim + append-only VerificationRecord; status is a monotone state machine (`unverified → verified|contested|refuted`), never a scalar; worker-pool skeleton reused with the 7-action registry collapsed to EXTRACT/VERIFY; composition = deterministic draft + one polish call behind 100% claim-coverage, no-new-numbers, and faithfulness audits; full keep/adapt/retire table over core/ and agents/. ~14 days, ~8 buildable now against fixtures.
> - Stages 3–4 → `docs/future/STAGE3_4_HETEROGENEITY_TRIAGE_SCALE_SPEC.md` — extractor/verifier/composer roles slot into the existing `engine_for(role)` contract; pre-registered decorrelation experiment (off-family catch-rate ≥ 15pp over same-family null, with role-reversal control); triage's single-call path IS condition F (shared evidence-assembly, `summary.json: triage` cost field); Colab/Longleaf run plan; claims-based memory (default OFF until its pre-registered eval passes). ~2.5–3.5 days per stage.
> - Seam notes: watch `Claim.confidence` — Stage 2 marks it explicitly non-load-bearing; any future patch that makes survival depend on it has smuggled a strength scalar back in. The Stage 2 ablation MUST run on Stage 1 synthetic corpora (running it on the old prompt set inherits the θ-dominance invalidity).

### Stage 1 — The synthetic-world eval (BUILD FIRST; everything gates on it)

The single change that fixes every invalid experiment to date: make the corpus
**unknowable from pretraining**.

- Generate fictional-world corpora (an invented company's filings, an invented
  country's statistics + news archive, a case file): internally consistent,
  seeded/deterministic, sized 1x/4x/16x against the baseline context.
- Checklists derived from generation-time ground truth: every item is a fact
  that exists **only** in the corpus. Mechanical scoring (atom/string match +
  citation-to-source verification). No LLM judge for the headline metric.
- **Validity gate:** condition B (no evidence) must score ≈ 0%. If B can answer
  from memory, the corpus failed; regenerate. This is the inverse of the flaw
  that killed the old prompt set.
- Then rerun the existing harness **unchanged**: A/B/E/F at 1x/4x/16x.

> **PILOT RESULT (2026-07-25, `eval/results/synth_company_1x/`, n=2, 1x scale — directional, not the n≥20 verdict):** A lost every prompt to every control on the mechanical scorer (atomic-fact: A 22% / B 44% / E 50% / F 56%; quantity: A 0% / F 26%), and condition A never produced a real answer — `cap_time` with **zero surviving clusters** both runs. The debate-tuned survival machinery does not function on fact-dense unfamiliar corpora at all. Consistent with the first branch below. Operational findings: one 1x ABEF pilot consumed the full Groq free-tier 500k TPD (the n≥20 fight requires paid Groq or all-local Colab serving per the Stage 3/4 spec); `verify_render`'s verbatim bar rejected 10/10 LLM-rendered docs (8B renderer paraphrases — needs a stronger renderer or normalized matching); B's in-run 44% vs its 3.5% Gate-1 score suggests the task prompts themselves leak registered facts — audit `synthetic_prompts_for_world()` before the fair fight.

**Decision gate (pre-registered):**
- If **F ≈ A** on synthetic corpora too → the current orchestration adds
  nothing even in its home regime. Do not iterate on the swarm; proceed to
  Stage 2 with the claims-ledger design replacing the signal store outright,
  and the old pipeline becomes the ablation baseline.
- If **A > F** at 16x → the coverage mechanism has value; Stage 2 hardens it.
- Either way, publish the Stage-1 numbers. Negative results from this harness
  are publishable; that is the point of the harness.

### Stage 2 — The claims ledger (replaces pheromones with procedure)

The unit of exchange stops being a "signal with strength" and becomes a
**typed claim with provenance**:

- Extraction workers read partitions → emit claims `{text, source_span,
  corpus_doc_id, confidence}`. No-leak preserved: workers see corpus + ledger
  artifacts, never each other's reasoning.
- **Verification is the survival mechanism:** each claim independently checked
  against its cited span (and against sampled *other* partitions for
  contradiction) — by a checker that did not write it. Survival = verification
  outcomes. No decay, no amplification, no pheromones. (The blackboard
  null-model question from `docs/CRITIQUE_LOOP_2026-07-06.md` §8 Thesis 5 is
  answered by construction: if strength dynamics matter, they must beat this
  ledger in ablation to earn re-entry.)
- **Deterministic composition** from surviving claims (the promoted extractive
  path); one constrained LLM polish pass, audited against the ledger, with the
  existing hard gate. The composer can rephrase; it cannot add or drop claims.

### Stage 3 — Heterogeneity where it pays

- Extractor and verifier from **different model families** (GroqRouter /
  HybridRouter already support this; the "cloud validator" item in DEFERRED.md
  was this idea). Measure error-decorrelation directly: rate of extractor
  errors caught by off-family vs same-family checkers.
- Triage front door: a cheap classifier/heuristic that routes
  single-call-sufficient tasks to a single call. The organization must never
  again pay 129x to lose a fight it didn't need to enter.

### Stage 4 — Scale and memory

- Scale runs on Colab A100/H100 (CLI workflow proven 2026-07-24) and Longleaf
  per the compute request — but only after Stage 1's gate passes something.
- Revisit the knowledge base as **organizational memory**: verified claims
  (not clusters) persist across runs with provenance and age-out. This is the
  third organizational advantage and currently unbuilt in the new design.

## 5. Rules of evidence (carried forward, amended)

All of CLAUDE.md's rules stand, plus:
1. **No mechanism without a pre-registered eval that could kill it.**
2. Every prompt set ships with a **B-from-memory validity check**; B ≥ 20% on
   the checklist invalidates the set.
3. Headline metrics are mechanical (grounded accuracy, citation
   precision/recall). LLM judges are for secondary prose comparisons only,
   off-family, both orders, agreement reported.
4. Cost multiple reported next to every win. A win at 129x must say so.

## 6. What we do not build (retired, with cause of death)

- Strength/pheromone dynamics as the survival mechanism — never beat its null
  model; re-entry only via ablation victory over the ledger (Stage 2).
- Partitioning-as-diversity — prior dominance (§2a). Partitioning remains as
  coverage logistics only.
- Same-model agent populations for "perspective diversity" — self-BLEU
  0.62–0.69; perspectives came from the prompt (E proved it).
- LLM-based synthesis as the primary renderer — lost to its own extractive
  fallback 4/6; demoted to constrained polish.
- LLM planner, speculative lattices, biomimicry primitives — already retired
  in CLAUDE.md; still retired.
- Competing on prose quality against any frontier model, ever.

---

*This document is the successor to the empirical program of
`STIGMERGIC_INFORMATION_PARTITIONING.md`. The theory doc stays: its
anti-deliberation argument is load-bearing in §1, and its §8 open questions —
especially the pretraining-leakage question it could not answer — are now
answered, mostly against the original design. That is what the harness was
for.*
