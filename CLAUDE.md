# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚡ CURRENT WORK STATE — READ FIRST (updated 2026-07-24)

**Staleness warning:** HEAD (`6d6a884`) is dated 2026-07-03. Everything below — ~2,300 insertions across ~20 files, all verified green — has been sitting uncommitted in the working tree for three weeks. Nothing has been run or committed since 2026-07-10. The single most protective first action in any session is committing this work in reviewable chunks.

A large uncommitted overhaul is in the working tree (run `git status` / `git diff` before anything else — nothing below is committed):

1. **`docs/CRITIQUE_LOOP_2026-07-06.md`** — the authoritative findings + rationale for every change (7 load-bearing findings; §8 = ordered plan; post-loop correction: the DEFAULT continuous pipeline never ran corpus partitioning at all — `partition_id` was the worker id).
2. **DONE + VERIFIED (641 tests, 0 failures):** tranches 1–2 (audit-ordering fix already in synthesizer.py, signed trail, dissent dup-count weighting, INITIAL-only novelty, grounded quality gate, render-signature stability fixes, fitness cleanups + re-signed consensus, eval condition F, judge provenance/family checks, `summary.json: partitioner`) **plus the big one: corpus/facet partition wiring into the DEFAULT `run_continuous_pipeline`** (`assemble_partitions()` in run_swarm.py, `USE_CORPUS_PARTITIONS` flag, partitions flow into `run_pool`→Worker→SCOUT evidence→INITIAL `partition_id` stamping) — smoke-verified 20/20 INITIALs carry content partition_ids, zero worker_* leakage. **This closes the core architectural gap**: corpus partitioning now actually runs in production, not just on `--legacy-rounds`.
3. **DONE + VERIFIED:** `eval/partition_probe.py` hardening (k=1 ZeroDiv fix, incremental crash-safe report writing, embedder/partitioner/output_lens provenance, degenerate-output filter, verbalized-sampling condition V, verdict withheld unless partitioner=="semantic"). 18/18 tests passed. Ready to re-run for real: `GROQ_API_KEY=... python -m eval.partition_probe --mini 20 --corpus retrieve --include-hot --v-condition`.
4. **DONE + VERIFIED:** faithfulness-audit hardening in `agents/synthesizer.py` — per-sentence citation attribution (`_sentence_at`, falls back to paragraph only under 8 words), a negation screen (`negation_mismatch` flag when a citing sentence negates a claim the source never negated), and a consequential `_AUDIT_HARD_GATE` (env `SWARM_AUDIT_HARD_GATE`, default 12) that discards flagged prose and ships the existing extractive-fallback rendering instead, with `renderer_audit.json: gated=true` when it fires. `core/clean_answer.py` now keeps referenced `[N]` markers in answer.txt and appends a compact **Sources** block (previously ALL citations were stripped — the shipped answer had zero attribution). 53 passed / 5 skipped on the targeted test set, no new skips.
5. **DONE + VERIFIED:** `docs/OVERCONTEXT_EVAL_PLAN.md` — the decisive A-vs-F over-context experiment. `eval/packs.py` (deterministic pack build/reuse), `run_swarm.py` `--corpus=pack:<path>` mode, `SWARM_DISABLE_LIVE_SEARCH=1` kill switch, `eval/ab_harness.py` flags `--prompt-set overcontext --pack-scale {1x,4x,16x}`, `eval/prompts.py: OVERCONTEXT_SET`. MOCK-verified A and F reference the identical pack (the property that makes the comparison valid); 22/22 targeted tests passed. **Real run not yet executed** — needs `GROQ_API_KEY`: `python -m eval.ab_harness --name overctx_16x --prompt-set overcontext --mini 8 --conditions ABEF --pack-scale 16x` then `python -m eval.judge eval/results/overctx_16x`. Kill criterion (pre-registered): if A doesn't beat F with Wilson lower bound > 0.5 at 16x, the compression thesis is falsified in its best-case regime.
6. **All four implementation tranches from the critique loop are complete and the full suite is green: 642 passed, 8 skipped, 0 failures** (verified 2026-07-10, end-to-end, after all changes including synthesizer hardening). Nothing is committed yet — consider committing in reviewable chunks (`git diff --stat` shows ~17 files across partitioning/convergence/fitness/eval/synthesizer) rather than one giant commit.
7. **Decision procedure — the path out of limbo (2026-07-24).** Verified state of the evidence: `eval/results/` contains ZERO real-LLM judged runs of the current design — `overctx_smoke/` is mock, both `partition_probe_20260706_*` runs are mock, and the one real-retrieval probe attempt (`partition_probe_20260703_214551/`) died before writing a report. Every judged loss in `hey fable.md` predates semantic partitioning, the clean-answer split, and conditions E/F. The project is not "failed"; it is **undecided, with the deciding experiment built and never run.** In order:
   1. **Commit the working tree** in reviewable chunks (see item 6). Three weeks of verified work in an uncommitted state is the largest live risk to the project.
   2. **Run the over-context experiment for real** (commands in item 5; needs only `GROQ_API_KEY`). It has a pre-registered kill criterion, so either outcome is a result: A beats F at 16x with Wilson LB > 0.5 → the compression thesis has its first supporting evidence, scale per the Longleaf request; A fails → the thesis is falsified in its best-case regime.
   3. **Branch on the outcome.** Win → real partition-probe run (`GROQ_API_KEY=... python -m eval.partition_probe --mini 20 --corpus retrieve --include-hot --v-condition`), then null-model ablation (CRITIQUE_LOOP §8 Thesis 5), then bigger compute. Loss → write the negative result honestly (the eval methodology itself — blind both-order judging, position-bias detection, attribution controls E/F, pre-registration — is publishable and reusable) and redirect effort to the salvageable assets: the eval harness, the retrieval stack, the faithfulness-audit/clean-answer machinery, and the MCP tool surface (`mcp_server.py`).
   Do not start new architectural work ahead of steps 1–2; every past cycle of building-before-measuring is how the repo accumulated three critique loops and zero decisive data points.

   **SUPERSEDED 2026-07-24 (end of day): the over-context run happened, the repaired verdict is in (see Empirical status), and the original bet is dead. The active roadmap is now `THEFUTURE.md` at the repo root** — the Organization Program: synthetic-world eval first (B-from-memory validity gate), then claims-ledger redesign, heterogeneous verification, triage. Read it before starting any new work.

## What this codebase is

A from-scratch rebuild of the original stigmergic multi-agent LLM pipeline that strictly enforces two principles:

1. **No-leak rule.** Agents only ever observe signals as artifacts (content + ID + structural metadata). Never another agent's reasoning chain, ancestry text, or chain-of-thought.
2. **Information partitioning as the diversity engine.** Diversity comes from *what each agent has been shown*, not from prompt or temperature tweaks. Scouts get disjoint corpus partitions; downstream roles use differentiated sampling strategies over the shared signal store.

This is the canonical pipeline, at the repository root. The original it replaced (`run_task.py` + the `swarm/` package) is preserved unmaintained under `legacy/`. Treat the no-leak rule as a hard architectural constraint when changing agents or the signal store.

## Empirical status — read before claiming wins

The core hypothesis (a swarm of partitioned agents out-reasons a single call of the same or a stronger model) is **currently unproven and losing on the head-to-head evidence**. `hey fable.md` (repo root) is the evidence-first audit, quoting the repo's own artifacts: 0 judged wins across all comparisons in `outputs/kb/` and `eval/results/`, `avg_verification_score` ≈ 0, `self_bleu` 0.62–0.69 (the field converges on near-duplicate claims), and substantive runs ending on `cap_time` rather than convergence. **Update 2026-07-06:** the failure mode has since flipped — 30/41 `driveoutputs/` runs halt on `quality` at ~70s (seconds after the MIN_TIME floor), 10 on `saturation` with 0 survivors, 1 on `cap_time`. The quality gate for debate/analysis now additionally requires one verification signal above the abstain plateau (`requires_grounding` in `SURVIVAL_TASK_PROFILES`) so a "quality" halt says something about grounding. Fixes landed since that audit: reader-answer split (`core/clean_answer.py`), judge normalization + neutral local judge, condition E attribution control, render-set stability halt, pre-call scout gate.

Rules of evidence for any future claim of a swarm win:
- It must come from `eval/ab_harness.py` conditions judged by `eval/judge.py` (blind, both orders). A verdict with `agreement: false` is position bias, not signal.
- The Wilson lower bound must clear 0.5 at a non-trivial n (the existing n=2–8 runs prove nothing either way).
- **Condition E is the attribution control**: a single direct call given the swarm's own synthesis instruction. If E matches A, the value was the prompt, not the orchestration. Never report A-vs-B without A-vs-E.
- **Condition F is the retrieval attribution control** (added 2026-07-06): a single direct call given evidence from the SAME retrieval stack the swarm uses. A-vs-B confounds orchestration with retrieval access (the swarm searches the web mid-run; B answers from memory). A-vs-F is the decisive comparison; A≈F at small evidence packs is expected — the compression thesis predicts A>F only when the pack outgrows a single context.
- **Pre-2026-07-03 negative results do not test the current design**: semantic corpus partitioning (`_partition_semantic`, commit aba21ef) postdates every judged run in `outputs/kb/`. Check `summary.json: partitioner` ("semantic" | "contiguous" | "") before citing a run as evidence about partitioning.
- **Know which pipeline you're reasoning about (verified 2026-07-06):** `partition_for_scouts` (and the facet web-partition block) runs only on the `--legacy-rounds` / phase-isolated paths. In the DEFAULT continuous pool, corpus partitioning never runs at all — `partition_id` is the depositing worker's agent_id (`core/worker_pool.py` ~1302–1314) and scout evidence is per-action live search via the query planner. `summary.json: partitioner` is `""` on every continuous run. The deployed diversity levers on the default path are: query-planner stance queries, multi-claim scout portfolio + novelty selection, and (with GroqRouter) model-family heterogeneity — NOT disjoint corpus partitions. Claims about "partitioning as the diversity engine" currently describe the legacy path and the probe, not production.
- Never report behavioral numbers from `outputs_mock/` or MOCK runs (P0.1).
- **JUDGE POSITION-LABEL BUG (found + fixed 2026-07-24, `eval/judge.py`): every historical A-vs-{C,D,E,F} tally is unreliable.** `judge_pairs()` stored the judge's generic position label ("A"/"B" = first/second argument) as the per-prompt winner without remapping to the real condition code, so every right-side win in a non-B comparison was silently tallied as a "tie" (harmless for A-vs-B only by coincidence of naming). Verified concretely: overctx_16x A-vs-E showed 6/6 "ties" pre-fix; the raw winners were literally "B" (= E won) on 4/6. Re-tally any past report citing A-vs-E/C/D/F ties before believing them. The `--pack`/hand-verdict path was unaffected.
- **2026-07-24 `eval/results/overctx_16x/` — first real over-context run, REPAIRED verdict (judge: `groq:openai/gpt-oss-120b`, off-family, both orders, agreement 6/6):** A-vs-B **0W/0T/6L (0%)**, A-vs-E **1W/0T/5L (17%, Wilson [0.03, 0.56])** — with a competent judge the swarm loses decisively to both the direct model and the synthesis-prompt control at 129× cost on this prompt set. **A-vs-F is NOT EVALUABLE (n=0)**: free-tier Groq structurally cannot serve pack-scale prompts (6k TPM cap vs 67–76k-token prompts starves the completion phase → mid-sentence truncation at ~332 tokens; full evidence in `eval/results/overctx_16x/F_BLOCKED.md`). The kill criterion is untested, not failed. Also still true: the 16x packs are real Tavily retrieval (74–97% of 384k-char target, after the `eval/packs.py` keyword-query rewrite); condition B hit 100% of the factual checklists from memory, so `must_include` items need a pack-dependent redesign; 4/6 condition-A answers shipped as extractive fallbacks (faithfulness hard gate fired), which plausibly contributed to the judged losses. To evaluate F: same Llama-3.1-8B weights at full context on Colab A100, or Groq paid tier.

## Commands

```bash
# Run the pipeline (entry point — there is no main.py / run_task.py here)
python run_swarm.py debate "Climate action is necessary"
python run_swarm.py analysis "What causes innovation?"
python run_swarm.py creative "Write a haiku about emergence"
python run_swarm.py problem_solving "How can cities reduce traffic?"
python run_swarm.py coding "Implement a binary search"

# Useful flags
python run_swarm.py debate "..." --mode=baseline           # non-stigmergic independent-agent baseline
python run_swarm.py debate "..." --corpus=placeholder      # skip retrieval, use engineered corpus
python run_swarm.py debate "..." --use-kb                  # OPT IN to cross-run knowledge base (off by default)
python run_swarm.py debate "..." --reset-kb                # quarantine existing KB entries
python run_swarm.py debate "..." --ignore-kb               # explicit no-op alias (KB already off by default)
python run_swarm.py debate "..." --show-partition-overlap  # surface Jaccard input-overlap diag
python run_swarm.py debate "..." --workers=4               # pool size (default 24; use ~4 on laptop GGUF)
python run_swarm.py debate "..." --legacy-rounds           # old round/phase scheduler (repro / GGUF laptop only)
python run_swarm.py debate "..." --synth-verbose           # old combined answer.txt (telemetry inline; default is clean reader answer + diagnostics.md)

# Develop without a GPU / model download
MOCK_LLM=1 python run_swarm.py debate "Test thesis"

# Swap the model
SWARM_MODEL="microsoft/phi-2" python run_swarm.py debate "..."
SWARM_MODEL="Qwen/Qwen2.5-3B-Instruct" python run_swarm.py debate "..."

# Tests (pytest) — run `pytest tests/ -q` for the current count; suite grows with each feature commit
pytest tests/                                              # full suite (~2 min with MOCK_LLM=1)
pytest tests/test_logit_dynamics.py -v                     # single file
pytest tests/test_no_leak_real_patterns.py::test_name -v   # single test

# Fast test run (all convergence gates disabled for subprocess tests)
MOCK_LLM=1 SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5 pytest tests/ -q

# Diagnostics and comparison
python diagnose.py                                          # signal-store + pipeline self-check
python tools/compare_runs.py outputs/RUN_A/summary.json outputs/RUN_B/summary.json  # grouped QUALITY/LATENCY/PROCESS diff + verdict
python tools/ab_run.py debate "..."                        # A/B harness: run stigmergic + --mode=baseline on one prompt, then compare
python tools/ab_run.py debate "..." --judge                # ...and run the pairwise LLM answer-text judge (needs a real LLM)
python tools/judge_answers.py "<prompt>" A/answer.txt B/answer.txt  # pairwise quality judge (position-bias-mitigated)
python kb_migrate.py                                        # knowledge-base schema migration
python synthesize.py                                        # re-render synthesis from a saved store

# Amplification-delta eval — the canonical swarm-vs-direct comparison (see Empirical status above)
MOCK_LLM=1 python -m eval.ab_harness --mini 8 --conditions AB     # plumbing check only
GROQ_API_KEY=... python -m eval.ab_harness --mini 8 --conditions ABCDEF  # real: A=swarm, B=direct M, C=best-of-N revise, D=strong M+, E=direct M + synthesis prompt (attribution control), F=direct M + same retrieved evidence (single-call RAG — the practitioner baseline)
python -m eval.ab_harness ... --backend local                     # single-GPU parity (B/C/D/E on the same local model as A)
python -m eval.judge eval/results/<exp>                           # blind pairwise judge, both orders, Wilson CIs → report.md
```

Mock-mode and real-model runs land in different directories on purpose — see Outputs below.

## Orientation: `docs/PIPELINE_MAP.md`

Start here when exploring or debugging a live run. It is a one-page table of every pipeline mechanism (1–26) with its source location, its knob/env var, the **exact log line to grep** to confirm it fired, and its current status (✅ working / 🆕 untested / ⚠️ weak link). It also carries a paste-ready grep/Select-String cheat-sheet and a per-run `summary.json` health checklist. When a feature "isn't firing," find its row, grep its log signature, then read the cited file — faster than searching blind. The map's "Known open issues" section is the live gap list (verification coverage, dead lattice fields, baseline A/B).

## Architecture

### Pipeline shape (`run_swarm.py` + `core/worker_pool.py`)

**The default is the continuous worker pool** (`run_continuous_pipeline` → `core/worker_pool.py:run_pool`): ~24 workers loop concurrently against the shared `SignalStore`. Each tick a worker picks an action — SCOUT / DEVELOP / CHAIN / CRITIQUE / OBJECT / VALIDATE / REFINE — via `choose_action()` (share-based balancing, optional cluster-local biases, pre-call scout gate). There are no rounds or phases; a `decay_loop` ticks decay/prune in parallel and `ConvergenceDetector` (`core/convergence.py`) decides when to halt. SCOUT and VALIDATE always search; DEVELOP searches when its sampled INITIAL has < 2 SUPPORT children.

The old round/phase scheduler is preserved behind `--legacy-rounds` (Phase A: scouts + validators, so VERIFICATION signals exist when the provenance boost is computed; Phase B: developers/critics/haters; then decay + prune + diversity logging). Use it only to reproduce old artifacts or on the GGUF laptop path where noted.

After halt, the `Synthesizer` reads the surviving signal DAG and produces the final answer. `core/clean_answer.py:split_answer()` then splits it: reader-facing Sections 1 (+2) → `answer.txt`; all field telemetry (Section 3, PROCESS NOTES, citation graph) → `diagnostics.md`. `--synth-verbose` / `SWARM_SYNTH_VERBOSE=1` restores the old combined output. The faithfulness audit still runs on the full pre-split text.

### Role activation (`ROLES_FOR_TASK` in `run_swarm.py`)

Task type gates which roles run. `creative` suppresses only Validator (no external facts to verify in poetry) but runs Hater with a craft-focused prompt — challenging derivative references, unearned allusions, grammatical breakage, and convergence without earned resonance. Without adversarial pressure, creative runs produce more clusters but with zero verification and collective agreement rather than genuine field pressure. `problem_solving` suppresses Validator; `debate`/`analysis` activate the full pipeline. Scout, Forager, and Synthesizer always run. The `coding` task swaps in `agents/coding_roles.py` (RequirementsScout, StaticCritic, ...) via `core/role_registry.py`.

### Signal store (`core/signal_store.py`)

Typed DAG of signals with strength dynamics. The no-leak rule is enforced here:

- Methods that walk provenance return **shapes** (lists of IDs, counts of typed ancestors), not rendered ancestor text.
- No `responses` field, no `parent_content` in metadata, no `get_dialogue_thread`.

Strength dynamics are gated by `USE_LOGIT_DYNAMICS` in `core/config.py`. The default path stores an internal `_logit` and applies additive deltas, projecting back through sigmoid. This fixed three real bugs documented at the top of the file: saturation crash, contrarian-drift via anti-decay, and order-dependence of decay × amplify. The legacy multiplicative path is preserved as a one-release escape hatch and exercised by `tests/test_logit_dynamics.py` — don't delete one without the other.

Dedup is a **string pre-screen only**: SequenceMatcher ratio > 0.95 against the 3 most-recent same-type signals from the last 5 minutes. On a hit the existing signal is amplified (+`DELTA_DEDUP_AMPLIFY` logit), its `metadata["dup_count"]` is incremented (projection weights OBJECTION dissent by `1+√dup_count` — the dissent self-dedup fix), and the deposit is rejected. The old embedding-cosine rejection path was removed: paraphrases now **join clusters** instead of being rejected — clustering, not dedup, is the absorption channel.

**Signed trail (2026-07-06, `USE_SIGNED_TRAIL`, default on).** An OBJECTION deposit dampens its target's cluster through the same `_amplify_cluster_trail` path a SUPPORT reinforces it (sign=-1). This is the store's only negative-evidence force — before it, no code path anywhere reduced strength in response to CRITIQUE/OBJECTION/failed VERIFICATION. `USE_EVAPORATION_DECAY` (default OFF — it trades away decay×amplify order-invariance, a deliberately fixed bug) is the ablation flag for ACO-style strength-proportional decay.

**Novelty accounting is INITIAL-only**: `novelty_rate` counts only INITIAL deposits that opened a new cluster. SUPPORT/OBJECTION prose churn no longer registers as exploration (it could previously hold novelty above the saturation floor forever).

**Partition invariant (hard enforcement).** Every INITIAL or SUPPORT signal *must* carry a non-empty `partition_id`. `deposit()` raises `AssertionError` if this is violated — this is intentional and loud. INITIAL signals get `partition_id` from `ScoutConfig.partition.partition_id` (set in `scout.py`). SUPPORT signals inherit it from their parent via the store's inheritance logic (lines 284–288 of `signal_store.py`), or explicitly via `deposit_meta["partition_id"]` which `base.py` now sets from the sampled parent's `partition_id` as a defensive redundancy. If you add a new agent role that deposits INITIAL or SUPPORT, you must supply `partition_id` in metadata or ensure the deposit has a parent that carries one.

**`sample_from_clusters()`** (Gap 1 stigmergy fix): cluster-aware sampling that biases toward the cluster whose centroid is closest to the worker's current semantic position. Workers that started developing claims in one region of idea-space keep developing in that region rather than jumping randomly. Gated by `USE_CLUSTER_AWARE_SAMPLING` in `config.py`.

**`_amplify_cluster_trail()`** (Gap 2): when a SUPPORT signal is deposited, all member signals of that cluster get a small amplification boost inside the lock. This is the pheromone-trail analogue — successful developments make the cluster more attractive to future workers. Gated by `USE_TRAIL_AMPLIFICATION`.

### Agents (`agents/`)

All agents inherit from `agents/base.py`. The base class enforces:

1. Prompts are built from exactly one of: the agent's own corpus partition (Scout only) or signals it sampled from the store. Plus the role instruction. Nothing else.
2. Only `Signal.content` may be rendered into prompts. IDs are opaque text for traceability.
3. Each iteration reports an `AgentContextRecord` listing the `chunk_ids` and `signal_ids` consumed — this is what `core/diversity.py` aggregates.

`agents/forager.py` is a backward-compat alias for `agents/developer.py` (the Forager → Developer rename is in-progress; new code should use Developer).

**Multi-claim scout sampling** (`SCOUT_CLAIMS_PER_CALL` in `core/config.py`, default 4). One scout call emits a numbered portfolio of K distinct claims (different angles: mechanism, counterexample, cost, stakeholder, quantitative, second-order); the deposit site (`worker_pool.py` SCOUT branch + `agents/scout.py` run loop) splits them (`split_scout_claims` in `core/actions.py`) and keeps the candidate **least similar to the recent INITIAL field** (`select_novel_claim` → `store.max_similarity_to_recent`, embedding cosine with string fallback). Rationale: a single-claim prompt takes the model's argmax claim, which is the same obvious thesis restatement for every worker — the root cause of the near-duplicate INITIAL fields seen in real Groq runs. Selection is code-side scalars only; no other agent's content enters the scout prompt (no-leak intact). The reseed hint now covers the scout's last `SCOUT_RESEED_DEPTH` (3) own deposits, not just the most recent. `MAX_TOKENS_SCOUT` was raised (200→400; small tiers 140→320) to fit the portfolio; `ACTION_REGISTRY[SCOUT].max_tokens` reads it from config.

A scratchpad-stripper (`_SCRATCHPAD_RE` in `base.py`) removes reasoning-tuned model preambles ("Alright, so I need to...", "Let me think...", "Step 1:") that leak chain-of-thought into deposits. Critical for DeepSeek-R1-Distill outputs.

**`base.py` deposit_meta partition_id carry-forward.** Before every `store.deposit()` call, `base.py` reads `partition_id` from the sampled signals and adds it to `deposit_meta` if not already present. This is a belt-and-suspenders guard: the store's own parent-inheritance logic (lines 284–288) is the primary mechanism, but if a signal is pruned or its parent reference is stale between `sample()` and `deposit()`, the carry-forward ensures the PARTITION LEAK assertion never fires spuriously on legitimate deposits.

**`developer.py` empty-strategy fallback.** `Developer.sample()` first tries `signals_with_few_children_of_type(INITIAL, SUPPORT, 2)` (underserved gap-fill). When that's empty, it uses the assigned sampling strategy. If the strategy returns `[]` (e.g., `stratified_extremes` finds no signals in the weak or strong strength strata — all INITIALs sit in the medium range), it falls back to `store.sample_weighted(INITIAL, 1)`. Without this fallback the developer deposits SUPPORT with no parent and no `partition_id`, triggering the partition leak assertion. **Do not remove this fallback.**

### Synthesizer (`agents/synthesizer.py`)

Two-layer read-out:

1. **`core/projection.py`** — pure-Python DAG projection. No LLM. Classifies clusters as `surviving` / `contested` / `weakly_supported` / `rejected_by_field`. Computes `support_diversity`, `dissent_pressure`, `verification_score`. Surviving clusters carry an `unverified` flag when no validator reached them.
2. **`agents/synthesizer.py`** — two-stage read-out on the sectioned path. Stage 1: one structured LLM call *per cluster* renders an evidence brief (isolates render failures). Stage 2 (`_compose_answer`): one bounded writer call receives ONLY the briefs + a scalar plan digest — never the store or corpus — and composes a coherent thesis-led Section 1, merging redundant claims and preserving citation tags. **Context-compression invariant:** composer input is O(K × brief), K = `RENDER_K` (8, `core/config.py`), each brief hard-capped at 200 words — independent of store size; the swarm remains the compression engine. Guards: minimum length + citation-tag retention ≥ 0.5; fallback chain is global composition → edge composition → plain join → deterministic extractive rendering (Section 1 is never empty, even on total API-token exhaustion). Hallucination safety comes from the post-hoc faithfulness audit (4-gram overlap per citation tag), not from structural isolation — **the audit runs on the pre-resolution text** (before `resolve_inline_citations` rewrites tags to `[N]` footnotes; running it after, as before 2026-07-06, meant it only ever audited the citation appendix). Section 2 consumes `plan.dissent_clusters` (planner-first ordering) and ends with bounded one-liners for quiet-minority clusters the planner demoted (`_MINORITY_ONELINER_CAP`), so distinct-but-uncontested positions reach the reader instead of dying in diagnostics.md. The answer leads with Section 1; field telemetry (cluster counts, topology coverage, genome stats) lands in a PROCESS NOTES section at the end. Sections 3 and 4 are deterministic.

**Planning — deterministic by default (commit 70d816d).** There are two cluster-selection planners, but the LLM one is now **retired to opt-in** after 3/3 real runs hit a context wall at planning:

- **`core/projection.build_plan()`** (pure-Python, deterministic): composite score + MMR over embeddings. No LLM. **This is the default path** — there is no LLM call at planning. Also called by `ConvergenceDetector` and any GPU-less path.
- **`Synthesizer._plan_synthesis()`** (LLM-based, in `agents/synthesizer.py`): sees a structural digest (IDs, counts, short previews) and returns JSON with `render_full` / `section3_only` / `merge_groups`. Falls back to `build_plan()` on failure. **Gated off by `USE_LLM_PLANNER` (`config.py`); enable only for ablation via `SWARM_USE_LLM_PLANNER=1`.**

If ablation shows the LLM planner gives no lift, delete `_plan_synthesis()` entirely (DEFERRED / PIPELINE_MAP open issue #4).

A faithfulness audit runs after rendering: each paragraph must have ≥4-gram overlap with the cluster content for every `[INITIAL_XXXXX]` citation tag it contains. Failures land in `renderer_audit.json`. External grounding (Wikipedia lookup per surviving cluster) is best-effort and tagged `[External context]` so it's distinguishable from agent content.

### Topology (`core/topology.py`)

Bounds-first exploration scaffolding. One LLM call before any scout runs produces an `AnswerSpaceTopology`:
- `AxisSpec` (name, values: 2–5 discrete categories) + `AnchorCorner` (coords, label, rationale)
- `generate_topology(task_prompt, task_type, llm)` — async, temp=0.1, 2 retries, returns `None` on parse failure
- `assign_topology_cells(topology, n_scouts)` — anchor cells first, then interior round-robin
- `format_cell_for_prompt(topology, cell)` — human-readable cell description for scout prompt

Scouts carry `topology_cell` and `topology_cell_desc` in `ScoutConfig`. INITIAL deposits carry `metadata["topology_coords"]`. `build_projection()` reads these to populate `topology_coverage` / `uncovered_cells` / `out_of_bounds_clusters` in `SynthesisProjection`.

`run_swarm.py` wiring: topology is generated after partition assembly, before rounds. Cell assignments are computed once and reused each round (`topology_cell_assignments`). Topology is also passed to `build_projection()` in the KB save section.

Task-type templates in `_TOPOLOGY_TEMPLATES` (debate/analysis/problem_solving/creative/coding). Falls back to `_TOPOLOGY_DEFAULT_TEMPLATE` for unknown types.

### Atom resolution + sensitivity (`core/projection.py`)

Projection dataclasses after `InterClusterEdge`:
- `AtomProjection` — individual SAFE atoms from VERIFICATION signal metadata
- `ClusterSensitivity` — robustness annotation per surviving cluster

`SynthesisProjection` carries: `topology`, `topology_coverage`, `uncovered_cells`, `out_of_bounds_clusters`, `atoms`, `cluster_sensitivities`.

> **History (2026-06-18):** the speculative multi-resolution lattice — `PropositionProjection`, `FrameProjection`, `CrossLevelEdge` and their builders `_build_frames`/`_build_cross_level_edges`, plus the `frames`/`propositions`/`cross_level_edges` fields — was **removed**. They were built every run but had no consumers (`frames` only fed the never-read `cross_level_edges`; `propositions` was never even populated). `atoms` and `cluster_sensitivities` were kept because `_build_genomes` consumes them to populate `genome.atoms` and `genome.sensitivity` — do not confuse these with the deleted lattice. Don't re-introduce the lattice without a consumer.

Live builders (all `O(n)` or `O(n log n)`, called at end of `build_projection()`):
- `_build_atoms(clusters, store)` — reads `vsig.metadata["atoms"]` from VERIFICATION signals → feeds `genome.atoms`
- `_build_sensitivities(clusters, store)` — simulates support removal; gated by `_SENSITIVITY_MAX_CLUSTERS=20`; feeds `genome.sensitivity` + a renderer annotation
- `_build_topology_coverage(clusters, store, topology)` — reads `metadata["topology_coords"]` from signals; drives Sections 5/6

### Synthesizer (`agents/synthesizer.py`) — topology/lattice extensions

New constants:
- `_LATTICE_RESOLUTION_BY_TASK` — maps task_type to atom/cluster/frame resolution level
- `_SYNTHESIZER_USE_DEBATE = True` — debate frame for alternatives cluster sets (now enabled)
- `SYNTHESIZER_EMIT_ALTERNATIVE = True` — alternative-of-the-best artifact (now enabled)

Renderer extensions (no new LLM calls except the pre-existing debate frame):
- **Topology preamble** in `_render_executive_summary` — coverage ratio, uncovered cell names
- **Sensitivity annotation** in `_render_cluster_position` — robustness, load_bearing signals, competing cluster, topology gap on removal
- **Section 5** (UNEXPLORED ANSWER-SPACE REGIONS) — deterministic, lists uncovered topology cells
- **Section 6** (OUT-OF-BOUNDS CLUSTERS) — clusters whose topology coords fall outside declared axes

### Configuration (`core/config.py`)

All tunables live in one validated module: agent counts, decay/amplify/prune thresholds, boost params, dedup thresholds, model name. Important env vars:

- `SWARM_MODEL` — model override (default `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`)
- `MOCK_LLM=1` — skip model load entirely

`LLM_CONCURRENCY = 1` is intentional for a 6 GB laptop GPU running 4-bit NF4. Synthesizer cluster calls will parallelize when this rises.

**Stigmergy gap feature flags** (now all `True` by default in `config.py` — this changed; older docs say False):
- `USE_CLUSTER_AWARE_SAMPLING` — Gap 1: `sample_from_clusters()` biases workers toward their semantic home cluster
- `USE_TRAIL_AMPLIFICATION` — Gap 2: SUPPORT deposits amplify the whole cluster (pheromone trail)
- `USE_LOCAL_ACTION_BIAS` — Gap 3: cluster-local state multipliers in `choose_action()`. The old "known-dead on the hot path" note is stale: biases are now computed *before* the primary `choose_action` call (worker_pool.py, ~line 1000), so the flag is live and ablatable.
- `USE_WORKER_POSITION` — Gap 4: workers track a centroid of their prior deposits and pass it to sampling

### LLM backends (`core/llm*.py`)

There is no single LLM class — `core/llm.py:make_llm()` is a factory that selects a *local engine* by `SWARM_BACKEND`, and `run_swarm.py` separately selects a *router* that may send different roles to different engines. Keep these two axes distinct.

**Local engine selection (`make_llm`, `SWARM_BACKEND`):**
- `vllm` (`core/llm_vllm.py`) — one non-quantized model in VRAM, AsyncLLMEngine batches internally (`_uses_internal_batching=True`, so callers skip the semaphore). Auto-selected on Colab (`COLAB=1`) or when a Colab-tier GPU is detected. Best for T4/L4/A100. Applies the model's HF chat template (kills the "Let me know if you need anything adjusted!" artifacts the GGUF path emits).
- `gguf` (`core/llm_gguf.py`) — llama-cpp-python with 4-bit GGUF weights, bypasses bitsandbytes entirely. The **default on the 6 GB laptop** (`SWARM_GGUF_REPO`/`SWARM_GGUF_FILE`/`SWARM_GPU_LAYERS`).
- `hf` — HuggingFace transformers + bitsandbytes 4-bit (`RealLLM`); needs ~14 GB transient RAM for fp16→4bit. The fallback, and what "anything else" maps to.
- `MOCK_LLM=1` short-circuits everything to `MockLLM`.
- Note: `SWARM_BACKEND` is **overloaded** — `run_swarm.py` accepts `hybrid`/`local` here to pick the *router*; `make_llm` treats those as "auto" so the local engine still gets vLLM on a Colab GPU. Load order is logged in a banner at startup.

**vLLM load cascade (`_try_vllm_cascade` / `_build_cascade` in `core/llm.py`).** On the vllm path, a forced-to-work fallback ladder walks progressively smaller/cheaper configs until one loads (T4 16 GB is right at the edge for Qwen-7B fp16). Each rung is **VRAM pre-checked**: rungs whose estimated weights exceed free VRAM are SKIPPED before the load is attempted (commit 66d2794), so the log shows `SKIP`/`attempt`/`SUCCESS`/`FAILED` per rung. Falls through to the hf cascade, then mock.

**Routers (selected in `run_swarm.py`, not `make_llm`):**
- `core/llm_groq.py` `GroqRouter` — activated by `GROQ_API_KEY`. **CORRECTION (audited 2026-07-24): the actual `_DEFAULT_GROQ_ROLE_MODELS` table (~lines 373–396) is ALL-Llama** — the "Llama scouts / Mixtral foragers / Gemma2 haters" family diversity this doc previously claimed does not exist in the shipped defaults (likely rotted when Groq deprecated Mixtral/Gemma hosting). Any past claim that Groq runs had model-family heterogeneity as a diversity lever is unsupported. The only confirmed off-family option currently hosted is Qwen (`qwen-qwq-32b`) — see `docs/future/STAGE3_4_HETEROGENEITY_TRIAGE_SCALE_SPEC.md`. Watch the free-tier RPM/TPD limits (measured 2026-07-24: 6k TPM, 500k TPD on `llama-3.1-8b-instant`).
- `core/llm_hybrid.py` `HybridRouter` — `SWARM_BACKEND=hybrid`. High-volume roles (scout/developer/critic/validator) + the **synthesizer** run on ONE local GPU model; only the low-volume hater goes to Groq for a different family. Synthesizer is local by default because its end-of-run token burst exhausts Groq's free TPD mid-render — a 14B that *completes* beats a 70B that fails halfway. Override with `SWARM_HYBRID_GROQ_ROLES`.
- `core/llm_router.py` — `HeterogeneousRouter` (laptop GGUF, load-one-model-per-phase, ~30–60 s/swap) and `LoRAHeterogeneousRouter` (Colab single base + per-role LoRA adapters in `loras/`, zero-cost swap; pass-through if `loras/` is empty — LoRA training is future work).

All routers share the contract `engine_for(role)` / `role_disabled` / `manifest` / `teardown` / `bundle_name`, so `worker_pool.py` and `run_swarm.py` route every role identically regardless of which router is active.

### Stock Swarm POC (`run_stock.py`, `core/stock_*.py`, `agents/stock_roles.py`)

A separate experimental pipeline — single-ticker price-direction prediction — that does **not** go through `run_swarm.py`. Entry point `run_stock.py` drives the round-based `core/stock_pipeline.py`. **Look-ahead rule (hard):** a historical `--as-of` REQUIRES `--db` (a `FrozenSnapshotProvider` point-in-time snapshot), because live yfinance fundamentals/news are "as of now" and would leak the future. Output (`answer.txt` + `prediction.json` + `signals.json`) lands in `outputs/stock_<SYMBOL>_<as_of>/`. Supporting modules: `stock_data.py` (yfinance behind a provider), `stock_verify.py`, `stock_compare.py`, `symbol_discovery.py`. See the POC plan in `docs/prompts/`. Status is early — treat as out-of-tree from the main swarm.

### Convergence (`core/convergence.py`)

All convergence thresholds are overridable via environment variables. This is critical for subprocess-based tests:

| Constant | Env var | Default |
|---|---|---|
| `MIN_TIME_S` | `SWARM_MIN_TIME_S` | `60.0` |
| `MIN_ITERATIONS` | `SWARM_MIN_ITERATIONS` | `50` |
| `MIN_INITIALS_FOR_HALT` | `SWARM_MIN_INITIALS_FOR_HALT` | `6` |
| `MIN_INTER_CLUSTER_EDGES` | `SWARM_MIN_INTER_CLUSTER_EDGES` | `0` (disabled) |
| `QUALITY_GROUNDING_VER_MIN` | `SWARM_QUALITY_GROUNDING_VER_MIN` | `0.55` |
| `SAT_NO_NEW_SURVIVING` | `SWARM_SAT_NO_NEW_SURVIVING` | `60` |
| `MAX_ITERATIONS` | `SWARM_MAX_ITERATIONS` | `2000` |
| `MAX_TIME_S` | `SWARM_MAX_TIME_S` | `900.0` |
| `NOVELTY_SAT_WINDOW` | `SWARM_NOVELTY_SAT_WINDOW` | `40` |
| `NOVELTY_SAT_FLOOR` | `SWARM_NOVELTY_SAT_FLOOR` | `0.05` |
| `RENDER_STABLE_ITERS` | `SWARM_RENDER_STABLE_ITERS` | `40` |

A novelty-saturation gate (`NOVELTY_SAT_*`) can short-circuit the quality-hold / no-new-surviving waits: when the last `NOVELTY_SAT_WINDOW` clusterable deposits opened essentially no new idea-space (`store.novelty_rate(...) < NOVELTY_SAT_FLOOR`), the detector treats the field as exhausted.

**Render-set stability halt (`RENDER_STABLE_ITERS`, 0 disables).** The primary defense against `cap_time` endings: near-duplicate "dust" clusters keep resetting the surviving/novelty counters (fragmentation masquerades as exploration), so the detector also watches the only thing the readout consumes — the top-`RENDER_K` set chosen by `build_plan()` plus each selected cluster's evidence counts. Unchanged for `RENDER_STABLE_ITERS` iterations ⇒ halt with reason `render_set_stable`. A companion **pre-call scout gate** (`SCOUT_GATE_*` in `core/config.py`, `scout_gate_engaged()` in `worker_pool.py`) demotes SCOUT draws to exploit actions once field novelty falls below `SCOUT_GATE_NOVELTY_FLOOR` (0.10) — the novelty check moved *ahead* of the LLM call instead of discarding its output afterwards. Skip count lands in `summary.json: scout_gate_skips`.

Tests that spawn subprocesses (test_phase_isolation, test_heterogeneous_routing, test_kb_default_off) must set `SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5 SWARM_MAX_ITERATIONS=20` etc. in the subprocess env dict, or they will time out waiting for the 60-second minimum wall. The `convergence.py` module reads these at import time.

### Retrieval (`core/retrieval.py`, `core/search_tool.py`, `core/query_planner.py`)

`CompositeRetriever` (`core/retrieval.py`): Wikipedia → Web → placeholder fallback. The `--corpus=placeholder` flag forces the engineered corpus; diversity numbers from placeholder mode are *not* empirical evidence — see DEFERRED.md P4.1.

Per-agent live search runs through `core/search_tool.py:search()`, a backend chain `Tavily → Stack Exchange (coding only) → DDG → follow-up → Wikipedia → Cohere` (`TAVILY_API_KEY` unset ⇒ DDG is primary; Wikipedia fallback means DDG is no longer a single point of failure). Result quality pipeline: source cap → relevance gate → BM25+dense RRF → fact-density + coding-domain priors → MMR → page enrichment (`SWARM_SEARCH_*` knobs). `core/query_planner.py` builds task-aware stance queries, mining fragments from high-strength INITIALs (stigmergic) plus step-back + HyDE, with a per-pool search budget. SEARCH-signal deposits carry the query, top URLs, and a content excerpt for traceability.

**Retrieval is the wall-clock bottleneck (blocking HTTP), not inference — and is the focus of the latency program (2026-06-18).** Key invariants to preserve:
- **Searches must stay off the asyncio event loop.** Every `_search(...)` in `worker_pool.py` is called via `await asyncio.to_thread(...)`. The whole worker pool runs on one event loop; a synchronous `requests.get`-based search blocks *all* workers and GPU scheduling. Do not call `_search` (or any blocking I/O) directly from a coroutine — wrap it.
- **Page enrichment runs on the FINAL ranked survivors**, via `_diversify(enrich_pages=True)` (DDG paths only — Tavily/Wikipedia return full text and pass `enrich_pages=False`). `search()` is a thin timing wrapper around `_search_impl`; `_ddg_search` no longer enriches inline. Don't move enrichment back ahead of ranking — it re-introduces fetching pages that MMR/relevance then discard.
- **Page fetches are memoized per run** (`_PAGE_CACHE`, url→text, thread-safe) and reset in `reset_search_stats()` at run start.
- **Retrieval timing** lands in `summary.json: timing` (`reset_search_stats`/`search_stats_snapshot` in `search_tool.py`); `search_fraction_of_wallclock` is the headline metric. See `docs/PIPELINE_MAP.md` row 27.
- **Synthesizer per-cluster renders size to the engine** (`Synthesizer._render_concurrency()`): all ≤6 briefs run concurrently on internal-batching engines (vLLM/Groq), serial on single-stream local engines. `_RENDER_SEM_SIZE` (= `LLM_CONCURRENCY`) is the local-only cap; don't let it throttle batching engines.

**Number-grounding gate (STORM-style, commit 70d816d).** `ungrounded_numbers()` in `core/actions.py` (gated in `worker_pool.py`) checks that any figure in a SCOUT/DEVELOP/CHAIN/REFINE deposit appears in the evidence the agent was shown (chunks / parent / task prompt). If evidence was present but the figure isn't in it, the deposit is **rejected as fabrication** (`REJECT … ungrounded figure(s) … (fabrication gate)`); if no evidence was shown at all, the figure is kept but tagged `numbers_grounded=false` and surfaced in briefs as `(UNSOURCED FIGURES — present as claimed)`. This exists because the particulars-demand prompt (#4) drove the model to fabricate specific city/figure claims.

### Cluster Genome (`core/projection.py`, `core/fitness.py`)

The cluster is the unit of selection. Each `ClusterProjection` carries a `ClusterGenome` assembled at the end of `build_projection()` by `_build_genomes()`. The genome is a typed heritable object that survives fission (via `centroid_at_formation` and atom inheritance) and recombines on merge.

Key genome fields:
- `atoms: list[AtomFact]` — SAFE-pipeline atomic propositions (verified against external sources). Deduplicated at word-Jaccard ≥ 0.70. Populated from `VERIFICATION.metadata["atoms"]` by `_build_atoms()`.
- `knowledge_base: ClusterKnowledgeBase` — aggregated SEARCH-signal lineage (3 channels: scout SEARCH by agent_id, developer SEARCH by parent_id, validator atom source_tags). `parametric_content_ratio` measures what fraction of members have no search lineage.
- `phenotype: Phenotype` — centroid, `centroid_at_formation` (snapshot set in `ClusterRegistry.create()` and `_reanchor()`), drift, stability, novelty_density.
- `sensitivity: GenomeSensitivity` — atom-level (not support-level) load-bearing atom IDs.
- `trajectory: FitnessTrajectory` — composite_fitness history accumulated across genome cache refreshes in `run_swarm.py`; enables `_trajectory_score()`.
- `composite_fitness: float` — 7-term compositor in `core/fitness.py`. Hard cap: `semantic_strength ≤ 0.35` (prevents LLM-judged term dominating). Tier 2: centroid_stability, novelty_density (model embeddings). Tier 3 only: entity_resolution (Wikidata, default off).

The genome drives selection: `_cp_priority()` and `build_plan()` use `composite_fitness` when available; the LLM planner digest includes `composite_fitness` and `grounding`; `_score_cohesive_candidate()` weights clusters by `composite_fitness`.

**Do not break:** the no-leak rule applies to genomes. Prompts may render `AtomFact.text` (proposition text), `atom_id`, and scalar fields. The `extracted_from` list carries signal IDs only — never ancestry text.

### Knowledge base (`core/knowledge_base.py`)

Cross-run consensus/rejection memory. **Default is OFF** — pass `--use-kb` to opt in (`run_swarm.py` flips `ignore_kb = not use_kb` in `main()`). `--ignore-kb` is now a redundant no-op alias kept for backward compat; `--reset-kb` quarantines existing entries before the run. **Schema v3** (bumped from v2) stores genome fields per entry: `genome_hash`, `genome_atoms`, `composite_fitness`, `fitness_breakdown`, `knowledge_base`. `kb_migrate.py` handles v2→v3 by adding null genome fields. Contradiction detection has two channels: (1) embedding cosine ≥ 0.75, (2) atom text word-Jaccard ≥ 0.50. When enabled, prior consensus claims are also registered as scout novelty references (`set_novelty_references`), steering new scouts away from already-known claims.

### Baseline mode (`core/baseline.py`)

`--mode=baseline` runs N independent agents with no signal store, no partitioning, no provenance boost. This is the A/B comparison condition for the stigmergic hypothesis. Output is shape-compatible with stigmergic runs so `tools/compare_runs.py` can diff them.

### MCP server (`mcp_server.py`, untracked)

A standalone FastMCP server exposing the pipeline as three tools — `run_swarm` (subprocess wrapper around `run_swarm.py`, returns the clean `answer.txt`), `get_swarm_diagnostics`, `list_swarm_runs` — over stdio (Claude Code/Desktop) or streamable-HTTP on port 8756. Intent per `docs/ODYSSEUS_MCP.md`: let an external agent workspace call the swarm as an on-demand heavyweight tool. Functional and self-contained; not wired into the core pipeline and not part of the eval story. This is the consumer-facing surface if the research thesis fails.

### Docs atlas (`docs/atlas/`, in progress)

A planned numbered code atlas; only `04_agents.md` exists (detailed map of `agents/` with a refinement-opportunities list: duplicated run loops, dead `_maybe_cloud_call` flag, Validator/Hater test gaps, fragile TYPE/PARENT parsing). Treat it as documentation-in-progress, not authority — `docs/PIPELINE_MAP.md` remains the orientation entry point.

## Outputs

- `outputs/` — real-LLM runs. Each run is a timestamped subdirectory containing `answer.txt` (clean reader answer: Sections 1–2 only), `diagnostics.md` (field telemetry moved out of the answer), `citations.json`, `kb_diff.json`, `lineage.dot`, `renderer_audit.json`, `round_log.json`, `run_meta.json`, `signals.json`, `summary.json`.
- `eval/results/` — amplification-delta experiments (`conditions.jsonl`, `scores.json`, `report.md`). The `plumbing_check/` subdir is MOCK-mode and carries no behavioral signal.
- `outputs_mock/` — `MOCK_LLM=1` runs. **Kept deliberately separate** (per P0.1) so mock artifacts cannot be confused with empirical evidence. MockLLM emits SHA1-seeded phrases regardless of input, so anything in `outputs_mock/` proves plumbing, not behavior.

## support_diversity — read this before writing tests or changing projection

`support_diversity` is computed in `_aggregate_cluster()` in `core/projection.py` as the size of a **union of independence discriminators** (one set, not a sum — the old `len(partitions) + len(strategy_names)` summed two different units of independence and inflated diversity; see the "Union, not sum" comment at the computation site):

- SUPPORT signals that carry `partition_id` contribute their `("partition", partition_id, depositor)` tuple
- partition-less support signals (CRITIQUE_POSITIVE, pre-partition-fix deposits) contribute `("agent", depositor_agent_id)` — four distinct critics endorsing is four independent reads; one agent depositing four times is one

**Critical for tests:** if multiple SUPPORT signals share the same `(partition_id, depositor)` pair, `support_diversity = 1` regardless of how many signals there are. Test helpers that use a fixed `partition_id = "test_partition_0"` for all deposits AND a fixed `depositor = "forager"` will always produce `support_diversity = 1`.

**Correct test pattern** (see `test_projection.py`, `test_kb_migration.py`, `test_knowledge_base.py`):
```python
for i, (content, strategy) in enumerate(forager_supports):
    _deposit(store, SUPPORT, content, 0.7, "forager", parent_id=init_id,
             metadata={"depositor_agent_id": f"forager_R1_{i}_{strategy}"})
```
The `_deposit` helper derives `partition_id = f"partition_{i}"` from the agent index in `depositor_agent_id`. With i=0,1,2,3 you get four distinct `(partition_id, depositor)` pairs → `support_diversity = 4`.

**Survival thresholds** (from `projection.py`):
- `weakly_supported`: `support_diversity < SURVIVAL_MIN_SUPPORT_DIVERSITY` (= 3)
- `surviving`: passes weakly_supported threshold + credibility gate
- `contested`: `SURVIVAL_CONTEST_MIN ≤ dissent_pressure < SURVIVAL_CONTEST_MAX` (0.5–1.5, log1p-transformed ratio)
- `rejected_by_field`: `dissent_pressure ≥ SURVIVAL_REJECT_DISSENT_PRESSURE` (= 1.5)

`dissent_pressure` uses `math.log1p(weighted_dissent / weighted_support)`. The log1p compression means the thresholds 0.5/1.5 correspond to dissent/support ratios of ~0.65 and ~3.5 respectively — not raw counts.

## Diversity metrics — read this before changing them

The Jaccard distance in `core/diversity.py` (`_partition_overlap_jaccard`, `_role_partition_overlap`, `_overall_partition_overlap`) measures **input-partition health** — did agents read disjoint chunks and signal IDs? It does *not* measure output diversity. Two agents reading disjoint inputs can produce identical outputs.

True output-diversity metrics live in `core/output_diversity.py`: `centroid_cosine_distance` (embedding-space) and `self_bleu` (lexical). Both are logged per round under `round_log.json:output_diversity`.

The old public names `role_diversity` / `overall_diversity` / `format_report` are kept as backward-compat aliases but they are partition-overlap functions. Partition-overlap is suppressed from the round log by default; surface with `--show-partition-overlap`.

## Constraints to respect

- The 6 GB consumer GPU (RTX 3060 Laptop, 4-bit NF4) is the **development environment**, not the research ceiling. The stated research priority is answer quality over cost and wall-clock — a provable quality win at a large cost multiple is acceptable; a cheap loss is not. Bigger-compute targets are A100/L40S (Longleaf) and Colab H100 (see `Stigmergic_Swarm__Compute_Request_Focused.md`). Multi-GPU model-serving infrastructure is still out of scope.
- Mock mode is for plumbing checks only — never report behavioral or diversity numbers from `outputs_mock/`.
- The colony biomimicry primitives, `dialogue_coordinator`, `Signal.responses`, `evaluate_insights_enhanced`, `deposit_with_context`, and the mode/phase/task-type/signal-type quadruple-classification system from the legacy pipeline (`legacy/`) were removed deliberately. Don't re-introduce them without a plan.
- See `DEFERRED.md` for the active list of known gaps (logit dynamics tuning, retriever calibration, A/B baseline runs, KB threshold calibration, etc.) before starting a new architectural change.
