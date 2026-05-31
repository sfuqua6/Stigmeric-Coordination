# Deferred Fixes — Things This Session Didn't Touch

<!-- Last verified: 2026-05-31. Update this date and move resolved items to
     the "Resolved" section when they land. -->


Items from the doctoral-critique plan that were _not_ integrated in the
patches dated this session. Roughly in order of impact-per-hour to fix.

Refs back to the plan use the M / m / R / Q numbering from the original
review and the P-phase IDs from the plan.

## Resolved Tasks (Colab Migration Sync)

- [x] **M10 Concurrency Gate**: Bypassed serialization semaphores on internal batching paths. Concurrent multi-agent inference runs cleanly over AsyncLLMEngine.
- [x] **Multi-GPU / High-VRAM Resource Scaling**: Scaled populations to match hardware capacities (T4 up to A100_80 tiers).
- [x] **P2.1 / R4 / M6 — logit-space strength dynamics** (resolved 2026-05-26). Implemented in `core/signal_store.py` behind `USE_LOGIT_DYNAMICS = True` (default on). Stores internal `_logit` field; all updates (decay, amplify, dedup, boost) apply additive deltas in logit space and project back through sigmoid. The legacy multiplicative path is preserved as `USE_LOGIT_DYNAMICS = False` and exercised by `tests/test_logit_dynamics.py`. See the FUTURE-CLAUDE NOTE at the top of `signal_store.py` for the three bugs this fixed.

## Resolved Tasks (2026-05-28 — Stigmergy Gaps + Test Suite)

- [x] **Stigmergy Gap 1 — Cluster-aware sampling** (`USE_CLUSTER_AWARE_SAMPLING`). Added `sample_from_clusters()` to `SignalStore`. Workers that have developed claims in a region of idea-space prefer to sample from clusters near their semantic centroid. Gated off by default; wire it into the worker pool when ablation data is available.

- [x] **Stigmergy Gap 2 — Trail amplification** (`USE_TRAIL_AMPLIFICATION`). Added `_amplify_cluster_trail()` called inside the lock when a SUPPORT signal is deposited. All cluster members get a small additive logit boost. This is the pheromone-reinforcement analogue: clusters that receive support become more attractive to future workers.

- [x] **Stigmergy Gap 3 — Local action biases** (`USE_LOCAL_ACTION_BIASES`). `choose_action()` in `worker_pool.py` now accepts `local_biases: dict[action, multiplier]` derived from cluster-local state (dissent pressure, support count). Workers in contested clusters bias toward OBJECT; workers in underserved clusters bias toward DEVELOP.

- [x] **Stigmergy Gap 4 — Worker semantic position** (`USE_WORKER_SEMANTIC_POSITION`). Each `Worker` tracks a centroid of its own deposit embeddings (`_position_centroid`). Passed to `_sample_initial()` and `_sample_underserved_initial()` so cluster-aware sampling knows where this worker "lives" in semantic space.

- [x] **Convergence env-var overrides** (`core/convergence.py`). All six convergence constants (`MIN_TIME_S`, `MIN_ITERATIONS`, `MIN_INITIALS_FOR_HALT`, `MIN_INTER_CLUSTER_EDGES`, `SAT_NO_NEW_SURVIVING`, `MAX_ITERATIONS`, `MAX_TIME_S`) are now overridable via env vars at import time. Subprocess-based tests that previously timed out (>120s) now complete in ~10s by passing `SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5 SWARM_MAX_ITERATIONS=20`.

- [x] **PARTITION LEAK assertion + developer fallback**. The `deposit()` method in `signal_store.py` raises `AssertionError` for any INITIAL or SUPPORT without `partition_id`. Root cause of the failure: `stratified_extremes` strategy returned `[]` when all INITIALs sat in the medium strength stratum, causing developer to deposit SUPPORT with no parent and no partition. Fixed by adding a `sample_weighted(INITIAL, 1)` fallback in `developer.py::sample()` and a defensive `partition_id` carry-forward in `base.py::run()`.

- [x] **P3 / R12 / R15 — Test suite** (resolved 2026-05-28). Full pytest suite now passes: **287 passed, 10 skipped, 0 failed** (~2 min with `MOCK_LLM=1`). Covered: dedup correctness, logit dynamics, no-leak assertion (positive + negative), provenance boost, convergence gating, phase-isolated orchestration (19-subprocess end-to-end), heterogeneous routing, KB save/load/dedup, planner selection, projection metrics, renderer audit coverage, output diversity JSON, coding roles, critique split, contradiction tracking.

- [x] **KB cumulative inflation cap** (`MAX_KB_DIVERSITY_BOOST` in `config.py`). `support_diversity` on KB entries is now capped at `MAX_KB_DIVERSITY_BOOST` (default: `2 * NUM_FORAGERS`) when merging repeated runs of the same cluster. Without the cap, repeated saves monotonically inflated diversity and biased the survival filter.

- [x] **KB `prune_before(date_str)`**. Added method to `KnowledgeBase` for age-based eviction of stale entries. Stopgap for the KB sunset problem — call `kb.prune_before("2026-01-01")` before save to drop pre-date entries.

## Active Research Directions

- [ ] **P4.2 A/B Empirical Baseline**: Evaluate the performance profiles of the multi-agent clusters versus unified monolithic generations across test sets.

## Architectural (next big work block)

- [x] **P2.2 / R2 — output-side diversity metric.** Resolved 2026-05-31.
  `output_diversity.py` existed with `centroid_cosine_distance` + `self_bleu`.
  Wired into `summary.json` for the continuous pipeline. Also added genome
  quality stats (avg/max composite_fitness, avg_grounding, total_atoms).

- [x] **P2.4 / M11 — real semantic clustering for haters.** Resolved 2026-05-31.
  `cluster_signals_dbscan` existed and was already used in the OBJECT action.
  OBJECT further upgraded to prefer genome-vulnerable clusters (high
  composite_fitness / low grounding ratio). VALIDATE upgraded to prefer
  high-fitness clusters with low mean atom verification_score.

## Empirical (hardware-bound, overnight)

- **P4.1 / R9 / M1 — real retriever instead of trivial corpus.** Until
  `trivial_corpus_from_thesis` is replaced with retrieved evidence
  (Wikipedia / web), the diversity numbers are tautological. The
  templates are deliberately separable per scout. Wire in a retriever
  before reporting any diversity claim.

- **P4.2 / R1 / M9 — A/B baseline protocol.** No empirical comparison
  has been run. Pre-register the conditions (single 7B baseline, no
  partitioning, no provenance boost, no contrarian boost, full pipeline)
  and queue overnight runs. ~45 hours of compute on the current CPU-only
  setup; ~5 hours with a working CUDA build of llama-cpp-python.

## Renderer and measurement gaps (session 2)

- **Renderer end-to-end testing requires real LLM.** `MockLLM` ignores
  the structured prompt and emits SHA1-seeded phrases regardless of input,
  so citation-tag preservation and prose quality cannot be validated with
  `MOCK_LLM=1`. The citation block is now post-stamped deterministically
  (not model-generated), so at minimum citations are guaranteed present.
  The prose quality test still requires a real inference run.

- **Scout novelty_per_iter is always empty with MockLLM.** Mock-mode
  produces near-identical outputs; every scout deposit after the first is
  dedup-rejected, so `own_embeddings` never accumulates and
  `novelty_per_iter` stays `[]`. Verify partition rotation effectiveness
  only on real-LLM runs. Watch `round_log.json` per-agent
  `novelty_per_iter` values — they should not monotonically decrease across
  iterations for scouts and should decrease for foragers/critics as the
  adaptive cap fires.

## Testing

- **Test suite complete.** 287 passed, 10 skipped, 0 failed as of 2026-05-28. The only remaining gap is real-LLM behavioral testing (renderer prose quality, citation faithfulness with actual model output) — see "Renderer and measurement gaps" above.

## Knowledge-base housekeeping (session 2 leftovers)

- **KB threshold calibration.** `_CLUSTER_SIM_THRESHOLD = 0.65`,
  `_KB_MATCH_THRESHOLD = 0.75`, `_KB_REJECTION_PENALTY = 0.5`, and all
  survival cutoffs (`support_diversity >= 2`, `dissent_pressure < 0.5`,
  contested band `0.5–1.5`, reject `> 1.5`) are hand-picked integers with
  no calibration data. Move them into `config.py` as named constants with
  a comment flagging them as needing empirical calibration. Calibrate once
  the real retriever is wired in (P4.1).

- **KB prior_consensus cumulative inflation.** Each save call adds
  `prior_cluster["support_diversity"]` to a matching new cluster without
  decay or a cap. Over many runs this monotonically inflates
  `support_diversity`, which biases the survival filter toward clusters that
  match prior survivors. Fix: either apply a discount per run
  (`accumulated_support_diversity *= discount_factor`) or cap at
  `2 * NUM_FORAGERS`. This is a correctness issue, not just a hygiene one,
  once KB files accumulate more than ~10 runs.

- **KB sunset / age-out.** There is no mechanism to expire KB entries.
  A cluster that was rejected 50 runs ago because of contextually outdated
  evidence will still penalise new runs. Options: time-to-live field;
  max entry count with LRU eviction; explicit `--prune-kb` CLI command.
  Add `"run_timestamp"` to entries (done) and a `--prune-kb-before DATE`
  flag as a stopgap.

## Smaller items not yet patched

- **m6 / P1.7 — RealLLM.\_input_device.** Currently uses
  `next(model.parameters()).device`; should be
  `model.get_input_embeddings().weight.device`. One-line fix in
  `core/llm.py`. Mostly cosmetic until you actually run the HF/bnb
  backend on a dispatched model.

- **m8 — Hater.parent_id_for_deposit links to strongest cluster
  representative.** Semantically the objection applies to the cluster,
  not the one signal. Options: link to none (orphan OBJECTION), or
  introduce a "cluster signal" type. Defer until M11 lands.

- **m10 — per-round agent_id collisions in diversity calculation.**
  Currently `scout_R1_0` and `scout_R2_0` are treated as different
  agents (correct), but cross-round comparison isn't well-defined.
  Cosmetic until cross-round analysis is added.

- **m11 — Signal dataclass thread safety.** Not exploitable in the
  current single-threaded asyncio path. Worth a docstring note;
  becomes real if/when concurrency goes above 1.

- **m13 — diversity report docstring inverted.** The doc says overall
  should be lower than per-role; at current population sizes it isn't.
  One-line docstring fix.

## Hardware-bound (out of scope until GPU)

- **M10 — real concurrency.** `LLM_CONCURRENCY=1` because
  `llama-cpp-python` is single-threaded per instance. Real parallelism
  needs either vLLM/TGI or multiple Llama instances on separate GPU
  partitions. Won't move on this laptop.

## Writing / framing

- **Q9 — task_type now changes behavior** (partial: roles are
  task-conditional after this session's patches), but the dynamics
  inside each active role are still identical across tasks. Worth
  revisiting per-role temperatures and per-task synthesizer prompts
  before the paper draft.

- **Cloud validator implementation.** The flag exists, the SDK is not
  wired. Implement against Anthropic Haiku 4.5 (paid, ~$0.005/run) or
  Gemini Flash 2.0 (free tier, 15 req/min). Decide first whether the
  fully-local property is worth preserving for the headline experiment;
  cloud validator is recommended only as a follow-up ablation.
