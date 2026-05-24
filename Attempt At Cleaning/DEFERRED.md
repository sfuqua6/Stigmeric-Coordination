# Deferred Fixes — Things This Session Didn't Touch

Items from the doctoral-critique plan that were _not_ integrated in the
patches dated this session. Roughly in order of impact-per-hour to fix.

Refs back to the plan use the M / m / R / Q numbering from the original
review and the P-phase IDs from the plan.

## Resolved Tasks (Colab Migration Sync)

- [x] **M10 Concurrency Gate**: Bypassed serialization semaphores on internal batching paths. Concurrent multi-agent inference runs cleanly over AsyncLLMEngine.
- [x] **Multi-GPU / High-VRAM Resource Scaling**: Scaled populations to match hardware capacities (T4 up to A100_80 tiers).

## Active Research Directions

- [ ] **P4.2 A/B Empirical Baseline**: Evaluate the performance profiles of the multi-agent clusters versus unified monolithic generations across test sets.

## Architectural (next big work block)

- **P2.1 / R4 / M6 — logit-space strength dynamics.** Multiplicative
  updates still saturate, just slower than before (AMPLIFY_FACTOR is now
  1.15 and dedup amp is 1.05). The real fix is `logit(s) += Δ` and
  `s = sigmoid(logit)`. Touches `signal_store.py` deposit/amplify/decay.
  Estimate: 1–2 hours. Until this lands, the synthesizer's ranking still
  collapses to near-uniform once a signal is hit twice.

- **P2.2 / R2 — output-side diversity metric.** `diversity.py` still
  only computes Jaccard over consumed signal IDs. The reviewer is right
  that this measures input partition, not output independence. Add
  pairwise centroid cosine distance over agent outputs in embedding
  space, plus optionally Self-BLEU. Estimate: 30–45 minutes.

- **P2.4 / M11 — real semantic clustering for haters.** `consensus_summary`
  returns top-K by strength, not an actual cluster. Replace with kNN over
  embeddings (e.g., DBSCAN or simple cosine-threshold component find).
  Estimate: ~1 hour using the embedder already loaded in SignalStore.

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

- **P3 / R12 / R15 — `tests/` directory.** None exist. Minimum set:
  (a) dedup correctness, (b) decay-then-prune invariants, (c) no-leak
  assertion (positive + negative cases), (d) provenance boost actually
  firing in a contrived ancestry, (e) the diversity-metric disambiguation
  test (paraphrased disjoint corpora should give high Jaccard, low
  centroid distance). Estimate: ~2 hours.

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
