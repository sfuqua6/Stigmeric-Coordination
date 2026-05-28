# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this codebase is

A from-scratch rebuild of the parent project's stigmergic multi-agent LLM pipeline that strictly enforces two principles:

1. **No-leak rule.** Agents only ever observe signals as artifacts (content + ID + structural metadata). Never another agent's reasoning chain, ancestry text, or chain-of-thought.
2. **Information partitioning as the diversity engine.** Diversity comes from *what each agent has been shown*, not from prompt or temperature tweaks. Scouts get disjoint corpus partitions; downstream roles use differentiated sampling strategies over the shared signal store.

This is the "cleaned" sibling of the top-level pipeline. Treat the no-leak rule as a hard architectural constraint when changing agents or the signal store.

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
python run_swarm.py debate "..." --ignore-kb               # disable cross-run knowledge base
python run_swarm.py debate "..." --reset-kb                # quarantine existing KB entries
python run_swarm.py debate "..." --show-partition-overlap  # surface Jaccard input-overlap diag

# Develop without a GPU / model download
MOCK_LLM=1 python run_swarm.py debate "Test thesis"

# Swap the model
SWARM_MODEL="microsoft/phi-2" python run_swarm.py debate "..."
SWARM_MODEL="Qwen/Qwen2.5-3B-Instruct" python run_swarm.py debate "..."

# Tests (pytest) — 287 pass, 10 skip, 0 fail as of 2026-05-28
pytest tests/                                              # full suite (~2 min with MOCK_LLM=1)
pytest tests/test_logit_dynamics.py -v                     # single file
pytest tests/test_no_leak_real_patterns.py::test_name -v   # single test

# Fast test run (all convergence gates disabled for subprocess tests)
MOCK_LLM=1 SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5 pytest tests/ -q

# Diagnostics and comparison
python diagnose.py                                          # signal-store + pipeline self-check
python tools/compare_runs.py outputs/RUN_A outputs/RUN_B    # side-by-side summary.json compare
python kb_migrate.py                                        # knowledge-base schema migration
python synthesize.py                                        # re-render synthesis from a saved store
```

Mock-mode and real-model runs land in different directories on purpose — see Outputs below.

## Architecture

### Pipeline shape (`run_swarm.py`)

Each round runs two phases against the shared `SignalStore`:

- **Phase A:** Scouts and Validators in parallel. Validators must run *before* downstream agents so VERIFICATION signals exist when the provenance boost is computed. (Reordering this breaks the boost — see the comment block in `run_swarm.py` for P2.3 / R6.)
- **Phase B:** Foragers/Developers, Critics, and Haters in parallel. They now see VERIFICATION signals from Phase A.
- After both phases: decay all signals, prune below `PRUNE_THRESHOLD`, log diversity metrics over `AgentContextRecord`s.

After `NUM_ROUNDS` rounds, the `Synthesizer` reads the surviving signal DAG and produces the final answer.

### Role activation (`ROLES_FOR_TASK` in `run_swarm.py`)

Task type gates which roles run. `creative` suppresses only Validator (no external facts to verify in poetry) but runs Hater with a craft-focused prompt — challenging derivative references, unearned allusions, grammatical breakage, and convergence without earned resonance. Without adversarial pressure, creative runs produce more clusters but with zero verification and collective agreement rather than genuine field pressure. `problem_solving` suppresses Validator; `debate`/`analysis` activate the full pipeline. Scout, Forager, and Synthesizer always run. The `coding` task swaps in `agents/coding_roles.py` (RequirementsScout, StaticCritic, ...) via `core/role_registry.py`.

### Signal store (`core/signal_store.py`)

Typed DAG of signals with strength dynamics. The no-leak rule is enforced here:

- Methods that walk provenance return **shapes** (lists of IDs, counts of typed ancestors), not rendered ancestor text.
- No `responses` field, no `parent_content` in metadata, no `get_dialogue_thread`.

Strength dynamics are gated by `USE_LOGIT_DYNAMICS` in `core/config.py`. The default path stores an internal `_logit` and applies additive deltas, projecting back through sigmoid. This fixed three real bugs documented at the top of the file: saturation crash, contrarian-drift via anti-decay, and order-dependence of decay × amplify. The legacy multiplicative path is preserved as a one-release escape hatch and exercised by `tests/test_logit_dynamics.py` — don't delete one without the other.

Dedup is similarity-based over signal content (sentence-transformers embeddings with a string-similarity fallback). On a near-duplicate, the existing signal is amplified and the new deposit is rejected.

**Partition invariant (hard enforcement).** Every INITIAL or SUPPORT signal *must* carry a non-empty `partition_id`. `deposit()` raises `AssertionError` if this is violated — this is intentional and loud. INITIAL signals get `partition_id` from `ScoutConfig.partition.partition_id` (set in `scout.py`). SUPPORT signals inherit it from their parent via the store's inheritance logic (lines 284–288 of `signal_store.py`), or explicitly via `deposit_meta["partition_id"]` which `base.py` now sets from the sampled parent's `partition_id` as a defensive redundancy. If you add a new agent role that deposits INITIAL or SUPPORT, you must supply `partition_id` in metadata or ensure the deposit has a parent that carries one.

**`sample_from_clusters()`** (Gap 1 stigmergy fix): cluster-aware sampling that biases toward the cluster whose centroid is closest to the worker's current semantic position. Workers that started developing claims in one region of idea-space keep developing in that region rather than jumping randomly. Gated by `USE_CLUSTER_AWARE_SAMPLING` in `config.py`.

**`_amplify_cluster_trail()`** (Gap 2): when a SUPPORT signal is deposited, all member signals of that cluster get a small amplification boost inside the lock. This is the pheromone-trail analogue — successful developments make the cluster more attractive to future workers. Gated by `USE_TRAIL_AMPLIFICATION`.

### Agents (`agents/`)

All agents inherit from `agents/base.py`. The base class enforces:

1. Prompts are built from exactly one of: the agent's own corpus partition (Scout only) or signals it sampled from the store. Plus the role instruction. Nothing else.
2. Only `Signal.content` may be rendered into prompts. IDs are opaque text for traceability.
3. Each iteration reports an `AgentContextRecord` listing the `chunk_ids` and `signal_ids` consumed — this is what `core/diversity.py` aggregates.

`agents/forager.py` is a backward-compat alias for `agents/developer.py` (the Forager → Developer rename is in-progress; new code should use Developer).

A scratchpad-stripper (`_SCRATCHPAD_RE` in `base.py`) removes reasoning-tuned model preambles ("Alright, so I need to...", "Let me think...", "Step 1:") that leak chain-of-thought into deposits. Critical for DeepSeek-R1-Distill outputs.

**`base.py` deposit_meta partition_id carry-forward.** Before every `store.deposit()` call, `base.py` reads `partition_id` from the sampled signals and adds it to `deposit_meta` if not already present. This is a belt-and-suspenders guard: the store's own parent-inheritance logic (lines 284–288) is the primary mechanism, but if a signal is pruned or its parent reference is stale between `sample()` and `deposit()`, the carry-forward ensures the PARTITION LEAK assertion never fires spuriously on legitimate deposits.

**`developer.py` empty-strategy fallback.** `Developer.sample()` first tries `signals_with_few_children_of_type(INITIAL, SUPPORT, 2)` (underserved gap-fill). When that's empty, it uses the assigned sampling strategy. If the strategy returns `[]` (e.g., `stratified_extremes` finds no signals in the weak or strong strength strata — all INITIALs sit in the medium range), it falls back to `store.sample_weighted(INITIAL, 1)`. Without this fallback the developer deposits SUPPORT with no parent and no `partition_id`, triggering the partition leak assertion. **Do not remove this fallback.**

### Synthesizer (`agents/synthesizer.py`)

Two-layer read-out:

1. **`core/projection.py`** — pure-Python DAG projection. No LLM. Classifies clusters as `surviving` / `contested` / `weakly_supported` / `rejected_by_field`. Computes `support_diversity`, `dissent_pressure`, `verification_score`. Surviving clusters carry an `unverified` flag when no validator reached them.
2. **`agents/synthesizer.py`** — one structured LLM call *per cluster*. There is no single pooled call. This makes cross-cluster hallucination structurally impossible and isolates render failures. Output has four sections (position synthesis, open questions, considered-and-filtered, citations); sections 3 and 4 are deterministic.

**Dual-planner architecture.** There are two cluster-selection planners:

- **`Synthesizer._plan_synthesis()`** (LLM-based, in `agents/synthesizer.py`): sees a structural digest (IDs, counts, short previews) and returns JSON with `render_full` / `section3_only` / `merge_groups`. Falls back to `build_plan()` on failure.
- **`core/projection.build_plan()`** (pure-Python, deterministic): composite score + MMR over embeddings. No LLM. Called by `ConvergenceDetector` and any path that needs a plan without a GPU.

Both implement the same conceptual role; the LLM planner has not yet been ablated against the Python planner. If ablations show no quality gain, retire `_plan_synthesis()` and always use `build_plan()`.

A faithfulness audit runs after rendering: each paragraph must have ≥4-gram overlap with the cluster content for every `[INITIAL_XXXXX]` citation tag it contains. Failures land in `renderer_audit.json`. External grounding (Wikipedia lookup per surviving cluster) is best-effort and tagged `[External context]` so it's distinguishable from agent content.

### Configuration (`core/config.py`)

All tunables live in one validated module: agent counts, decay/amplify/prune thresholds, boost params, dedup thresholds, model name. Important env vars:

- `SWARM_MODEL` — model override (default `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`)
- `MOCK_LLM=1` — skip model load entirely

`LLM_CONCURRENCY = 1` is intentional for a 6 GB laptop GPU running 4-bit NF4. Synthesizer cluster calls will parallelize when this rises.

**Stigmergy gap feature flags** (all `False` by default, gated in `config.py`):
- `USE_CLUSTER_AWARE_SAMPLING` — Gap 1: `sample_from_clusters()` biases workers toward their semantic home cluster
- `USE_TRAIL_AMPLIFICATION` — Gap 2: SUPPORT deposits amplify the whole cluster (pheromone trail)
- `USE_LOCAL_ACTION_BIASES` — Gap 3: cluster-local state multipliers in `choose_action()`
- `USE_WORKER_SEMANTIC_POSITION` — Gap 4: workers track a centroid of their prior deposits and pass it to sampling

### Convergence (`core/convergence.py`)

All convergence thresholds are overridable via environment variables. This is critical for subprocess-based tests:

| Constant | Env var | Default |
|---|---|---|
| `MIN_TIME_S` | `SWARM_MIN_TIME_S` | `60.0` |
| `MIN_ITERATIONS` | `SWARM_MIN_ITERATIONS` | `50` |
| `MIN_INITIALS_FOR_HALT` | `SWARM_MIN_INITIALS_FOR_HALT` | `6` |
| `MIN_INTER_CLUSTER_EDGES` | `SWARM_MIN_INTER_CLUSTER_EDGES` | `1` |
| `SAT_NO_NEW_SURVIVING` | `SWARM_SAT_NO_NEW_SURVIVING` | `60` |
| `MAX_ITERATIONS` | `SWARM_MAX_ITERATIONS` | `2000` |
| `MAX_TIME_S` | `SWARM_MAX_TIME_S` | `900.0` |

Tests that spawn subprocesses (test_phase_isolation, test_heterogeneous_routing, test_kb_default_off) must set `SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5 SWARM_MAX_ITERATIONS=20` etc. in the subprocess env dict, or they will time out waiting for the 60-second minimum wall. The `convergence.py` module reads these at import time.

### Retrieval (`core/retrieval.py`)

`CompositeRetriever`: Wikipedia → Web → placeholder fallback. The `--corpus=placeholder` flag forces the engineered corpus; diversity numbers from placeholder mode are *not* empirical evidence — see DEFERRED.md P4.1.

### Knowledge base (`core/knowledge_base.py`)

Cross-run consensus/rejection memory. Persisted between runs and influences strength dynamics on subsequent invocations. `--ignore-kb` disables for the current run; `--reset-kb` quarantines existing entries. Hand-picked thresholds (`_CLUSTER_SIM_THRESHOLD`, `_KB_MATCH_THRESHOLD`, `_KB_REJECTION_PENALTY`) need calibration once the real retriever is producing empirical data — see DEFERRED.md.

### Baseline mode (`core/baseline.py`)

`--mode=baseline` runs N independent agents with no signal store, no partitioning, no provenance boost. This is the A/B comparison condition for the stigmergic hypothesis. Output is shape-compatible with stigmergic runs so `tools/compare_runs.py` can diff them.

## Outputs

- `outputs/` — real-LLM runs. Each run is a timestamped subdirectory containing `answer.txt`, `citations.json`, `kb_diff.json`, `lineage.dot`, `renderer_audit.json`, `round_log.json`, `run_meta.json`, `signals.json`, `summary.json`.
- `outputs_mock/` — `MOCK_LLM=1` runs. **Kept deliberately separate** (per P0.1) so mock artifacts cannot be confused with empirical evidence. MockLLM emits SHA1-seeded phrases regardless of input, so anything in `outputs_mock/` proves plumbing, not behavior.

## support_diversity — read this before writing tests or changing projection

`support_diversity` is computed in `_aggregate_cluster()` in `core/projection.py` as:

```
total_support_diversity = len(support_partitions) + len(strategy_names)
```

where:
- `support_partitions` = set of distinct `(partition_id, depositor)` pairs from SUPPORT signals that carry `partition_id`
- `strategy_names` = set of distinct action/strategy names from SUPPORT signals that **do NOT** carry `partition_id` (fallback path for pre-partition-fix deposits)

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

- Hardware target is a single 6 GB consumer GPU (RTX 3060 Laptop, 4-bit NF4). Multi-GPU and model-serving are explicitly out of scope.
- Mock mode is for plumbing checks only — never report behavioral or diversity numbers from `outputs_mock/`.
- The colony biomimicry primitives, `dialogue_coordinator`, `Signal.responses`, `evaluate_insights_enhanced`, `deposit_with_context`, and the mode/phase/task-type/signal-type quadruple-classification system from the parent project were removed deliberately (see README "What was removed and why"). Don't re-introduce them without a plan.
- See `DEFERRED.md` for the active list of known gaps (logit dynamics tuning, retriever calibration, A/B baseline runs, KB threshold calibration, etc.) before starting a new architectural change.
