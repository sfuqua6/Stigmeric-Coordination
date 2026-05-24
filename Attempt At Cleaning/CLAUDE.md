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

# Tests (pytest)
pytest tests/                                              # full suite
pytest tests/test_logit_dynamics.py -v                     # single file
pytest tests/test_no_leak_real_patterns.py::test_name -v   # single test

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

### Agents (`agents/`)

All agents inherit from `agents/base.py`. The base class enforces:

1. Prompts are built from exactly one of: the agent's own corpus partition (Scout only) or signals it sampled from the store. Plus the role instruction. Nothing else.
2. Only `Signal.content` may be rendered into prompts. IDs are opaque text for traceability.
3. Each iteration reports an `AgentContextRecord` listing the `chunk_ids` and `signal_ids` consumed — this is what `core/diversity.py` aggregates.

`agents/forager.py` is a backward-compat alias for `agents/developer.py` (the Forager → Developer rename is in-progress; new code should use Developer).

A scratchpad-stripper (`_SCRATCHPAD_RE` in `base.py`) removes reasoning-tuned model preambles ("Alright, so I need to...", "Let me think...", "Step 1:") that leak chain-of-thought into deposits. Critical for DeepSeek-R1-Distill outputs.

### Synthesizer (`agents/synthesizer.py`)

Two-layer read-out:

1. **`core/projection.py`** — pure-Python DAG projection. No LLM. Classifies clusters as `surviving` / `contested` / `weakly_supported` / `rejected_by_field`. Computes `support_diversity`, `dissent_pressure`, `verification_score`. Surviving clusters carry an `unverified` flag when no validator reached them.
2. **`agents/synthesizer.py`** — one structured LLM call *per cluster*. There is no single pooled call. This makes cross-cluster hallucination structurally impossible and isolates render failures. Output has four sections (position synthesis, open questions, considered-and-filtered, citations); sections 3 and 4 are deterministic.

A faithfulness audit runs after rendering: each paragraph must have ≥4-gram overlap with the cluster content for every `[INITIAL_XXXXX]` citation tag it contains. Failures land in `renderer_audit.json`. External grounding (Wikipedia lookup per surviving cluster) is best-effort and tagged `[External context]` so it's distinguishable from agent content.

### Configuration (`core/config.py`)

All tunables live in one validated module: agent counts, decay/amplify/prune thresholds, boost params, dedup thresholds, model name. Important env vars:

- `SWARM_MODEL` — model override (default `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`)
- `MOCK_LLM=1` — skip model load entirely

`LLM_CONCURRENCY = 1` is intentional for a 6 GB laptop GPU running 4-bit NF4. Synthesizer cluster calls will parallelize when this rises.

### Retrieval (`core/retrieval.py`)

`CompositeRetriever`: Wikipedia → Web → placeholder fallback. The `--corpus=placeholder` flag forces the engineered corpus; diversity numbers from placeholder mode are *not* empirical evidence — see DEFERRED.md P4.1.

### Knowledge base (`core/knowledge_base.py`)

Cross-run consensus/rejection memory. Persisted between runs and influences strength dynamics on subsequent invocations. `--ignore-kb` disables for the current run; `--reset-kb` quarantines existing entries. Hand-picked thresholds (`_CLUSTER_SIM_THRESHOLD`, `_KB_MATCH_THRESHOLD`, `_KB_REJECTION_PENALTY`) need calibration once the real retriever is producing empirical data — see DEFERRED.md.

### Baseline mode (`core/baseline.py`)

`--mode=baseline` runs N independent agents with no signal store, no partitioning, no provenance boost. This is the A/B comparison condition for the stigmergic hypothesis. Output is shape-compatible with stigmergic runs so `tools/compare_runs.py` can diff them.

## Outputs

- `outputs/` — real-LLM runs. Each run is a timestamped subdirectory containing `answer.txt`, `citations.json`, `kb_diff.json`, `lineage.dot`, `renderer_audit.json`, `round_log.json`, `run_meta.json`, `signals.json`, `summary.json`.
- `outputs_mock/` — `MOCK_LLM=1` runs. **Kept deliberately separate** (per P0.1) so mock artifacts cannot be confused with empirical evidence. MockLLM emits SHA1-seeded phrases regardless of input, so anything in `outputs_mock/` proves plumbing, not behavior.

## Diversity metrics — read this before changing them

The Jaccard distance in `core/diversity.py` (`_partition_overlap_jaccard`, `_role_partition_overlap`, `_overall_partition_overlap`) measures **input-partition health** — did agents read disjoint chunks and signal IDs? It does *not* measure output diversity. Two agents reading disjoint inputs can produce identical outputs.

True output-diversity metrics live in `core/output_diversity.py`: `centroid_cosine_distance` (embedding-space) and `self_bleu` (lexical). Both are logged per round under `round_log.json:output_diversity`.

The old public names `role_diversity` / `overall_diversity` / `format_report` are kept as backward-compat aliases but they are partition-overlap functions. Partition-overlap is suppressed from the round log by default; surface with `--show-partition-overlap`.

## Constraints to respect

- Hardware target is a single 6 GB consumer GPU (RTX 3060 Laptop, 4-bit NF4). Multi-GPU and model-serving are explicitly out of scope.
- Mock mode is for plumbing checks only — never report behavioral or diversity numbers from `outputs_mock/`.
- The colony biomimicry primitives, `dialogue_coordinator`, `Signal.responses`, `evaluate_insights_enhanced`, `deposit_with_context`, and the mode/phase/task-type/signal-type quadruple-classification system from the parent project were removed deliberately (see README "What was removed and why"). Don't re-introduce them without a plan.
- See `DEFERRED.md` for the active list of known gaps (logit dynamics tuning, retriever calibration, A/B baseline runs, KB threshold calibration, etc.) before starting a new architectural change.
