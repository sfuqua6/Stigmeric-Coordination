> **Status note.** This implementation is the apparatus for an experimental
> comparison across four levers (partition, model, strategy, trace-hiding).
> The experiment is partially run: see `outputs/` for stigmergic runs and
> `outputs/baseline_*` for non-stigmergic comparison runs. The headline
> diversity metric is output-side centroid cosine distance + Self-BLEU,
> NOT input-side Jaccard. The Jaccard numbers remain in the code under
> their honest name (`_partition_overlap_jaccard`) but are not reported as
> diversity. See `DEFERRED.md` for known limits and Section "Why we
> deprecated the Jaccard diversity number" in this README.

---

# Attempt At Cleaning

Reference implementation of stigmergic multi-agent LLM coordination through information partitioning. Companion to `../STIGMERGIC_INFORMATION_PARTITIONING.md` at the project root.

## What this is

A from-scratch rebuild of the swarm pipeline that treats multi-agent coordination
as the system, and treats _every_ source of inter-agent decorrelation as an
independently measurable lever. The levers we instrument are:

1. **Information partitioning.** Scouts receive disjoint slices of a retrieved
   corpus. They cannot see each other's slices.
2. **Model heterogeneity.** Different roles run on different local LLM
   checkpoints from different families and size classes, sequentially loaded
   per phase so a 6 GB consumer GPU can host the ensemble across time rather
   than in space.
3. **Strategy heterogeneity.** Foragers/critics/validators use different
   sampling strategies over the shared signal store.
4. **Trace-only coordination.** Agents see deposited signals as artifacts;
   they never see another agent's reasoning chain. (The no-leak rule.)

The dissertation claim is NOT "information partitioning is the diversity
engine." The claim is "multi-agent coordination produces emergent task
performance from multiple independently-ablatable sources of diversity, and
the architecture lets us measure which sources contribute."

This framing matters for what counts as evidence. A single-source claim
would require us to control all confounds and demonstrate the chosen lever
in isolation. A multi-source claim requires us to ablate each lever and
report per-lever contribution. The current experimental apparatus (with
the baseline-mode comparison from prior work) is designed for the latter.

## Run it

```bash
# from this folder
python run_swarm.py debate "Climate action is necessary"
python run_swarm.py analysis "What causes innovation?"
python run_swarm.py creative "Write a haiku about emergence"
python run_swarm.py problem_solving "How can cities reduce traffic?"
```

To develop without a GPU, set `MOCK_LLM=1`:

```bash
MOCK_LLM=1 python run_swarm.py debate "Test thesis"
```

## Execution Modes

The stigmergic swarm supports two deployment modes optimized for hardware configurations:

1. **Laptop Mode (GGUF Path)**
   - **Backend**: `llama-cpp-python` / single serialized sequence.
   - **Constraints**: Quantized `Q4_K_M` model stack, `LLM_CONCURRENCY=1`.
   - **Trigger**: Automatic fallback if no GPU acceleration is present, or forced local initialization.

2. **Colab / Cloud Compute Mode (vLLM Path)**
   - **Backend**: `vLLM` high-throughput async engine leveraging unified fp16/bf16 weights.
   - **Execution**: Concurrent multi-agent execution loops bypass manual synchronization semaphores.
   - **Trigger**: Automatically scales agent populations and sets precision modes based on runtime hardware detection.

### Environment Variable Configurations

- `COLAB=1` : Forces the system to evaluate vLLM paths and expand agent cluster coverage parameters.
- `SWARM_MODEL="your-repo/your-model"` : Overrides automatic tier selection and forces a specific Hugging Face model template path.

### Installation for Parallel Processing Capabilities

```bash
pip install vllm torch
```

## Experimental design

The four levers are independently controlled via CLI flags:

--mode={stigmergic, baseline} trace-hiding on/off
--corpus={real, placeholder} real retriever vs. engineered corpus
--heterogeneous enable role→model routing per `configs/heterogeneous.json`
--strategy-variant={diverse,single} strategy library: full vs. single-strategy

A factorial sweep across these gives the ablation table the dissertation needs.
The driver for the sweep is `tools/sweep.py` (see §6 below).

## What was kept from the parent project

- Strength dynamics (decay / amplify / prune) on the signal store.
- Provenance-aware boost when a child descends from verified ancestry.
- Stratified and weighted sampling primitives.
- The agent role taxonomy (Scout, Forager, Critic, Hater, Validator, Synthesizer).
- External-source validator pattern.

## What was removed and why

- The `swarm/colony/` module (PheromoneField, FungusGarden, TrailNetwork, CasteSystem, Stridulation): not wired into the parent pipeline. Decorative biomimicry.
- `dialogue_coordinator.py`, Signal.responses, `get_dialogue_thread`: explicitly anti-stigmergic — they let agents reconstruct argument chains.
- `evaluate_insights_enhanced` critic mode: dead code in the parent and a stigmergy violation by design (renders full provenance text into prompts).
- The mode/phase/task-type/signal-type quadruple classification system: collapsed to one task config and universal signal types.
- `deposit_with_context` injecting `parent_content` into child metadata: removed. A child may know it has a parent (by ID); it may not see what the parent said.
- All explicit `asyncio.sleep(random.uniform(...))` polling delays: replaced with proper event-loop yielding and signal events.

### What this build adds

- **Per-role model routing.** `--heterogeneous` enables sequential
  per-phase loading of distinct GGUF models per role. See
  `configs/heterogeneous.json` for the default assignment and
  `core/llm_router.py` for the load/unload mechanics. Total disk
  footprint for the default assignment is ~33 GB across six models.
- **Per-role diversity logging.** `round_log.json` now includes
  `diversity.cross_model_delta` so the contribution of model
  heterogeneity (vs. partition heterogeneity vs. strategy heterogeneity)
  can be isolated in ablation analysis.
- **Cloud-validator stub.** `--cloud-validator={anthropic,gemini}`
  flag wired through but not implemented. To enable, see comments
  in `agents/validator.py`.
- **Phase-isolated execution (`--isolated`).** Splits the pipeline
  away from in-process parallelism: each phase of each round runs
  in its own subprocess, persists the `SignalStore` to
  `store_state.json`, and exits. The next subprocess loads that
  checkpoint and runs the next phase. Exactly one model is resident
  in memory at any time — the canonical workaround for
  llama-cpp-python's incomplete model unload on Windows / 16 GB-RAM
  hardware. Orchestrator: `python tools/run_isolated.py debate
"..." [--heterogeneous]`. `run_meta.json` records
  `execution_mode: "phase_isolated"` so isolated and in-process
  runs are distinguishable in cross-run analysis.
- **Path overrides for non-local environments (Colab, scratch dirs).**
  Four env vars redirect where the pipeline reads and writes:
  `SWARM_OUTPUTS_BASE_DIR` (run output trees), `SWARM_MODELS_DIR`
  (GGUF files), `SWARM_KB_DIR` (cross-run knowledge base),
  `SWARM_RETRIEVAL_CACHE_DIR` (Wikipedia/Web cache). A turnkey
  Colab notebook is at `notebooks/colab_setup.ipynb` — handles
  Drive mount, model download from Hugging Face, env-var wiring,
  and either execution mode (in-process heterogeneous or
  phase-isolated).

## Folder layout

```
Attempt At Cleaning/
├── README.md                  (this file)
├── run_swarm.py               (pipeline orchestrator; --mode, --corpus flags)
├── core/
│   ├── config.py              (small, validated tunables)
│   ├── signal_types.py        (INITIAL, SUPPORT, CRITIQUE_POSITIVE/NEGATIVE, ...)
│   ├── signal_store.py        (typed DAG, logit-space dynamics, DBSCAN clustering)
│   ├── intake.py              (corpus partitioner)
│   ├── sampling.py            (differentiated sampling strategies)
│   ├── diversity.py           (partition-overlap metrics — input-health only)
│   ├── output_diversity.py    (centroid cosine distance, Self-BLEU)
│   ├── baseline.py            (non-stigmergic independent-agent baseline)
│   ├── retrieval.py           (Wikipedia -> Web -> placeholder CompositeRetriever)
│   ├── projection.py          (DAG projection: surviving/contested/weakly_supported)
│   ├── knowledge_base.py      (cross-run consensus/rejection memory)
│   ├── role_registry.py       (task-type -> role class mapping)
│   └── llm.py                 (model wrapper + MockLLM)
├── agents/
│   ├── base.py                (BaseAgent — enforces no-leak rule)
│   ├── scout.py               (corpus-partition-conditioned)
│   ├── developer.py           (signal-trace-conditioned; renamed from Forager)
│   ├── forager.py             (backward-compat alias for developer.py)
│   ├── critic.py              (CRITIQUE_POSITIVE / CRITIQUE_NEGATIVE routing)
│   ├── hater.py               (DBSCAN cluster targeting, no agent reasoning)
│   ├── validator.py           (signed-avg external grounding)
│   ├── synthesizer.py         (ranked cluster render + executive summary)
│   └── coding_roles.py        (coding task: RequirementsScout, StaticCritic, ...)
└── tools/
    └── compare_runs.py        (side-by-side summary.json comparison)
```

## What this implementation does _not_ yet do

- **A run of the actual experiment.** The baseline coordinator and stigmergic pipeline are wired and tested, but no GPU run with a real model has been performed. All per-round numbers in `outputs_mock/` prove only plumbing.
- **Ablation of the provenance boost.** The boost is on by default; there is no flag to disable it for a controlled comparison.
- **Multi-GPU or model-serving scaling.** By design — this targets a single 6GB consumer GPU (NVIDIA RTX 3060 Laptop, 4-bit NF4).

## The Jaccard metric was renamed (not removed)

Earlier versions reported a "diversity" number per round that was actually
**Jaccard distance over agent context sets** — which corpus chunks and signal
IDs each agent consumed. That is a partition-health diagnostic (did agents read
disjoint inputs?) not an output-diversity metric (did agents produce different
ideas?). Two agents who read completely different chunks can still produce
identical outputs; the Jaccard number wouldn't catch it.

The functions are now named `_partition_overlap_jaccard`, `_role_partition_overlap`,
and `_overall_partition_overlap` in `core/diversity.py`. The old public names
are kept as backward-compat aliases. Partition overlap is suppressed from the
round log by default; enable it with `--show-partition-overlap` when debugging
corpus partitioning.

The **actual** output diversity metrics are in `core/output_diversity.py`:

- `centroid_cosine_distance` — average cosine distance from each deposit's
  embedding to the group centroid. Uses `sentence-transformers` when available,
  falls back to bag-of-words TF vectors.
- `self_bleu` — average BLEU of each deposit against all others. Low = diverse.

Both are logged per round in `round_log.json` under `output_diversity`.

See the research note at the project root for the full motivation.
