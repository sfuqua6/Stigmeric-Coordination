> **Status note (P0.3).** This implementation is a working apparatus for
> an experiment that has not yet been performed. All artifacts in
> `outputs_mock/` are from MockLLM and prove only plumbing, not behavior.
> Real retrieval (Wikipedia → Web → placeholder fallback) is now wired in
> via `core/retrieval.py`; use `--corpus=placeholder` to revert to the
> engineered corpus, though diversity numbers from it are not evidence.
> A non-stigmergic baseline mode is available via `--mode=baseline`;
> run both modes and compare with `python tools/compare_runs.py`.
> See `DEFERRED.md` for the full list of non-integrated items.

---

# Attempt At Cleaning

Reference implementation of stigmergic multi-agent LLM coordination through information partitioning. Companion to `../STIGMERGIC_INFORMATION_PARTITIONING.md` at the project root.

## What this is

A from-scratch rebuild of the swarm pipeline that respects two principles the parent project gestures at but does not enforce:

1. **No-leak rule.** Agents observe deposited signals as artifacts (content + ID + structural metadata only). Agents never see another agent's reasoning chain, ancestry text, or chain-of-thought.
2. **Information partitioning as the diversity engine.** Diversity comes from what each agent has been shown, not from what role prompt or temperature it was given. Scouts get disjoint corpus partitions; downstream agents get differentiated sampling strategies over the shared signal store.

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

## What this implementation does *not* yet do

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
