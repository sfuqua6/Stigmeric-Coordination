> **Status note (P0.3).** This implementation is a working apparatus for
> an experiment that has not yet been performed. All artifacts in
> `outputs_mock/` are from MockLLM and prove only plumbing, not behavior.
> The Jaccard numbers logged per round measure *input-side* signal-ID
> overlap, not output-side informational independence — a meaningfully
> different quantity, addressed in DEFERRED.md (R2). The trivial
> placeholder corpus in `core/intake.py` is engineered to make
> partitioning succeed by construction; diversity numbers from it should
> not be cited as evidence until a real retriever is wired in (R9).
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
├── run_swarm.py               (pipeline orchestrator)
├── core/
│   ├── config.py              (small, validated tunables)
│   ├── signal_types.py        (universal types only)
│   ├── signal_store.py        (typed DAG, trace-only API)
│   ├── intake.py              (corpus partitioner)
│   ├── sampling.py            (differentiated sampling strategies)
│   ├── diversity.py           (Jaccard distance over agent context sets)
│   └── llm.py                 (DeepSeek-R1-Distill wrapper + MockLLM)
└── agents/
    ├── base.py                (BaseAgent — enforces no-leak rule)
    ├── scout.py               (corpus-partition-conditioned)
    ├── forager.py             (signal-trace-conditioned, varied sampling)
    ├── critic.py              (artifact evaluation, no chain rendering)
    ├── hater.py               (consensus-cluster gradient, no agent reasoning)
    ├── validator.py           (external grounding via Wikipedia/web)
    └── synthesizer.py         (final read-out from surviving signals)
```

## What this implementation does *not* yet do

- An A/B benchmark against a deliberative baseline. The infrastructure is in place; the experiment is not run.
- Ablation of the provenance boost.
- Beyond-Jaccard information-theoretic measures of inter-agent independence.
- Multi-GPU or model-serving scaling. By design — this targets a single 6GB consumer GPU.

See the research note at the project root for the full motivation.
