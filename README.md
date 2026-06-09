# AI Swarm Mechanics

**Stigmergic multi-agent LLM pipeline with strict information partitioning**

Agents coordinate only through a shared signal store — like ants depositing pheromones. Two hard architectural rules drive everything:

1. **No-leak rule.** Agents only ever observe signals as artifacts (content + ID + structural metadata) — never another agent's reasoning chain, ancestry text, or chain-of-thought.
2. **Information partitioning as the diversity engine.** Diversity comes from *what each agent has been shown*, not from prompt or temperature tweaks. Scouts get disjoint corpus partitions; downstream roles use differentiated sampling strategies over the shared signal store.

## Quick start

```bash
pip install -r requirements.txt

# Run a task (entry point is run_swarm.py)
python run_swarm.py debate "Climate action is necessary"
python run_swarm.py analysis "What causes innovation?"
python run_swarm.py creative "Write a haiku about emergence"
python run_swarm.py problem_solving "How can cities reduce traffic?"
python run_swarm.py coding "Implement a binary search"

# Develop without a GPU / model download
MOCK_LLM=1 python run_swarm.py debate "Test thesis"

# Swap the model (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
SWARM_MODEL="Qwen/Qwen2.5-3B-Instruct" python run_swarm.py debate "..."
```

Output lands in `outputs/<task>_<timestamp>/` (`answer.txt`, `summary.json`, `signals.json`, lineage and audit artifacts). Mock runs land in `outputs_mock/` — kept deliberately separate so mock artifacts are never mistaken for empirical evidence.

Useful flags: `--mode=baseline` (independent agents, no stigmergy — the A/B comparison condition), `--corpus=placeholder`, `--ignore-kb`, `--reset-kb`, `--show-partition-overlap`.

## How it works

Each round runs two phases against the shared `SignalStore`:

- **Phase A:** Scouts (disjoint corpus partitions → INITIAL signals) and Validators (fact-check against external sources → VERIFICATION signals) run in parallel.
- **Phase B:** Developers, Critics, and Haters run in parallel — they see Phase A's VERIFICATION signals, which feed the provenance boost.
- After both phases: signal decay, pruning, diversity-metric logging.

After all rounds, a pure-Python DAG projection classifies clusters (`surviving` / `contested` / `weakly_supported` / `rejected_by_field`), and the Synthesizer renders the final answer with one structured LLM call per cluster — making cross-cluster hallucination structurally impossible. Cross-run consensus persists in a knowledge base; clusters carry a typed genome with composite fitness driving selection.

## Repository layout

```
run_swarm.py          # Entry point (there is no main.py / run_task.py here)
core/                 # Signal store, projection, fitness, topology, config, convergence
agents/               # Scout, Developer, Critic, Hater, Validator, Synthesizer
tests/                # pytest suite (~304 tests; run with MOCK_LLM=1)
tools/                # compare_runs.py, maintenance scripts
eval/                 # Evaluation datasets and results
notebooks/            # Colab runners (hybrid local-GPU + Groq backend)
docs/                 # Design docs, reviews, prompts, research notes, reference runs
legacy/               # The original pipeline (run_task.py + swarm/) — unmaintained,
                      #   preserved for reference; see legacy/README.md
```

## Testing

```bash
# Full suite (~2 min with mock LLM)
MOCK_LLM=1 SWARM_MIN_TIME_S=0 SWARM_MIN_ITERATIONS=5 pytest tests/ -q

# Self-check the signal store and pipeline wiring
python diagnose.py
```

## Documentation

- **`CLAUDE.md`** — architecture reference (signal store, projection, genome, convergence)
- **`USAGE.md`** — usage guide
- **`DEFERRED.md`** — known gaps and deferred work
- **`docs/`** — design docs, architecture reviews, research notes

## Constraints

- Hardware target is a single 6 GB consumer GPU (4-bit NF4). A Groq/hybrid backend is available for the Colab notebooks.
- Mock mode proves plumbing, not behavior — never report results from `outputs_mock/`.

## License

MIT License
