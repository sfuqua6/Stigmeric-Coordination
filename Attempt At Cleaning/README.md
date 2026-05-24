# AI Swarm Mechanics

**Stigmergic multi-agent system for collaborative intelligence**

A pure event-driven swarm where agents communicate through signal deposits, generating novel insights through adversarial collaboration.

## What Makes This Different

| Traditional LLM | This Swarm System |
|----------------|-------------------|
| Single query → single response | Multiple agents explore concurrently |
| No self-critique | Haters challenge weak signals |
| No fact-checking | Validators verify against Wikipedia/web |
| No refinement | Multi-round iterative improvement |
| Opaque reasoning | Full signal provenance tracked |
| Prone to groupthink | Adversarial pressure prevents consensus bias |

## Quick Start (3 Steps)

### 1. Install Dependencies

```bash
# Core dependencies (required)
pip install torch transformers

# Optional but recommended
pip install requests beautifulsoup4 wikipedia sentence-transformers
```

### 2. Run a Task

```bash
python run_task.py debate "Climate action is necessary"
```

**Available task types:**
- `debate` - Argue a thesis with pro/con evidence
- `creative` - Generate creative content (poems, stories)
- `analysis` - Analyze research questions
- `problem_solving` - Propose and evaluate solutions

### 3. Check Results

Output saved to `outputs/debate_TIMESTAMP/`

## How It Works (Stigmergic Collaboration)

1. **Scouts** (4 agents) explore and generate initial ideas
2. **Foragers** (4 agents) develop those ideas with supporting details
3. **Critics** (2 agents) evaluate quality and adjust strength
4. **Haters** (2 agents) challenge weak arguments with objections
5. **Validators** (1 agent) fact-check claims against external sources
6. **Synthesizer** combines the best signals into final output

All communication is **stigmergic** - agents deposit signals and react to others' signals, with no direct messaging.

### Pure Event-Driven Architecture

- **No artificial sleep delays** - agents react immediately to signal deposits
- **Event-driven coordination** - agents wait for signal events, not polling
- **Concurrent execution** - all agents run in parallel using async/await
- **Signal strength** - successful ideas amplify, weak ones decay and get pruned

Like ants depositing pheromones, agents deposit signals that others discover and build upon.

## Architecture

### Module Structure

```
swarm/                          # Main package
├── core/
│   ├── config.py              # System configuration
│   ├── signal_store.py        # Signal storage and retrieval
│   ├── signal_types.py        # Signal type definitions
│   ├── task_config.py         # Task-specific configs
│   ├── round_coordinator.py   # Multi-round orchestration
│   └── dialogue_coordinator.py # Agent dialogue system
├── agents/
│   ├── scout.py               # Initial idea generation
│   ├── forager.py             # Idea development
│   ├── critic.py              # Quality evaluation
│   ├── hater.py               # Adversarial challenge
│   ├── validator.py           # Fact checking
│   ├── pruner.py              # Signal cleanup
│   └── synthesizer.py         # Final synthesis
├── llm/
│   └── simple_llm.py          # LLM interface with caching
├── validation/
│   └── external_sources.py    # Wikipedia/web verification
└── retrieval/
    └── advanced_retriever.py  # 100K+ word knowledge ingestion
```

**Entry Point:** `run_task.py` (NOT main.py!)

## Configuration

Edit `swarm/core/config.py`:

```python
# Agent population
NUM_SCOUTS = 4
NUM_FORAGERS = 4
NUM_CRITICS = 2
NUM_HATERS = 2

# Runtime behavior
MAX_ITERATIONS = 50
NUM_ROUNDS = 3

# Experimental features (optional)
USE_SIMPLE_SCOUTS = False      # Spatial movement (experimental)
USE_SPATIAL_STORE = False      # Locality constraints (experimental)
USE_REAL_VALIDATOR = True      # External verification (recommended)
USE_ADVANCED_RETRIEVER = True  # Knowledge ingestion (recommended)
```

## Examples

### Debate Task

```bash
python run_task.py debate "Remote work increases productivity"
```

Agents will:
- Generate claims and counter-claims
- Find supporting evidence
- Challenge weak arguments
- Verify facts against Wikipedia
- Synthesize balanced perspective

### Creative Task

```bash
python run_task.py creative "Write a haiku about AI"
```

Agents will:
- Generate draft ideas
- Refine and elaborate
- Critique for quality
- Challenge clichés
- Synthesize best elements

### Analysis Task

```bash
python run_task.py analysis "What causes innovation?"
```

Agents will:
- Propose findings
- Find supporting research
- Challenge unsupported claims
- Verify facts
- Synthesize insights

## Performance

- **Async execution:** All agents run concurrently
- **Event-driven:** Pure stigmergic communication (no sleep delays)
- **Caching:** LLM responses cached to avoid redundant generation
- **Fact-checking:** Optional Wikipedia/web verification
- **Knowledge retrieval:** Optional 100K+ word ingestion per round

**Typical runtime:** 2-5 minutes for 3 rounds of 50 iterations

## Testing

```bash
# Quick sanity test (no LLM required)
python test_pipeline_sanity.py

# Should see: "11 passed, 0 failed"

# Unit tests comparing swarm to single-LLM baselines
python -m unittest tests.test_swarm_vs_llm_benchmarks -v

# Should see: "10 tests passed" (5 test classes)
```

### Unit Test Coverage

The `tests/test_swarm_vs_llm_benchmarks.py` file tests swarm mechanics **without running actual LLMs**, allowing comparison to published benchmarks:

1. **Adversarial Validation** (comparable to TruthfulQA)
   - Haters reduce strength of unsupported claims
   - Multiple objections create adversarial pressure

2. **Iterative Refinement** (comparable to MMLU)
   - Foragers build on scout signals
   - Multi-round signal evolution improves quality

3. **Consensus Prevention** (comparable to HumanEval)
   - Diverse scout signals prevent groupthink
   - Weighted sampling maintains diversity

4. **Provenance Tracking**
   - Full chains traceable from root to leaves
   - Branching provenance supported

5. **Event-Driven Scalability**
   - Signal events trigger immediately (no polling)
   - Concurrent agent execution scales linearly

## Troubleshooting

### "ModuleNotFoundError: torch"
```bash
pip install torch transformers
```

### "CUDA out of memory"
Edit `config.py`:
```python
DEVICE = "cpu"  # Force CPU mode
```

### "No output generated"
Check `outputs/` directory for partial results. System has 5-level fallback for synthesis failures.

### Slow performance
Reduce iterations:
```python
MAX_ITERATIONS = 20  # Down from 50
NUM_ROUNDS = 2       # Down from 3
```

## Advanced: Custom Task Configs

Create custom task types in `swarm/core/task_config.py`:

```python
CUSTOM_TASK = TaskConfig(
    task_type="custom",
    task_prompt="Your prompt here",
    signal_types={
        "initial": "OBSERVATION",
        "support": "ANALYSIS",
        "critique": "CRITIQUE",
        "objection": "COUNTER"
    },
    scout_prompt_template="Generate observation: {task_prompt}",
    forager_prompt_template="Analyze: {parent_content}",
    # ... etc
)
```

## Documentation

- **`GET_STARTED_ACCURATE.md`** - Comprehensive getting started guide
- **`research/CORRECTED_FINDINGS.md`** - Known issues and fixes
- **`research/PERFORMANCE_ANALYSIS.md`** - Performance bottlenecks and solutions
- **`research/COMPREHENSIVE_IMPLEMENTATION_ANALYSIS.md`** - Dream vs reality analysis
- **`research/TECHNICAL_DEBT_AUDIT.md`** - Technical debt tracking

## What's Next

See `research/CORRECTED_FINDINGS.md` for known issues and roadmap.

**The system works! It's fast, event-driven, and produces genuinely novel insights through stigmergic collaboration.**

## Contributing

Contributions welcome! Key areas for improvement:

1. **Performance optimization** - Make embeddings lazy/batched, selective cache clearing
2. **Test coverage** - Expand unit tests for all agent types
3. **Documentation** - Add examples and tutorials
4. **Task types** - Create new task configurations

## License

MIT License

---

**Built with stigmergic principles:** Agents communicate through the environment, not with each other. Like ants following pheromone trails, they deposit signals and react to what others have left behind.
