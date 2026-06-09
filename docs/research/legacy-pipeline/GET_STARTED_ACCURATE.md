# Getting Started with AI Swarm Mechanics

**Last Updated:** 2025-11-17
**Status:** Accurate guide for current codebase

---

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

---

## What Actually Happens

1. **Scouts** (4 agents) explore and generate initial ideas
2. **Foragers** (4 agents) develop those ideas with supporting details  
3. **Critics** (2 agents) evaluate quality and adjust strength
4. **Haters** (2 agents) challenge weak arguments with objections
5. **Validators** (1 agent) fact-check claims against external sources
6. **Synthesizer** combines the best signals into final output

All communication is **stigmergic** - agents deposit signals and react to others' signals, with no direct messaging.

---

## Module Structure (Actual)

```
swarm/                          # Main package (NOT swarm_debate!)
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

---

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

---

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

---

## How It's Different from Regular LLMs

| Traditional LLM | Swarm System |
|----------------|--------------|
| Single query → single response | Multiple agents explore concurrently |
| No self-critique | Haters challenge weak signals |
| No fact-checking | Validators verify against Wikipedia/web |
| No refinement | Multi-round iterative improvement |
| Opaque reasoning | Full signal provenance tracked |
| Prone to groupthink | Adversarial pressure prevents consensus bias |

---

## Performance

- **Async execution:** All agents run concurrently
- **Event-driven:** Pure stigmergic communication (no sleep delays)
- **Caching:** LLM responses cached to avoid redundant generation
- **Fact-checking:** Optional Wikipedia/web verification
- **Knowledge retrieval:** Optional 100K+ word ingestion per round

**Typical runtime:** 2-5 minutes for 3 rounds of 50 iterations

---

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

---

## Testing

```bash
# Quick sanity test (no LLM required)
python test_pipeline_sanity.py

# Should see: "11 passed, 0 failed"
```

---

## Advanced: Task Configs

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

---

## What's Next

See `research/CORRECTED_FINDINGS.md` for known issues and roadmap.

**The system works! It's fast, event-driven, and produces genuinely novel insights through stigmergic collaboration.**
