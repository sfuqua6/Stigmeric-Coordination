# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stigmergic multi-agent swarm system where LLM-powered agents communicate through signal deposits (like ant pheromones) rather than direct messaging. Agents explore, develop, critique, and synthesize ideas through iterative rounds.

## Two Codebases

This repo contains **two separate implementations** with different entry points:

| | `run_task.py` (main) | `Attempt At Cleaning/run_swarm.py` (new) |
|---|---|---|
| Task types | debate, creative, analysis, problem_solving | + **coding** |
| Architecture | monolithic orchestrator | strict no-leak, information-partitioning |
| Coding support | **not implemented** | full (RequirementsScout, CodeDeveloper, StaticCritic, TestValidator, CodeSynthesizer) |
| CLAUDE.md | this file | `Attempt At Cleaning/CLAUDE.md` |

## Commands

```bash
# Main pipeline (run_task.py)
python run_task.py debate "Climate action is necessary"
python run_task.py analysis "What causes innovation?"
python run_task.py creative "Write a haiku about AI"
python run_task.py problem_solving "How can we make cities more sustainable?"
python run_task.py hyper_test   # fast end-to-end validation (2 rounds, 5 iter, no GPU needed)

# New pipeline (Attempt At Cleaning/)
cd "Attempt At Cleaning"
python run_swarm.py coding "Implement a binary search"
MOCK_LLM=1 python run_swarm.py debate "Test thesis"   # no GPU needed

# Pipeline sanity test (no LLM/GPU required, custom test framework)
python tests/test_pipeline_sanity.py

# Unit tests (unittest framework)
python -m unittest tests.test_swarm_vs_llm_benchmarks -v
python -m unittest tests.test_colony -v
python -m unittest tests.test_critic_signal_generation -v

# Install dependencies
pip install torch transformers
pip install requests beautifulsoup4 wikipedia sentence-transformers  # optional
```

## Architecture

### Main Pipeline (`run_task.py`)
A ~1000-line async orchestrator. **Protect this file.** Changes affect every task type.

**Execution flow per round:**
1. Keyword extraction → web searches (`DynamicRetriever`)
2. Stage 1: Scouts (parallel) → deposit `INITIAL` signals
3. Stage 2: Foragers + Critics (parallel, depend on scouts)
4. Stage 3: Haters (depend on stage 2)
5. Background: Validators, Pruners, decay/prune loop, `DialogueCoordinator`
6. `Synthesizer` collapses surviving signals into a round synthesis
7. Repeat for `NUM_ROUNDS` (default 3), then cross-round synthesis

### Signal Store (`swarm/core/signal_store.py`)
Central coordination mechanism. Agents deposit typed signals with strength values. Signals decay, get amplified by corroboration, and are pruned when weak. Uses `RLock` (not `Lock`) because `deposit→get_ancestors` is reentrant. FAISS optional for semantic similarity.

### Stage Coordinator (`swarm/core/stage_coordinator.py`)
Manages parallel execution across `AdaptiveLLMPool`. Agents in the same stage run concurrently via `prepare_prompt()` / `process_result()` (the staged API); stages execute sequentially by dependency graph.

### Key Modules
- `swarm/core/config.py` — all tunables. Imported with `from swarm.core.config import *` everywhere.
- `swarm/core/task_config.py` — `ADAPTIVE_CONFIG` is the single config used for all task types. Legacy per-type configs are aliases. **`coding` is not registered here** — this is the root cause of coding task failures.
- `swarm/llm/simple_llm.py` — LLM with response cache
- `swarm/llm/llm_pool.py` — `AdaptiveLLMPool` for parallel LLM calls
- `swarm/agents/` — one file per role, all inherit `base_agent.py`. `Scout.assess_strength_creative()` scores by prose indicators (study, data, research) — this **actively penalizes code**.
- `swarm/colony/` — leaf-cutter ant biomimicry primitives (pheromone fields, caste system, trail networks). Stigmergic backbone.

### Why Coding Fails in `run_task.py`
1. **No `coding` task config**: `get_task_config("coding")` silently returns `ADAPTIVE_CONFIG` (prose prompts).
2. **Prose-oriented prompts**: Scouts are told "Generate a clear, specific, and well-developed idea" — not "Write a Python function."
3. **`assess_strength_creative()` rejects code**: Scores by presence of "study", "data", "research", "because" — code snippets score ~0.35 and may never clear `MIN_DEPOSIT_STRENGTH`.
4. **No code extraction**: No fenced block detection, no `ast.parse()`, no `py_compile` validation.
5. **Synthesizer emits prose**: Final output is a paragraph, not assembled runnable code.

### `Attempt At Cleaning/` Architecture
Implements the coding task with specialized roles loaded via `core/role_registry.py`:
- `RequirementsScout` — emits acceptance criteria from task prompt (no corpus needed)
- `CodeDeveloper` — deposits fenced Python `SUPPORT` signals, rejects prose
- `StaticCritic` — runs `ast.parse()` on code blocks; strength = 1 − (errors/attempts)
- `EdgeCaseHater` — cycles through a canonical edge-case list and asks if the code fails on each
- `TestValidator` — writes and runs a pytest function via subprocess (5s timeout); strength ∈ {0, 0.5, 1.0}
- `CodeSynthesizer` — assembles surviving code blocks, runs `py_compile`, falls back to strongest single SUPPORT

Also enforces the **no-leak rule**: agents never see another agent's ancestry text or reasoning chain — only signal content + IDs.

## Configuration Pattern
`swarm/core/config.py` uses `from swarm.core.config import *` throughout. All tuning is via module-level constants with inline `assert` validation. Feature flags (`USE_SIMPLE_SCOUTS`, `USE_SPATIAL_STORE`, `USE_REAL_VALIDATOR`, `USE_ADVANCED_RETRIEVER`) gate experimental imports at the top of `run_task.py`.

## File Organization
- `run_task.py` — main pipeline (do not casually refactor)
- `Attempt At Cleaning/` — new architecture with coding support; has its own `CLAUDE.md`
- `tests/` — unit tests (unittest + custom sanity harness)
- `swarm/` — core package
- `archive/` — historical entry points, old tools, unused modules
- `research/` — design docs, performance analysis
- `outputs/` — generated at runtime

## Constraints
- GPU target: NVIDIA RTX 3060 Laptop (6GB VRAM). Models use 4-bit NF4 quantization.
- Default model: `microsoft/phi-2` (2.7B params) in `run_task.py`; `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` in `Attempt At Cleaning/`.
- Tests must run without GPU/LLM — they mock the LLM layer or use hardcoded outputs.
- The `from config import *` pattern is intentional and pervasive. Don't refactor it without a plan.
