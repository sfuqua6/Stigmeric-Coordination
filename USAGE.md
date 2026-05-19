# Usage Guide

## Quick Start

```bash
# from the Attempt At Cleaning/ directory
python run_swarm.py debate "Climate action is necessary"
python run_swarm.py analysis "What causes innovation?"
python run_swarm.py creative "Write a haiku about emergence"
python run_swarm.py problem_solving "How can cities reduce traffic?"
python run_swarm.py coding "Implement a binary search"
```

Output lands in `outputs/<task>_<timestamp>/`.

---

## Installation

**Minimum (CPU / mock mode):**
```bash
pip install torch transformers
```

**Recommended (retrieval + embeddings):**
```bash
pip install torch transformers requests beautifulsoup4 wikipedia sentence-transformers
```

**Full Colab / vLLM stack:**
```bash
pip install -r requirements-colab.txt
```

**GPU quantization (reduces VRAM from ~14 GB to ~3.5 GB):**
```bash
pip install bitsandbytes
```

---

## Task Types

| Task | What it does | Roles active |
|------|-------------|--------------|
| `debate` | Argue a thesis, pro/con, with fact-checking | Full pipeline |
| `analysis` | Explore a research question | Full pipeline |
| `problem_solving` | Propose and evaluate solutions | No Validator |
| `creative` | Generate creative content | No Hater or Validator |
| `coding` | Generate and review code | Coding-specific roles |

---

## CLI Flags

### Mode and corpus

```bash
# Run non-stigmergic independent-agent baseline (A/B comparison condition)
python run_swarm.py debate "..." --mode=baseline

# Skip live retrieval, use an engineered placeholder corpus
# (diversity numbers from this mode are NOT empirical evidence)
python run_swarm.py debate "..." --corpus=placeholder
```

### Model heterogeneity

```bash
# Route different roles to different GGUF models (see configs/heterogeneous.json)
python run_swarm.py debate "..." --heterogeneous

# Phase-isolated execution: one model in memory at a time
# Use this on Windows / 16 GB RAM where llama-cpp-python doesn't fully unload
python tools/run_isolated.py debate "..." --heterogeneous
```

### Strategy and diagnostics

```bash
# Single sampling strategy across all roles (ablation)
python run_swarm.py debate "..." --strategy-variant=single

# Show Jaccard input-partition overlap in the round log (debug only)
python run_swarm.py debate "..." --show-partition-overlap
```

### Knowledge base

```bash
# Disable cross-run knowledge base for this run
python run_swarm.py debate "..." --ignore-kb

# Quarantine existing KB entries before running
python run_swarm.py debate "..." --reset-kb
```

---

## Environment Variables

| Variable | Effect |
|----------|--------|
| `MOCK_LLM=1` | Skip model load; use a deterministic mock. Proves plumbing only. |
| `SWARM_MODEL="repo/name"` | Override the model (default: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`) |
| `COLAB=1` | Force vLLM path and expand agent population for cloud hardware |
| `SWARM_OUTPUTS_BASE_DIR` | Redirect run output trees |
| `SWARM_MODELS_DIR` | Redirect GGUF model storage |
| `SWARM_KB_DIR` | Redirect cross-run knowledge base |
| `SWARM_RETRIEVAL_CACHE_DIR` | Redirect Wikipedia/Web cache |

**Develop without a GPU:**
```bash
MOCK_LLM=1 python run_swarm.py debate "Test thesis"
```

**Swap the model:**
```bash
SWARM_MODEL="microsoft/phi-2" python run_swarm.py debate "..."
SWARM_MODEL="Qwen/Qwen2.5-3B-Instruct" python run_swarm.py debate "..."
```

---

## Execution Modes

### Laptop (GGUF path)
- Backend: `llama-cpp-python`, serialized (one model at a time)
- `LLM_CONCURRENCY=1` — enforced in `core/config.py`
- Quantized `Q4_K_M` models, fits on 6 GB VRAM (RTX 3060 Laptop)
- If `llama-cpp-python` lacks GPU acceleration, falls back to CPU automatically

### Colab / Cloud (vLLM path)
- Set `COLAB=1` to activate
- Backend: vLLM async engine, `fp16`/`bf16` weights
- Concurrent multi-agent execution
- See `notebooks/colab_setup.ipynb` for a turnkey setup (Drive mount, model download, env wiring)

### Phase-isolated execution
- Splits each pipeline phase into its own subprocess
- Persists `SignalStore` to `store_state.json` between phases
- Exactly one model in memory at any time — the fix for Windows model-unload issues
```bash
python tools/run_isolated.py debate "..." [--heterogeneous]
```

---

## Configuration

Edit `core/config.py` for tuning:

```python
# Agent population
NUM_SCOUTS = 4
NUM_FORAGERS = 4
NUM_CRITICS = 2
NUM_HATERS = 2

# Runtime behavior
NUM_ROUNDS = 3

# Signal dynamics
PRUNE_THRESHOLD = 0.1      # Signals below this strength are removed
USE_LOGIT_DYNAMICS = True  # Logit-space strength updates (fixes saturation bugs)

# Experimental flags
USE_KNOWLEDGE_BASE = False # Cross-run memory (default off)
```

---

## Output Files

Each run produces a timestamped directory under `outputs/` (real runs) or `outputs_mock/` (mock runs):

| File | Contents |
|------|----------|
| `answer.txt` | Final synthesized answer |
| `summary.json` | High-level metrics and top signals |
| `signals.json` | Full signal DAG |
| `round_log.json` | Per-round metrics including `output_diversity` |
| `citations.json` | External source citations |
| `renderer_audit.json` | Faithfulness audit (paragraph ↔ citation overlap check) |
| `kb_diff.json` | Knowledge base changes this run made |
| `run_meta.json` | Execution mode, flags, model used |

**Note:** `outputs_mock/` runs prove plumbing only — MockLLM emits SHA1-seeded phrases regardless of input. Never report diversity or quality numbers from mock runs.

---

## Tools

```bash
# Compare two runs side-by-side
python tools/compare_runs.py outputs/RUN_A outputs/RUN_B

# Factorial sweep across experimental levers (generates the ablation table)
python tools/sweep.py

# Re-render synthesis from a saved signal store (no re-run needed)
python synthesize.py

# Signal-store and pipeline self-check
python diagnose.py

# Knowledge-base schema migration
python kb_migrate.py
```

---

## Testing

```bash
# Full test suite (195 passed, 7 skipped as of last run)
pytest tests/

# Single file
pytest tests/test_logit_dynamics.py -v

# Single test
pytest tests/test_no_leak_real_patterns.py::test_name -v
```

Tests run without a GPU or model download. Mock LLM is wired automatically in the test harness.

---

## Troubleshooting

### "CUDA out of memory"
Reduce concurrency or switch to CPU:
```python
# core/config.py
LLM_CONCURRENCY = 1
DEVICE = "cpu"
```
Or enable quantization:
```bash
pip install bitsandbytes
# Then set use_quantization in your model config
```

### "No module named torch" / missing dependencies
```bash
pip install torch transformers
```

### Model won't unload between phases (Windows)
Use phase-isolated mode:
```bash
python tools/run_isolated.py debate "..."
```

### Retrieval returns no results
The pipeline falls back: Wikipedia → Web → placeholder corpus. If all three fail, `--corpus=placeholder` forces the engineered corpus (note: diversity numbers are not valid in this mode).

### Low validator scores
See `validator_diagnostic.md` for known keyphrase extraction and score parsing issues with prioritized fixes.

### Slow first run
The first run downloads the model and warms the retrieval cache. Subsequent runs reuse `retrieval_cache/` and `search_cache/`.

---

## Experimental Levers (for ablation study)

The four independently-controllable levers:

| Lever | Flag | What it controls |
|-------|------|-----------------|
| Trace-hiding | `--mode={stigmergic,baseline}` | Stigmergic signal store vs. independent agents |
| Corpus | `--corpus={real,placeholder}` | Live retrieval vs. engineered corpus |
| Model heterogeneity | `--heterogeneous` | Per-role model routing via `configs/heterogeneous.json` |
| Strategy heterogeneity | `--strategy-variant={diverse,single}` | Full strategy library vs. single strategy |

A factorial sweep across all four generates the ablation table. Driver: `tools/sweep.py`.
