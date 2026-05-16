"""Configuration for the cleaned stigmergic implementation."""

import os

# ---------------------------------------------------------------------------
# Colab tier detection — drives model selection, populations, dtype, etc.
# ---------------------------------------------------------------------------
# Detect GPU tier so we can pick reasonable defaults for Colab T4 / L4 / A100
# without forcing the user to set every knob. _TIER is None off-Colab (laptop /
# CPU dev), or "t4" | "l4" | "a100_40" | "a100_80" | "unknown" on Colab.
# Force the Colab path on a non-Colab host with COLAB=1.

def _detect_colab_tier():
    """Return one of ('t4', 'l4', 'a100_40', 'a100_80', 'unknown') or None."""
    forced = bool(os.environ.get("COLAB", "").strip() not in ("", "0", "false", "False"))
    try:
        import torch
        if not torch.cuda.is_available():
            return "unknown" if forced else None
        name = torch.cuda.get_device_name(0).lower()
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if "a100" in name:
            return "a100_80" if mem_gb > 60 else "a100_40"
        if "l4" in name:
            return "l4"
        if "t4" in name:
            return "t4"
        return "unknown" if forced else None
    except Exception:
        return "unknown" if forced else None


_TIER = _detect_colab_tier()

# Drop reasoning-distilled models on Colab — the </think> / scratchpad
# pathology lives there and the base agent's _SCRATCHPAD_RE is a partial
# mitigation, not a fix. Plain -Instruct variants are cleaner.
_MODEL_BY_TIER = {
    "t4":      "Qwen/Qwen2.5-7B-Instruct",
    "l4":      "Qwen/Qwen2.5-14B-Instruct",
    "a100_40": "Qwen/Qwen2.5-32B-Instruct",
    "a100_80": "Qwen/Qwen2.5-32B-Instruct",
    "unknown": "Qwen/Qwen2.5-7B-Instruct",
}

_DTYPE_BY_TIER = {
    "t4":      "float16",
    "l4":      "float16",
    "a100_40": "bfloat16",
    "a100_80": "bfloat16",
    "unknown": "float16",
}

# Chat-template identifiers per model. vLLM picks up the HF chat template
# from the tokenizer automatically; this dict is informational + a check
# against typos in the manifest.
CHAT_TEMPLATES = {
    "Qwen/Qwen2.5-7B-Instruct":  "qwen",
    "Qwen/Qwen2.5-14B-Instruct": "qwen",
    "Qwen/Qwen2.5-32B-Instruct": "qwen",
}

# ---------------------------------------------------------------------------
# Model & hardware
# ---------------------------------------------------------------------------

# Default (laptop / non-Colab) model: GGUF-friendly DeepSeek 7B.
# Override with the SWARM_MODEL env var.
_DEFAULT_LAPTOP_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME = (
    os.environ.get("SWARM_MODEL")
    or (_MODEL_BY_TIER[_TIER] if _TIER is not None else _DEFAULT_LAPTOP_MODEL)
)

# vLLM dtype: bfloat16 on A100 (faster on Ampere), float16 elsewhere.
VLLM_DTYPE = os.environ.get("VLLM_DTYPE") or _DTYPE_BY_TIER.get(_TIER or "unknown", "float16")

# Sized for a 6GB laptop GPU running a small model in 4-bit NF4.
LLM_CONCURRENCY = 1

# Mock mode: set MOCK_LLM=1 to skip model loading entirely (CI / dev).
USE_MOCK_LLM = os.environ.get("MOCK_LLM", "").strip() not in ("", "0", "false", "False")

# Per-agent generation length budget.
MAX_TOKENS_SCOUT = 100
MAX_TOKENS_FORAGER = 160
MAX_TOKENS_CRITIC = 120
MAX_TOKENS_HATER = 160
MAX_TOKENS_VALIDATOR = 80
# Per-cluster call budget for the synthesizer. Each surviving/contested cluster
# gets its own LLM call capped at this value; total output scales with cluster
# count (target 1200-1500 tokens across 2-4 clusters).
# §6d: Raised from 400 to 600. Truncations in the faithfulness audit were caused
# by the 400-token cap cutting paragraphs mid-sentence. Legacy value preserved.
MAX_TOKENS_SYNTHESIZER_LEGACY = 400
MAX_TOKENS_SYNTHESIZER = 600

# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

NUM_SCOUTS = 4#*5
NUM_FORAGERS = 4#*5
NUM_CRITICS = 2#*5
NUM_HATERS = 2#*5
NUM_VALIDATORS = 1#*5
NUM_ROUNDS = 3#*8
ITERATIONS_PER_ROUND = 8*2

# Scout saturation cap: stop after this many successful INITIAL deposits per round.
# Prevents scouts from burning 6-7 iterations generating near-duplicate claims.
# The 8-iteration ceiling (ITERATIONS_PER_ROUND) still applies as an absolute cap.
SCOUT_MAX_DEPOSITS_PER_ROUND = 3#*5

# Characters from a scout's last successful deposit to include in the re-seed hint.
# Steers the scout away from its most recent claim without injecting other agents' work.
SCOUT_RESEED_CHARS = 175

# ---------------------------------------------------------------------------
# Signal store dynamics
# ---------------------------------------------------------------------------

DECAY_RATE = 0.05
PRUNE_THRESHOLD = 0.30  # was 0.15; signals with no corroboration now actually leave (m12)
AMPLIFY_FACTOR = 1.15  # was 1.3; lowered to slow strength saturation (M6/R4 partial)
EXPLORATION_BONUS = 0.3
DIVERSITY_THRESHOLD = 0.85

BOOST_THRESHOLD = 0.7
BOOST_BETA = 0.2

# ---------------------------------------------------------------------------
# Logit-space dynamics (DEFERRED.md P2.1 / R4 / M6)
# ---------------------------------------------------------------------------
# Strength is stored in [0,1] but updated in logit space via additive deltas.
# This avoids the saturation problem of repeated multiplicative amplification
# (strengths pinned at 1.0 lose all ordering) and — critically — makes
# contrarian signals subject to a real, ongoing decay so dissent_pressure
# does not accumulate indefinitely.
#
# Set USE_LOGIT_DYNAMICS = False to fall back to the legacy multiplicative
# code path for A/B comparison (one-release escape hatch).
USE_LOGIT_DYNAMICS = True
DELTA_DECAY = -0.10              # per-round logit decrement for non-contrarian signals
DELTA_DECAY_CONTRARIAN = -0.04   # contrarians decay slower but DO decay
DELTA_AMPLIFY = 0.30             # per-corroboration logit increment
DELTA_DEDUP_AMPLIFY = 0.10       # per-dedup-hit logit increment
DELTA_BOOST_BETA = 0.60          # provenance boost: delta = BOOST_BETA * avg_ver_strength

# Maximum number of surviving clusters the synthesizer renders in full (§6a).
# Tail clusters beyond this limit appear in Section 3 (filtered) only.
MAX_RENDERED_CLUSTERS = 4*8

assert isinstance(USE_LOGIT_DYNAMICS, bool)
assert DELTA_DECAY < 0 and DELTA_DECAY_CONTRARIAN < 0
assert DELTA_DECAY <= DELTA_DECAY_CONTRARIAN  # non-contrarians decay at least as fast
assert DELTA_AMPLIFY > 0 and DELTA_DEDUP_AMPLIFY > 0 and DELTA_BOOST_BETA > 0

# ---------------------------------------------------------------------------
# Intake / partitioning
# ---------------------------------------------------------------------------

CHUNK_WORDS = 600
CHUNK_OVERLAP = 80
CHUNKS_PER_SCOUT_MAX = 4*2

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

assert 0.0 < DECAY_RATE < 1.0
assert 0.0 < PRUNE_THRESHOLD < 1.0
assert PRUNE_THRESHOLD >= DECAY_RATE
assert AMPLIFY_FACTOR >= 1.0
assert 0.0 <= EXPLORATION_BONUS <= 1.0
assert 0.0 <= DIVERSITY_THRESHOLD <= 1.0
assert 0.0 <= BOOST_THRESHOLD <= 1.0
assert 0.0 <= BOOST_BETA <= 1.0
assert NUM_SCOUTS > 0 and NUM_FORAGERS > 0
assert NUM_CRITICS >= 0 and NUM_HATERS >= 0 and NUM_VALIDATORS >= 0
assert NUM_ROUNDS > 0 and ITERATIONS_PER_ROUND > 0
assert CHUNK_WORDS > 0 and CHUNK_OVERLAP >= 0 and CHUNK_OVERLAP < CHUNK_WORDS
assert LLM_CONCURRENCY >= 1
assert SCOUT_MAX_DEPOSITS_PER_ROUND >= 1
assert SCOUT_RESEED_CHARS >= 0

# ---------------------------------------------------------------------------
# Heterogeneous model routing (Pattern 1: sequential per-phase loading)
# ---------------------------------------------------------------------------

import json as _json  # local alias to avoid clobbering other json imports
from pathlib import Path as _Path

# Default: homogeneous — every role uses SWARM_MODEL.
USE_HETEROGENEOUS = False

# Models directory. Override with SWARM_MODELS_DIR env var.
MODELS_DIR = _Path(os.environ.get("SWARM_MODELS_DIR", "models")).resolve()

# Default heterogeneous assignment. Loaded from configs/heterogeneous.json
# at startup when --heterogeneous is passed. Keep these path fragments —
# the loader joins them with MODELS_DIR.
DEFAULT_HETEROGENEOUS_ASSIGNMENT = {
    "scout":        "Qwen2.5-7B-Instruct-Q5_K_M.gguf",
    "forager":      "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
    "developer":    "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",   # alias if §5 rename landed
    "critic":       "Phi-3.5-mini-instruct-Q4_K_M.gguf",
    "hater":        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "validator":    "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
    "synthesizer":  "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
}
