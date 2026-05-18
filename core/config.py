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
    # A100 uses 14B fp16: 28 GB weights leaves ~52 GB for KV cache vs. 32B's
    # 11 GB. KV pressure on 32B was forcing max_model_len=1024 and truncating
    # generations mid-sentence — capacity was being wasted.
    "a100_40": "Qwen/Qwen2.5-14B-Instruct",
    "a100_80": "Qwen/Qwen2.5-14B-Instruct",
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

# Tier-aware concurrency. On Colab/vLLM the AsyncLLMEngine batches internally
# so this is the cap on in-flight async generate() calls — bigger is fine, the
# vLLM scheduler queues. On laptop GGUF this MUST stay at 1 (llama-cpp-python
# is single-threaded per Llama instance).
LLM_CONCURRENCY = 1 if _TIER is None else 32

# Mock mode: set MOCK_LLM=1 to skip model loading entirely (CI / dev).
USE_MOCK_LLM = os.environ.get("MOCK_LLM", "").strip() not in ("", "0", "false", "False")

# Per-agent generation length budget. Raised in Phase 1F (A100 + max_model_len=4096
# supports these without truncation). Laptop overrides via SWARM_MAX_TOKENS_* env
# vars if needed; the Colab tier block below tightens these further per tier.
MAX_TOKENS_SCOUT = 200
MAX_TOKENS_FORAGER = 300
MAX_TOKENS_CRITIC = 200
MAX_TOKENS_HATER = 300
MAX_TOKENS_VALIDATOR = 150
# Per-cluster call budget for the synthesizer. Each surviving/contested cluster
# gets its own LLM call capped at this value; total output scales with cluster
# count.
MAX_TOKENS_SYNTHESIZER_LEGACY = 400
MAX_TOKENS_SYNTHESIZER = 1500

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
# Colab tier overrides (Phase E + F + G of the Colab migration)
# ---------------------------------------------------------------------------
# When a Colab tier is detected, override the laptop defaults above with
# values that exploit real concurrency: more scouts/foragers/critics/haters
# for richer cluster coverage; lower iterations per agent because the
# non-quantized model produces denser content per call; larger token
# budgets per agent; narrower per-scout partitions (smaller CHUNK_WORDS so
# more chunks exist to feed more scouts at fewer chunks each).
#
# Laptop values above are preserved verbatim — the overrides only fire when
# _TIER is not None.

_POP_BY_TIER = {
    "t4":       {"scouts": 6,  "foragers": 6,  "critics": 3, "haters": 2, "validators": 2},
    "l4":       {"scouts": 8,  "foragers": 8,  "critics": 4, "haters": 3, "validators": 2},
    "a100_40":  {"scouts": 10, "foragers": 10, "critics": 4, "haters": 4, "validators": 3},
    "a100_80":  {"scouts": 12, "foragers": 12, "critics": 5, "haters": 4, "validators": 3},
    "unknown":  {"scouts": 6,  "foragers": 6,  "critics": 3, "haters": 2, "validators": 2},
}

_ITER_BY_TIER = {
    "t4":      10,
    "l4":      8,
    "a100_40": 8,
    "a100_80": 6,
    "unknown": 10,
}

_CHUNKS_BY_TIER = {
    "t4":      4,
    "l4":      3,
    "a100_40": 3,
    "a100_80": 2,
    "unknown": 4,
}

if _TIER is not None:
    _pop = _POP_BY_TIER[_TIER]
    NUM_SCOUTS     = _pop["scouts"]
    NUM_FORAGERS   = _pop["foragers"]
    NUM_CRITICS    = _pop["critics"]
    NUM_HATERS     = _pop["haters"]
    NUM_VALIDATORS = _pop["validators"]
    # Three rounds gives logit-space strength dynamics time to differentiate
    # clusters; two rounds is too few for the contrarian-decay path to matter;
    # four is wasted wall-clock at these populations.
    NUM_ROUNDS = 3
    ITERATIONS_PER_ROUND = _ITER_BY_TIER[_TIER]
    # More scouts → fewer needed per scout to saturate the corpus partition.
    SCOUT_MAX_DEPOSITS_PER_ROUND = 2 if NUM_SCOUTS >= 8 else 3
    # Narrower partitions: keep each scout focused on a small slice.
    CHUNKS_PER_SCOUT_MAX = _CHUNKS_BY_TIER[_TIER]
    # Smaller chunks so the corpus yields enough partitions to feed the
    # larger scout population at the lower per-scout chunk count.
    CHUNK_WORDS = 400
    # Token budgets: Phase 1F raises the budgets on A100 where max_model_len=4096
    # supports them without truncating context. Smaller tiers (t4/l4) keep the
    # tighter caps because their KV cache cannot absorb the larger generations.
    if _TIER in ("a100_40", "a100_80"):
        MAX_TOKENS_SCOUT       = 200
        MAX_TOKENS_FORAGER     = 300
        MAX_TOKENS_CRITIC      = 200
        MAX_TOKENS_HATER       = 300
        MAX_TOKENS_VALIDATOR   = 150
        MAX_TOKENS_SYNTHESIZER = 1500
    else:
        MAX_TOKENS_SCOUT       = 140
        MAX_TOKENS_FORAGER     = 200
        MAX_TOKENS_CRITIC      = 150
        MAX_TOKENS_HATER       = 200
        MAX_TOKENS_VALIDATOR   = 100
        MAX_TOKENS_SYNTHESIZER = 800

    # Phase H: strength dynamics recalibrated for higher deposit volume.
    # Laptop deltas were tuned for ~30 deposits/round; Colab tiers run
    # 80–150 deposits/round, so amplification fires more often and decay
    # needs to bite harder to prevent saturation at 1.0.
    DELTA_AMPLIFY          = 0.20
    DELTA_DEDUP_AMPLIFY    = 0.07
    DELTA_DECAY            = -0.12
    DELTA_DECAY_CONTRARIAN = -0.05
    DELTA_BOOST_BETA       = 0.45

# Phase I — cluster threshold. Lower on laptop (0.55, justified by the
# in-code comment in projection.py) because quantized-model diversity at
# 0.65 fragments into 17 narrow clusters. On Colab tiers, fp16 Qwen-Instruct
# produces sufficiently distinct claims that a tighter threshold is safe.
CLUSTER_SIM_THRESHOLD = 0.72 if _TIER is not None else 0.55

# ---------------------------------------------------------------------------
# Survival-filter thresholds (consumed by core/projection.py)
# ---------------------------------------------------------------------------
# Exposed here (not in projection.py) so the notebook can override via env
# vars without editing projection.py. projection.py imports these directly.
#
#   support_diversity >= SURVIVAL_MIN_SUPPORT_DIVERSITY     -> not weakly_supported
#   dissent_pressure  >  SURVIVAL_REJECT_DISSENT_PRESSURE   -> rejected_by_field
#   dissent_pressure in [SURVIVAL_CONTEST_MIN, SURVIVAL_CONTEST_MAX] -> contested
#   credibility gate: passes if ANY of:
#     verification_score >= SURVIVAL_VERIFY_MIN
#     len(dissent_set)   >= SURVIVAL_DISSENT_MIN (=1)
#     support_diversity  >= SURVIVAL_BROAD_SUPPORT
SURVIVAL_MIN_SUPPORT_DIVERSITY = 3
SURVIVAL_REJECT_DISSENT_PRESSURE = 1.5
SURVIVAL_CONTEST_MIN = 0.5
SURVIVAL_CONTEST_MAX = 1.5
SURVIVAL_VERIFY_MIN = 0.3
SURVIVAL_DISSENT_MIN = 1
SURVIVAL_BROAD_SUPPORT = 4

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
# Env-var overrides (notebook playground knobs)
# ---------------------------------------------------------------------------
# Anything below can be overridden by setting the matching SWARM_* env var
# BEFORE the pipeline imports core.config. This is the entry point for the
# Colab playground cells — keep the names stable so the notebook can rely
# on them.

def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[config] WARNING: {name}={raw!r} is not an int; using {default}")
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[config] WARNING: {name}={raw!r} is not a float; using {default}")
        return default


# Round / iteration knobs
NUM_ROUNDS                    = _int_env("SWARM_NUM_ROUNDS",           NUM_ROUNDS)
ITERATIONS_PER_ROUND          = _int_env("SWARM_ITERATIONS_PER_ROUND", ITERATIONS_PER_ROUND)

# Agent population knobs
NUM_SCOUTS                    = _int_env("SWARM_NUM_SCOUTS",     NUM_SCOUTS)
NUM_FORAGERS                  = _int_env("SWARM_NUM_FORAGERS",   NUM_FORAGERS)
NUM_CRITICS                   = _int_env("SWARM_NUM_CRITICS",    NUM_CRITICS)
NUM_HATERS                    = _int_env("SWARM_NUM_HATERS",     NUM_HATERS)
NUM_VALIDATORS                = _int_env("SWARM_NUM_VALIDATORS", NUM_VALIDATORS)
SCOUT_MAX_DEPOSITS_PER_ROUND  = _int_env("SWARM_SCOUT_MAX_DEPOSITS", SCOUT_MAX_DEPOSITS_PER_ROUND)

# Token budgets
MAX_TOKENS_SCOUT       = _int_env("SWARM_MAX_TOKENS_SCOUT",       MAX_TOKENS_SCOUT)
MAX_TOKENS_FORAGER     = _int_env("SWARM_MAX_TOKENS_FORAGER",     MAX_TOKENS_FORAGER)
MAX_TOKENS_CRITIC      = _int_env("SWARM_MAX_TOKENS_CRITIC",      MAX_TOKENS_CRITIC)
MAX_TOKENS_HATER       = _int_env("SWARM_MAX_TOKENS_HATER",       MAX_TOKENS_HATER)
MAX_TOKENS_VALIDATOR   = _int_env("SWARM_MAX_TOKENS_VALIDATOR",   MAX_TOKENS_VALIDATOR)
MAX_TOKENS_SYNTHESIZER = _int_env("SWARM_MAX_TOKENS_SYNTHESIZER", MAX_TOKENS_SYNTHESIZER)

# Intake
CHUNK_WORDS            = _int_env("SWARM_CHUNK_WORDS",          CHUNK_WORDS)
CHUNKS_PER_SCOUT_MAX   = _int_env("SWARM_CHUNKS_PER_SCOUT_MAX", CHUNKS_PER_SCOUT_MAX)
LLM_CONCURRENCY        = _int_env("SWARM_LLM_CONCURRENCY",      LLM_CONCURRENCY)

# Survival-filter thresholds
SURVIVAL_MIN_SUPPORT_DIVERSITY   = _int_env(  "SWARM_SURVIVAL_MIN_SUPPORT_DIVERSITY",   SURVIVAL_MIN_SUPPORT_DIVERSITY)
SURVIVAL_REJECT_DISSENT_PRESSURE = _float_env("SWARM_SURVIVAL_REJECT_DISSENT_PRESSURE", SURVIVAL_REJECT_DISSENT_PRESSURE)
SURVIVAL_CONTEST_MIN             = _float_env("SWARM_SURVIVAL_CONTEST_MIN",             SURVIVAL_CONTEST_MIN)
SURVIVAL_CONTEST_MAX             = _float_env("SWARM_SURVIVAL_CONTEST_MAX",             SURVIVAL_CONTEST_MAX)
SURVIVAL_VERIFY_MIN              = _float_env("SWARM_SURVIVAL_VERIFY_MIN",              SURVIVAL_VERIFY_MIN)
SURVIVAL_BROAD_SUPPORT           = _int_env(  "SWARM_SURVIVAL_BROAD_SUPPORT",           SURVIVAL_BROAD_SUPPORT)
CLUSTER_SIM_THRESHOLD            = _float_env("SWARM_CLUSTER_SIM_THRESHOLD",            CLUSTER_SIM_THRESHOLD)

# Re-validate after overrides so a bad value crashes early rather than
# producing weird mid-run behavior.
assert NUM_SCOUTS > 0 and NUM_FORAGERS > 0
assert NUM_CRITICS >= 0 and NUM_HATERS >= 0 and NUM_VALIDATORS >= 0
assert NUM_ROUNDS > 0 and ITERATIONS_PER_ROUND > 0
assert CHUNK_WORDS > 0 and CHUNKS_PER_SCOUT_MAX >= 1
assert LLM_CONCURRENCY >= 1
assert SCOUT_MAX_DEPOSITS_PER_ROUND >= 1
assert SURVIVAL_MIN_SUPPORT_DIVERSITY >= 1
assert SURVIVAL_REJECT_DISSENT_PRESSURE >= SURVIVAL_CONTEST_MAX
assert SURVIVAL_CONTEST_MIN < SURVIVAL_CONTEST_MAX
assert 0.0 <= SURVIVAL_VERIFY_MIN <= 1.0
assert 0.0 < CLUSTER_SIM_THRESHOLD < 1.0

# ---------------------------------------------------------------------------
# Heterogeneous model routing (Pattern 1: sequential per-phase loading)
# ---------------------------------------------------------------------------

import json as _json  # local alias to avoid clobbering other json imports
from pathlib import Path as _Path

# Default: homogeneous — every role uses SWARM_MODEL.
USE_HETEROGENEOUS = False

# B1: T4 cannot run heterogeneous routing. A single 7B AWQ weighs ~4.5 GB
# and leaves room for ~3 GB of KV cache and activation memory in T4's 16 GB
# VRAM. Multiple models or even multiple LoRA adapters of significant rank
# can't co-reside. Force-off and log — defensive guard against anyone
# setting USE_HETEROGENEOUS=True via env or downstream config. The L4 and
# A100 tiers have headroom for the LoRA path (core/llm_router.py).
if _TIER == "t4" and USE_HETEROGENEOUS:
    print("[config] heterogeneous suppressed on T4 — single-model required by VRAM")
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
