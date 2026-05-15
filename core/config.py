"""Configuration for the cleaned stigmergic implementation."""

import os

# ---------------------------------------------------------------------------
# Model & hardware
# ---------------------------------------------------------------------------

# Default model: ~3 GB download, fits comfortably on a laptop with 4-5 GB free.
# Override with the SWARM_MODEL env var, e.g.:
#     $env:SWARM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"    # ~15 GB
#     $env:SWARM_MODEL = "microsoft/phi-2"                            # ~5.4 GB
#     $env:SWARM_MODEL = "Qwen/Qwen2.5-3B-Instruct"                   # ~6 GB
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

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
MAX_TOKENS_SYNTHESIZER = 400

# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

NUM_SCOUTS = 4
NUM_FORAGERS = 4
NUM_CRITICS = 2
NUM_HATERS = 2
NUM_VALIDATORS = 1
NUM_ROUNDS = 3
ITERATIONS_PER_ROUND = 8

# Scout saturation cap: stop after this many successful INITIAL deposits per round.
# Prevents scouts from burning 6-7 iterations generating near-duplicate claims.
# The 8-iteration ceiling (ITERATIONS_PER_ROUND) still applies as an absolute cap.
SCOUT_MAX_DEPOSITS_PER_ROUND = 3

# Characters from a scout's last successful deposit to include in the re-seed hint.
# Steers the scout away from its most recent claim without injecting other agents' work.
SCOUT_RESEED_CHARS = 100

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

assert isinstance(USE_LOGIT_DYNAMICS, bool)
assert DELTA_DECAY < 0 and DELTA_DECAY_CONTRARIAN < 0
assert DELTA_DECAY <= DELTA_DECAY_CONTRARIAN  # non-contrarians decay at least as fast
assert DELTA_AMPLIFY > 0 and DELTA_DEDUP_AMPLIFY > 0 and DELTA_BOOST_BETA > 0

# ---------------------------------------------------------------------------
# Intake / partitioning
# ---------------------------------------------------------------------------

CHUNK_WORDS = 600
CHUNK_OVERLAP = 80
CHUNKS_PER_SCOUT_MAX = 4

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
