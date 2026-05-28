"""Configuration for the cleaned stigmergic implementation."""

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Colab tier detection — drives model selection, populations, dtype, etc.
# ---------------------------------------------------------------------------
# Detect GPU tier so we can pick reasonable defaults for Colab T4 / L4 / A100
# without forcing the user to set every knob. _TIER is None off-Colab (laptop /
# CPU dev), or "t4" | "l4" | "a100_40" | "a100_80" | "unknown" on Colab.
# Force the Colab path on a non-Colab host with COLAB=1.

_VALID_TIERS = ("t4", "l4", "a100_40", "a100_80", "h100", "blackwell", "unknown")


def _detect_colab_tier():
    """Return one of ('t4', 'l4', 'a100_40', 'a100_80', 'h100', 'blackwell', 'unknown') or None.

    SWARM_TIER env var overrides auto-detection entirely. Set it in a Colab
    cell when the GPU name doesn't match the expected patterns:
        import os; os.environ["SWARM_TIER"] = "a100_80"

    'blackwell' covers both consumer Blackwell (RTX 50-series, sm_120) and
    datacenter Blackwell (B100/B200, sm_100). The distinction doesn't matter
    for our purposes — both need FlashInfer disabled when paired with
    CUDA < 12.9 (the common Colab state as of 2026-Q2).
    """
    # Manual override — highest priority.
    forced_tier = os.environ.get("SWARM_TIER", "").strip().lower()
    if forced_tier:
        if forced_tier in _VALID_TIERS:
            print(f"[config] SWARM_TIER override: tier={forced_tier!r}")
            return forced_tier
        print(f"[config] WARNING: SWARM_TIER={forced_tier!r} is not a recognised tier "
              f"({_VALID_TIERS}); ignoring and auto-detecting.")

    forced = bool(os.environ.get("COLAB", "").strip() not in ("", "0", "false", "False"))
    try:
        import torch
        if not torch.cuda.is_available():
            return "unknown" if forced else None
        name = torch.cuda.get_device_name(0).lower()
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Name-based detection first (fast, deterministic).
        if "a100" in name:
            return "a100_80" if mem_gb > 60 else "a100_40"
        if "h100" in name or "h200" in name:
            return "h100"
        if "b100" in name or "b200" in name or "blackwell" in name:
            return "blackwell"
        if "l4" in name:
            return "l4"
        if "t4" in name:
            return "t4"
        # Compute-capability fallback for unrecognized names. Major sm version
        # maps to architecture: 7 = Volta, 7.5 = Turing (T4), 8.0 = Ampere
        # (A100), 8.9 = Ada (L4/RTX 40xx), 9.0 = Hopper (H100/H200),
        # 10.x = Blackwell datacenter (B100/B200), 12.x = Blackwell consumer
        # (RTX 50-series).
        try:
            major, _minor = torch.cuda.get_device_capability(0)
        except Exception:
            major = None
        if major is not None:
            if major >= 10:
                return "blackwell"
            if major == 9:
                return "h100"
            if major == 8 and mem_gb > 30:
                return "a100_80" if mem_gb > 60 else "a100_40"
        return "unknown" if forced else None
    except Exception:
        return "unknown" if forced else None


_TIER = _detect_colab_tier()

# Drop reasoning-distilled models on Colab — the </think> / scratchpad
# pathology lives there and the base agent's _SCRATCHPAD_RE is a partial
# mitigation, not a fix. Plain -Instruct variants are cleaner.
_MODEL_BY_TIER = {
    "t4":        "Qwen/Qwen2.5-7B-Instruct",
    "l4":        "Qwen/Qwen2.5-14B-Instruct",
    # A100 runs 32B fp16. max_model_len=4096 (Phase 1B) + enforce_eager=False
    # let the 32B path fit on a100_80 without per-call truncation.
    "a100_40":   "Qwen/Qwen2.5-32B-Instruct",
    "a100_80":   "Qwen/Qwen2.5-32B-Instruct",
    "h100":      "Qwen/Qwen2.5-32B-Instruct",
    "blackwell": "Qwen/Qwen2.5-32B-Instruct",
    "unknown":   "Qwen/Qwen2.5-7B-Instruct",
}

_DTYPE_BY_TIER = {
    "t4":        "float16",
    "l4":        "float16",
    "a100_40":   "bfloat16",
    "a100_80":   "bfloat16",
    "h100":      "bfloat16",
    "blackwell": "float16",   # Blackwell + older CUDA on Colab; fp16 is safer than bf16
    "unknown":   "float16",
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

# Maximum cumulative support_diversity boost a KB prior-consensus entry may
# contribute to a matching cluster across all runs. Caps the rich-get-richer
# dynamic: without this, a cluster that survives N times accumulates N×(+1)
# to its stored support_diversity in the KB and effectively never faces the
# weakly_supported gate again. Empirically calibrate once real-retriever runs
# accumulate; 4 is a conservative floor (= ~SURVIVAL_BROAD_SUPPORT).
MAX_KB_DIVERSITY_BOOST = 4

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
# Interaction-based aging (replacing pure wall-clock decay)
# ---------------------------------------------------------------------------
# Signals younger than MIN_INTERACTIONS_BEFORE_DECAY pool iterations skip
# decay AND don't contribute to dissent_pressure. Wall-clock decay punished
# late-deposited signals for the swarm's finite budget — they couldn't
# accumulate supports fast enough before the run ended. Interaction-based
# aging measures "field activity during the signal's lifetime"; youth grace
# adds a strict floor so signals always get N iterations to mature.
#
# Calibrated against pool throughput (~50–200 iters/sec on Colab A100):
# 5 iterations ≈ tens of milliseconds of grace, which is enough for several
# concurrent workers to read and respond before decay can bite. Bare
# integer defaults here; the env-var override block at the bottom of the
# file widens these to SWARM_MIN_INTERACTIONS_BEFORE_DECAY /
# SWARM_INTERACTION_AGE_HALFLIFE once _int_env is defined.
MIN_INTERACTIONS_BEFORE_DECAY = 5
INTERACTION_AGE_HALFLIFE = 200
assert MIN_INTERACTIONS_BEFORE_DECAY >= 0
assert INTERACTION_AGE_HALFLIFE > 0

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
    "t4":        {"scouts": 6,  "foragers": 6,  "critics": 3, "haters": 2, "validators": 2},
    "l4":        {"scouts": 8,  "foragers": 8,  "critics": 4, "haters": 3, "validators": 2},
    "a100_40":   {"scouts": 10, "foragers": 10, "critics": 4, "haters": 4, "validators": 3},
    "a100_80":   {"scouts": 12, "foragers": 12, "critics": 5, "haters": 4, "validators": 3},
    "h100":      {"scouts": 12, "foragers": 12, "critics": 5, "haters": 4, "validators": 3},
    "blackwell": {"scouts": 12, "foragers": 12, "critics": 5, "haters": 4, "validators": 3},
    "unknown":   {"scouts": 6,  "foragers": 6,  "critics": 3, "haters": 2, "validators": 2},
}

_ITER_BY_TIER = {
    "t4":        10,
    "l4":        8,
    "a100_40":   8,
    "a100_80":   6,
    "h100":      6,
    "blackwell": 6,
    "unknown":   10,
}

_CHUNKS_BY_TIER = {
    "t4":        4,
    "l4":        3,
    "a100_40":   3,
    "a100_80":   2,
    "h100":      2,
    "blackwell": 2,
    "unknown":   4,
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
    # Token budgets: A100 path uses max_model_len=8192/16384 (Phase: context
    # bump) so per-cluster synthesizer calls can produce richer paragraphs
    # without bumping the input ceiling. Smaller tiers (t4/l4) keep tighter
    # caps because their KV cache cannot absorb the larger generations.
    if _TIER in ("a100_40", "a100_80", "h100", "blackwell"):
        MAX_TOKENS_SCOUT       = 200
        MAX_TOKENS_FORAGER     = 300
        MAX_TOKENS_CRITIC      = 200
        MAX_TOKENS_HATER       = 300
        MAX_TOKENS_VALIDATOR   = 200   # 150 -> 200 (engages/quality JSON + reasoning fits)
        MAX_TOKENS_SYNTHESIZER = 2000  # 1500 -> 2000 (anti-parametric prompt is longer)
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
# Fix S — Search diversity
# ---------------------------------------------------------------------------
N_FACETS = 6                   # facets generated by FacetPlanner per run
MMR_LAMBDA = 0.7               # λ for Maximal Marginal Relevance reranking
MAX_CHUNKS_PER_SOURCE = 2      # source-diversity cap per partition
HYBRID_RETRIEVAL = True        # enable BM25+dense RRF combination
RRF_K = 60                     # RRF smoothing constant
WEB_PARTITION_COUNT = 2        # scouts assigned dedicated web-search partitions
TAVILY_QUERIES_PER_RUN = 4     # cap on live Tavily calls per run (free-tier guard)

# ---------------------------------------------------------------------------
# Fix C' — ClusterRegistry (deposit-time clustering)
# ---------------------------------------------------------------------------
CLUSTER_JOIN_THRESHOLD = 0.88     # cosine sim above which a signal joins an existing cluster
CLUSTER_SPLIT_THRESHOLD = 0.78    # sim below which a member is ejected during reanchor
CLUSTER_REANCHOR_EVERY = 8        # recompute medoid every N deposits into a cluster
CLUSTERING_ENABLED_TYPES = {"INITIAL", "SUPPORT", "OBJECTION"}

# ---------------------------------------------------------------------------
# Stigmergy upgrade flags (Gaps 1–4)
# ---------------------------------------------------------------------------

# Gap 1: cluster-aware sampling — workers follow semantic trails instead of
# sampling from a flat list of all INITIAL signals.
USE_CLUSTER_AWARE_SAMPLING: bool = True

# Gap 2: trail amplification — when a SUPPORT lands against a cluster,
# sibling members receive a small logit-space boost.
USE_TRAIL_AMPLIFICATION: bool = True
# Fraction of DELTA_AMPLIFY applied to cluster siblings when a SUPPORT lands.
# Scaled by 1/cluster_size so large clusters don't compound unboundedly.
DELTA_CLUSTER_TRAIL: float = 0.03

# Gap 3: local action bias — local cluster field state biases choose_action()
# per-worker rather than relying solely on global share enforcement.
USE_LOCAL_ACTION_BIAS: bool = True
LOCAL_BIAS_DEVELOP_LOW_DIVERSITY: float = 2.0   # cluster support_diversity < 2
LOCAL_BIAS_REFINE_HAS_DISSENT: float = 2.0      # cluster has OBJECTION/CRITIQUE_NEG
LOCAL_BIAS_VALIDATE_NO_VERIFY: float = 2.0      # cluster has no VERIFICATION
LOCAL_BIAS_CHAIN_DEEP_SUPPORT: float = 1.5      # cluster support_count >= 3

# Gap 4: worker semantic position — running centroid of own deposit embeddings
# so workers can specialize into niches and cluster sampling is position-biased.
USE_WORKER_POSITION: bool = True
WORKER_POSITION_WINDOW: int = 8

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

# Per-task survival profile. For factual tasks (coding has unit-test ground
# truth) the credibility gate requires external verification or dissent or
# broad support. For non-factual tasks (debate, analysis, problem_solving,
# creative) external sources don't corroborate philosophical or interpretive
# claims — survival is instead achievable through internal coherence:
# support_depth >= credibility_chain_depth counts as a credibility signal,
# and the verification requirement is dropped entirely.
SURVIVAL_TASK_PROFILES = {
    # coding: requires external test verification (ground truth exists).
    # support_diversity_min=2 because only DEVELOP+CHAIN action types naturally
    # appear for code — the global default of 3 trapped 86% of clusters at the
    # weakly_supported gate in a 319-iteration run that produced zero output.
    "coding":          {"requires_verification": True,  "credibility_chain_depth": 999,
                        "support_diversity_min": 2},
    "debate":          {"requires_verification": False, "credibility_chain_depth": 3},
    "analysis":        {"requires_verification": False, "credibility_chain_depth": 3},
    "problem_solving": {"requires_verification": False, "credibility_chain_depth": 3},
    "creative":        {"requires_verification": False, "credibility_chain_depth": 2},
}
# Default profile for unknown task types: behave like analysis.
SURVIVAL_DEFAULT_PROFILE = {"requires_verification": False, "credibility_chain_depth": 3}
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
N_FACETS              = _int_env(  "SWARM_N_FACETS",              N_FACETS)
MMR_LAMBDA            = _float_env("SWARM_MMR_LAMBDA",            MMR_LAMBDA)
MAX_CHUNKS_PER_SOURCE = _int_env(  "SWARM_MAX_CHUNKS_PER_SOURCE", MAX_CHUNKS_PER_SOURCE)
RRF_K                 = _int_env(  "SWARM_RRF_K",                 RRF_K)
WEB_PARTITION_COUNT   = _int_env(  "SWARM_WEB_PARTITION_COUNT",   WEB_PARTITION_COUNT)
TAVILY_QUERIES_PER_RUN= _int_env(  "SWARM_TAVILY_QUERIES_PER_RUN",TAVILY_QUERIES_PER_RUN)
HYBRID_RETRIEVAL      = os.environ.get("SWARM_HYBRID_RETRIEVAL", "").strip() not in ("0", "false", "False") if os.environ.get("SWARM_HYBRID_RETRIEVAL") else HYBRID_RETRIEVAL
CLUSTER_JOIN_THRESHOLD           = _float_env("SWARM_CLUSTER_JOIN_THRESHOLD",           CLUSTER_JOIN_THRESHOLD)
CLUSTER_SPLIT_THRESHOLD          = _float_env("SWARM_CLUSTER_SPLIT_THRESHOLD",          CLUSTER_SPLIT_THRESHOLD)
CLUSTER_REANCHOR_EVERY           = _int_env(  "SWARM_CLUSTER_REANCHOR_EVERY",           CLUSTER_REANCHOR_EVERY)

# ---------------------------------------------------------------------------
# Fix P — Planner (position-space SynthesisPlan)
# ---------------------------------------------------------------------------
RENDER_K             = 3    # max clusters rendered in Section 1 (position synthesis)
DISSENT_K            = 3    # max clusters rendered in Section 2 (dissent)
PLANNER_MMR_LAMBDA   = 0.6  # λ for MMR cluster diversity selection in build_plan()
VERIFICATION_WEIGHT  = 0.5  # weight of verification_score in composite cluster score
DISSENT_WEIGHT       = 0.3  # weight of dissent_pressure penalty in composite score

RENDER_K            = _int_env(  "SWARM_RENDER_K",            RENDER_K)
DISSENT_K           = _int_env(  "SWARM_DISSENT_K",           DISSENT_K)
PLANNER_MMR_LAMBDA  = _float_env("SWARM_PLANNER_MMR_LAMBDA",  PLANNER_MMR_LAMBDA)
VERIFICATION_WEIGHT = _float_env("SWARM_VERIFICATION_WEIGHT", VERIFICATION_WEIGHT)
DISSENT_WEIGHT      = _float_env("SWARM_DISSENT_WEIGHT",      DISSENT_WEIGHT)

assert RENDER_K >= 1
assert DISSENT_K >= 0
assert 0.0 < PLANNER_MMR_LAMBDA < 1.0
assert 0.0 <= VERIFICATION_WEIGHT <= 1.0
assert 0.0 <= DISSENT_WEIGHT <= 1.0

# ---------------------------------------------------------------------------
# Fix R — Renderer guards
# ---------------------------------------------------------------------------
RENDER_POSITION_MAX_WORDS = 80   # hard word cap for Section-1 position paragraphs
RENDER_DISSENT_MAX_WORDS  = 90   # hard word cap for Section-2 dissent paragraphs

RENDER_POSITION_MAX_WORDS = _int_env("SWARM_RENDER_POSITION_MAX_WORDS", RENDER_POSITION_MAX_WORDS)
RENDER_DISSENT_MAX_WORDS  = _int_env("SWARM_RENDER_DISSENT_MAX_WORDS",  RENDER_DISSENT_MAX_WORDS)

assert RENDER_POSITION_MAX_WORDS >= 20, (
    f"RENDER_POSITION_MAX_WORDS={RENDER_POSITION_MAX_WORDS} is implausibly small"
)
assert RENDER_DISSENT_MAX_WORDS >= 20, (
    f"RENDER_DISSENT_MAX_WORDS={RENDER_DISSENT_MAX_WORDS} is implausibly small"
)

# Interaction-based aging knobs (env override).
MIN_INTERACTIONS_BEFORE_DECAY    = _int_env("SWARM_MIN_INTERACTIONS_BEFORE_DECAY", MIN_INTERACTIONS_BEFORE_DECAY)
INTERACTION_AGE_HALFLIFE         = _int_env("SWARM_INTERACTION_AGE_HALFLIFE",      INTERACTION_AGE_HALFLIFE)

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
assert 0.0 < CLUSTER_JOIN_THRESHOLD < 1.0
assert 0.0 < CLUSTER_SPLIT_THRESHOLD < CLUSTER_JOIN_THRESHOLD
assert CLUSTER_REANCHOR_EVERY >= 1

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

# Config file for heterogeneous routing. Mutated at runtime by --small so
# the router picks up the small assignment without an env var.
HETEROGENEOUS_CONFIG_FILE: str = "heterogeneous.json"

# Compact single model used by --small on the single-model path and as the
# backbone of the small vLLM bundles.
SMALL_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"

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


# ---------------------------------------------------------------------------
# Task-conditional model bundles (multi-engine resident routing)
# ---------------------------------------------------------------------------
# A "bundle" is a coherent set of vLLM engines loaded simultaneously and
# routed per-role for a given task type. Unlike DEFAULT_HETEROGENEOUS_ASSIGNMENT
# (sequential swap), bundle engines all stay resident for the run; the
# MultiEngineRouter dispatches each Worker.generate() call to the engine its
# role implies. Fits the A100 80 GB envelope.
#
# Each engine kwargs dict is forwarded to VLLMBackend(__init__); recognised
# keys: model, dtype, quantization, max_num_seqs, max_model_len,
# gpu_memory_utilization, enforce_eager, kv_cache_dtype, enable_chunked_prefill,
# speculative_config, plus any extra_engine_args.

MODEL_BUNDLES = {
    "debate_analysis": {
        # Per-engine gpu_memory_utilization is PINNED here because the
        # auto-budget can't see model size. 14B fp16 needs ~35% of an 80 GB
        # GPU just for weights; 32B AWQ needs ~22%; 7B fp16 needs ~20%.
        # Sum ~77%, leaving ~3% slack for activations and the speculative
        # draft (when re-enabled).
        "primary":   {"model": "Qwen/Qwen2.5-14B-Instruct",       "dtype": "float16",
                      "max_num_seqs": 24, "max_model_len": 4096,
                      "gpu_memory_utilization": 0.35},
        # Default reasoner is Qwen2.5-32B-Instruct-AWQ (official Qwen org,
        # vocab 152064 → matches the 14B primary, so the dissent-routing
        # path can share tokenizer state if we ever turn speculative
        # decoding back on). Previously defaulted to
        # casperhansen/QwQ-32B-Preview-AWQ, which preserved QwQ reasoning
        # but leaked mixed-language tokens (Chinese 发达 inside English
        # objections) — the Preview checkpoint is not production-ready.
        # Override via env to opt back into QwQ if you accept the artifact.
        "reasoner":  {"model": os.environ.get(
                          "SWARM_REASONER_MODEL",
                          "Qwen/Qwen2.5-32B-Instruct-AWQ"),
                      "dtype": "float16",
                      "quantization": "awq", "max_num_seqs": 12, "max_model_len": 4096,
                      "gpu_memory_utilization": 0.22},
        "fast":      {"model": "Qwen/Qwen2.5-7B-Instruct",        "dtype": "float16",
                      "max_num_seqs": 32, "max_model_len": 2048,
                      "gpu_memory_utilization": 0.20},
    },
    "coding": {
        "primary":   {"model": "Qwen/Qwen2.5-Coder-32B-Instruct", "dtype": "float16",
                      "quantization": "fp8", "max_num_seqs": 16, "max_model_len": 4096,
                      "gpu_memory_utilization": 0.45},
        "secondary": {"model": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
                      "dtype": "float16", "max_num_seqs": 16, "max_model_len": 4096,
                      "gpu_memory_utilization": 0.25},
        "fast":      {"model": "Qwen/Qwen2.5-Coder-7B-Instruct",  "dtype": "float16",
                      "max_num_seqs": 32, "max_model_len": 2048,
                      "gpu_memory_utilization": 0.20},
    },
    "creative": {
        "primary":   {"model": "mistralai/Mistral-Nemo-Instruct-2407", "dtype": "float16",
                      "max_num_seqs": 24, "max_model_len": 4096,
                      "gpu_memory_utilization": 0.40},
        "fast":      {"model": "Qwen/Qwen2.5-7B-Instruct",        "dtype": "float16",
                      "max_num_seqs": 32, "max_model_len": 2048,
                      "gpu_memory_utilization": 0.25},
    },
    "research_ensemble": {  # family-diverse for ensemble runs
        "primary":   {"model": "Qwen/Qwen2.5-14B-Instruct",        "dtype": "float16",
                      "gpu_memory_utilization": 0.35},
        "secondary": {"model": "meta-llama/Llama-3.1-8B-Instruct", "dtype": "float16",
                      "gpu_memory_utilization": 0.22},
        "tertiary":  {"model": "mistralai/Mistral-Nemo-Instruct-2407", "dtype": "float16",
                      "gpu_memory_utilization": 0.30},
    },
    # --small bundles: 3B-class models that fit on a T4 (16 GB) or any L4/A100.
    # Two engine slots (primary=Qwen2.5-3B, fast=Phi-3.5-mini) preserve
    # family diversity at <6 GB combined VRAM weight.
    "small": {
        "primary": {"model": "Qwen/Qwen2.5-3B-Instruct",        "dtype": "float16",
                    "max_num_seqs": 48, "max_model_len": 2048,
                    "gpu_memory_utilization": 0.55},
        "fast":    {"model": "microsoft/Phi-3.5-mini-instruct",  "dtype": "float16",
                    "max_num_seqs": 64, "max_model_len": 2048,
                    "gpu_memory_utilization": 0.30},
    },
    "small_coding": {
        "primary": {"model": "Qwen/Qwen2.5-Coder-3B-Instruct",  "dtype": "float16",
                    "max_num_seqs": 48, "max_model_len": 2048,
                    "gpu_memory_utilization": 0.55},
        "fast":    {"model": "microsoft/Phi-3.5-mini-instruct",  "dtype": "float16",
                    "max_num_seqs": 64, "max_model_len": 2048,
                    "gpu_memory_utilization": 0.30},
    },
    # A100-80GB coding bundle: primary=Qwen2.5-Coder-32B fp8 (~32 GiB),
    # fast=Qwen2.5-Coder-7B float16 (~14 GiB). Total budget 0.83 → 66 GB.
    # Used automatically as fallback when the full "coding" bundle OOMs
    # (i.e. on A100-80GB, which can't hold 32B + 16B-MoE simultaneously).
    # The MoE secondary (DeepSeek-Coder-V2-Lite) is dropped; hater routes
    # to the 7B fast engine instead — losing family diversity vs the H100
    # bundle but fitting cleanly on 80 GB.
    "a100_coding": {
        "primary": {"model": "Qwen/Qwen2.5-Coder-32B-Instruct", "dtype": "float16",
                    "quantization": "fp8", "max_num_seqs": 16, "max_model_len": 4096,
                    "gpu_memory_utilization": 0.55},
        "fast":    {"model": "Qwen/Qwen2.5-Coder-7B-Instruct",  "dtype": "float16",
                    "max_num_seqs": 32, "max_model_len": 2048,
                    "gpu_memory_utilization": 0.28},
    },
}

# Maps task_type → bundle name. Unknown task types fall back to
# "debate_analysis" (full-featured) so the router can always pick *some*
# bundle. Override via --bundle in run_swarm.py.
TASK_TO_BUNDLE = {
    "debate":          "debate_analysis",
    "analysis":        "debate_analysis",
    "problem_solving": "debate_analysis",
    "creative":        "creative",
    "coding":          "coding",
}

TASK_TO_BUNDLE_SMALL = {
    "debate":          "small",
    "analysis":        "small",
    "problem_solving": "small",
    "creative":        "small",
    "coding":          "small_coding",
}

# Tier-aware bundle overrides. Consulted by make_bundle_router() before
# TASK_TO_BUNDLE so the right bundle is selected upfront — before any
# engine is loaded — rather than attempting an in-process cascade after
# a partial OOM (at which point the already-loaded engine's VRAM cannot
# be reclaimed). Explicit --bundle=NAME on the CLI always wins.
#
# A100-80GB: the full 'coding' bundle (32B fp8 + 16B-MoE + 7B) OOMs
# because the MoE secondary needs ~29 GiB on top of the already-loaded
# 32B primary (~32 GiB). 'a100_coding' drops the MoE and fits in ~66 GB.
TASK_TO_BUNDLE_TIER_OVERRIDE: dict = {
    "a100_80": {
        "coding": "a100_coding",
    },
    "a100_40": {
        "coding":          "small_coding",
        "debate":          "small",
        "analysis":        "small",
        "problem_solving": "small",
    },
}

# Role → engine routing. Each entry maps a role name to:
#   - a string engine key ("primary"|"reasoner"|"fast"|"secondary"|"tertiary"),
#   - None → role is disabled for this bundle (the action chooser drops
#     the corresponding action),
#   - a list of strings → round-robin across engines for that role across
#     iterations (used by research_ensemble.scout).
ROLE_TO_ENGINE = {
    "debate_analysis": {
        "scout":       "primary",
        "forager":     "primary",
        "developer":   "primary",   # alias for forager
        "critic":      "reasoner",
        "hater":       "reasoner",
        "validator":   "fast",
        "synthesizer": "primary",
    },
    "coding": {
        "scout":       "primary",
        "forager":     "primary",
        "developer":   "primary",
        "critic":      "primary",
        "hater":       "secondary",   # different family for adversarial
        "validator":   "fast",
        "synthesizer": "primary",
    },
    "creative": {
        "scout":       "primary",   # Mistral-Nemo: creative generation
        "forager":     "primary",
        "developer":   "primary",
        "critic":      "fast",      # Qwen2.5-7B: different family for genuine eval diversity
        "hater":       "fast",      # Qwen2.5-7B: craft-focused adversarial (clichés, broken lines)
        "validator":   None,        # disabled: no external facts to verify in poetry
        "synthesizer": "primary",   # Mistral-Nemo: creative synthesis
    },
    "research_ensemble": {
        "scout":       ["primary", "secondary", "tertiary"],  # round-robin across families
        "forager":     "primary",
        "developer":   "primary",
        "critic":      "secondary",
        "hater":       "tertiary",
        "validator":   "primary",
        "synthesizer": "primary",
    },
    "small": {
        "scout":       "primary",
        "forager":     "fast",
        "developer":   "fast",
        "critic":      "fast",
        "hater":       "primary",   # different family from critic for adversarial diversity
        "validator":   "fast",
        "synthesizer": "primary",
    },
    "small_coding": {
        "scout":       "primary",
        "forager":     "primary",
        "developer":   "primary",
        "critic":      "fast",
        "hater":       "fast",
        "validator":   "fast",
        "synthesizer": "primary",
    },
    "a100_coding": {
        "scout":       "primary",
        "forager":     "primary",
        "developer":   "primary",
        "critic":      "fast",
        "hater":       "fast",    # no MoE secondary on A100-80; route to 7B fast
        "validator":   "fast",
        "synthesizer": "primary",
    },
}

# Per-engine SamplingParams defaults. Per-role temperature/max_tokens
# overrides apply on top via SamplingParams(**engine_defaults, **role_override).
# Llama-family engines keep a lower repetition_penalty than Qwen because Llama
# fp16 amplifies the 1.15 penalty into degenerate token loops at temp >=0.7.
SAMPLING_PER_ENGINE = {
    "primary":   {"temperature": 0.7, "top_p": 0.92, "repetition_penalty": 1.15},
    "reasoner":  {"temperature": 0.6, "top_p": 0.95, "repetition_penalty": 1.10},
    "fast":      {"temperature": 0.4, "top_p": 0.92, "repetition_penalty": 1.10},
    "secondary": {"temperature": 0.7, "top_p": 0.92, "repetition_penalty": 1.10},
    "tertiary":  {"temperature": 0.7, "top_p": 0.92, "repetition_penalty": 1.10},
}

# Action → role mapping for worker_pool. Used by the MultiEngineRouter to
# pick the right engine for each Worker.iterate() call.
ACTION_TO_ROLE = {
    "SCOUT":    "scout",
    "DEVELOP":  "forager",
    "CHAIN":    "forager",
    "CRITIQUE": "critic",
    "OBJECT":   "hater",
    "VALIDATE": "validator",
    "REFINE":   "synthesizer",
}

# Feature flag for the bundle path. Defaults off; flipped on by
# run_swarm.py main() when --bundle is passed or USE_MODEL_BUNDLES=1 env is set.
USE_MODEL_BUNDLES = os.environ.get("USE_MODEL_BUNDLES", "").strip() not in ("", "0", "false", "False")
# Active bundle name; set by run_swarm.py once task_type is parsed. None
# means "auto-pick from TASK_TO_BUNDLE[task_type]".
ACTIVE_BUNDLE: Optional[str] = None  # populated at runtime

# Speculative-decoding draft model used on the synthesizer-targeted engine.
# vLLM ≥ 0.5 supports `speculative_config` on AsyncEngineArgs. Drafts are
# verified by the primary engine, so wall-clock falls 1.8-2.2× on long
# generations (synthesizer cluster calls) with no quality regression.
#
# DEFAULT IS DISABLED. Speculative decoding requires the draft model to
# share the target model's vocabulary. Qwen2.5 splits vocab across sizes —
# 0.5B/1.5B/3B use vocab 151936, while 7B/14B/32B use 152064 — so there
# is no small Qwen2.5 model that pairs with the bundle's primary engine.
# Opt back in by setting SWARM_SPECULATIVE_DRAFT to a vocab-matched draft
# (e.g. "Qwen/Qwen2.5-7B-Instruct" for a 14B target gets ~1.4× but the
# draft is itself big enough that the win is small).
SPECULATIVE_DRAFT_MODEL = os.environ.get("SWARM_SPECULATIVE_DRAFT", "")
SPECULATIVE_NUM_TOKENS = _int_env("SWARM_SPECULATIVE_NUM_TOKENS", 5)
# Engine key that the speculative draft attaches to. "primary" is the
# default — that's the engine the synthesizer routes to in every bundle.
SPECULATIVE_TARGET_ENGINE = "primary"
