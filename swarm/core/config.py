"""Configuration for stigmergic swarm system."""

# Device detection with automatic fallback
try:
    import torch
    # Try to use CUDA if available
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print(f"[CONFIG] GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"[CONFIG] Using CUDA (device {torch.cuda.current_device()})")
    else:
        DEVICE = "cpu"
        print("[CONFIG] No GPU detected, using CPU mode")
except ImportError:
    DEVICE = "cpu"  # Fallback to CPU if torch not available
    print("[CONFIG] PyTorch not found, using CPU mode")
except Exception as e:
    DEVICE = "cpu"  # Fallback to CPU on any CUDA errors
    print(f"[CONFIG] CUDA initialization failed ({e}), falling back to CPU mode")


# Model settings
MODEL_NAME = "microsoft/phi-2"  # 2.7B params, better reasoning
MAX_TOKENS = 512  # Maximum tokens per generation (Phi-2 context window)

# Per-agent temperatures for diversity
TEMP_SCOUT = 0.9  # High exploration
TEMP_FORAGER = 0.7  # Balanced development
TEMP_CRITIC = 0.6  # Focused analysis
TEMP_HATER = 0.85  # High adversarial creativity
TEMP_SYNTHESIZER = 0.6  # Focused synthesis (lower = more consistent)

# VALIDATION: Temperature ranges
assert 0.0 <= TEMP_SCOUT <= 2.0, f"TEMP_SCOUT must be in [0, 2], got {TEMP_SCOUT}"
assert 0.0 <= TEMP_FORAGER <= 2.0, f"TEMP_FORAGER must be in [0, 2], got {TEMP_FORAGER}"
assert 0.0 <= TEMP_CRITIC <= 2.0, f"TEMP_CRITIC must be in [0, 2], got {TEMP_CRITIC}"
assert 0.0 <= TEMP_HATER <= 2.0, f"TEMP_HATER must be in [0, 2], got {TEMP_HATER}"
assert 0.0 <= TEMP_SYNTHESIZER <= 2.0, f"TEMP_SYNTHESIZER must be in [0, 2], got {TEMP_SYNTHESIZER}"

# Swarm settings
NUM_SCOUTS = 4  # Exploration agents
NUM_FORAGERS = 4  # Following agents
NUM_CRITICS = 2  # Critic agents
NUM_HATERS = 2  # Adversarial agents
NUM_VALIDATORS = 1  # Fact-checking agents (new!)
NUM_PRUNERS = 1  # Signal quality management agents (new!)
MAX_ITERATIONS = 50  # More cycles for richer debate
ITERATION_DELAY = 0.5  # Seconds between agent actions

# VALIDATION: Agent counts must be non-negative
assert NUM_SCOUTS > 0, f"NUM_SCOUTS must be positive, got {NUM_SCOUTS}"
assert NUM_FORAGERS > 0, f"NUM_FORAGERS must be positive, got {NUM_FORAGERS}"
assert NUM_CRITICS >= 0, f"NUM_CRITICS must be non-negative, got {NUM_CRITICS}"
assert NUM_HATERS >= 0, f"NUM_HATERS must be non-negative, got {NUM_HATERS}"
assert NUM_VALIDATORS >= 0, f"NUM_VALIDATORS must be non-negative, got {NUM_VALIDATORS}"
assert NUM_PRUNERS >= 0, f"NUM_PRUNERS must be non-negative, got {NUM_PRUNERS}"
assert MAX_ITERATIONS > 0, f"MAX_ITERATIONS must be positive, got {MAX_ITERATIONS}"
assert ITERATION_DELAY >= 0, f"ITERATION_DELAY must be non-negative, got {ITERATION_DELAY}"

# Signal settings
DECAY_RATE = 0.05  # Strength reduction per iteration (5%)
PRUNE_THRESHOLD = 0.15  # Remove signals below this
AMPLIFY_FACTOR = 1.3  # Boost when corroborated (30%)

# VALIDATION: Signal dynamics
assert 0.0 < DECAY_RATE < 1.0, f"DECAY_RATE must be in (0, 1), got {DECAY_RATE}"
assert 0.0 < PRUNE_THRESHOLD < 1.0, f"PRUNE_THRESHOLD must be in (0, 1), got {PRUNE_THRESHOLD}"
assert PRUNE_THRESHOLD >= DECAY_RATE, f"PRUNE_THRESHOLD ({PRUNE_THRESHOLD}) should be >= DECAY_RATE ({DECAY_RATE}) to avoid instant pruning"
assert AMPLIFY_FACTOR >= 1.0, f"AMPLIFY_FACTOR must be >= 1.0 (boost, not reduction), got {AMPLIFY_FACTOR}"

# Thresholds (lowered so foragers/critics/hater actually deposit signals)
MIN_DEPOSIT_STRENGTH = 0.3  # Only deposit if above this (lowered from 0.4)
MIN_AMPLIFY_STRENGTH = 0.4  # Only amplify if developed idea is strong (lowered from 0.5)

# VALIDATION: Strength thresholds
assert 0.0 <= MIN_DEPOSIT_STRENGTH <= 1.0, f"MIN_DEPOSIT_STRENGTH must be in [0, 1], got {MIN_DEPOSIT_STRENGTH}"
assert 0.0 <= MIN_AMPLIFY_STRENGTH <= 1.0, f"MIN_AMPLIFY_STRENGTH must be in [0, 1], got {MIN_AMPLIFY_STRENGTH}"

# Debate topic
THESIS = "Climate change requires immediate global action to prevent catastrophic warming"

# Output settings
TOP_N_RESULTS = 5  # Show top N signals at end

# Agent generation settings (max tokens per agent type)
MAX_TOKENS_SCOUT = 80  # Scout exploration output length
MAX_TOKENS_FORAGER = 150  # Forager development output length
MAX_TOKENS_CRITIC = 150  # Critic analysis output length
MAX_TOKENS_HATER = 150  # Hater counterargument output length
MAX_TOKENS_SYNTHESIZER = 200  # Synthesizer final answer length

# Diversity and exploration settings
DIVERSITY_THRESHOLD = 0.95  # Similarity threshold for duplicate detection (0.0-1.0)
                            # Higher = only reject very similar signals
EXPLORATION_BONUS = 0.3  # Weight bonus for under-visited signals (0.0-1.0)
                         # Higher = more exploration of minority opinions

# VALIDATION: Diversity and exploration bounds
assert 0.0 <= DIVERSITY_THRESHOLD <= 1.0, f"DIVERSITY_THRESHOLD must be in [0, 1], got {DIVERSITY_THRESHOLD}"
assert 0.0 <= EXPLORATION_BONUS <= 1.0, f"EXPLORATION_BONUS must be in [0, 1], got {EXPLORATION_BONUS}"

# Contrarian boost settings (offset echo chamber effects)
CONTRARIAN_BOOST = 1.10  # Multiplier for contrarian signal types during decay (1.0-1.2)
                         # 1.10 = 10% boost to critiques/counter-evidence

# LLM cache settings
LLM_CACHE_SIZE = 1000  # Number of cached LLM responses (default/analytical tasks)
ENABLE_LLM_CACHE = True  # Enable caching (disable for more diversity, slower)

# Task-specific cache settings (overrides for creative/exploratory tasks)
CREATIVE_CACHE_SIZE = 50  # Much smaller cache for creative tasks (target <50% hit rate)
CREATIVE_CACHE_ENABLED = True  # Keep enabled but very small to allow some efficiency

# Knowledge retrieval settings
ENABLE_KNOWLEDGE_RETRIEVAL = False  # Enable Wikipedia/web search for agents (experimental)
KNOWLEDGE_CACHE_SIZE = 100  # Number of cached knowledge queries

# Experimental Features (Runtime Switchable)
# These are research features that can be enabled/disabled via config
USE_SIMPLE_SCOUTS = False  # Use SimpleScouts with spatial movement (Experimental - Phase 2)
USE_SPATIAL_STORE = False  # Use SpatialSignalStore with locality constraints (Experimental - Phase 3)

# Production Features (Recommended)
USE_REAL_VALIDATOR = True  # External source validation (Recommended for research tasks)
                           # Requires: pip install requests beautifulsoup4 wikipedia duckduckgo-search

# SimpleScout settings (Phase 2)
SIMPLE_SCOUT_PERCEPTION_RADIUS = 10.0  # How far scouts can perceive signals
SIMPLE_SCOUT_MOVEMENT_SPEED = 2.0  # Movement speed per iteration
SIMPLE_SCOUT_MAX_TOKENS = 20  # Tiny LLM queries (reduced from 70)
SPATIAL_GRID_DIMENSIONS = 100  # Size of spatial grid (100×100)

# RealValidator settings (Phase 4)
DYNAMIC_KB_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to consider fact "known"
DYNAMIC_KB_MAX_FACTS = 10000  # Maximum facts to store in knowledge base

# Advanced Knowledge Retrieval (Recommended for research tasks)
USE_ADVANCED_RETRIEVER = True  # Use AdvancedRetriever with 100K+ word ingestion per round
                               # Requires: pip install requests beautifulsoup4
ADVANCED_RETRIEVAL_TARGET_WORDS = 100000  # Target words to ingest per round (100K recommended)
ADVANCED_RETRIEVAL_MIN_SOURCES = 3  # Minimum sources to check per keyword
ADVANCED_RETRIEVAL_TEMP_DIR = "research/temp"  # Temporary directory for scraped content


# Configuration validation
def validate_config():
    """Validate configuration settings for consistency.

    Raises:
        ValueError: If configuration is invalid
    """
    errors = []

    # SimpleScouts require SpatialSignalStore (scouts have positions)
    if USE_SIMPLE_SCOUTS and not USE_SPATIAL_STORE:
        errors.append(
            "Invalid config: USE_SIMPLE_SCOUTS=True requires USE_SPATIAL_STORE=True\n"
            "  Reason: SimpleScouts have spatial positions and need locality-based signal access"
        )

    # SpatialSignalStore works with both scout types (optional validation)
    # This is valid: USE_SPATIAL_STORE=True with USE_SIMPLE_SCOUTS=False
    # (regular scouts can use spatial store too)

    # RealValidator requires knowledge base (warning, not error)
    if USE_REAL_VALIDATOR:
        try:
            from ..validation.external_sources import REAL_APIS_AVAILABLE
            if not REAL_APIS_AVAILABLE:
                print("[WARNING] USE_REAL_VALIDATOR=True but real APIs not available")
                print("[WARNING] Install dependencies: pip install requests beautifulsoup4")
        except ImportError:
            print("[WARNING] USE_REAL_VALIDATOR=True but validation module not found")

    # Raise combined error if any validation failed
    if errors:
        error_msg = "Configuration validation failed:\n\n" + "\n\n".join(errors)
        raise ValueError(error_msg)


# Run validation on import
validate_config()
