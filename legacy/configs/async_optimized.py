"""Optimized async configuration preset for maximum performance."""

import torch

# Auto-detect CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# Async configuration with all optimizations enabled
ASYNC_OPTIMIZED_CONFIG = {
    # Async execution settings
    "enable_async": True,
    "max_concurrent_agents": 5,          # Run 5 agents in parallel (adjust based on GPU)

    # Model optimization (CRITICAL for memory)
    "lazy_load": True,                   # Instant startup
    "use_quantization": CUDA_AVAILABLE,  # Enable 4-bit quantization only if CUDA available
    "quantization_bits": 4,              # 4-bit = ~3.5GB, 8-bit = ~7GB, full = ~14GB
    "enable_caching": True,              # Cache LLM responses

    # Model settings
    "model_name": "microsoft/phi-2",     # 2.7B params, better reasoning
    "device": "cuda" if CUDA_AVAILABLE else "cpu",
    "max_tokens": 512,
    "default_temperature": 0.7,

    # Agent population
    "max_agents_per_type": 5,
    "initial_agents_per_type": 2,

    # Termination conditions
    "max_total_actions": 100,
    "max_runtime_minutes": 20,
    "min_claims_for_convergence": 10,
    "required_evidence_per_claim": 2,
    "required_critiques_per_claim": 1,

    # Performance thresholds
    "spawn_threshold": 0.75,
    "death_threshold": 0.25,
    "ema_alpha": 0.15,
}


# Fast testing configuration (small limits, no quantization)
FAST_TEST_CONFIG = {
    # Async execution settings
    "enable_async": True,
    "max_concurrent_agents": 3,

    # Model optimization
    "lazy_load": True,
    "use_quantization": False,           # Disable for faster startup
    "quantization_bits": 4,
    "enable_caching": True,

    # Model settings
    "model_name": "microsoft/phi-2",     # 2.7B params, better reasoning
    "device": "cuda" if CUDA_AVAILABLE else "cpu",
    "max_tokens": 512,
    "default_temperature": 0.7,

    # Agent population
    "max_agents_per_type": 3,
    "initial_agents_per_type": 1,        # Minimal population

    # Termination conditions (reduced for fast testing)
    "max_total_actions": 20,             # Very small for quick tests
    "max_runtime_minutes": 5,
    "min_claims_for_convergence": 3,
    "required_evidence_per_claim": 1,
    "required_critiques_per_claim": 1,

    # Performance thresholds
    "spawn_threshold": 0.75,
    "death_threshold": 0.25,
    "ema_alpha": 0.15,
}


# CPU-only configuration (no quantization, slower but works everywhere)
CPU_CONFIG = {
    # Async execution settings
    "enable_async": True,
    "max_concurrent_agents": 3,          # Lower concurrency for CPU

    # Model optimization
    "lazy_load": True,
    "use_quantization": False,           # Quantization requires CUDA
    "quantization_bits": 4,
    "enable_caching": True,

    # Model settings
    "model_name": "microsoft/phi-2",     # 2.7B params, better reasoning
    "device": "cpu",                     # Force CPU mode
    "max_tokens": 512,
    "default_temperature": 0.7,

    # Agent population
    "max_agents_per_type": 4,
    "initial_agents_per_type": 2,

    # Termination conditions
    "max_total_actions": 30,
    "max_runtime_minutes": 30,           # CPU is slower
    "min_claims_for_convergence": 5,
    "required_evidence_per_claim": 2,
    "required_critiques_per_claim": 1,

    # Performance thresholds
    "spawn_threshold": 0.75,
    "death_threshold": 0.25,
    "ema_alpha": 0.15,
}


# Maximum quality configuration (no shortcuts, slower but best results)
MAX_QUALITY_CONFIG = {
    # Async execution settings
    "enable_async": True,
    "max_concurrent_agents": 3,          # Lower concurrency for stability

    # Model optimization
    "lazy_load": True,
    "use_quantization": False,           # Full precision for best quality
    "quantization_bits": 4,
    "enable_caching": False,             # No caching for variety

    # Model settings
    "model_name": "microsoft/phi-2",     # 2.7B params, better reasoning
    "device": "cuda" if CUDA_AVAILABLE else "cpu",
    "max_tokens": 512,
    "default_temperature": 0.7,

    # Agent population
    "max_agents_per_type": 6,
    "initial_agents_per_type": 3,

    # Termination conditions (higher for quality)
    "max_total_actions": 200,            # More iterations
    "max_runtime_minutes": 60,
    "min_claims_for_convergence": 20,    # More claims
    "required_evidence_per_claim": 3,    # More evidence
    "required_critiques_per_claim": 2,   # More critiques

    # Performance thresholds
    "spawn_threshold": 0.8,              # Higher bar for spawning
    "death_threshold": 0.2,              # Lower bar for death
    "ema_alpha": 0.15,
}
