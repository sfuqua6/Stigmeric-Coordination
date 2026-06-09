"""LLM Factory - Unified LLM System

This factory creates and configures the complete LLM system:
- Provider selection (vLLM, TGI, SimpleLLM)
- Provider pool with health checks
- Request batching
- Circuit breakers
- Metrics collection

Usage:
    factory = LLMFactory(config)
    llm_system = await factory.create()
    response = await llm_system.generate(request)
"""

import logging
from typing import Dict, Optional, List
from swarm.llm.provider import LLMProvider, LLMProviderType, GenerationRequest, GenerationResponse
from swarm.llm.pool import ProviderPool
from swarm.llm.batcher import RequestBatcher, AdaptiveBatcher
from swarm.llm.vllm_provider import VLLMProvider
from swarm.llm.simple_provider import SimpleLLMProvider

logger = logging.getLogger(__name__)


class LLMSystem:
    """Unified LLM system combining all strategies.

    Provides a single interface for text generation with:
    - Multiple provider support (vLLM, SimpleLLM)
    - Provider pool with health checks
    - Automatic request batching
    - Circuit breakers
    - Metrics collection
    """

    def __init__(
        self,
        pool: ProviderPool,
        batcher: Optional[RequestBatcher] = None,
        enable_batching: bool = True
    ):
        """Initialize LLM system.

        Args:
            pool: Provider pool
            batcher: Request batcher (optional)
            enable_batching: Enable request batching
        """
        self.pool = pool
        self.batcher = batcher
        self.enable_batching = enable_batching and batcher is not None

        logger.info(f"[LLM SYSTEM] Initialized (batching={'enabled' if self.enable_batching else 'disabled'})")

    async def start(self):
        """Start the LLM system."""
        await self.pool.start()

        if self.batcher:
            await self.batcher.start()

        logger.info("[LLM SYSTEM] Started")

    async def stop(self):
        """Stop the LLM system."""
        if self.batcher:
            await self.batcher.stop()

        await self.pool.stop()

        logger.info("[LLM SYSTEM] Stopped")

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from prompt.

        Args:
            request: Generation request

        Returns:
            Generation response
        """
        if self.enable_batching:
            # Use batcher for automatic batching
            return await self.batcher.generate(request)
        else:
            # Direct pool access
            return await self.pool.generate(request)

    def get_status(self) -> Dict:
        """Get system status.

        Returns:
            Status dictionary with pool and batcher info
        """
        status = {
            "pool": self.pool.get_pool_status(),
            "batching_enabled": self.enable_batching
        }

        if self.batcher:
            status["batcher"] = self.batcher.get_metrics()

        return status


class LLMFactory:
    """Factory for creating configured LLM systems.

    Handles:
    - Provider creation (vLLM, SimpleLLM)
    - Pool configuration
    - Batcher configuration
    - Automatic fallbacks
    """

    @staticmethod
    async def create(config: Dict) -> LLMSystem:
        """Create and configure complete LLM system.

        Args:
            config: Configuration dictionary with keys:
                Provider Selection:
                - provider_type: "vllm", "simple" (default: "simple")
                - model_name: HuggingFace model ID
                - device: "cuda" or "cpu"

                Pool Configuration:
                - pool_size: Number of provider instances (default: 1)
                - health_check_interval: Seconds between checks (default: 60)

                Batching Configuration:
                - enable_batching: Enable request batching (default: False)
                - batch_size: Max requests per batch (default: 8)
                - batch_timeout_ms: Batch timeout (default: 100)
                - adaptive_batching: Use adaptive batcher (default: False)

                vLLM Specific:
                - tensor_parallel_size: Number of GPUs (default: auto)
                - max_num_seqs: Max concurrent seqs (default: 256)

                SimpleLLM Specific:
                - use_quantization: Enable quantization (default: None=auto)
                - enable_cache: Enable response cache (default: True)
                - cache_size: Max cache entries (default: 100)

        Returns:
            Configured LLM system
        """
        provider_type = config.get("provider_type", "simple")
        pool_size = config.get("pool_size", 1)
        health_check_interval = config.get("health_check_interval", 60.0)
        enable_batching = config.get("enable_batching", False)

        logger.info(f"[FACTORY] Creating LLM system with provider: {provider_type}")
        logger.info(f"[FACTORY] Pool size: {pool_size}, Batching: {enable_batching}")

        # Create providers
        providers = []
        for i in range(pool_size):
            provider = await LLMFactory._create_provider(provider_type, config, instance_id=i)
            if provider:
                providers.append(provider)

        if not providers:
            logger.error("[FACTORY] Failed to create any providers")
            raise RuntimeError("No providers available")

        logger.info(f"[FACTORY] Created {len(providers)} provider(s)")

        # Create pool
        pool = ProviderPool(providers, health_check_interval=health_check_interval)

        # Create batcher if enabled
        batcher = None
        if enable_batching and len(providers) > 0:
            # Use first provider for batching
            primary_provider = providers[0]

            if config.get("adaptive_batching", False):
                batcher = AdaptiveBatcher(
                    provider=primary_provider,
                    initial_batch_size=config.get("batch_size", 8),
                    initial_timeout_ms=config.get("batch_timeout_ms", 100.0),
                    target_latency_ms=config.get("target_latency_ms", 500.0)
                )
                logger.info("[FACTORY] Created adaptive batcher")
            else:
                batcher = RequestBatcher(
                    provider=primary_provider,
                    batch_size=config.get("batch_size", 8),
                    batch_timeout_ms=config.get("batch_timeout_ms", 100.0)
                )
                logger.info("[FACTORY] Created standard batcher")

        # Create system
        system = LLMSystem(pool=pool, batcher=batcher, enable_batching=enable_batching)

        logger.info("[FACTORY] LLM system created successfully")
        return system

    @staticmethod
    async def _create_provider(provider_type: str, config: Dict, instance_id: int = 0) -> Optional[LLMProvider]:
        """Create a single provider instance.

        Args:
            provider_type: Type of provider ("vllm", "simple")
            config: Provider configuration
            instance_id: Instance number (for logging)

        Returns:
            Provider instance or None if creation failed
        """
        logger.info(f"[FACTORY] Creating provider {instance_id}: {provider_type}")

        try:
            if provider_type == "vllm":
                provider = VLLMProvider(config)

            elif provider_type == "simple":
                provider = SimpleLLMProvider(config)

            else:
                logger.error(f"[FACTORY] Unknown provider type: {provider_type}")
                return None

            # Initialize provider
            success = await provider.initialize()

            if not success:
                logger.error(f"[FACTORY] Provider {instance_id} initialization failed")
                return None

            logger.info(f"[FACTORY] Provider {instance_id} created successfully")
            return provider

        except Exception as e:
            logger.error(f"[FACTORY] Error creating provider {instance_id}: {e}")
            return None

    @staticmethod
    def create_config_preset(preset: str) -> Dict:
        """Create configuration from preset.

        Args:
            preset: Preset name ("vllm_optimal", "simple_fast", "simple_safe", "testing")

        Returns:
            Configuration dictionary
        """
        if preset == "vllm_optimal":
            # vLLM with optimal settings for maximum throughput
            return {
                "provider_type": "vllm",
                "model_name": "Qwen/Qwen2.5-7B-Instruct",
                "pool_size": 1,  # vLLM handles concurrency internally
                "enable_batching": False,  # vLLM handles batching internally
                "tensor_parallel_size": None,  # Auto-detect GPUs
                "max_num_seqs": 256,
                "gpu_memory_utilization": 0.9
            }

        elif preset == "simple_fast":
            # SimpleLLM with batching for better throughput
            return {
                "provider_type": "simple",
                "model_name": "Qwen/Qwen2.5-7B-Instruct",
                "device": "cuda",
                "pool_size": 1,
                "enable_batching": True,
                "batch_size": 8,
                "batch_timeout_ms": 100.0,
                "adaptive_batching": True,
                "use_quantization": True,
                "enable_cache": True,
                "cache_size": 100
            }

        elif preset == "simple_safe":
            # SimpleLLM with isolation (multiple instances)
            return {
                "provider_type": "simple",
                "model_name": "Qwen/Qwen2.5-7B-Instruct",
                "device": "cuda",
                "pool_size": 3,  # Multiple instances for isolation
                "enable_batching": False,
                "use_quantization": True,
                "enable_cache": True,
                "cache_size": 50
            }

        elif preset == "testing":
            # Fast configuration for testing
            return {
                "provider_type": "simple",
                "model_name": "gpt2",  # Tiny model
                "device": "cpu",
                "pool_size": 1,
                "enable_batching": False,
                "use_quantization": False,
                "enable_cache": False
            }

        else:
            raise ValueError(f"Unknown preset: {preset}")
