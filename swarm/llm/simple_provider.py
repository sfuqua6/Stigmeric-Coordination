"""SimpleLLM Provider Adapter - Legacy Compatibility

Wraps the existing SimpleLLM class in the new LLMProvider interface.
This allows gradual migration from SimpleLLM to the new architecture.
"""

import time
import logging
from typing import List, Dict
from swarm.llm.provider import (
    LLMProvider, LLMProviderType,
    GenerationRequest, GenerationResponse
)
from swarm.llm.simple_llm import SimpleLLM

logger = logging.getLogger(__name__)


class SimpleLLMProvider(LLMProvider):
    """Adapter for legacy SimpleLLM class.

    Wraps SimpleLLM to provide the new LLMProvider interface while
    maintaining backward compatibility.
    """

    def __init__(self, config: Dict):
        """Initialize SimpleLLM provider.

        Args:
            config: Configuration dictionary with keys:
                - model_name: HuggingFace model ID
                - device: "cuda" or "cpu"
                - enable_cache: Enable response caching
                - cache_size: Maximum cache entries
                - use_quantization: Enable 8-bit quantization
        """
        super().__init__(LLMProviderType.SIMPLE, config)

        self.model_name = config.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        self.device = config.get("device", "cuda")
        self.enable_cache = config.get("enable_cache", True)
        self.cache_size = config.get("cache_size", 100)
        self.use_quantization = config.get("use_quantization", None)

        self.llm: SimpleLLM = None

        logger.info(f"[SIMPLE PROVIDER] Initialized with model: {self.model_name}")

    async def initialize(self) -> bool:
        """Initialize SimpleLLM.

        Returns:
            True if successful
        """
        try:
            self.llm = SimpleLLM(
                model_name=self.model_name,
                device=self.device,
                enable_cache=self.enable_cache,
                cache_size=self.cache_size,
                use_quantization=self.use_quantization
            )

            await self.llm.load()

            self._initialized = True
            logger.info("[SIMPLE PROVIDER] SimpleLLM loaded successfully")
            return True

        except Exception as e:
            logger.error(f"[SIMPLE PROVIDER] Failed to initialize: {e}")
            return False

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from prompt.

        Args:
            request: Generation request

        Returns:
            Generation response
        """
        if not self._initialized:
            return GenerationResponse(
                text="",
                success=False,
                latency_ms=0,
                tokens_generated=0,
                error="SimpleLLM not initialized"
            )

        start_time = time.time()

        try:
            # Generate using SimpleLLM
            result = await self.llm.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                use_cache=request.use_cache
            )

            latency_ms = (time.time() - start_time) * 1000

            if result is None:
                # SimpleLLM returns None on failure
                response = GenerationResponse(
                    text="",
                    success=False,
                    latency_ms=latency_ms,
                    tokens_generated=0,
                    error="SimpleLLM generation returned None"
                )
            else:
                # Estimate tokens generated (SimpleLLM doesn't return this)
                tokens_generated = len(result.split())  # Rough estimate

                response = GenerationResponse(
                    text=result,
                    success=True,
                    latency_ms=latency_ms,
                    tokens_generated=tokens_generated,
                    cached=False  # SimpleLLM doesn't expose cache hits
                )

            self._update_metrics(response)
            return response

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"[SIMPLE PROVIDER] Generation error: {e}")

            response = GenerationResponse(
                text="",
                success=False,
                latency_ms=latency_ms,
                tokens_generated=0,
                error=str(e)
            )

            self._update_metrics(response)
            return response

    async def cleanup(self):
        """Cleanup SimpleLLM."""
        if self.llm is not None:
            try:
                await self.llm.cleanup()
            except Exception as e:
                logger.warning(f"[SIMPLE PROVIDER] Cleanup warning: {e}")

            self.llm = None
            self._initialized = False
            logger.info("[SIMPLE PROVIDER] Cleaned up")
