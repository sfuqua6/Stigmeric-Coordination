"""LLM Provider Interface - Strategy 1: Model Serving Abstraction

This module provides the base interface for all LLM providers (vLLM, TGI, SimpleLLM).
Allows swapping between different serving strategies without changing agent code.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import time


class LLMProviderType(Enum):
    """Types of LLM providers."""
    SIMPLE = "simple"  # Legacy SimpleLLM
    VLLM = "vllm"  # vLLM inference server
    TGI = "tgi"  # Text Generation Inference
    RAY_SERVE = "ray"  # Ray Serve


@dataclass
class GenerationRequest:
    """Request for text generation."""
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    use_cache: bool = True
    request_id: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class GenerationResponse:
    """Response from text generation."""
    text: str
    success: bool
    latency_ms: float
    tokens_generated: int
    error: Optional[str] = None
    cached: bool = False
    provider_metadata: Optional[Dict] = None


@dataclass
class HealthStatus:
    """Health status of an LLM provider."""
    healthy: bool
    failures: int = 0
    successes: int = 0
    avg_latency_ms: float = 0.0
    last_error: Optional[str] = None
    last_check: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.0


class LLMProvider(ABC):
    """Base interface for all LLM providers.

    This abstraction allows implementing different serving strategies:
    - SimpleLLM: Direct model loading (legacy)
    - vLLM: High-throughput inference server
    - TGI: Production-grade inference
    - Ray Serve: Distributed inference

    All providers support:
    - Async generation
    - Health checks
    - Metrics collection
    - Error recovery
    """

    def __init__(self, provider_type: LLMProviderType, config: Dict):
        """Initialize provider.

        Args:
            provider_type: Type of this provider
            config: Provider-specific configuration
        """
        self.provider_type = provider_type
        self.config = config
        self._health = HealthStatus(healthy=True)
        self._initialized = False

        # Metrics
        self._total_requests = 0
        self._total_latency = 0.0
        self._total_tokens = 0

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider (load model, connect to server, etc).

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from prompt.

        Args:
            request: Generation request

        Returns:
            Generation response with result or error
        """
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup resources (unload model, close connections, etc)."""
        pass

    async def health_check(self) -> HealthStatus:
        """Check provider health.

        Returns:
            Current health status
        """
        self._health.last_check = time.time()
        return self._health

    def get_metrics(self) -> Dict:
        """Get provider metrics.

        Returns:
            Dictionary of metrics
        """
        avg_latency = self._total_latency / max(self._total_requests, 1)
        avg_tokens = self._total_tokens / max(self._total_requests, 1)

        return {
            "provider_type": self.provider_type.value,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "avg_latency_ms": avg_latency,
            "avg_tokens_per_request": avg_tokens,
            "success_rate": self._health.success_rate,
            "health": {
                "healthy": self._health.healthy,
                "failures": self._health.failures,
                "successes": self._health.successes,
                "last_error": self._health.last_error
            }
        }

    def _update_metrics(self, response: GenerationResponse):
        """Update internal metrics after generation.

        Args:
            response: Generation response
        """
        self._total_requests += 1
        self._total_latency += response.latency_ms
        self._total_tokens += response.tokens_generated

        if response.success:
            self._health.successes += 1
            self._health.failures = 0  # Reset consecutive failures
        else:
            self._health.failures += 1
            self._health.last_error = response.error

            # Mark unhealthy after 3 consecutive failures
            if self._health.failures >= 3:
                self._health.healthy = False

    def reset_health(self):
        """Reset health status (called after successful recovery)."""
        self._health.healthy = True
        self._health.failures = 0
        self._health.last_error = None

    @property
    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        return self._health.healthy

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized


class BatchableProvider(LLMProvider):
    """Extended provider interface with batch support.

    Providers that can batch multiple requests together should inherit
    from this class and implement batch_generate().
    """

    @abstractmethod
    async def batch_generate(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        """Generate text for multiple prompts in a batch.

        Args:
            requests: List of generation requests

        Returns:
            List of generation responses (same order as requests)
        """
        pass

    @abstractmethod
    def supports_batching(self) -> bool:
        """Check if this provider supports batching.

        Returns:
            True if batch_generate() is implemented
        """
        pass
