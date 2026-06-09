"""LLM Pool - Strategy 2: Agent Isolation with Health Checks

This module provides a pool of LLM providers with:
- Isolated providers (failures don't cascade)
- Automatic health checks
- Circuit breakers
- Automatic recovery
- Load balancing
"""

import asyncio
import time
import logging
from typing import List, Optional, Dict
from collections import deque
from swarm.llm.provider import LLMProvider, GenerationRequest, GenerationResponse, HealthStatus

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker for LLM provider failures.

    States:
    - CLOSED: Normal operation
    - OPEN: Too many failures, block requests
    - HALF_OPEN: Testing if provider recovered
    """

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.failures = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.opened_at = 0.0

    def record_success(self):
        """Record successful request."""
        if self.state == "HALF_OPEN":
            # Recovery successful
            self.state = "CLOSED"
            self.failures = 0
            logger.info("[CIRCUIT BREAKER] Circuit closed - provider recovered")

        self.failures = max(0, self.failures - 1)  # Gradually recover

    def record_failure(self):
        """Record failed request."""
        self.failures += 1

        if self.failures >= self.failure_threshold and self.state == "CLOSED":
            self.state = "OPEN"
            self.opened_at = time.time()
            logger.warning(f"[CIRCUIT BREAKER] Circuit opened after {self.failures} failures")

    def can_attempt(self) -> bool:
        """Check if requests can be attempted.

        Returns:
            True if circuit allows requests
        """
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            # Check if recovery timeout elapsed
            if time.time() - self.opened_at >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("[CIRCUIT BREAKER] Circuit half-open - testing recovery")
                return True
            return False

        # HALF_OPEN: allow one request to test
        return True

    def get_state(self) -> Dict:
        """Get circuit breaker state.

        Returns:
            State dictionary
        """
        return {
            "state": self.state,
            "failures": self.failures,
            "failure_threshold": self.failure_threshold,
            "time_since_open": time.time() - self.opened_at if self.state == "OPEN" else 0
        }


class ProviderPool:
    """Pool of LLM providers with isolation and health management."""

    def __init__(self, providers: List[LLMProvider], health_check_interval: float = 60.0):
        """Initialize provider pool.

        Args:
            providers: List of LLM providers
            health_check_interval: Seconds between health checks
        """
        self.providers = providers
        self.health_check_interval = health_check_interval

        # Circuit breakers per provider
        self.circuit_breakers = {
            id(provider): CircuitBreaker() for provider in providers
        }

        # Request queue for load balancing
        self.request_queue = asyncio.Queue()

        # Provider usage tracking for load balancing
        self.provider_usage = {id(provider): 0 for provider in providers}
        self.provider_locks = {id(provider): asyncio.Lock() for provider in providers}

        # Health check task
        self._health_check_task = None
        self._running = False

        logger.info(f"[POOL] Initialized with {len(providers)} providers")

    async def start(self):
        """Start the pool (initializes providers and health checking)."""
        self._running = True

        # Initialize all providers
        for provider in self.providers:
            try:
                await provider.initialize()
                logger.info(f"[POOL] Initialized provider: {provider.provider_type.value}")
            except Exception as e:
                logger.error(f"[POOL] Failed to initialize provider {provider.provider_type.value}: {e}")
                self.circuit_breakers[id(provider)].record_failure()

        # Start health check background task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("[POOL] Health check task started")

    async def stop(self):
        """Stop the pool and cleanup providers."""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Cleanup all providers
        for provider in self.providers:
            try:
                await provider.cleanup()
            except Exception as e:
                logger.error(f"[POOL] Error cleaning up provider: {e}")

        logger.info("[POOL] Stopped")

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using least-loaded healthy provider.

        Args:
            request: Generation request

        Returns:
            Generation response
        """
        # Get healthy provider
        provider = await self._get_healthy_provider()

        if provider is None:
            # No healthy providers available
            return GenerationResponse(
                text="",
                success=False,
                latency_ms=0,
                tokens_generated=0,
                error="No healthy providers available",
                cached=False
            )

        provider_id = id(provider)
        circuit_breaker = self.circuit_breakers[provider_id]

        # Check circuit breaker
        if not circuit_breaker.can_attempt():
            logger.warning(f"[POOL] Circuit breaker open for provider {provider.provider_type.value}")
            return GenerationResponse(
                text="",
                success=False,
                latency_ms=0,
                tokens_generated=0,
                error=f"Circuit breaker {circuit_breaker.state}",
                cached=False
            )

        # Track usage
        async with self.provider_locks[provider_id]:
            self.provider_usage[provider_id] += 1

        # Generate with provider
        try:
            response = await provider.generate(request)

            if response.success:
                circuit_breaker.record_success()
            else:
                circuit_breaker.record_failure()
                logger.warning(f"[POOL] Provider {provider.provider_type.value} generation failed: {response.error}")

            return response

        except Exception as e:
            circuit_breaker.record_failure()
            logger.error(f"[POOL] Provider {provider.provider_type.value} raised exception: {e}")

            return GenerationResponse(
                text="",
                success=False,
                latency_ms=0,
                tokens_generated=0,
                error=str(e),
                cached=False
            )

        finally:
            # Release usage tracking
            async with self.provider_locks[provider_id]:
                self.provider_usage[provider_id] -= 1

    async def _get_healthy_provider(self) -> Optional[LLMProvider]:
        """Get least-loaded healthy provider.

        Returns:
            Provider or None if no healthy providers
        """
        healthy_providers = [
            (p, self.provider_usage[id(p)])
            for p in self.providers
            if p.is_healthy and self.circuit_breakers[id(p)].can_attempt()
        ]

        if not healthy_providers:
            return None

        # Return least-loaded provider
        provider, _ = min(healthy_providers, key=lambda x: x[1])
        return provider

    async def _health_check_loop(self):
        """Background task that periodically checks provider health."""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_providers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[POOL] Health check error: {e}")

    async def _check_all_providers(self):
        """Check health of all providers."""
        for provider in self.providers:
            try:
                health = await provider.health_check()

                if not health.healthy:
                    logger.warning(f"[POOL] Provider {provider.provider_type.value} unhealthy: {health.last_error}")

                    # Attempt recovery
                    await self._attempt_recovery(provider)

            except Exception as e:
                logger.error(f"[POOL] Health check failed for {provider.provider_type.value}: {e}")

    async def _attempt_recovery(self, provider: LLMProvider):
        """Attempt to recover a failed provider.

        Args:
            provider: Provider to recover
        """
        logger.info(f"[POOL] Attempting recovery of {provider.provider_type.value}")

        try:
            # Cleanup current state
            await provider.cleanup()

            # Wait a bit
            await asyncio.sleep(2)

            # Re-initialize
            success = await provider.initialize()

            if success:
                provider.reset_health()
                self.circuit_breakers[id(provider)].record_success()
                logger.info(f"[POOL] Successfully recovered {provider.provider_type.value}")
            else:
                logger.warning(f"[POOL] Recovery failed for {provider.provider_type.value}")

        except Exception as e:
            logger.error(f"[POOL] Recovery attempt failed for {provider.provider_type.value}: {e}")

    def get_pool_status(self) -> Dict:
        """Get status of all providers in pool.

        Returns:
            Pool status dictionary
        """
        provider_statuses = []

        for provider in self.providers:
            provider_id = id(provider)
            circuit_breaker = self.circuit_breakers[provider_id]

            provider_statuses.append({
                "type": provider.provider_type.value,
                "healthy": provider.is_healthy,
                "initialized": provider.is_initialized,
                "current_load": self.provider_usage[provider_id],
                "circuit_breaker": circuit_breaker.get_state(),
                "metrics": provider.get_metrics()
            })

        return {
            "total_providers": len(self.providers),
            "healthy_providers": sum(1 for p in self.providers if p.is_healthy),
            "providers": provider_statuses
        }
