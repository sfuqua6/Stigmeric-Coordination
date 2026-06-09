"""Request Batching - Strategy 3: Batch Multiple Requests

This module provides automatic request batching for improved GPU utilization:
- Collects requests within a time window
- Batches them together for parallel processing
- 5-10x better GPU utilization
- Lower latency per request
"""

import asyncio
import time
import logging
from typing import List, Dict, Optional
from collections import deque
from swarm.llm.provider import LLMProvider, BatchableProvider, GenerationRequest, GenerationResponse

logger = logging.getLogger(__name__)


class RequestBatcher:
    """Automatic request batching for LLM providers.

    Collects requests within a time window and processes them as a batch
    for better GPU utilization and lower per-request latency.
    """

    def __init__(
        self,
        provider: LLMProvider,
        batch_size: int = 8,
        batch_timeout_ms: float = 100.0,
        max_queue_size: int = 256
    ):
        """Initialize request batcher.

        Args:
            provider: LLM provider to batch requests for
            batch_size: Maximum requests per batch
            batch_timeout_ms: Max milliseconds to wait before processing batch
            max_queue_size: Maximum pending requests
        """
        self.provider = provider
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms / 1000.0  # Convert to seconds
        self.max_queue_size = max_queue_size

        # Check if provider supports batching
        self.supports_batching = isinstance(provider, BatchableProvider) and provider.supports_batching()

        # Request queue
        self.pending_requests: deque = deque()
        self.queue_lock = asyncio.Lock()
        self.queue_not_empty = asyncio.Event()

        # Background batcher task
        self._batcher_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self._total_requests = 0
        self._total_batches = 0
        self._total_batch_size = 0

        if not self.supports_batching:
            logger.warning(f"[BATCHER] Provider {provider.provider_type.value} does not support batching - will process sequentially")

    async def start(self):
        """Start the batcher background task."""
        self._running = True
        self._batcher_task = asyncio.create_task(self._batch_processor())
        logger.info(f"[BATCHER] Started for provider {self.provider.provider_type.value}")

    async def stop(self):
        """Stop the batcher and process remaining requests."""
        self._running = False

        if self._batcher_task:
            self._batcher_task.cancel()
            try:
                await self._batcher_task
            except asyncio.CancelledError:
                pass

        # Process any remaining requests
        if self.pending_requests:
            await self._process_pending_batch()

        logger.info(f"[BATCHER] Stopped")

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text with automatic batching.

        Args:
            request: Generation request

        Returns:
            Generation response
        """
        # Create future for this request
        future = asyncio.Future()

        # Add to queue
        async with self.queue_lock:
            if len(self.pending_requests) >= self.max_queue_size:
                # Queue full - return error
                logger.warning("[BATCHER] Queue full, rejecting request")
                return GenerationResponse(
                    text="",
                    success=False,
                    latency_ms=0,
                    tokens_generated=0,
                    error="Batcher queue full"
                )

            self.pending_requests.append((request, future))
            self._total_requests += 1

            # Signal batcher if batch is full
            if len(self.pending_requests) >= self.batch_size:
                self.queue_not_empty.set()

        # Wait for result
        return await future

    async def _batch_processor(self):
        """Background task that processes batched requests."""
        while self._running:
            try:
                # Wait for batch to fill or timeout
                try:
                    await asyncio.wait_for(
                        self.queue_not_empty.wait(),
                        timeout=self.batch_timeout_ms
                    )
                except asyncio.TimeoutError:
                    pass

                # Process batch
                await self._process_pending_batch()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[BATCHER] Error in batch processor: {e}")
                await asyncio.sleep(1.0)  # Back off on error

    async def _process_pending_batch(self):
        """Process all pending requests as a batch."""
        # Get batch
        async with self.queue_lock:
            if not self.pending_requests:
                self.queue_not_empty.clear()
                return

            batch_items = []
            while self.pending_requests and len(batch_items) < self.batch_size:
                batch_items.append(self.pending_requests.popleft())

            self.queue_not_empty.clear()

        if not batch_items:
            return

        # Process batch
        requests = [item[0] for item in batch_items]
        futures = [item[1] for item in batch_items]

        self._total_batches += 1
        self._total_batch_size += len(requests)

        logger.debug(f"[BATCHER] Processing batch of {len(requests)} requests")

        start_time = time.time()

        try:
            if self.supports_batching:
                # Use provider's batch generation
                responses = await self.provider.batch_generate(requests)
            else:
                # Fall back to sequential processing
                responses = []
                for request in requests:
                    response = await self.provider.generate(request)
                    responses.append(response)

            # Distribute results to futures
            for future, response in zip(futures, responses):
                if not future.done():
                    future.set_result(response)

            batch_time = (time.time() - start_time) * 1000
            logger.debug(f"[BATCHER] Batch completed in {batch_time:.1f}ms ({batch_time/len(requests):.1f}ms per request)")

        except Exception as e:
            logger.error(f"[BATCHER] Batch processing failed: {e}")

            # Set error for all futures
            error_response = GenerationResponse(
                text="",
                success=False,
                latency_ms=0,
                tokens_generated=0,
                error=f"Batch processing error: {e}"
            )

            for future in futures:
                if not future.done():
                    future.set_result(error_response)

    def get_metrics(self) -> Dict:
        """Get batcher metrics.

        Returns:
            Metrics dictionary
        """
        avg_batch_size = self._total_batch_size / max(self._total_batches, 1)

        return {
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "avg_batch_size": avg_batch_size,
            "current_queue_size": len(self.pending_requests),
            "max_queue_size": self.max_queue_size,
            "batch_size": self.batch_size,
            "supports_batching": self.supports_batching,
            "provider_type": self.provider.provider_type.value
        }


class AdaptiveBatcher(RequestBatcher):
    """Request batcher with adaptive batch sizing.

    Automatically adjusts batch size and timeout based on:
    - Request rate
    - Latency targets
    - Provider performance
    """

    def __init__(
        self,
        provider: LLMProvider,
        initial_batch_size: int = 8,
        initial_timeout_ms: float = 100.0,
        target_latency_ms: float = 500.0,
        **kwargs
    ):
        """Initialize adaptive batcher.

        Args:
            provider: LLM provider
            initial_batch_size: Starting batch size
            initial_timeout_ms: Starting timeout
            target_latency_ms: Target latency per request
            **kwargs: Additional arguments for RequestBatcher
        """
        super().__init__(provider, initial_batch_size, initial_timeout_ms, **kwargs)

        self.target_latency_ms = target_latency_ms
        self.min_batch_size = 1
        self.max_batch_size = 32
        self.min_timeout_ms = 10.0
        self.max_timeout_ms = 500.0

        # Performance tracking
        self._recent_latencies: deque = deque(maxlen=100)
        self._last_adjustment_time = 0.0
        self._adjustment_interval = 10.0  # Adjust every 10 seconds

    async def _process_pending_batch(self):
        """Process batch and adapt parameters."""
        start_time = time.time()

        await super()._process_pending_batch()

        # Track latency
        if self._total_batches > 0:
            batch_latency = (time.time() - start_time) * 1000
            per_request_latency = batch_latency / max(len(self.pending_requests) if self.pending_requests else 1, 1)
            self._recent_latencies.append(per_request_latency)

        # Adapt parameters periodically
        if time.time() - self._last_adjustment_time >= self._adjustment_interval:
            await self._adapt_parameters()
            self._last_adjustment_time = time.time()

    async def _adapt_parameters(self):
        """Adapt batch size and timeout based on performance."""
        if len(self._recent_latencies) < 10:
            return

        avg_latency = sum(self._recent_latencies) / len(self._recent_latencies)

        # Adjust batch size
        if avg_latency > self.target_latency_ms * 1.2:
            # Too slow - reduce batch size
            old_size = self.batch_size
            self.batch_size = max(self.min_batch_size, self.batch_size - 2)
            if old_size != self.batch_size:
                logger.info(f"[ADAPTIVE BATCHER] Reduced batch size: {old_size} -> {self.batch_size} (latency: {avg_latency:.1f}ms)")

        elif avg_latency < self.target_latency_ms * 0.8:
            # Fast enough - increase batch size
            old_size = self.batch_size
            self.batch_size = min(self.max_batch_size, self.batch_size + 2)
            if old_size != self.batch_size:
                logger.info(f"[ADAPTIVE BATCHER] Increased batch size: {old_size} -> {self.batch_size} (latency: {avg_latency:.1f}ms)")

        # Adjust timeout based on request rate
        recent_queue_sizes = len(self.pending_requests)
        if recent_queue_sizes > self.batch_size * 2:
            # High load - reduce timeout (process faster)
            old_timeout = self.batch_timeout_ms * 1000
            self.batch_timeout_ms = max(self.min_timeout_ms / 1000, self.batch_timeout_ms * 0.8)
            new_timeout = self.batch_timeout_ms * 1000
            if abs(old_timeout - new_timeout) > 1:
                logger.info(f"[ADAPTIVE BATCHER] Reduced timeout: {old_timeout:.1f}ms -> {new_timeout:.1f}ms")

        elif recent_queue_sizes < self.batch_size:
            # Low load - increase timeout (wait for more requests)
            old_timeout = self.batch_timeout_ms * 1000
            self.batch_timeout_ms = min(self.max_timeout_ms / 1000, self.batch_timeout_ms * 1.2)
            new_timeout = self.batch_timeout_ms * 1000
            if abs(old_timeout - new_timeout) > 1:
                logger.info(f"[ADAPTIVE BATCHER] Increased timeout: {old_timeout:.1f}ms -> {new_timeout:.1f}ms")
