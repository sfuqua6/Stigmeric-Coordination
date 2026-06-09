"""vLLM Provider - High-throughput inference server

This provider wraps vLLM for maximum throughput:
- Automatic request batching (handled by vLLM)
- PagedAttention for efficient KV cache
- Multi-GPU support
- 2-10x throughput vs SimpleLLM
"""

import asyncio
import time
import logging
from typing import List, Dict, Optional
from swarm.llm.provider import (
    LLMProvider, BatchableProvider, LLMProviderType,
    GenerationRequest, GenerationResponse
)

logger = logging.getLogger(__name__)


class VLLMProvider(BatchableProvider):
    """vLLM-based provider for high-throughput inference.

    Features:
    - Automatic continuous batching
    - PagedAttention (efficient memory)
    - Multi-GPU tensor parallelism
    - Up to 10x throughput vs SimpleLLM
    """

    def __init__(self, config: Dict):
        """Initialize vLLM provider.

        Args:
            config: Configuration dictionary with keys:
                - model_name: HuggingFace model ID
                - tensor_parallel_size: Number of GPUs (default: auto)
                - max_num_seqs: Max concurrent sequences (default: 256)
                - gpu_memory_utilization: GPU memory to use 0-1 (default: 0.9)
                - dtype: Model dtype (default: auto)
        """
        super().__init__(LLMProviderType.VLLM, config)

        self.model_name = config.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        self.tensor_parallel_size = config.get("tensor_parallel_size", None)
        self.max_num_seqs = config.get("max_num_seqs", 256)
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.9)
        self.dtype = config.get("dtype", "auto")

        self.llm = None
        self.sampling_params_cache: Dict[str, any] = {}

        logger.info(f"[VLLM] Initialized with model: {self.model_name}")

    async def initialize(self) -> bool:
        """Initialize vLLM engine.

        Returns:
            True if successful
        """
        try:
            # Import vLLM (only when needed)
            from vllm import LLM
            import torch

            # Auto-detect tensor parallel size
            if self.tensor_parallel_size is None:
                self.tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

            logger.info(f"[VLLM] Loading model with tensor_parallel_size={self.tensor_parallel_size}")

            # Initialize vLLM engine
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                max_num_seqs=self.max_num_seqs,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                trust_remote_code=True
            )

            self._initialized = True
            logger.info(f"[VLLM] Model loaded successfully")
            return True

        except ImportError as e:
            logger.error(f"[VLLM] vLLM not installed: {e}")
            logger.error("[VLLM] Install with: pip install vllm")
            return False

        except Exception as e:
            logger.error(f"[VLLM] Failed to initialize: {e}")
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
                error="vLLM not initialized"
            )

        start_time = time.time()

        try:
            from vllm import SamplingParams

            # Create sampling params (cached by key)
            params_key = f"{request.max_tokens}_{request.temperature}_{request.top_p}_{request.top_k}"
            if params_key not in self.sampling_params_cache:
                self.sampling_params_cache[params_key] = SamplingParams(
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty
                )

            sampling_params = self.sampling_params_cache[params_key]

            # Generate (vLLM handles batching automatically)
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                self.llm.generate,
                [request.prompt],
                sampling_params
            )

            # Extract result
            output = outputs[0]
            generated_text = output.outputs[0].text
            tokens_generated = len(output.outputs[0].token_ids)

            latency_ms = (time.time() - start_time) * 1000

            response = GenerationResponse(
                text=generated_text,
                success=True,
                latency_ms=latency_ms,
                tokens_generated=tokens_generated,
                cached=False,
                provider_metadata={
                    "finish_reason": output.outputs[0].finish_reason
                }
            )

            self._update_metrics(response)
            return response

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"[VLLM] Generation error: {e}")

            response = GenerationResponse(
                text="",
                success=False,
                latency_ms=latency_ms,
                tokens_generated=0,
                error=str(e)
            )

            self._update_metrics(response)
            return response

    async def batch_generate(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        """Generate text for multiple prompts (batched).

        vLLM handles batching automatically, but we provide explicit batch
        support for compatibility.

        Args:
            requests: List of generation requests

        Returns:
            List of generation responses
        """
        if not self._initialized:
            error_response = GenerationResponse(
                text="",
                success=False,
                latency_ms=0,
                tokens_generated=0,
                error="vLLM not initialized"
            )
            return [error_response] * len(requests)

        start_time = time.time()

        try:
            from vllm import SamplingParams

            # Group requests by sampling params
            # vLLM is most efficient when sampling params are the same
            param_groups: Dict[str, List] = {}

            for i, request in enumerate(requests):
                params_key = f"{request.max_tokens}_{request.temperature}_{request.top_p}_{request.top_k}"
                if params_key not in param_groups:
                    param_groups[params_key] = []
                param_groups[params_key].append((i, request))

            # Generate for each param group
            results = [None] * len(requests)

            for params_key, group_requests in param_groups.items():
                indices = [item[0] for item in group_requests]
                reqs = [item[1] for item in group_requests]

                # Create sampling params
                if params_key not in self.sampling_params_cache:
                    first_req = reqs[0]
                    self.sampling_params_cache[params_key] = SamplingParams(
                        max_tokens=first_req.max_tokens,
                        temperature=first_req.temperature,
                        top_p=first_req.top_p,
                        top_k=first_req.top_k,
                        repetition_penalty=first_req.repetition_penalty
                    )

                sampling_params = self.sampling_params_cache[params_key]

                # Generate batch
                prompts = [req.prompt for req in reqs]
                loop = asyncio.get_event_loop()
                outputs = await loop.run_in_executor(
                    None,
                    self.llm.generate,
                    prompts,
                    sampling_params
                )

                # Distribute results
                for idx, output in zip(indices, outputs):
                    generated_text = output.outputs[0].text
                    tokens_generated = len(output.outputs[0].token_ids)
                    latency_ms = (time.time() - start_time) * 1000

                    results[idx] = GenerationResponse(
                        text=generated_text,
                        success=True,
                        latency_ms=latency_ms,
                        tokens_generated=tokens_generated,
                        cached=False,
                        provider_metadata={
                            "finish_reason": output.outputs[0].finish_reason
                        }
                    )

            # Update metrics for all responses
            for response in results:
                self._update_metrics(response)

            logger.debug(f"[VLLM] Batch of {len(requests)} completed in {(time.time() - start_time)*1000:.1f}ms")

            return results

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"[VLLM] Batch generation error: {e}")

            error_response = GenerationResponse(
                text="",
                success=False,
                latency_ms=latency_ms,
                tokens_generated=0,
                error=str(e)
            )

            for _ in requests:
                self._update_metrics(error_response)

            return [error_response] * len(requests)

    async def cleanup(self):
        """Cleanup vLLM engine."""
        if self.llm is not None:
            # vLLM doesn't have explicit cleanup
            # Just clear reference and let GC handle it
            self.llm = None
            self._initialized = False

            # Force CUDA cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"[VLLM] CUDA cleanup warning: {e}")

            logger.info("[VLLM] Cleaned up")

    def supports_batching(self) -> bool:
        """vLLM supports explicit batching.

        Returns:
            True
        """
        return True
