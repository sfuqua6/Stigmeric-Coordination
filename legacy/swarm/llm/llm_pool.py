"""Adaptive LLM pool for parallel execution with CUDA memory management.

This module provides a pool of LLM instances that automatically adapts to
available CUDA memory, enabling true parallel processing of agent prompts
while respecting hardware constraints.
"""

import asyncio
import time
from typing import List, Optional, Tuple
from .simple_llm import SimpleLLM


class AdaptiveLLMPool:
    """LLM pool that adapts to available CUDA memory.

    Key features:
    - Query CUDA memory at startup
    - Allocate 1-4 instances based on available memory
    - Load balancing across instances
    - Batch prompts for parallel execution
    - Graceful degradation to single instance if memory constrained
    """

    def __init__(self, model_name: str, device: str = "cuda",
                 enable_cache: bool = True, cache_size: int = 100):
        """Initialize LLM pool (instances created during initialize()).

        Args:
            model_name: Model to load (e.g., "Qwen/Qwen2.5-3B-Instruct")
            device: Device to use ("cuda" or "cpu")
            enable_cache: Enable LLM caching
            cache_size: Cache size for each instance
        """
        self.model_name = model_name
        self.device = device
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.instances: List[SimpleLLM] = []
        self.instance_busy: List[bool] = []  # Track which instances are busy

    async def initialize(self) -> int:
        """Initialize pool with optimal size based on CUDA memory.

        Returns:
            Number of instances created
        """
        if self.device != "cuda":
            # CPU mode: use single instance
            print("[LLM_POOL] CPU mode: creating single instance")
            instance = SimpleLLM(self.model_name, self.device,
                               enable_cache=self.enable_cache,
                               cache_size=self.cache_size)
            await instance.load()
            self.instances.append(instance)
            self.instance_busy.append(False)
            print("[LLM_POOL] Pool ready with 1 instance (CPU)\n")
            return 1

        # CUDA mode: detect memory and create optimal pool
        try:
            import torch

            if not torch.cuda.is_available():
                print("[LLM_POOL] CUDA not available, falling back to CPU")
                return await self._initialize_cpu_fallback()

            # Query CUDA memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            allocated = torch.cuda.memory_allocated(0) / 1e9
            available_memory = total_memory - allocated

            print(f"[LLM_POOL] CUDA memory: {available_memory:.1f}GB available / {total_memory:.1f}GB total")

            # Estimate memory per instance based on model name
            memory_per_instance = self._estimate_model_memory()
            print(f"[LLM_POOL] Estimated memory per instance: {memory_per_instance:.1f}GB")

            # Calculate optimal pool size
            # Reserve 2GB for activations/overhead
            usable_memory = max(0, available_memory - 2.0)
            max_instances = int(usable_memory / memory_per_instance)
            pool_size = max(1, min(4, max_instances))  # Between 1 and 4

            print(f"[LLM_POOL] Creating pool with {pool_size} instances...")

            # Load instances sequentially (safer for memory management)
            for i in range(pool_size):
                try:
                    instance = SimpleLLM(self.model_name, self.device,
                                       enable_cache=self.enable_cache,
                                       cache_size=self.cache_size)
                    await instance.load()
                    self.instances.append(instance)
                    self.instance_busy.append(False)

                    # Report memory after each load
                    if self.device == "cuda":
                        allocated = torch.cuda.memory_allocated(0) / 1e9
                        print(f"[LLM_POOL] Instance {i+1}/{pool_size} loaded (CUDA: {allocated:.1f}GB used)")

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[LLM_POOL] Out of memory loading instance {i+1}, stopping at {i} instances")
                        if i == 0:
                            # If we can't even load one instance, fall back to CPU
                            print("[LLM_POOL] Cannot load even one instance on CUDA, falling back to CPU")
                            return await self._initialize_cpu_fallback()
                        break
                    else:
                        raise

            actual_pool_size = len(self.instances)
            print(f"[LLM_POOL] Pool ready with {actual_pool_size} instances\n")
            return actual_pool_size

        except ImportError:
            print("[LLM_POOL] PyTorch not available, falling back to CPU")
            return await self._initialize_cpu_fallback()

    async def _initialize_cpu_fallback(self) -> int:
        """Fallback initialization for CPU mode."""
        print("[LLM_POOL] Initializing CPU fallback with single instance")
        instance = SimpleLLM(self.model_name, "cpu",
                           enable_cache=self.enable_cache,
                           cache_size=self.cache_size)
        await instance.load()
        self.instances.append(instance)
        self.instance_busy.append(False)
        print("[LLM_POOL] Pool ready with 1 instance (CPU)\n")
        return 1

    def _estimate_model_memory(self) -> float:
        """Estimate memory needed per model instance (GB).

        Returns:
            Estimated memory in GB
        """
        model_lower = self.model_name.lower()

        # Extract parameter count from model name
        if "3b" in model_lower or "3-b" in model_lower:
            return 6.0  # ~6GB for 3B parameter model
        elif "7b" in model_lower or "7-b" in model_lower:
            return 14.0  # ~14GB for 7B parameter model
        elif "14b" in model_lower or "14-b" in model_lower:
            return 28.0  # ~28GB for 14B parameter model
        elif "1.5b" in model_lower or "1-5b" in model_lower:
            return 4.0  # ~4GB for 1.5B parameter model
        else:
            # Conservative default
            return 8.0

    def get_pool_size(self) -> int:
        """Get number of LLM instances in pool.

        Returns:
            Number of instances
        """
        return len(self.instances)

    async def generate(self, prompt: str, max_tokens: int = 150,
                      temperature: float = 0.7) -> str:
        """Generate response using least-busy instance.

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        # Select least-busy instance
        instance_idx = self._select_instance()
        instance = self.instances[instance_idx]

        # Mark as busy
        self.instance_busy[instance_idx] = True

        try:
            # Generate
            result = await instance.generate(prompt, max_tokens, temperature)
            return result
        finally:
            # Mark as free
            self.instance_busy[instance_idx] = False

    async def generate_batch(self, prompts: List[str], max_tokens: int = 150,
                            temperature: float = 0.7) -> List[str]:
        """Generate responses for multiple prompts in parallel.

        Distributes prompts across all instances for true parallelism.

        Args:
            prompts: List of prompts
            max_tokens: Max tokens per prompt
            temperature: Sampling temperature

        Returns:
            List of generated texts (same order as prompts)
        """
        if not prompts:
            return []

        pool_size = len(self.instances)

        # Single prompt: use regular generate
        if len(prompts) == 1:
            return [await self.generate(prompts[0], max_tokens, temperature)]

        # Multiple prompts: distribute across pool
        tasks = []
        for i, prompt in enumerate(prompts):
            # Round-robin distribution across instances
            instance_idx = i % pool_size
            instance = self.instances[instance_idx]

            # Create task
            task = instance.generate(prompt, max_tokens, temperature)
            tasks.append(task)

        # Execute all in parallel
        results = await asyncio.gather(*tasks)
        return results

    async def generate_with_stage_batching(self, stage_name: str,
                                          agent_prompts: List[Tuple[str, str, int, float]]) -> List[str]:
        """Generate for all agents in a stage with optimal batching.

        Args:
            stage_name: Name of stage (for logging)
            agent_prompts: List of (agent_id, prompt, max_tokens, temperature)

        Returns:
            List of results (same order as agent_prompts)
        """
        if not agent_prompts:
            return []

        print(f"[LLM_POOL] Stage '{stage_name}': Processing {len(agent_prompts)} prompts across {len(self.instances)} instances")

        # Distribute prompts across instances
        pool_size = len(self.instances)
        batches = [[] for _ in range(pool_size)]

        for i, agent_prompt in enumerate(agent_prompts):
            instance_idx = i % pool_size
            batches[instance_idx].append((i, agent_prompt))  # Store index for reordering

        # Process each batch on its instance (in parallel)
        async def process_batch(instance_idx: int, batch: List[Tuple[int, Tuple]]) -> List[Tuple[int, str]]:
            """Process a batch of prompts on one instance."""
            instance = self.instances[instance_idx]
            results = []
            for original_idx, (agent_id, prompt, max_tokens, temperature) in batch:
                result = await instance.generate(prompt, max_tokens, temperature)
                results.append((original_idx, result))
            return results

        # Execute all batches in parallel
        batch_tasks = [
            process_batch(i, batch)
            for i, batch in enumerate(batches) if batch
        ]
        batch_results = await asyncio.gather(*batch_tasks)

        # Flatten and reorder results to match input order
        all_results = {}
        for batch_result in batch_results:
            for original_idx, result in batch_result:
                all_results[original_idx] = result

        # Return in original order
        ordered_results = [all_results[i] for i in range(len(agent_prompts))]

        print(f"[LLM_POOL] Stage '{stage_name}': Complete ({len(ordered_results)} responses)")
        return ordered_results

    def _select_instance(self) -> int:
        """Select least-busy instance (round-robin among free instances).

        Returns:
            Index of selected instance
        """
        # Find free instances
        free_instances = [i for i, busy in enumerate(self.instance_busy) if not busy]

        if free_instances:
            # Return first free instance
            return free_instances[0]
        else:
            # All busy - return first (will wait)
            return 0

    def get_stats(self) -> dict:
        """Get pool statistics.

        Returns:
            Dict with pool_size, busy_instances, free_instances
        """
        return {
            "pool_size": len(self.instances),
            "busy_instances": sum(self.instance_busy),
            "free_instances": len(self.instances) - sum(self.instance_busy)
        }

    def get_cache_stats(self) -> dict:
        """Get aggregated cache statistics from all instances.

        Returns:
            Dict with aggregated cache stats
        """
        if not self.instances:
            return {"total_calls": 0, "cache_hits": 0, "hit_rate": 0.0}

        # Aggregate stats from all instances
        total_calls = 0
        cache_hits = 0

        for instance in self.instances:
            stats = instance.get_cache_stats()
            total_calls += stats.get("total_calls", 0)
            cache_hits += stats.get("cache_hits", 0)

        hit_rate = cache_hits / total_calls if total_calls > 0 else 0.0

        return {
            "total_calls": total_calls,
            "cache_hits": cache_hits,
            "hit_rate": hit_rate,
            "pool_size": len(self.instances)
        }
