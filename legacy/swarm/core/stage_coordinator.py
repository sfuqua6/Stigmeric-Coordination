"""Stage-based execution coordinator for parallel agent processing.

This module coordinates agent execution in stages based on signal dependencies,
enabling true parallel processing while respecting the dependency graph.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from .signal_store import SignalStore
from ..llm.llm_pool import AdaptiveLLMPool


class StageCoordinator:
    """Coordinates stage-based parallel execution of agents.

    Each stage:
    1. Waits for dependencies (previous stage completion)
    2. Batches all agents in stage
    3. Executes in parallel across LLM pool
    4. Validates outputs
    5. Signals next stage to start

    This enables parallelism while respecting signal dependencies:
    - Scouts run in parallel (no dependencies)
    - Foragers/Critics run in parallel (depend on scouts)
    - Haters run in parallel (depend on foragers/critics)
    """

    def __init__(self, llm_pool: AdaptiveLLMPool, signal_store: SignalStore):
        """Initialize stage coordinator.

        Args:
            llm_pool: Pool of LLM instances for parallel execution
            signal_store: Shared signal store for agent coordination
        """
        self.llm_pool = llm_pool
        self.signal_store = signal_store

        # Stage completion events for dependency management
        self.stage_complete: Dict[str, asyncio.Event] = {}
        self.stage_stats: Dict[str, dict] = {}

    def register_stage(self, stage_name: str):
        """Register a stage and create its completion event.

        Args:
            stage_name: Name of stage to register
        """
        if stage_name not in self.stage_complete:
            self.stage_complete[stage_name] = asyncio.Event()
            print(f"[STAGE_COORD] Registered stage: {stage_name}")

    async def run_stage(self, stage_name: str, agents: List[Any],
                       depends_on: Optional[List[str]] = None,
                       max_iterations: int = 1) -> Dict[str, Any]:
        """Run a stage of agents in parallel.

        Args:
            stage_name: Name of stage (for logging)
            agents: List of agent objects
            depends_on: List of stage names this stage depends on
            max_iterations: Number of iterations to run agents

        Returns:
            Stage statistics (agents, prompts, signals, time)
        """
        # Register this stage if not already registered
        self.register_stage(stage_name)

        # Wait for dependencies
        if depends_on:
            print(f"[STAGE] {stage_name}: Waiting for dependencies {depends_on}")
            for dep in depends_on:
                if dep not in self.stage_complete:
                    self.register_stage(dep)
                await self.stage_complete[dep].wait()
            print(f"[STAGE] {stage_name}: Dependencies satisfied")

        print(f"\n[STAGE] {stage_name}: Starting with {len(agents)} agents ({max_iterations} iterations)")
        stage_start = time.time()

        # Run agents for specified iterations
        signals_deposited = 0
        total_prompts = 0

        for iteration in range(max_iterations):
            # Collect all prompts from agents for this iteration
            agent_prompts = []
            for agent in agents:
                # Each agent prepares its prompt (no LLM call yet)
                if hasattr(agent, 'prepare_prompt'):
                    prompt_data = await agent.prepare_prompt(self.signal_store)
                    if prompt_data:
                        agent_prompts.append((agent, prompt_data))
                else:
                    # Fallback for agents not yet refactored
                    print(f"[STAGE] {stage_name}: Agent {agent.id} not yet refactored for staged execution")

            if not agent_prompts:
                print(f"[STAGE] {stage_name}: No prompts generated in iteration {iteration + 1}")
                continue

            total_prompts += len(agent_prompts)
            print(f"[STAGE] {stage_name}: Iteration {iteration + 1}/{max_iterations} - {len(agent_prompts)} prompts collected")

            # Extract prompt tuples for batch processing
            prompt_tuples = [prompt_data for _, prompt_data in agent_prompts]

            # Batch execute across LLM pool
            results = await self.llm_pool.generate_with_stage_batching(
                f"{stage_name}_iter{iteration+1}", prompt_tuples
            )

            # Process results (deposit signals)
            for (agent, _), result in zip(agent_prompts, results):
                if result and hasattr(agent, 'process_result'):
                    deposited = await agent.process_result(result, self.signal_store)
                    if deposited:
                        signals_deposited += 1

        stage_time = time.time() - stage_start
        print(f"[STAGE] {stage_name}: Complete in {stage_time:.2f}s "
              f"({signals_deposited} signals deposited from {total_prompts} prompts)")

        # Mark stage as complete
        self.stage_complete[stage_name].set()

        # Store stats
        stats = {
            "stage": stage_name,
            "agents": len(agents),
            "prompts": total_prompts,
            "signals_deposited": signals_deposited,
            "time": stage_time,
            "iterations": max_iterations
        }
        self.stage_stats[stage_name] = stats

        return stats

    async def run_parallel_stages(self, stages: List[tuple]) -> Dict[str, dict]:
        """Run multiple stages in parallel (if they have same dependencies).

        Args:
            stages: List of (stage_name, agents, depends_on, max_iterations) tuples

        Returns:
            Dict of stage_name -> stats
        """
        # Group stages by dependencies
        dep_groups: Dict[tuple, List[tuple]] = {}
        for stage_name, agents, depends_on, max_iterations in stages:
            dep_key = tuple(sorted(depends_on)) if depends_on else ()
            if dep_key not in dep_groups:
                dep_groups[dep_key] = []
            dep_groups[dep_key].append((stage_name, agents, depends_on, max_iterations))

        # Run each dependency group in parallel
        all_stats = {}
        for dep_key, group_stages in dep_groups.items():
            print(f"\n[STAGE_COORD] Running {len(group_stages)} stages in parallel (dependencies: {list(dep_key) if dep_key else 'none'})")

            # Create tasks for all stages in this group
            tasks = []
            for stage_name, agents, depends_on, max_iterations in group_stages:
                task = self.run_stage(stage_name, agents, depends_on, max_iterations)
                tasks.append(task)

            # Execute in parallel
            results = await asyncio.gather(*tasks)

            # Collect stats
            for stats in results:
                all_stats[stats["stage"]] = stats

        return all_stats

    def get_stage_stats(self, stage_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a stage or all stages.

        Args:
            stage_name: Specific stage name, or None for all stages

        Returns:
            Stage statistics
        """
        if stage_name:
            return self.stage_stats.get(stage_name, {})
        return self.stage_stats

    def reset(self):
        """Reset all stage completion events and stats.

        Call this between rounds if reusing the coordinator.
        """
        for event in self.stage_complete.values():
            event.clear()
        self.stage_stats.clear()
        print("[STAGE_COORD] Reset all stages")

    def get_summary(self) -> str:
        """Get summary of all stage executions.

        Returns:
            Formatted summary string
        """
        if not self.stage_stats:
            return "No stages executed yet"

        summary = ["Stage Execution Summary:"]
        summary.append("=" * 60)

        total_time = 0
        total_signals = 0
        total_prompts = 0

        for stage_name, stats in self.stage_stats.items():
            summary.append(f"\n{stage_name}:")
            summary.append(f"  Agents: {stats['agents']}")
            summary.append(f"  Prompts: {stats['prompts']}")
            summary.append(f"  Signals: {stats['signals_deposited']}")
            summary.append(f"  Time: {stats['time']:.2f}s")
            summary.append(f"  Efficiency: {stats['signals_deposited']/stats['prompts']*100 if stats['prompts'] > 0 else 0:.1f}% deposit rate")

            total_time += stats['time']
            total_signals += stats['signals_deposited']
            total_prompts += stats['prompts']

        summary.append("\n" + "=" * 60)
        summary.append(f"Total Time: {total_time:.2f}s")
        summary.append(f"Total Prompts: {total_prompts}")
        summary.append(f"Total Signals: {total_signals}")
        summary.append(f"Overall Efficiency: {total_signals/total_prompts*100 if total_prompts > 0 else 0:.1f}% deposit rate")

        return "\n".join(summary)


class StagedAgent:
    """Mixin class for agents supporting staged execution.

    Agents should inherit from this and implement:
    - prepare_prompt(): Prepare prompt without LLM call
    - process_result(): Process LLM result and deposit signal
    """

    async def prepare_prompt(self, signal_store: SignalStore) -> Optional[tuple]:
        """Prepare prompt for this agent (no LLM call).

        Returns:
            (agent_id, prompt, max_tokens, temperature) or None if nothing to do
        """
        raise NotImplementedError("Subclass must implement prepare_prompt()")

    async def process_result(self, result: str, signal_store: SignalStore) -> bool:
        """Process LLM result and deposit signal.

        Args:
            result: Generated text from LLM
            signal_store: Where to deposit

        Returns:
            True if signal deposited, False otherwise
        """
        raise NotImplementedError("Subclass must implement process_result()")
