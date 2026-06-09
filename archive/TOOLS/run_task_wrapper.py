"""Simplified wrapper for running swarm tasks for benchmarking.

This wrapper provides a clean interface for comparative evaluation,
extracting the core swarm functionality without the full run_task.py complexity.
"""

import asyncio
import time
from typing import Dict
from pathlib import Path

from swarm.core.signal_store import SignalStore
from swarm.core.config import *
from swarm.core.task_config import create_custom_task
from swarm.llm.simple_llm import SimpleLLM
from swarm.agents.scout import Scout
from swarm.agents.forager import Forager
from swarm.agents.critic import Critic
from swarm.agents.synthesizer import Synthesizer
from swarm.retrieval.simple_web_search import web_search


async def run_swarm_simple(prompt: str, task_type: str = 'analysis', max_iterations: int = 20, quick_mode: bool = False) -> Dict:
    """Run swarm in simplified mode for benchmarking.

    Args:
        prompt: Task prompt
        task_type: Task type (debate, creative, analysis, problem_solving)
        max_iterations: Maximum iterations
        quick_mode: If True, use minimal agents for fast testing (3 scouts, 1 forager, 1 critic)

    Returns:
        Dictionary with:
        - signal_store: Final signal store
        - synthesis: Synthesized output
        - llm_generations: Count of LLM calls
        - total_signals: Total signals created
        - runtime: Execution time in seconds
    """
    start_time = time.time()

    # Create task config
    task_config = create_custom_task(task_type, prompt)

    # Initialize signal store
    signal_store = SignalStore(
        decay_rate=DECAY_RATE,
        prune_threshold=PRUNE_THRESHOLD,
        diversity_threshold=DIVERSITY_THRESHOLD,
        exploration_bonus=EXPLORATION_BONUS
    )

    # Initialize LLM
    llm = SimpleLLM(MODEL_NAME, DEVICE, enable_cache=ENABLE_LLM_CACHE, cache_size=LLM_CACHE_SIZE)
    await llm.load()

    # Create agents (quick mode uses minimal agents for speed)
    if quick_mode:
        num_scouts = 3
        num_foragers = 1
        num_critics = 1
        print(f"[SWARM] Quick mode: {num_scouts} scouts, {num_foragers} foragers, {num_critics} critics")
    else:
        num_scouts = 5
        num_foragers = 2
        num_critics = 1

    scouts = [
        Scout(f"Scout_{i}", task_config.signal_types["initial"], task_config.task_prompt)
        for i in range(num_scouts)
    ]

    foragers_support = [
        Forager(
            f"Forager_Support_{i}",
            mode="creative",
            output_type=task_config.signal_types["support"],
            thesis=task_config.task_prompt
        )
        for i in range(num_foragers)
    ]
    for f in foragers_support:
        f.input_type = task_config.signal_types["initial"]

    critics = [
        Critic(f"Critic_{i}", mode="creative", thesis=task_config.task_prompt)
        for i in range(num_critics)
    ]
    for c in critics:
        c.evaluate_type = task_config.signal_types["initial"]

    # Launch agents
    print(f"[SWARM] Launching {num_scouts} scouts, {num_foragers} foragers, {num_critics} critics...")
    print(f"[SWARM] Running for {max_iterations} iterations...")

    tasks = []

    # Scouts
    for scout in scouts:
        tasks.append(asyncio.create_task(
            scout.run(signal_store, llm, MIN_DEPOSIT_STRENGTH, max_iterations, web_search_fn=web_search)
        ))

    # Foragers
    for forager in foragers_support:
        tasks.append(asyncio.create_task(
            forager.run(signal_store, llm, max_actions=max_iterations)
        ))

    # Critics
    for critic in critics:
        tasks.append(asyncio.create_task(
            critic.run(signal_store, llm, max_actions=max_iterations)
        ))

    # Environment process
    async def environment_process():
        contrarian_types = [task_config.signal_types["critique"]]

        for iteration in range(max_iterations):
            await asyncio.sleep(ITERATION_DELAY)
            signal_store.decay_all(contrarian_types=contrarian_types)
            signal_store.prune_weak()

            if (iteration + 1) % 5 == 0:
                stats = signal_store.get_stats()
                print(f"[SWARM] [ITER {iteration+1:02d}] Signals: {stats['total_signals']}, "
                      f"Avg strength: {stats['avg_strength']:.2f}")

    tasks.append(asyncio.create_task(environment_process()))

    # Wait for completion
    await asyncio.gather(*tasks, return_exceptions=True)

    # Synthesize
    print("[SWARM] Synthesizing results...")
    synthesizer = Synthesizer("Synthesizer", task_config.task_prompt)
    synthesis = await synthesizer.synthesize(
        signal_store,
        llm,
        task_config.signal_types,
        temperature=TEMP_SYNTHESIZER
    )

    # Get metrics
    cache_stats = llm.get_cache_stats()
    signal_stats = signal_store.get_stats()

    runtime = time.time() - start_time

    print(f"[SWARM] Complete in {runtime:.2f}s")
    print(f"[SWARM] Signals: {signal_stats['total_signals']}")
    print(f"[SWARM] LLM generations: {cache_stats['generation_successes'] + cache_stats['generation_failures']}")

    return {
        'signal_store': signal_store,
        'synthesis': synthesis,
        'llm_generations': cache_stats['generation_successes'] + cache_stats['generation_failures'],
        'total_signals': signal_stats['total_signals'],
        'runtime': runtime,
        'cache_stats': cache_stats,
        'signal_stats': signal_stats
    }
