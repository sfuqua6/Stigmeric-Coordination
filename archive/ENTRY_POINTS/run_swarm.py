"""Run the stigmergic swarm system with full agent registry."""

import asyncio
import time
from swarm.core.signal_store import SignalStore
from swarm.core.config import *
from swarm.llm.simple_llm import SimpleLLM
from swarm.agents.scout import Scout
from swarm.agents.forager import Forager
from swarm.agents.critic import Critic
from swarm.agents.hater import Hater


async def run_swarm():
    """Run the stigmergic swarm system."""

    print("=" * 70)
    print("STIGMERGIC SWARM SYSTEM v1.0")
    print("=" * 70)
    print(f"\nThesis: {THESIS}")
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME} on {DEVICE}")
    print(f"  Agents: {NUM_SCOUTS} scouts, {NUM_FORAGERS} foragers, "
          f"{NUM_CRITICS} critics, {NUM_HATERS} haters")
    print(f"  Temperatures: Scout={TEMP_SCOUT}, Forager={TEMP_FORAGER}, "
          f"Critic={TEMP_CRITIC}, Hater={TEMP_HATER}")
    print(f"  Iterations: {MAX_ITERATIONS}")
    print(f"  Decay rate: {DECAY_RATE}, Prune threshold: {PRUNE_THRESHOLD}")
    print(f"  Thresholds: Deposit={MIN_DEPOSIT_STRENGTH}, Amplify={MIN_AMPLIFY_STRENGTH}\n")

    # Initialize components with diversity checking and exploration bonuses
    signal_store = SignalStore(
        decay_rate=DECAY_RATE,
        prune_threshold=PRUNE_THRESHOLD,
        diversity_threshold=0.95,  # Only reject 95%+ similar signals
        exploration_bonus=0.3  # Boost under-visited signals
    )
    llm = SimpleLLM(MODEL_NAME, DEVICE)

    # Load model once
    print("[INIT] Loading language model...")
    start = time.time()
    await llm.load()
    print(f"[INIT] Model loaded in {time.time() - start:.1f}s\n")

    # Create agents - full agent registry
    scouts = [
        Scout(f"Scout_Claim_{i}", "CLAIM", THESIS) for i in range(NUM_SCOUTS)
    ]

    # Evidence foragers (CLAIM → EVIDENCE)
    foragers_evidence = [
        Forager(f"Forager_Evidence_{i}", "CLAIM", "EVIDENCE", THESIS)
        for i in range(NUM_FORAGERS // 2)
    ]

    # Critic foragers (CLAIM → CRITIQUE) - new!
    foragers_critique = [
        Forager(f"Forager_Critique_{i}", "CLAIM", "CRITIQUE", THESIS)
        for i in range(NUM_FORAGERS // 2)
    ]

    # Critic agents
    critics = [
        Critic(f"Critic_{i}", THESIS) for i in range(NUM_CRITICS)
    ]

    # Hater agents (adversarial)
    haters = [
        Hater(f"Hater_{i}", THESIS) for i in range(NUM_HATERS)
    ]

    foragers = foragers_evidence + foragers_critique
    all_agents = scouts + foragers + critics + haters

    print(f"[INIT] Created {len(scouts)} scouts, {len(foragers)} foragers "
          f"({len(foragers_evidence)} evidence, {len(foragers_critique)} critique), "
          f"{len(critics)} critics, {len(haters)} haters\n")

    # Launch all agents concurrently (decentralized!)
    print("[START] Launching swarm (agents run independently)...\n")

    tasks = []

    # Scout tasks
    for scout in scouts:
        task = asyncio.create_task(
            scout.run(signal_store, llm,
                     min_strength=MIN_DEPOSIT_STRENGTH,
                     max_actions=MAX_ITERATIONS)
        )
        tasks.append(task)

    # Forager tasks
    for forager in foragers:
        task = asyncio.create_task(
            forager.run(signal_store, llm, max_actions=MAX_ITERATIONS)
        )
        tasks.append(task)

    # Critic tasks
    for critic in critics:
        task = asyncio.create_task(
            critic.run(signal_store, llm, max_actions=MAX_ITERATIONS)
        )
        tasks.append(task)

    # Hater tasks (adversarial)
    for hater in haters:
        task = asyncio.create_task(
            hater.run(signal_store, llm,
                     min_strength=MIN_DEPOSIT_STRENGTH,
                     max_actions=MAX_ITERATIONS,
                     temperature=TEMP_HATER)
        )
        tasks.append(task)

    # Environmental processes (decay and pruning)
    async def environment_process():
        """Periodic decay and pruning."""
        for iteration in range(MAX_ITERATIONS):
            await asyncio.sleep(ITERATION_DELAY)

            # Decay signals with contrarian boost to offset echo effects
            contrarian_types = ["CRITIQUE", "COUNTER_EVIDENCE"]
            remaining = signal_store.decay_all(contrarian_types=contrarian_types)

            # Prune weak signals
            pruned = signal_store.prune_weak()

            # Stats
            stats = signal_store.get_stats()
            print(f"\n[ITER {iteration+1:02d}] Environment update:")
            print(f"  Signals: {stats['total_signals']} "
                  f"(pruned {pruned}, avg strength {stats['avg_strength']:.2f})")
            print(f"  By type: {stats['by_type']}")

    tasks.append(asyncio.create_task(environment_process()))

    # Wait for all tasks to complete - use return_exceptions to allow individual agent failures
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Log any exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[ERROR] Task {i} failed: {type(result).__name__}: {result}")

    # Final results
    print("\n" + "=" * 70)
    print("SWARM COMPLETE - RESULTS")
    print("=" * 70)

    # Show cache and generation stats
    cache_stats = llm.get_cache_stats()
    print(f"\n--- LLM Statistics ---")
    print(f"Cache hits: {cache_stats['cache_hits']}")
    print(f"Cache misses: {cache_stats['cache_misses']}")
    print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"Cache size: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
    print(f"\nGeneration successes: {cache_stats['generation_successes']}")
    print(f"Generation failures: {cache_stats['generation_failures']}")
    print(f"Success rate: {cache_stats['success_rate']:.2%}")

    # Show top signals of each type (including new types)
    for signal_type in ["CLAIM", "EVIDENCE", "CRITIQUE", "COUNTER_EVIDENCE"]:
        top_signals = signal_store.get_top_signals(signal_type, TOP_N_RESULTS)

        if top_signals:
            print(f"\n--- Top {signal_type}s (by signal strength) ---")
            for i, signal in enumerate(top_signals, 1):
                print(f"\n{i}. [Strength: {signal.strength:.3f}, Visits: {signal.visits}]")
                print(f"   {signal.content[:200]}")
                if len(signal.content) > 200:
                    print(f"   ...")
        else:
            print(f"\n--- No {signal_type}s generated ---")

    # Final stats
    final_stats = signal_store.get_stats()
    print(f"\n--- Final Statistics ---")
    print(f"Total signals: {final_stats['total_signals']}")
    print(f"By type: {final_stats['by_type']}")
    print(f"Average strength: {final_stats['avg_strength']:.3f}")
    print(f"Strongest signal: {final_stats['strongest']:.3f}")

    print("\n" + "=" * 70)
    print("Stigmergic coordination complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_swarm())
