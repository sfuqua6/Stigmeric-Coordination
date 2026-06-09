"""Stigmergic swarm system - main entry point."""

import asyncio
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from swarm.core.signal_store import SignalStore
from swarm.core.config import *
from swarm.llm.simple_llm import SimpleLLM
from swarm.agents.scout import Scout
from swarm.agents.forager import Forager


async def run_swarm():
    """Run the stigmergic swarm system."""

    print("=" * 70)
    print("STIGMERGIC SWARM SYSTEM v1.0")
    print("=" * 70)
    print(f"\nThesis: {THESIS}")
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME} on {DEVICE}")
    print(f"  Scouts: {NUM_SCOUTS}, Foragers: {NUM_FORAGERS}")
    print(f"  Iterations: {MAX_ITERATIONS}")
    print(f"  Decay rate: {DECAY_RATE}, Prune threshold: {PRUNE_THRESHOLD}\n")

    # Initialize components
    signal_store = SignalStore(decay_rate=DECAY_RATE, prune_threshold=PRUNE_THRESHOLD)
    llm = SimpleLLM(MODEL_NAME, DEVICE)

    # Load model once
    print("[INIT] Loading language model...")
    start = time.time()
    await llm.load()
    print(f"[INIT] Model loaded in {time.time() - start:.1f}s\n")

    # Create agents
    scouts = [
        Scout(f"Scout_Claim_{i}", "CLAIM", THESIS) for i in range(NUM_SCOUTS)
    ]

    foragers = [
        Forager(f"Forager_Evidence_{i}", "CLAIM", "EVIDENCE", THESIS)
        for i in range(NUM_FORAGERS)
    ]

    print(f"[INIT] Created {len(scouts)} scouts and {len(foragers)} foragers\n")

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

    # Environmental processes (decay and pruning)
    async def environment_process():
        """Periodic decay and pruning."""
        for iteration in range(MAX_ITERATIONS):
            await asyncio.sleep(ITERATION_DELAY)

            # Decay signals
            remaining = signal_store.decay_all()

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

    # Show top signals of each type
    for signal_type in ["CLAIM", "EVIDENCE", "CRITIQUE"]:
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
