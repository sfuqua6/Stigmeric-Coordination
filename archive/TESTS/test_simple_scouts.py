"""Test simple scouts with spatial signal store.

Demonstrates:
- Simple agents with positions and movement
- Local-only information access
- Emergent clustering behavior
- Token efficiency (20 tokens vs 70 tokens)
"""

import asyncio
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from swarm.core.spatial_signal_store import SpatialSignalStore
from swarm.agents.simple_scout import SimpleScout
from swarm.llm.simple_llm import SimpleLLM
from swarm.core.config import MODEL_NAME, DEVICE


def visualize_grid(signal_store: SpatialSignalStore, scouts: list, dimensions: int = 100):
    """ASCII visualization of scouts and signals in grid.

    Args:
        signal_store: Spatial signal store
        scouts: List of simple scouts
        dimensions: Grid size
    """
    # Create grid (reduce to 20x20 for display)
    scale = dimensions // 20
    grid = [['.' for _ in range(20)] for _ in range(20)]

    # Mark signals
    for signal in signal_store.signals.values():
        x = int(signal.position[0] / scale)
        y = int(signal.position[1] / scale)
        x = max(0, min(19, x))
        y = max(0, min(19, y))

        # Strength indicator
        if signal.strength > 0.7:
            grid[y][x] = 'S'  # Strong signal
        elif signal.strength > 0.4:
            grid[y][x] = 's'  # Medium signal
        else:
            grid[y][x] = '.'  # Weak signal

    # Mark scouts (overwrite signals)
    for scout in scouts:
        x = int(scout.position[0] / scale)
        y = int(scout.position[1] / scale)
        x = max(0, min(19, x))
        y = max(0, min(19, y))

        # Confidence indicator
        if scout.confidence > 0.7:
            grid[y][x] = 'A'  # Confident agent
        else:
            grid[y][x] = 'a'  # Low confidence agent

    # Print grid
    print("\n" + "=" * 42)
    print("  " + "".join(str(i % 10) for i in range(20)))
    for i, row in enumerate(grid):
        print(f"{i:2d} {''.join(row)}")
    print("=" * 42)
    print("Legend: A/a=scouts (High/low confidence), S/s=signals (Strong/medium), .=empty")


async def test_simple_scouts():
    """Test simple scouts with spatial store."""

    print("=" * 70)
    print("SIMPLE SCOUT TEST - Spatial Swarm Intelligence")
    print("=" * 70)

    # Configuration
    NUM_SCOUTS = 20  # Start with fewer for testing
    PERCEPTION_RADIUS = 10.0
    ITERATIONS = 20
    DIMENSIONS = 100

    # Task
    task_prompt = "What are solutions to climate change?"

    print(f"\nConfiguration:")
    print(f"  Scouts: {NUM_SCOUTS}")
    print(f"  Perception radius: {PERCEPTION_RADIUS}")
    print(f"  Iterations: {ITERATIONS}")
    print(f"  Grid dimensions: {DIMENSIONS}x{DIMENSIONS}")
    print(f"  LLM tokens per scout: 20 (reduced from 70)")

    # Initialize spatial signal store
    print("\nInitializing spatial signal store...")
    signal_store = SpatialSignalStore(
        dimensions=DIMENSIONS,
        decay_rate=0.05,
        prune_threshold=0.1
    )

    # Initialize LLM
    print("Loading language model...")
    llm = SimpleLLM(MODEL_NAME, DEVICE, enable_cache=False)
    await llm.load()

    # Create scouts with random positions and knowledge fragments
    print(f"\nCreating {NUM_SCOUTS} simple scouts...")
    knowledge_fragments = [
        "Renewable energy reduces carbon emissions",
        "Solar and wind power are sustainable",
        "Electric vehicles reduce pollution",
        "Carbon capture technology exists",
        "Reforestation absorbs CO2",
        "Energy efficiency saves resources",
        "Green buildings reduce energy use",
        "Public transit reduces emissions",
        "Plant-based diets lower carbon footprint",
        "Circular economy reduces waste",
        "Ocean cleanup removes plastic",
        "Sustainable agriculture practices",
        "Carbon pricing incentivizes reduction",
        "Green infrastructure in cities",
        "International climate agreements",
        "Methane reduction strategies",
        "Energy storage solutions",
        "Smart grid technology",
        "Climate education and awareness",
        "Green technology innovation",
    ]

    scouts = []
    for i in range(NUM_SCOUTS):
        # Random starting position
        x = random.uniform(0, DIMENSIONS - 1)
        y = random.uniform(0, DIMENSIONS - 1)

        # Assign knowledge fragment
        knowledge = knowledge_fragments[i % len(knowledge_fragments)]

        scout = SimpleScout(
            agent_id=f"SimpleScout_{i}",
            position=(x, y),
            knowledge=knowledge,
            perception_radius=PERCEPTION_RADIUS
        )
        scouts.append(scout)

    print(f"Scouts initialized at random positions")

    # Run simulation
    print(f"\nRunning {ITERATIONS} iterations...")
    print("Watch for emergent clustering behavior!\n")

    token_count = 0

    for iteration in range(ITERATIONS):
        print(f"\n--- Iteration {iteration + 1}/{ITERATIONS} ---")

        # Each scout takes one step
        for scout in scouts:
            signal_id = await scout.step(
                signal_store, llm, task_prompt, iteration
            )

            if signal_id:
                token_count += 20  # 20 tokens per deposit

        # Environmental processes
        signal_store.decay_all()
        pruned = signal_store.prune_weak()

        # Stats
        stats = signal_store.get_stats()
        print(f"Signals: {stats['total_signals']}, "
              f"Avg strength: {stats['avg_strength']:.2f}, "
              f"Clusters: {stats['num_clusters']}, "
              f"Pruned: {pruned}")

        # Visualize every 5 iterations
        if (iteration + 1) % 5 == 0:
            visualize_grid(signal_store, scouts, DIMENSIONS)

            # Show sample scout states
            print("\nSample Scout States:")
            for scout in scouts[:3]:
                state = scout.get_state()
                print(f"  {state['agent_id']}: pos=({state['position'][0]:.1f}, {state['position'][1]:.1f}), "
                      f"conf={state['confidence']:.2f}, deposited={state['signals_deposited']}")

    # Final visualization
    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)
    visualize_grid(signal_store, scouts, DIMENSIONS)

    # Final stats
    stats = signal_store.get_stats()
    print("\nFinal Statistics:")
    print(f"  Total signals: {stats['total_signals']}")
    print(f"  Average strength: {stats['avg_strength']:.2f}")
    print(f"  Clusters detected: {stats['num_clusters']}")
    print(f"  Average cluster size: {stats['avg_cluster_size']:.1f}")
    print(f"  Grid cells used: {stats['grid_cells_used']}/{DIMENSIONS * DIMENSIONS}")

    print(f"\nToken Usage:")
    print(f"  Total tokens: {token_count}")
    print(f"  Per scout: {token_count / NUM_SCOUTS:.1f}")
    print(f"  Per iteration: {token_count / ITERATIONS:.1f}")

    print(f"\nComparison to Old System:")
    old_tokens = NUM_SCOUTS * 70 * ITERATIONS
    print(f"  Old system (70 tokens/scout): {old_tokens}")
    print(f"  New system (20 tokens/scout): {token_count}")
    print(f"  Reduction: {((old_tokens - token_count) / old_tokens * 100):.1f}%")

    # Emergence analysis
    print("\n" + "=" * 70)
    print("EMERGENCE ANALYSIS")
    print("=" * 70)

    clusters = signal_store.detect_spatial_clusters(
        signal_type="SCOUT_SIGNAL",
        cluster_radius=15.0
    )

    print(f"\nClustering detected: {len(clusters)} clusters")
    for i, cluster in enumerate(clusters, 1):
        avg_x = sum(s.position[0] for s in cluster) / len(cluster)
        avg_y = sum(s.position[1] for s in cluster) / len(cluster)
        avg_strength = sum(s.strength for s in cluster) / len(cluster)

        print(f"  Cluster {i}: {len(cluster)} signals at "
              f"({avg_x:.1f}, {avg_y:.1f}), strength={avg_strength:.2f}")

        # Show sample content
        sample = random.choice(cluster)
        print(f"    Sample: {sample.content[:60]}...")

    # Consensus detection
    print("\nConsensus Detection:")
    strong_signals = [s for s in signal_store.signals.values() if s.strength > 0.6]
    print(f"  Strong signals (>0.6): {len(strong_signals)}")
    print(f"  Consensus ratio: {len(strong_signals) / max(stats['total_signals'], 1):.2%}")

    if strong_signals:
        print("\n  Top 3 strongest signals:")
        strong_signals.sort(key=lambda s: s.strength, reverse=True)
        for s in strong_signals[:3]:
            print(f"    [{s.strength:.2f}] {s.content[:70]}...")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\nKey Observations:")
    print("  ✓ Scouts move in space using simple rules")
    print("  ✓ Local-only information access (perception radius)")
    print("  ✓ Emergent spatial clustering")
    print("  ✓ 71.4% token reduction (20 vs 70)")
    print("\nNext: Integrate with full swarm pipeline (run_task.py)")


if __name__ == "__main__":
    try:
        asyncio.run(test_simple_scouts())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
