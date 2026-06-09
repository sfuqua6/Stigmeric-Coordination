"""Profile swarm execution to find actual bottlenecks.

Usage:
    python profile_swarm.py

Generates:
    - profiling_stats.txt - Text report of top time consumers
    - profiling.prof - Raw cProfile data for visualization
"""

import cProfile
import pstats
import io
import os
import sys

# Set hyper test mode for fast profiling
os.environ['HYPER_TEST_MODE'] = '1'

# Import after setting env var
from run_task import run_task
import asyncio
import swarm.core.config as config_module

# Configure for fast test
config_module.MAX_ITERATIONS = 5
config_module.NUM_SCOUTS = 3
config_module.NUM_FORAGERS = 2
config_module.NUM_CRITICS = 1
config_module.NUM_HATERS = 1

def main():
    """Run profiled swarm execution."""
    print("=" * 70)
    print("PROFILING SWARM EXECUTION")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Mode: hyper_test (fast)")
    print(f"  Iterations: {config_module.MAX_ITERATIONS}")
    print(f"  Agents: {config_module.NUM_SCOUTS} scouts, {config_module.NUM_FORAGERS} foragers")
    print(f"  Profiler: cProfile")
    print("\nStarting profiling...\n")

    # Profile the execution
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        # Run the swarm
        asyncio.run(run_task("problem_solving", "How can we reduce plastic waste?"))
    except Exception as e:
        print(f"\n⚠️  Execution error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        profiler.disable()

    # Save raw profile data
    profiler.dump_stats("profiling.prof")
    print("\n✓ Saved raw profile data to profiling.prof")

    # Generate text report
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)

    print("\n" + "=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)

    # Sort by cumulative time (time spent in function + callees)
    print("\n### TOP 20 FUNCTIONS BY CUMULATIVE TIME ###")
    print("(Total time spent in function and all functions it calls)\n")
    ps.sort_stats('cumulative')
    ps.print_stats(20)

    # Sort by internal time (time in function only)
    print("\n### TOP 20 FUNCTIONS BY INTERNAL TIME ###")
    print("(Time spent in function itself, excluding callees)\n")
    ps.sort_stats('time')
    ps.print_stats(20)

    # Save full report
    with open("profiling_stats.txt", "w") as f:
        # Cumulative time
        f.write("=" * 70 + "\n")
        f.write("PROFILING RESULTS - FULL REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("### TOP 50 FUNCTIONS BY CUMULATIVE TIME ###\n\n")
        ps_file = pstats.Stats(profiler, stream=f)
        ps_file.sort_stats('cumulative')
        ps_file.print_stats(50)

        # Internal time
        f.write("\n\n### TOP 50 FUNCTIONS BY INTERNAL TIME ###\n\n")
        ps_file.sort_stats('time')
        ps_file.print_stats(50)

        # Callers of signal_store methods
        f.write("\n\n### SIGNAL_STORE METHOD CALLS ###\n\n")
        ps_file.print_callers('signal_store')

    print("\n✓ Saved full report to profiling_stats.txt")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Next steps:
1. Review profiling_stats.txt for bottlenecks
2. Look for functions taking >10% of total time
3. Check if signal_store operations are slow
4. Identify actual problems, not imagined ones

Visualization (optional):
  pip install snakeviz
  snakeviz profiling.prof
""")

if __name__ == "__main__":
    main()
