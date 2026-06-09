"""Monolith-breaking document processor - main orchestration.

This module implements distributed document processing that breaks the
context window monolith problem through stigmergic swarm intelligence.
"""

import asyncio
import time
from pathlib import Path
from typing import List, Optional

# Optional numpy import with fallback
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[WARN] NumPy not available - using fallback Gini coefficient calculation")

from .documents import DocumentProcessor
from .core.signal_store import SignalStore
from .core.config import DEVICE, DECAY_RATE, PRUNE_THRESHOLD, DIVERSITY_THRESHOLD, EXPLORATION_BONUS
from .core.agent_wrapper import AgentExecutionWrapper
from .llm.simple_llm import SimpleLLM
from .agents.scout import Scout
from .agents.forager import Forager
from .agents.gatherer import Gatherer
from .agents.critic import Critic
from .agents.hater import Hater
from .agents.synthesizer import Synthesizer
from .agents.monitor import SwarmMonitor


def calculate_gini_coefficient(strengths: List[float]) -> float:
    """Calculate Gini coefficient of strength distribution.

    Measures inequality in strength distribution. 0 = uniform, 1 = winner-take-all.
    Target is >0.7 for convergence.

    Args:
        strengths: List of signal strengths

    Returns:
        Gini coefficient (0.0-1.0)
    """
    if not strengths or len(strengths) < 2:
        return 0.0

    strengths = sorted(strengths)
    n = len(strengths)

    if HAS_NUMPY:
        # Efficient numpy implementation
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * strengths)) / (n * np.sum(strengths)) - (n + 1) / n
    else:
        # Pure Python fallback
        sum_strengths = sum(strengths)
        if sum_strengths == 0:
            return 0.0

        weighted_sum = sum((i + 1) * s for i, s in enumerate(strengths))
        return (2 * weighted_sum) / (n * sum_strengths) - (n + 1) / n


async def run_document_swarm(
    document_paths: List[str],
    model_name: str = "microsoft/phi-2",
    num_foragers: int = 50,
    num_gatherers: int = 20,
    num_critics: int = 20,
    num_haters: int = 10,
    max_iterations: int = 500,
    convergence_threshold: float = 0.7,
    enable_mcp: bool = False
):
    """Run distributed document processing swarm.

    Args:
        document_paths: List of paths to documents to process
        model_name: HuggingFace model name
        num_foragers: Number of forager agents
        num_gatherers: Number of gatherer agents
        num_critics: Number of critic agents
        num_haters: Number of hater agents
        max_iterations: Maximum total iterations
        convergence_threshold: Gini coefficient threshold for convergence
        enable_mcp: Enable MCP for external validation

    Returns:
        Dictionary with results
    """
    print("=" * 80)
    print("STIGMERGIC MONOLITH-BREAKING DOCUMENT PROCESSOR")
    print("=" * 80)

    start_time = time.time()

    # ========================================================================
    # PHASE 1: DOCUMENT LOADING
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: DOCUMENT LOADING")
    print("=" * 80)

    processor = DocumentProcessor(target_tokens=2000, tolerance=500)
    sections = processor.process_documents(document_paths)

    if not sections:
        print("\nERROR: No sections created from documents")
        return None

    total_tokens = sum(s.token_count for s in sections)
    print(f"\n[PHASE 1] Document loading complete:")
    print(f"  Documents: {len(document_paths)}")
    print(f"  Sections: {len(sections)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Avg tokens/section: {total_tokens // len(sections)}")

    # ========================================================================
    # PHASE 2: INFRASTRUCTURE INITIALIZATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: INFRASTRUCTURE INITIALIZATION")
    print("=" * 80)

    # Create signal store
    signal_store = SignalStore(
        decay_rate=DECAY_RATE,
        prune_threshold=PRUNE_THRESHOLD,
        diversity_threshold=DIVERSITY_THRESHOLD,
        exploration_bonus=EXPLORATION_BONUS
    )

    # Create LLM wrapper with quantization
    llm = SimpleLLM(model_name, DEVICE, enable_cache=True, cache_size=50)

    print(f"\n[PHASE 2] Loading model: {model_name}")
    await llm.load()

    # Create MCP client if enabled
    mcp_client = None
    if enable_mcp:
        try:
            from .knowledge.mcp_client import MCPClient
            mcp_client = MCPClient()
            print(f"[PHASE 2] MCP client initialized")
        except ImportError:
            print(f"[PHASE 2] WARNING: MCP client not available, validation will be limited")

    # Create SwarmMonitor for health monitoring and self-healing
    monitor = SwarmMonitor(agent_id="SwarmMonitor")
    print(f"[PHASE 2] SwarmMonitor initialized (self-healing enabled)")

    print(f"\n[PHASE 2] Infrastructure ready")

    # Launch monitor as background task
    monitor_task = asyncio.create_task(monitor.run(signal_store, max_iterations=100))
    print(f"[PHASE 2] SwarmMonitor launched as background task")

    # ========================================================================
    # PHASE 3: DISTRIBUTED READING (PARALLEL INPUT PROCESSING)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: DISTRIBUTED READING")
    print("=" * 80)

    print(f"\n[PHASE 3] Creating {len(sections)} scouts (1 per section)")

    # Create one scout per section
    scouts = [
        Scout(f"Scout_{i}", section=section)
        for i, section in enumerate(sections)
    ]

    # Launch ALL scouts simultaneously with robust error handling
    print(f"[PHASE 3] Launching {len(scouts)} scouts in parallel...")
    scout_start = time.time()

    wrapper = AgentExecutionWrapper(enable_retry=True, max_retries=1)
    await wrapper.run_agent_swarm(scouts, signal_store, llm, max_actions=1)

    scout_time = time.time() - scout_start

    # Report results
    stats = signal_store.get_stats()
    observation_count = stats['by_type'].get('OBSERVATION', 0)

    print(f"\n[PHASE 3] Distributed reading complete ({scout_time:.1f}s)")
    print(f"  Observations extracted: {observation_count}")
    print(f"  Avg observations/section: {observation_count / len(sections):.1f}")
    print(f"  Throughput: {len(sections) / scout_time:.1f} sections/second")

    if observation_count == 0:
        print("\nERROR: No observations extracted! Check model output.")
        return None

    # ========================================================================
    # PHASE 4: PATTERN FINDING (DISTRIBUTED REASONING)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: PATTERN FINDING")
    print("=" * 80)

    print(f"\n[PHASE 4] Creating {num_foragers} foragers for pattern discovery")

    foragers = [
        Forager(f"Forager_{i}", mode="document")
        for i in range(num_foragers)
    ]

    # Run foragers for pattern finding phase
    pattern_iterations = 150
    print(f"[PHASE 4] Running pattern finding for {pattern_iterations} iterations...")

    # Launch foragers with robust wrapper (non-blocking)
    async def run_foragers_with_monitoring():
        """Run foragers while monitoring progress."""
        wrapper = AgentExecutionWrapper(enable_retry=True, max_retries=1)
        forager_task = asyncio.create_task(
            wrapper.run_agent_swarm(foragers, signal_store, llm, max_actions=pattern_iterations)
        )

        # Monitor progress while foragers run
        for i in range(0, pattern_iterations, 50):
            remaining = min(50, pattern_iterations - i)

            if i > 0:
                stats = signal_store.get_stats()
                insight_count = stats['by_type'].get('INSIGHT', 0)
                print(f"\n[PHASE 4] Iteration {i}: {insight_count} insights discovered")

            await asyncio.sleep(remaining * 0.5)

        # Ensure foragers finish
        await forager_task

        # Stop any still-running foragers
        for forager in foragers:
            forager.stop()

    await run_foragers_with_monitoring()

    stats = signal_store.get_stats()
    insight_count = stats['by_type'].get('INSIGHT', 0)

    print(f"\n[PHASE 4] Pattern finding complete")
    print(f"  Insights discovered: {insight_count}")
    print(f"  Avg insights/forager: {insight_count / num_foragers:.1f}")

    # ========================================================================
    # PHASE 5: VALIDATION LOOP (DISTRIBUTED EVIDENCE GATHERING)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 5: VALIDATION LOOP")
    print("=" * 80)

    print(f"\n[PHASE 5] Creating validation agents:")
    print(f"  Gatherers: {num_gatherers}")
    print(f"  Critics: {num_critics}")
    print(f"  Haters: {num_haters}")

    # Create validation agents
    gatherers = [
        Gatherer(f"Gatherer_{i}", mcp_client=mcp_client)
        for i in range(num_gatherers)
    ]

    critics = [
        Critic(f"Critic_{i}", mode="document")
        for i in range(num_critics)
    ]

    haters = [
        Hater(f"Hater_{i}", task_prompt="Challenge insights")
        for i in range(num_haters)
    ]

    validation_iterations = 200  # Increased to match forager power
    print(f"[PHASE 5] Running validation for {validation_iterations} iterations...")
    print(f"[PHASE 5] Enhanced mode: Critics use full context, Haters target consensus")

    # Launch validation agents with robust wrapper
    async def run_validation_with_monitoring():
        """Run validation agents while monitoring progress."""
        all_validators = gatherers + critics + haters
        wrapper = AgentExecutionWrapper(enable_retry=True, max_retries=1)

        # Create tasks for different agent types (they have different signatures)
        async def run_gatherer(g):
            return await wrapper.run_agent_safe(g, signal_store, llm, max_actions=validation_iterations)

        async def run_critic(c):
            # Critics now use enhanced_context=True by default (full provenance, reasoned critiques)
            return await wrapper.run_agent_safe(c, signal_store, llm, max_actions=validation_iterations)

        async def run_hater(h):
            # Haters now use target_consensus=True by default (challenge groupthink, not strongest)
            return await wrapper.run_agent_safe(h, signal_store, llm,
                                               max_actions=validation_iterations,
                                               temperature=0.85,
                                               target_consensus=True)

        # Launch all validators
        tasks = []
        tasks.extend([run_gatherer(g) for g in gatherers])
        tasks.extend([run_critic(c) for c in critics])
        tasks.extend([run_hater(h) for h in haters])

        validation_task = asyncio.create_task(asyncio.gather(*tasks, return_exceptions=True))

        # Monitor progress
        for i in range(0, validation_iterations, 50):
            remaining = min(50, validation_iterations - i)

            if i > 0:
                stats = signal_store.get_stats()
                evidence_count = stats['by_type'].get('EVIDENCE', 0)
                objection_count = stats['by_type'].get('OBJECTION', 0)
                print(f"\n[PHASE 5] Iteration {i}:")
                print(f"  Evidence gathered: {evidence_count}")
                print(f"  Objections raised: {objection_count}")
                print(f"  Avg strength: {stats['avg_strength']:.2f}")

            await asyncio.sleep(remaining * 0.6)

        # Ensure all finish
        await validation_task

        # Stop any still-running agents
        for gatherer in gatherers:
            gatherer.stop()
        for critic in critics:
            critic.stop()
        for hater in haters:
            hater.stop()

        # Print wrapper stats
        wrapper.print_summary()

    await run_validation_with_monitoring()

    stats = signal_store.get_stats()
    evidence_count = stats['by_type'].get('EVIDENCE', 0)
    objection_count = stats['by_type'].get('OBJECTION', 0)

    print(f"\n[PHASE 5] Validation complete")
    print(f"  Evidence gathered: {evidence_count}")
    print(f"  Objections raised: {objection_count}")

    # ========================================================================
    # PHASE 5B: AGENT DIALOGUE (Foragers defend, Haters engage)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 5B: AGENT DIALOGUE")
    print("=" * 80)

    print(f"\n[PHASE 5B] Enabling forager-hater dialogue...")
    dialogue_rounds = 3  # Multiple rounds for sustained dialogue

    for dialogue_round in range(dialogue_rounds):
        print(f"\n[PHASE 5B] Dialogue round {dialogue_round + 1}/{dialogue_rounds}")

        # Foragers defend their insights against objections
        async def run_forager_defense():
            """Run defense for all foragers."""
            tasks = []
            for forager in foragers:
                tasks.append(forager.defend_insights(signal_store, llm))
            await asyncio.gather(*tasks, return_exceptions=True)

        # Haters engage in dialogue with responses
        async def run_hater_dialogue():
            """Run dialogue for all haters."""
            tasks = []
            for hater in haters:
                tasks.append(hater.engage_in_dialogue(signal_store, llm))
            await asyncio.gather(*tasks, return_exceptions=True)

        # Run both in sequence (foragers defend first, then haters respond)
        await run_forager_defense()
        await run_hater_dialogue()

        # Check dialogue stats
        stats = signal_store.get_stats()
        total_signals = sum(stats['by_type'].values())
        print(f"  Total signals after dialogue: {total_signals}")

    print(f"\n[PHASE 5B] Dialogue complete")
    print(f"  Dialogue rounds: {dialogue_rounds}")
    print(f"  Participants: {len(foragers)} foragers + {len(haters)} haters")

    # ========================================================================
    # PHASE 6: CONVERGENCE MONITORING
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 6: CONVERGENCE ANALYSIS")
    print("=" * 80)

    # Calculate Gini coefficient for INSIGHT signals
    insights = [s for s in signal_store.get_all_signals() if s.type == "INSIGHT"]
    if insights:
        insight_strengths = [s.strength for s in insights]
        gini = calculate_gini_coefficient(insight_strengths)

        print(f"\n[PHASE 6] Convergence metrics:")
        print(f"  Total insights: {len(insights)}")
        print(f"  Gini coefficient: {gini:.3f} (target: {convergence_threshold})")
        print(f"  Strength range: {min(insight_strengths):.2f} - {max(insight_strengths):.2f}")

        # Show top insights
        top_insights = sorted(insights, key=lambda s: s.strength, reverse=True)[:10]
        print(f"\n[PHASE 6] Top 10 insights by strength:")
        for i, insight in enumerate(top_insights, 1):
            print(f"  {i}. {insight.id}: {insight.strength:.3f}")

        converged = gini >= convergence_threshold
        print(f"\n[PHASE 6] Convergence: {'YES' if converged else 'NO'}")
    else:
        print(f"\n[PHASE 6] WARNING: No insights created")
        gini = 0.0
        converged = False

    # ========================================================================
    # PHASE 7: SYNTHESIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 7: SYNTHESIS")
    print("=" * 80)

    if insights:
        # Get top insights for synthesis
        top_insights = signal_store.get_top_signals("INSIGHT", n=10)

        print(f"\n[PHASE 7] Synthesizing from top {len(top_insights)} insights")

        # Create synthesizer
        synthesizer = Synthesizer(
            "Synthesizer_Final",
            task_prompt="Synthesize insights from document analysis"
        )

        # Create signal context
        signal_context = {
            'INSIGHT': top_insights
        }

        # Generate synthesis
        synthesis = await synthesizer.synthesize(
            signal_store,
            llm,
            {'insights': 'INSIGHT'},
            temperature=0.6
        )

        print(f"\n[PHASE 7] Synthesis complete ({len(synthesis) if synthesis else 0} characters)")
    else:
        synthesis = "No synthesis generated - insufficient insights"

    # ========================================================================
    # PHASE 7B: STOP MONITOR AND PRINT HEALTH SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 7B: SWARM HEALTH REPORT")
    print("=" * 80)

    # Stop monitor
    monitor.stop()
    await monitor_task  # Wait for it to finish

    # Print comprehensive health summary
    monitor.print_summary()

    # ========================================================================
    # PHASE 8: OUTPUT AND DIAGNOSTICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 8: OUTPUT")
    print("=" * 80)

    total_time = time.time() - start_time

    # Final statistics
    final_stats = signal_store.get_stats()

    print(f"\n[FINAL STATISTICS]")
    print(f"  Total processing time: {total_time:.1f}s")
    print(f"  Input tokens: {total_tokens}")
    print(f"  Output characters: {len(synthesis) if synthesis else 0}")
    print(f"  Compression ratio: {total_tokens / (len(synthesis) / 4 if synthesis else 1):.1f}:1")
    print(f"\n  Signal counts by type:")
    for signal_type, count in sorted(final_stats['by_type'].items()):
        print(f"    {signal_type}: {count}")
    print(f"\n  Strength distribution:")
    print(f"    Average: {final_stats['avg_strength']:.3f}")
    print(f"    Maximum: {final_stats['strongest']:.3f}")
    print(f"    Gini coefficient: {gini:.3f}")

    # ========================================================================
    # DISPLAY SYNTHESIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SYNTHESIS OUTPUT")
    print("=" * 80)
    print(f"\n{synthesis}\n")

    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)

    # Return results
    return {
        'synthesis': synthesis,
        'top_insights': top_insights if insights else [],
        'statistics': final_stats,
        'gini_coefficient': gini,
        'converged': converged,
        'processing_time': total_time,
        'compression_ratio': total_tokens / (len(synthesis) / 4 if synthesis else 1)
    }


async def run_document_swarm_from_sections(
    sections: List,
    model_name: str = "microsoft/phi-2",
    num_foragers: int = 50,
    num_gatherers: int = 20,
    num_critics: int = 20,
    num_haters: int = 10,
    max_iterations: int = 500,
    convergence_threshold: float = 0.7,
    enable_mcp: bool = False
):
    """Run distributed document processing swarm from pre-loaded sections.

    This is an optimized version for web ingestion and other scenarios where
    document sections are already loaded. Skips Phase 1 (document loading).

    Args:
        sections: List of DocumentSection objects (already loaded)
        model_name: HuggingFace model name
        num_foragers: Number of forager agents
        num_gatherers: Number of gatherer agents
        num_critics: Number of critic agents
        num_haters: Number of hater agents
        max_iterations: Maximum total iterations
        convergence_threshold: Gini coefficient threshold for convergence
        enable_mcp: Enable MCP for external validation

    Returns:
        Dictionary with results
    """
    print("=" * 80)
    print("STIGMERGIC MONOLITH-BREAKING DOCUMENT PROCESSOR")
    print("(FROM PRE-LOADED SECTIONS)")
    print("=" * 80)

    start_time = time.time()

    # ========================================================================
    # PHASE 1: SKIPPED (Sections Already Loaded)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: DOCUMENT LOADING (SKIPPED - Using Pre-loaded Sections)")
    print("=" * 80)

    if not sections:
        print("\nERROR: No sections provided")
        return None

    total_tokens = sum(s.token_count for s in sections)
    print(f"\n[PHASE 1] Pre-loaded sections:")
    print(f"  Sections: {len(sections)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Avg tokens/section: {total_tokens // len(sections)}")

    # ========================================================================
    # PHASE 2: INFRASTRUCTURE INITIALIZATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: INFRASTRUCTURE INITIALIZATION")
    print("=" * 80)

    # Create signal store
    signal_store = SignalStore(
        decay_rate=DECAY_RATE,
        prune_threshold=PRUNE_THRESHOLD,
        diversity_threshold=DIVERSITY_THRESHOLD,
        exploration_bonus=EXPLORATION_BONUS
    )

    # Create LLM wrapper with quantization
    llm = SimpleLLM(model_name, DEVICE, enable_cache=True, cache_size=50)

    print(f"\n[PHASE 2] Loading model: {model_name}")
    await llm.load()

    # Create MCP client if enabled
    mcp_client = None
    if enable_mcp:
        try:
            from .knowledge.mcp_client import MCPClient
            mcp_client = MCPClient()
            print(f"[PHASE 2] MCP client initialized")
        except ImportError:
            print(f"[PHASE 2] WARNING: MCP client not available, validation will be limited")

    # Create SwarmMonitor for health monitoring and self-healing
    monitor = SwarmMonitor(agent_id="SwarmMonitor")
    print(f"[PHASE 2] SwarmMonitor initialized (self-healing enabled)")

    print(f"\n[PHASE 2] Infrastructure ready")

    # Launch monitor as background task
    monitor_task = asyncio.create_task(monitor.run(signal_store, max_iterations=100))
    print(f"[PHASE 2] SwarmMonitor launched as background task")

    # ========================================================================
    # PHASE 3: DISTRIBUTED READING (PARALLEL INPUT PROCESSING)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: DISTRIBUTED READING")
    print("=" * 80)

    print(f"\n[PHASE 3] Creating {len(sections)} scouts (1 per section)")

    # Create one scout per section
    scouts = [
        Scout(f"Scout_{i}", section=section)
        for i, section in enumerate(sections)
    ]

    # Launch ALL scouts simultaneously with robust error handling
    print(f"[PHASE 3] Launching {len(scouts)} scouts in parallel...")
    scout_start = time.time()

    wrapper = AgentExecutionWrapper(enable_retry=True, max_retries=1)
    await wrapper.run_agent_swarm(scouts, signal_store, llm, max_actions=1)

    scout_time = time.time() - scout_start

    # Report results
    stats = signal_store.get_stats()
    observation_count = stats['by_type'].get('OBSERVATION', 0)

    print(f"\n[PHASE 3] Distributed reading complete ({scout_time:.1f}s)")
    print(f"  Observations extracted: {observation_count}")
    print(f"  Avg observations/section: {observation_count / len(sections):.1f}")
    print(f"  Throughput: {len(sections) / scout_time:.1f} sections/second")

    if observation_count == 0:
        print("\nERROR: No observations extracted! Check model output.")
        return None

    # ========================================================================
    # PHASES 4-8: Continue with standard pipeline
    # ========================================================================
    # The rest of the pipeline is identical to run_document_swarm
    # So we'll call the shared processing logic

    # PHASE 4: Pattern Finding
    print("\n" + "=" * 80)
    print("PHASE 4: PATTERN FINDING")
    print("=" * 80)

    print(f"\n[PHASE 4] Creating {num_foragers} foragers for pattern discovery")

    foragers = [
        Forager(f"Forager_{i}", mode="document")
        for i in range(num_foragers)
    ]

    pattern_iterations = 150
    print(f"[PHASE 4] Running pattern finding for {pattern_iterations} iterations...")

    # Launch foragers with robust wrapper (non-blocking)
    async def run_foragers_with_monitoring():
        """Run foragers while monitoring progress."""
        wrapper = AgentExecutionWrapper(enable_retry=True, max_retries=1)
        forager_task = asyncio.create_task(
            wrapper.run_agent_swarm(foragers, signal_store, llm, max_actions=pattern_iterations)
        )

        # Monitor progress while foragers run
        for i in range(0, pattern_iterations, 50):
            remaining = min(50, pattern_iterations - i)

            if i > 0:
                stats = signal_store.get_stats()
                insight_count = stats['by_type'].get('INSIGHT', 0)
                print(f"\n[PHASE 4] Iteration {i}: {insight_count} insights discovered")

            await asyncio.sleep(remaining * 0.5)

        # Ensure foragers finish
        await forager_task

        # Stop any still-running foragers
        for forager in foragers:
            forager.stop()

    await run_foragers_with_monitoring()

    stats = signal_store.get_stats()
    insight_count = stats['by_type'].get('INSIGHT', 0)

    print(f"\n[PHASE 4] Pattern finding complete")
    print(f"  Insights discovered: {insight_count}")
    print(f"  Avg insights/forager: {insight_count / num_foragers:.1f}")

    # PHASE 5: Validation
    print("\n" + "=" * 80)
    print("PHASE 5: VALIDATION LOOP")
    print("=" * 80)

    print(f"\n[PHASE 5] Creating validation agents:")
    print(f"  Gatherers: {num_gatherers}")
    print(f"  Critics: {num_critics}")
    print(f"  Haters: {num_haters}")

    gatherers = [
        Gatherer(f"Gatherer_{i}", mcp_client=mcp_client)
        for i in range(num_gatherers)
    ]

    critics = [
        Critic(f"Critic_{i}", mode="document")
        for i in range(num_critics)
    ]

    haters = [
        Hater(f"Hater_{i}", task_prompt="Challenge insights")
        for i in range(num_haters)
    ]

    validation_iterations = 200  # Increased to match forager power
    print(f"[PHASE 5] Running validation for {validation_iterations} iterations...")
    print(f"[PHASE 5] Enhanced mode: Critics use full context, Haters target consensus")

    # Launch validation agents with robust wrapper
    async def run_validation_with_monitoring():
        """Run validation agents while monitoring progress."""
        all_validators = gatherers + critics + haters
        wrapper = AgentExecutionWrapper(enable_retry=True, max_retries=1)

        # Create tasks for different agent types (they have different signatures)
        async def run_gatherer(g):
            return await wrapper.run_agent_safe(g, signal_store, llm, max_actions=validation_iterations)

        async def run_critic(c):
            # Critics now use enhanced_context=True by default (full provenance, reasoned critiques)
            return await wrapper.run_agent_safe(c, signal_store, llm, max_actions=validation_iterations)

        async def run_hater(h):
            # Haters now use target_consensus=True by default (challenge groupthink, not strongest)
            return await wrapper.run_agent_safe(h, signal_store, llm,
                                               max_actions=validation_iterations,
                                               temperature=0.85,
                                               target_consensus=True)

        # Launch all validators
        tasks = []
        tasks.extend([run_gatherer(g) for g in gatherers])
        tasks.extend([run_critic(c) for c in critics])
        tasks.extend([run_hater(h) for h in haters])

        validation_task = asyncio.create_task(asyncio.gather(*tasks, return_exceptions=True))

        # Monitor progress
        for i in range(0, validation_iterations, 50):
            remaining = min(50, validation_iterations - i)

            if i > 0:
                stats = signal_store.get_stats()
                evidence_count = stats['by_type'].get('EVIDENCE', 0)
                objection_count = stats['by_type'].get('OBJECTION', 0)
                print(f"\n[PHASE 5] Iteration {i}:")
                print(f"  Evidence gathered: {evidence_count}")
                print(f"  Objections raised: {objection_count}")
                print(f"  Avg strength: {stats['avg_strength']:.2f}")

            await asyncio.sleep(remaining * 0.6)

        # Ensure all finish
        await validation_task

        # Stop any still-running agents
        for gatherer in gatherers:
            gatherer.stop()
        for critic in critics:
            critic.stop()
        for hater in haters:
            hater.stop()

        # Print wrapper stats
        wrapper.print_summary()

    await run_validation_with_monitoring()

    stats = signal_store.get_stats()
    evidence_count = stats['by_type'].get('EVIDENCE', 0)
    objection_count = stats['by_type'].get('OBJECTION', 0)

    print(f"\n[PHASE 5] Validation complete")
    print(f"  Evidence gathered: {evidence_count}")
    print(f"  Objections raised: {objection_count}")

    # ========================================================================
    # PHASE 5B: AGENT DIALOGUE (Foragers defend, Haters engage)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 5B: AGENT DIALOGUE")
    print("=" * 80)

    print(f"\n[PHASE 5B] Enabling forager-hater dialogue...")
    dialogue_rounds = 3  # Multiple rounds for sustained dialogue

    for dialogue_round in range(dialogue_rounds):
        print(f"\n[PHASE 5B] Dialogue round {dialogue_round + 1}/{dialogue_rounds}")

        # Foragers defend their insights against objections
        async def run_forager_defense():
            """Run defense for all foragers."""
            tasks = []
            for forager in foragers:
                tasks.append(forager.defend_insights(signal_store, llm))
            await asyncio.gather(*tasks, return_exceptions=True)

        # Haters engage in dialogue with responses
        async def run_hater_dialogue():
            """Run dialogue for all haters."""
            tasks = []
            for hater in haters:
                tasks.append(hater.engage_in_dialogue(signal_store, llm))
            await asyncio.gather(*tasks, return_exceptions=True)

        # Run both in sequence (foragers defend first, then haters respond)
        await run_forager_defense()
        await run_hater_dialogue()

        # Check dialogue stats
        stats = signal_store.get_stats()
        total_signals = sum(stats['by_type'].values())
        print(f"  Total signals after dialogue: {total_signals}")

    print(f"\n[PHASE 5B] Dialogue complete")
    print(f"  Dialogue rounds: {dialogue_rounds}")
    print(f"  Participants: {len(foragers)} foragers + {len(haters)} haters")

    # PHASE 6: Convergence
    print("\n" + "=" * 80)
    print("PHASE 6: CONVERGENCE ANALYSIS")
    print("=" * 80)

    insights = [s for s in signal_store.get_all_signals() if s.type == "INSIGHT"]
    if insights:
        insight_strengths = [s.strength for s in insights]
        gini = calculate_gini_coefficient(insight_strengths)

        print(f"\n[PHASE 6] Convergence metrics:")
        print(f"  Total insights: {len(insights)}")
        print(f"  Gini coefficient: {gini:.3f} (target: {convergence_threshold})")
        print(f"  Strength range: {min(insight_strengths):.2f} - {max(insight_strengths):.2f}")

        top_insights = sorted(insights, key=lambda s: s.strength, reverse=True)[:10]
        print(f"\n[PHASE 6] Top 10 insights by strength:")
        for i, insight in enumerate(top_insights, 1):
            print(f"  {i}. {insight.id}: {insight.strength:.3f}")

        converged = gini >= convergence_threshold
        print(f"\n[PHASE 6] Convergence: {'YES' if converged else 'NO'}")
    else:
        print(f"\n[PHASE 6] WARNING: No insights created")
        gini = 0.0
        converged = False

    # PHASE 7: Synthesis
    print("\n" + "=" * 80)
    print("PHASE 7: SYNTHESIS")
    print("=" * 80)

    if insights:
        top_insights = signal_store.get_top_signals("INSIGHT", n=10)

        print(f"\n[PHASE 7] Synthesizing from top {len(top_insights)} insights")

        synthesizer = Synthesizer(
            "Synthesizer_Final",
            task_prompt="Synthesize insights from document analysis"
        )

        signal_context = {
            'INSIGHT': top_insights
        }

        synthesis = await synthesizer.synthesize(
            signal_store,
            llm,
            {'insights': 'INSIGHT'},
            temperature=0.6
        )

        print(f"\n[PHASE 7] Synthesis complete ({len(synthesis) if synthesis else 0} characters)")
    else:
        synthesis = "No synthesis generated - insufficient insights"

    # PHASE 8: Output
    print("\n" + "=" * 80)
    print("PHASE 8: OUTPUT")
    print("=" * 80)

    total_time = time.time() - start_time

    final_stats = signal_store.get_stats()

    print(f"\n[FINAL STATISTICS]")
    print(f"  Total processing time: {total_time:.1f}s")
    print(f"  Input tokens: {total_tokens}")
    print(f"  Output characters: {len(synthesis) if synthesis else 0}")
    print(f"  Compression ratio: {total_tokens / (len(synthesis) / 4 if synthesis else 1):.1f}:1")
    print(f"\n  Signal counts by type:")
    for signal_type, count in sorted(final_stats['by_type'].items()):
        print(f"    {signal_type}: {count}")
    print(f"\n  Strength distribution:")
    print(f"    Average: {final_stats['avg_strength']:.3f}")
    print(f"    Maximum: {final_stats['strongest']:.3f}")
    print(f"    Gini coefficient: {gini:.3f}")

    print("\n" + "=" * 80)
    print("SYNTHESIS OUTPUT")
    print("=" * 80)
    print(f"\n{synthesis}\n")

    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)

    return {
        'synthesis': synthesis,
        'top_insights': top_insights if insights else [],
        'statistics': final_stats,
        'gini_coefficient': gini,
        'converged': converged,
        'processing_time': total_time,
        'compression_ratio': total_tokens / (len(synthesis) / 4 if synthesis else 1)
    }


async def main():
    """Example usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m swarm.monolith_breaking <document1.pdf> <document2.pdf> ...")
        print("\nExample:")
        print("  python -m swarm.monolith_breaking papers/research1.pdf papers/research2.pdf")
        return

    document_paths = sys.argv[1:]

    # Verify files exist
    for path in document_paths:
        if not Path(path).exists():
            print(f"ERROR: File not found: {path}")
            return

    # Run processing
    result = await run_document_swarm(
        document_paths=document_paths,
        num_foragers=50,
        num_gatherers=20,
        num_critics=20,
        max_iterations=500,
        convergence_threshold=0.7,
        enable_mcp=False  # Set to True when MCP client is available
    )

    if result:
        print(f"\n\nProcessing completed successfully!")
        print(f"Compression ratio: {result['compression_ratio']:.1f}:1")
        print(f"Convergence: {result['converged']}")


if __name__ == "__main__":
    asyncio.run(main())
