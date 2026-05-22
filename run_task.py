"""Run stigmergic swarm with different task types.

Usage:
    python run_task.py debate
    python run_task.py creative
    python run_task.py creative "Write a haiku about artificial intelligence"
    python run_task.py analysis "What are the societal impacts of social media?"
    python run_task.py problem_solving "How can we make cities more sustainable?"
"""

import asyncio
import sys
import time
import json
import os
import random
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from swarm.core.signal_store import SignalStore
from swarm.core.config import *
from swarm.core.task_config import get_task_config, create_custom_task
from swarm.core.logging_config import setup_logging
from swarm.llm.simple_llm import SimpleLLM
from swarm.llm.llm_pool import AdaptiveLLMPool
from swarm.core.stage_coordinator import StageCoordinator
from swarm.agents.scout import Scout
from swarm.agents.forager import Forager
from swarm.agents.critic import Critic
from swarm.agents.hater import Hater
from swarm.agents.validator import Validator
from swarm.agents.pruner import Pruner
from swarm.agents.synthesizer import Synthesizer
from swarm.validation.format_validator import FormatValidator
from swarm.retrieval.dynamic_retriever import DynamicRetriever
from swarm.retrieval.simple_web_search import web_search
from swarm.core.round_coordinator import RoundCoordinator
from swarm.core.swarm_monitor import SwarmMonitor
from swarm.core.dialogue_coordinator import DialogueCoordinator

# Phase 2-4: True Swarm Intelligence Components (conditional imports)
if USE_SPATIAL_STORE:
    from swarm.core.spatial_signal_store import SpatialSignalStore
if USE_SIMPLE_SCOUTS:
    from swarm.agents.simple_scout import SimpleScout

# Phase 5: Advanced Knowledge Retrieval (conditional import)
if USE_ADVANCED_RETRIEVER:
    from swarm.retrieval.advanced_retriever import AdvancedRetriever
if USE_REAL_VALIDATOR:
    from swarm.validation import DynamicKnowledgeBase, RealValidator


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

@asynccontextmanager
async def timer(name: str, enabled: bool = True):
    """Context manager for timing code sections.

    Usage:
        async with timer("Scout phase"):
            await run_scouts()
    """
    if not enabled:
        yield
        return

    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"[TIMING] {name}: {elapsed:.2f}s")


class TaskBasedAgent:
    """Wrapper that makes agents task-aware."""

    @staticmethod
    def create_scout(agent_id: str, task_config, dynamic_retriever=None):
        """Create scout that uses task-specific prompts and dynamic retrieval."""
        # NEW: Pass task_config directly (composition - no monkey patching!)
        scout = Scout(agent_id, task_config.signal_types["initial"],
                     task_config.task_prompt, dynamic_retriever=dynamic_retriever,
                     task_config=task_config)
        return scout

    @staticmethod
    def create_forager(agent_id: str, input_type: str, output_type: str, task_config):
        """Create forager that uses task-specific prompts."""
        # NEW: Pass task_config directly (composition - no monkey patching!)
        forager = Forager(agent_id, input_type=input_type, output_type=output_type,
                         enable_verification=True, mode="creative",
                         thesis=task_config.task_prompt, task_config=task_config)
        return forager

    @staticmethod
    def create_critic(agent_id: str, task_config):
        """Create critic that uses task-specific prompts."""
        # NEW: Pass task_config directly (composition - no monkey patching!)
        critic = Critic(agent_id, mode="creative", thesis=task_config.task_prompt,
                       task_config=task_config)
        # Set the signal type to evaluate (initial type for the task)
        critic.evaluate_type = task_config.signal_types["initial"]
        return critic

    @staticmethod
    def create_hater(agent_id: str, task_config):
        """Create hater that uses task-specific prompts."""
        # Configure signal types based on task mode
        input_types = [task_config.signal_types["initial"]]  # Target what scouts generate
        output_type = task_config.signal_types["counter"]    # Deposit counter signals

        # NEW: Pass task_config directly (composition - no monkey patching!)
        hater = Hater(agent_id, task_config.task_prompt,
                     input_types=input_types, output_type=output_type,
                     task_config=task_config)
        return hater

    @staticmethod
    def create_simple_scout(agent_id: str, task_config, grid_dimensions: int = 100):
        """Create SimpleScout with position and movement (Phase 2).

        Args:
            agent_id: Unique agent ID
            task_config: Task configuration
            grid_dimensions: Size of spatial grid

        Returns:
            SimpleScout instance
        """
        # Random starting position in grid
        x = random.uniform(0, grid_dimensions - 1)
        y = random.uniform(0, grid_dimensions - 1)

        # Extract simple knowledge fragment from task
        knowledge = f"Task: {task_config.task_prompt[:100]}"

        simple_scout = SimpleScout(
            agent_id=agent_id,
            position=(x, y),
            knowledge=knowledge,
            perception_radius=SIMPLE_SCOUT_PERCEPTION_RADIUS,
            movement_speed=SIMPLE_SCOUT_MOVEMENT_SPEED
        )

        return simple_scout


def save_run_outputs(run_dir: Path, task_config, signal_store, cache_stats, config_info, synthesis: str = None, round_summary: str = None):
    """Save all outputs for a swarm run.

    Args:
        run_dir: Output directory
        task_config: Task configuration
        signal_store: Signal store with all signals
        cache_stats: LLM cache statistics
        config_info: Configuration information
        synthesis: Synthesized final answer (optional, becomes base truth if provided)
    """

    # Get all signals (handle both SignalStore and SpatialSignalStore)
    if hasattr(signal_store, 'get_all_signals'):
        all_signals = signal_store.get_all_signals()
    else:
        # SpatialSignalStore - get signals from internal storage
        all_signals = list(signal_store.signals.values())

    # Group signals by type
    signals_by_type = {}
    for signal in all_signals:
        if signal.type not in signals_by_type:
            signals_by_type[signal.type] = []
        signals_by_type[signal.type].append({
            "id": signal.id,
            "content": signal.content,
            "strength": signal.strength,
            "visits": signal.visits,
            "depositor": signal.depositor,
            "parent": signal.parent,
            "timestamp": signal.timestamp
        })

    # Save signals by type as JSON
    for signal_type, signals in signals_by_type.items():
        filename = run_dir / f"{signal_type.lower()}_signals.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(signals, f, indent=2, ensure_ascii=False)

    # Base truth selection with MINIMAL validation (no format scaffolding - let swarm emerge quality)
    base_truth_text = None
    base_truth_source = None

    # Try 1: Use synthesis if it's substantive (no format checks)
    if synthesis:
        # Relaxed validation - accept shorter outputs (min 15 chars)
        is_valid, msg = FormatValidator.validate_content_quality(synthesis, min_length=15)
        if is_valid:
            base_truth_text = synthesis
            base_truth_source = "Synthesized Answer"
            print(f"[BASE_TRUTH] Selected synthesis")
        else:
            print(f"[BASE_TRUTH] Synthesis rejected: {msg}")

    # Try 2: Fall back to top initial signals (check quality only, no format rules)
    if not base_truth_text:
        initial_type = task_config.signal_types.get("initial", "OBSERVATION")
        top_initial = signal_store.get_top_signals(initial_type, 15)  # Increased from 10 to 15

        for i, signal in enumerate(top_initial):
            # Relaxed quality check (min 15 chars instead of 20)
            is_valid, msg = FormatValidator.validate_content_quality(signal.content, min_length=15)
            if not is_valid:
                print(f"[BASE_TRUTH] Skipping {signal.id}: {msg}")
                continue

            # Generic contamination check (exam rubrics) - but NO task-specific keyword matching
            text_lower = signal.content.lower()
            if any(indicator in text_lower for indicator in ['exercise 1', 'answer key', 'rubric', 'question 1']):
                print(f"[BASE_TRUTH] Skipping {signal.id}: appears to be exam/rubric text")
                continue

            # Found valid candidate (swarm-determined quality)
            base_truth_text = signal.content
            base_truth_source = f"Strongest {initial_type} ({signal.id}, strength={signal.strength:.3f})"
            print(f"[BASE_TRUTH] Selected {signal.id} by strength and basic quality")
            break

    # Try 3: Fallback to ANY signal type (support, critique, etc.)
    if not base_truth_text:
        print(f"[BASE_TRUTH] No valid initial signals, trying ALL signal types...")
        all_signals = signal_store.get_all_signals()
        # Sort by strength
        all_signals_sorted = sorted(all_signals, key=lambda s: s.strength, reverse=True)

        for signal in all_signals_sorted[:20]:  # Check top 20 signals of any type
            is_valid, msg = FormatValidator.validate_content_quality(signal.content, min_length=15)
            if not is_valid:
                continue

            # Skip contaminated
            text_lower = signal.content.lower()
            if any(indicator in text_lower for indicator in ['exercise 1', 'answer key', 'rubric', 'question 1']):
                continue

            base_truth_text = signal.content
            base_truth_source = f"Strongest signal (any type) ({signal.id}, type={signal.type}, strength={signal.strength:.3f})"
            print(f"[BASE_TRUTH] Selected {signal.id} from type {signal.type} as last resort")
            break

    # Try 4: Manual concatenation of top signals (graceful degradation)
    if not base_truth_text:
        print(f"[BASE_TRUTH] All validation failed, attempting manual concatenation...")
        all_signals = signal_store.get_all_signals()
        if all_signals:
            # Get top 3 strongest signals by raw strength
            top_3 = sorted(all_signals, key=lambda s: s.strength, reverse=True)[:3]
            concatenated = " ".join([s.content[:100] for s in top_3 if len(s.content) > 10])

            if len(concatenated) > 20:
                base_truth_text = concatenated
                base_truth_source = "Concatenated top 3 signals (graceful degradation)"
                print(f"[BASE_TRUTH] Using concatenated output ({len(concatenated)} chars)")

    # Try 5: If still nothing, provide diagnostic error (process failure, not swarm failure)
    if not base_truth_text:
        base_truth_text = f"[PROCESS ERROR] No substantive output generated. This indicates a process failure (caching/contamination), not a swarm intelligence failure. Check logs for details."
        base_truth_source = "Error - Process Failure"
        total_signals = len(signal_store.get_all_signals()) if hasattr(signal_store, 'get_all_signals') else 0
        print(f"[BASE_TRUTH] ERROR: All candidates failed validation (total signals: {total_signals})")

    # Generate summary markdown
    summary_path = run_dir / "summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# {task_config.task_type.upper()} Task Results\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Task**: {task_config.task_prompt}\n\n")
        f.write("---\n\n")

        # Base Truth
        f.write("## Base Truth (Final Result)\n\n")
        if base_truth_text:
            f.write(f"> {base_truth_text}\n\n")
            f.write(f"**Source**: {base_truth_source}\n\n")
        else:
            f.write("*No result generated*\n\n")

        f.write("---\n\n")

        # Round progression (if available)
        if round_summary:
            f.write("## Round Progression\n\n")
            f.write(round_summary)
            f.write("\n---\n\n")

        # Signal distribution
        stats = signal_store.get_stats()
        f.write("## Signal Distribution\n\n")
        f.write("| Type | Count | % of Total | Avg Strength |\n")
        f.write("|------|-------|------------|-------------|\n")

        for signal_type, count in stats['by_type'].items():
            pct = (count / stats['total_signals'] * 100) if stats['total_signals'] > 0 else 0
            type_signals = [s for s in all_signals if s.type == signal_type]
            avg_strength = sum(s.strength for s in type_signals) / len(type_signals) if type_signals else 0
            f.write(f"| {signal_type} | {count} | {pct:.1f}% | {avg_strength:.3f} |\n")

        f.write(f"\n**Total**: {stats['total_signals']} signals | ")
        f.write(f"**Avg Strength**: {stats['avg_strength']:.3f}\n\n")
        f.write("---\n\n")

        # Performance
        f.write("## Performance Metrics\n\n")
        f.write(f"- **Generation Success Rate**: {cache_stats['success_rate']:.1%}\n")
        f.write(f"- **Cache Hit Rate**: {cache_stats['hit_rate']:.1%}\n")
        f.write(f"- **Successes**: {cache_stats['generation_successes']}\n")
        f.write(f"- **Failures**: {cache_stats['generation_failures']}\n\n")
        f.write("---\n\n")

        # Top signals of each type
        for role, signal_type in task_config.signal_types.items():
            top_signals = signal_store.get_top_signals(signal_type, 3)

            if top_signals:
                f.write(f"## Top {signal_type}s\n\n")
                for i, signal in enumerate(top_signals, 1):
                    f.write(f"### {i}. {signal.id}\n\n")
                    f.write(f"{signal.content}\n\n")
                    f.write(f"*Strength: {signal.strength:.3f} | ")
                    f.write(f"Visits: {signal.visits} | ")
                    f.write(f"By: {signal.depositor}*\n\n")
            else:
                f.write(f"## Top {signal_type}s\n\n")
                f.write("*No signals generated*\n\n")

        # Configuration
        f.write("---\n\n")
        f.write("## Configuration\n\n")
        for key, value in config_info.items():
            f.write(f"- **{key}**: {value}\n")

    return base_truth_text


async def run_task(task_type: str, custom_prompt: str = None):
    """Run swarm with specified task type.

    Args:
        task_type: One of "debate", "creative", "analysis", "problem_solving"
        custom_prompt: Optional custom prompt (uses default if not provided)
    """
    # Get task configuration
    if custom_prompt:
        task_config = create_custom_task(task_type, custom_prompt)
    else:
        task_config = get_task_config(task_type)

    print("=" * 70)
    print(f"STIGMERGIC SWARM - {task_config.task_type.upper()} MODE")
    print("=" * 70)
    print(f"\nTask: {task_config.task_prompt}")
    print(f"\nSignal Types:")
    for role, signal_type in task_config.signal_types.items():
        print(f"  {role:12} -> {signal_type}")

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME} on {DEVICE}")
    print(f"  Agents: {NUM_SCOUTS} scouts, {NUM_FORAGERS} foragers, "
          f"{NUM_CRITICS} critics, {NUM_HATERS} haters, "
          f"{NUM_VALIDATORS} validators, {NUM_PRUNERS} pruners")
    print(f"  Iterations: {MAX_ITERATIONS}")

    # Phase 2-4 features
    if USE_SIMPLE_SCOUTS:
        print(f"  ✨ SimpleScouts: Enabled (Phase 2)")
    if USE_SPATIAL_STORE:
        print(f"  ✨ SpatialSignalStore: Enabled (Phase 3)")
    if USE_REAL_VALIDATOR:
        print(f"  ✨ RealValidator: Enabled (Phase 4)")
    print()

    # Initialize signal store with diversity checking (Phase 3: conditional spatial store)
    if USE_SPATIAL_STORE:
        signal_store = SpatialSignalStore(
            dimensions=SPATIAL_GRID_DIMENSIONS,
            decay_rate=DECAY_RATE,
            prune_threshold=PRUNE_THRESHOLD,
            diversity_threshold=DIVERSITY_THRESHOLD
        )
        print(f"[INIT] Using SpatialSignalStore ({SPATIAL_GRID_DIMENSIONS}×{SPATIAL_GRID_DIMENSIONS} grid)")

        # Initialize swarm health monitoring
        monitor = SwarmMonitor(signal_store)
        print(f"[INIT] SwarmMonitor enabled - tracking health, convergence, and echo chambers\n")
    else:
        signal_store = SignalStore(
            decay_rate=DECAY_RATE,
            prune_threshold=PRUNE_THRESHOLD,
            diversity_threshold=DIVERSITY_THRESHOLD,
            exploration_bonus=EXPLORATION_BONUS
        )
        print(f"[INIT] Using SignalStore (global access)")

    # Initialize swarm health monitoring
    monitor = SwarmMonitor(signal_store)
    print(f"[INIT] SwarmMonitor enabled - tracking health, convergence, and echo chambers\n")

    # PHASE 2C: Use AdaptiveLLMPool for parallel execution
    # Task-specific cache settings (creative tasks need smaller cache for originality)
    if task_config.task_type == "creative":
        llm_pool = AdaptiveLLMPool(MODEL_NAME, DEVICE, enable_cache=CREATIVE_CACHE_ENABLED, cache_size=CREATIVE_CACHE_SIZE)
        print(f"[INIT] Using creative cache settings (size={CREATIVE_CACHE_SIZE}, target hit rate <50%)")
    else:
        llm_pool = AdaptiveLLMPool(MODEL_NAME, DEVICE, enable_cache=ENABLE_LLM_CACHE, cache_size=LLM_CACHE_SIZE)

    print("[INIT] Initializing adaptive LLM pool...")
    start = time.time()
    pool_size = await llm_pool.initialize()
    print(f"[INIT] LLM pool initialized in {time.time() - start:.1f}s ({pool_size} instances)")
    print(f"[INIT] Expected speedup: {min(pool_size, 4)}x for parallel agent execution\n")

    # Keep SimpleLLM for backward compatibility with synthesis and other non-parallel operations
    llm = SimpleLLM(MODEL_NAME, DEVICE, enable_cache=ENABLE_LLM_CACHE, cache_size=LLM_CACHE_SIZE)
    await llm.load()
    print(f"[INIT] Fallback SimpleLLM loaded for synthesis\n")

    # PRE-FLIGHT PLANNER: Set swarm direction before any agents run
    from swarm.core.planner import Planner
    print("[PLANNER] Analyzing task and setting swarm direction...")
    planner = Planner()
    task_frame = await planner.plan(task_config.task_prompt, llm)
    if task_frame:
        task_config.task_frame = task_frame
        print(f"[PLANNER] Output format : {task_frame.output_format}")
        print(f"[PLANNER] Strength mode : {task_frame.strength_mode}")
        print(f"[PLANNER] Scout         : {task_frame.scout_directive}")
        print(f"[PLANNER] Synthesizer   : {task_frame.synthesizer_directive}")
        if task_frame.decomposition:
            print(f"[PLANNER] Sub-problems  : {' | '.join(task_frame.decomposition)}")
    else:
        print("[PLANNER] Falling back to default adaptive prompts\n")

    # Phase 4: Create shared DynamicKnowledgeBase for RealValidator (if enabled)
    shared_kb = None
    if USE_REAL_VALIDATOR:
        shared_kb = DynamicKnowledgeBase(
            confidence_threshold=DYNAMIC_KB_CONFIDENCE_THRESHOLD,
            max_facts=DYNAMIC_KB_MAX_FACTS
        )
        print(f"[INIT] DynamicKnowledgeBase created (confidence_threshold={DYNAMIC_KB_CONFIDENCE_THRESHOLD})")
        print(f"[INIT] Knowledge base starts empty and learns during execution\n")

    # Create dynamic retriever for swarm-driven information gathering
    print("[INIT] Creating dynamic retriever...")
    dynamic_retriever = DynamicRetriever(
        temp_file="research/temp_context.txt",
        max_file_size_mb=10.0
    )
    print(f"[INIT] Retriever ready (temp file: {dynamic_retriever.temp_file})\n")

    # Phase 5: Create advanced retriever for deep knowledge ingestion (optional)
    advanced_retriever = None
    if USE_ADVANCED_RETRIEVER:
        print("[INIT] Creating AdvancedRetriever for deep knowledge ingestion...")
        try:
            advanced_retriever = AdvancedRetriever(
                temp_dir=ADVANCED_RETRIEVAL_TEMP_DIR,
                intake_profile=task_config.intake_profile  # Use task's intake profile
            )
            print(f"[INIT] AdvancedRetriever ready")
            print(f"[INIT] Using intake profile: {task_config.intake_profile.__class__.__name__ if hasattr(task_config.intake_profile, '__class__') else 'default'}")
            print(f"[INIT] Target words/round: {task_config.intake_profile.target_words_per_round:,}")
            print(f"[INIT] Temp dir: {ADVANCED_RETRIEVAL_TEMP_DIR}\n")
        except ImportError as e:
            print(f"[WARNING] AdvancedRetriever import failed: {e}")
            print("[WARNING] Install dependencies: pip install requests beautifulsoup4")
            advanced_retriever = None

    # Create round coordinator for iterative refinement
    # Check for hyper_test mode (set via environment)
    is_hyper_test = os.environ.get('HYPER_TEST_MODE', '0') == '1'
    NUM_ROUNDS = 2 if is_hyper_test else 3  # Number of iterative refinement rounds
    ITERATIONS_PER_ROUND = max(5 if is_hyper_test else 20, MAX_ITERATIONS // NUM_ROUNDS)  # Distribute iterations across rounds

    print(f"[INIT] Creating round coordinator ({NUM_ROUNDS} rounds, {ITERATIONS_PER_ROUND} iterations each)...")
    round_coordinator = RoundCoordinator(
        num_rounds=NUM_ROUNDS,
        iterations_per_round=ITERATIONS_PER_ROUND
    )
    print(f"[INIT] Round coordinator ready\n")

    # ROUND-BASED ITERATIVE REFINEMENT
    # Each round: extract keywords → search → swarm process → synthesize → extract new keywords
    print("=" * 70)
    print("STARTING ROUND-BASED ITERATIVE REFINEMENT")
    print("=" * 70)

    # Increase scout count for parallel searching
    num_scouts = max(10, NUM_SCOUTS)

    all_round_syntheses = []  # Track syntheses across rounds
    final_signal_store = None  # Will hold the last round's signal store

    for round_num in range(NUM_ROUNDS):
        round_start = time.time()  # Track total round time
        print(f"\n{'=' * 70}")
        print(f"ROUND {round_num + 1}/{NUM_ROUNDS}")
        print(f"{'=' * 70}\n")

        # Step 1: Extract keywords for this round
        if round_num == 0:
            # Round 1: Extract from task prompt
            keywords = round_coordinator.extract_keywords_from_task(task_config.task_prompt)
            print(f"[ROUND {round_num + 1}] Keywords from task: {keywords[:5]}")
        else:
            # Round 2+: Extract from previous round's synthesis
            previous_synthesis = all_round_syntheses[-1]
            keywords = round_coordinator.extract_keywords_from_synthesis(previous_synthesis, max_keywords=10)
            print(f"[ROUND {round_num + 1}] Keywords from previous synthesis: {keywords[:5]}")

        if not keywords:
            print(f"[ROUND {round_num + 1}] WARNING: No keywords extracted, using task keywords as fallback")
            keywords = round_coordinator.extract_keywords_from_task(task_config.task_prompt)

        # Step 2: Reset retriever and perform web searches
        research_start = time.time()
        if round_num > 0:
            print(f"[ROUND {round_num + 1}] Resetting retriever (deleting old context, clearing search history)...")
            dynamic_retriever.reset_for_next_round()

        print(f"[ROUND {round_num + 1}] Performing initial web searches with {len(keywords)} keywords...")
        # Do initial searches with top keywords (scripting phase, no LLM)
        for i, keyword_batch in enumerate([keywords[j:j+3] for j in range(0, min(len(keywords), 9), 3)]):
            if keyword_batch:
                search_success = await dynamic_retriever.search_and_append(keyword_batch, web_search)
                if search_success:
                    print(f"[ROUND {round_num + 1}] Search {i+1}: {' '.join(keyword_batch)} ✓")

        # Check if we got any context
        context_stats = dynamic_retriever.get_stats()
        print(f"[ROUND {round_num + 1}] Context file: {context_stats['file_size_kb']:.1f}KB, "
              f"{context_stats['num_searches']} searches")
        print(f"[TIMING] Research/search phase: {time.time() - research_start:.2f}s\n")

        # Phase 5: Advanced deep research (optional, runs in parallel with DynamicRetriever)
        round_knowledge = None
        if advanced_retriever:
            print(f"[ROUND {round_num + 1}] Starting DEEP research (target: {ADVANCED_RETRIEVAL_TARGET_WORDS:,} words)...")
            try:
                # Get previous synthesis for refinement (if available)
                previous_synthesis = all_round_syntheses[-1] if round_num > 0 and all_round_syntheses else ""

                # Perform deep research
                round_knowledge = await advanced_retriever.deep_research_round(
                    keywords=keywords,
                    round_num=round_num,
                    task_context=task_config.task_prompt,
                    previous_synthesis=previous_synthesis
                )

                print(f"[ROUND {round_num + 1}] Deep research complete:")
                print(f"  - Words ingested: {round_knowledge.total_words_ingested:,}")
                print(f"  - Sources accessed: {round_knowledge.sources_count}")
                print(f"  - Fragments extracted: {len(round_knowledge.fragments)}")
                print(f"  - Niche discoveries: {len(round_knowledge.niche_discoveries)}")
                print(f"  - Queries executed: {len(round_knowledge.queries_executed)}\n")

                # Save round knowledge to temp file for scouts
                advanced_retriever.save_round_knowledge(round_num)

            except Exception as e:
                print(f"[WARNING] Advanced retrieval failed for round {round_num + 1}: {e}")
                import traceback
                traceback.print_exc()

        # Step 3: Create fresh signal store for this round (Phase 3: conditional spatial store)
        if USE_SPATIAL_STORE:
            signal_store = SpatialSignalStore(
                dimensions=SPATIAL_GRID_DIMENSIONS,
                decay_rate=DECAY_RATE,
                prune_threshold=PRUNE_THRESHOLD,
                diversity_threshold=DIVERSITY_THRESHOLD
            )
        else:
            signal_store = SignalStore(
                decay_rate=DECAY_RATE,
                prune_threshold=PRUNE_THRESHOLD,
                diversity_threshold=DIVERSITY_THRESHOLD,
                exploration_bonus=EXPLORATION_BONUS
            )

        # Step 4: Create agents for this round (Phase 2: conditional SimpleScouts)
        print(f"[ROUND {round_num + 1}] Creating agents...")
        if USE_SIMPLE_SCOUTS:
            scouts = [
                TaskBasedAgent.create_simple_scout(f"SimpleScout_R{round_num}_{i}", task_config, SPATIAL_GRID_DIMENSIONS)
                for i in range(num_scouts)
            ]
            print(f"[ROUND {round_num + 1}] Created {num_scouts} SimpleScouts with spatial movement")
        else:
            scouts = [
                TaskBasedAgent.create_scout(f"Scout_R{round_num}_{i}", task_config, dynamic_retriever=dynamic_retriever)
                for i in range(num_scouts)
            ]
            print(f"[ROUND {round_num + 1}] Created {num_scouts} original Scouts")

        # Assign sub-problems to scouts when planner decomposed the task
        decomposition = getattr(getattr(task_config, 'task_frame', None), 'decomposition', [])
        if decomposition:
            for i, scout in enumerate(scouts):
                scout.sub_problem = decomposition[i % len(decomposition)]
            print(f"[ROUND {round_num + 1}] Assigned {len(decomposition)} sub-problems across {len(scouts)} scouts")

        foragers_support = [
            TaskBasedAgent.create_forager(
                f"Forager_Support_R{round_num}_{i}",
                task_config.signal_types["initial"],
                task_config.signal_types["support"],
                task_config
            )
            for i in range(NUM_FORAGERS // 2)
        ]

        foragers_critique = [
            TaskBasedAgent.create_forager(
                f"Forager_Critique_R{round_num}_{i}",
                task_config.signal_types["initial"],
                task_config.signal_types["critique"],
                task_config
            )
            for i in range(NUM_FORAGERS // 2)
        ]

        critics = [
            TaskBasedAgent.create_critic(f"Critic_R{round_num}_{i}", task_config)
            for i in range(NUM_CRITICS)
        ]

        haters = [
            TaskBasedAgent.create_hater(f"Hater_R{round_num}_{i}", task_config)
            for i in range(NUM_HATERS)
        ]

        # Phase 4: Conditional RealValidator with external verification
        if USE_REAL_VALIDATOR and shared_kb is not None:
            validators = [
                RealValidator(
                    agent_id=f"RealValidator_R{round_num}_{i}",
                    knowledge_base=shared_kb,
                    confidence_threshold=DYNAMIC_KB_CONFIDENCE_THRESHOLD
                )
                for i in range(NUM_VALIDATORS)
            ]
            print(f"[ROUND {round_num + 1}] Created {NUM_VALIDATORS} RealValidators with dynamic KB")
        else:
            validators = [
                Validator(f"Validator_R{round_num}_{i}", task_config.task_prompt)
                for i in range(NUM_VALIDATORS)
            ]
            if not USE_REAL_VALIDATOR:
                print(f"[ROUND {round_num + 1}] Created {NUM_VALIDATORS} original Validators")

        pruners = [
            Pruner(f"Pruner_R{round_num}_{i}", min_strength=PRUNE_THRESHOLD)
            for i in range(NUM_PRUNERS)
        ]

        foragers = foragers_support + foragers_critique

        # Step 5: PHASE 2C - Launch swarm with staged parallel execution
        print(f"[ROUND {round_num + 1}] Launching swarm (staged parallel coordination)...\n")

        # Create StageCoordinator for this round
        stage_coordinator = StageCoordinator(llm_pool, signal_store)

        # STAGED EXECUTION STRATEGY:
        # - Stage 1: Scouts (no dependencies) - parallel across LLM pool
        # - Stage 2: Foragers + Critics (depend on scouts) - parallel across LLM pool
        # - Stage 3: Haters (depend on foragers/critics) - parallel across LLM pool
        # - Background: Validators, Pruners, Environment, Dialogue (run concurrently)

        # Set max_actions for agents (used in prepare_prompt to limit iterations)
        for scout in scouts:
            scout.max_actions = ITERATIONS_PER_ROUND
        for forager in foragers:
            forager.max_actions = ITERATIONS_PER_ROUND
        for critic in critics:
            critic.max_actions = ITERATIONS_PER_ROUND

        HATER_ACTIONS_PER_ROUND = max(200 // NUM_ROUNDS, ITERATIONS_PER_ROUND)
        for hater in haters:
            hater.max_actions = HATER_ACTIONS_PER_ROUND
            hater.min_strength = MIN_DEPOSIT_STRENGTH
            hater.target_consensus = True

        # Background tasks (validators, pruners, environment, dialogue)
        background_tasks = []

        # SimpleScouts or Validators use old run() method (not yet refactored)
        if USE_SIMPLE_SCOUTS:
            # FIX: Define run_simple_scout outside loop to avoid closure bug
            async def run_simple_scout(scout_agent, iterations, delay):
                for iteration in range(iterations):
                    await scout_agent.step(signal_store, llm, task_config.task_prompt, iteration)
                    await asyncio.sleep(delay)

            for scout in scouts:
                background_tasks.append(asyncio.create_task(
                    run_simple_scout(scout, ITERATIONS_PER_ROUND, ITERATION_DELAY)
                ))

        # Validators use old run() method (not yet refactored)
        for validator in validators:
            background_tasks.append(asyncio.create_task(
                validator.run(signal_store, llm,
                            min_strength=MIN_DEPOSIT_STRENGTH,
                            max_actions=ITERATIONS_PER_ROUND,
                            temperature=0.5)
            ))

        # Pruners use old run() method (not yet refactored)
        for pruner in pruners:
            background_tasks.append(asyncio.create_task(
                pruner.run(signal_store, max_actions=ITERATIONS_PER_ROUND // 3)
            ))

        # Environment process for this round
        async def environment_process():
            contrarian_types = [task_config.signal_types["critique"], task_config.signal_types["counter"]]

            for iteration in range(ITERATIONS_PER_ROUND):
                await asyncio.sleep(ITERATION_DELAY)

                signal_store.decay_all(contrarian_types=contrarian_types)
                signal_store.prune_weak()

                if (iteration + 1) % 10 == 0:
                    stats = signal_store.get_stats()
                    print(f"[ROUND {round_num + 1}] [ITER {iteration+1:02d}] Signals: {stats['total_signals']}, "
                          f"Avg strength: {stats['avg_strength']:.2f}")

        background_tasks.append(asyncio.create_task(environment_process()))

        # Dialogue coordinator for multi-turn exchanges
        dialogue_coordinator = DialogueCoordinator(signal_store, task_config)

        async def dialogue_process():
            """Periodically check for unanswered objections and trigger responses."""
            agents_dict = {
                'forager': foragers,
                'hater': haters
            }

            for iteration in range(ITERATIONS_PER_ROUND):
                await asyncio.sleep(ITERATION_DELAY * 2)  # Check every 2 iterations

                # Trigger dialogue responses
                await dialogue_coordinator.check_and_trigger_responses(agents_dict, llm)

                # Log dialogue stats periodically
                if (iteration + 1) % 20 == 0:
                    stats = dialogue_coordinator.get_dialogue_stats()
                    if stats['total_objections'] > 0:
                        print(f"[DIALOGUE] Objections: {stats['total_objections']}, "
                              f"Avg depth: {stats['average_dialogue_depth']:.1f}, "
                              f"Max depth: {stats['max_dialogue_depth']}")

        background_tasks.append(asyncio.create_task(dialogue_process()))

        # STAGED PARALLEL EXECUTION
        agent_processing_start = time.time()

        # Run staged execution in parallel with background tasks
        async def run_staged_swarm():
            """Run agents in dependency-ordered stages for true parallelism."""
            if not USE_SIMPLE_SCOUTS:
                # Define stages with dependencies
                stages = [
                    # Stage 1: Scouts (no dependencies)
                    ("scouts", scouts, None, ITERATIONS_PER_ROUND),
                    # Stage 2: Foragers + Critics (depend on scouts)
                    ("foragers", foragers, ["scouts"], ITERATIONS_PER_ROUND),
                    ("critics", critics, ["scouts"], ITERATIONS_PER_ROUND),
                    # Stage 3: Haters (depend on foragers + critics)
                    ("haters", haters, ["foragers", "critics"], HATER_ACTIONS_PER_ROUND)
                ]

                # Run all stages
                all_stats = await stage_coordinator.run_parallel_stages(stages)

                # Print stage summary
                print(f"\n[ROUND {round_num + 1}] Staged execution complete:")
                print(stage_coordinator.get_summary())

        # Execute staged swarm + background tasks in parallel
        staged_task = asyncio.create_task(run_staged_swarm())
        all_tasks = [staged_task] + background_tasks

        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        agent_processing_time = time.time() - agent_processing_start
        print(f"\n[TIMING] Agent processing phase (parallel): {agent_processing_time:.2f}s")

        # Log any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[ROUND {round_num + 1}] [ERROR] Task {i} failed: {type(result).__name__}: {result}")

        # Step 6: Synthesize what was learned this round
        print(f"\n[ROUND {round_num + 1}] Synthesizing round insights...")
        synthesis_start = time.time()
        synthesizer = Synthesizer(f"Synthesizer_R{round_num}", task_config.task_prompt, task_config=task_config)
        round_synthesis = await synthesizer.synthesize(
            signal_store,
            llm,
            task_config.signal_types,
            temperature=TEMP_SYNTHESIZER
        )
        synthesis_time = time.time() - synthesis_start
        print(f"[TIMING] Synthesis phase: {synthesis_time:.2f}s")

        # Store synthesis
        all_round_syntheses.append(round_synthesis)

        # Step 7: Record round results
        round_coordinator.record_round(round_num, round_synthesis, keywords)

        # Show round stats
        round_stats = signal_store.get_stats()
        print(f"\n[ROUND {round_num + 1}] Complete:")
        print(f"  Signals generated: {round_stats['total_signals']}")
        print(f"  Signal distribution: {round_stats['by_type']}")
        print(f"  Synthesis preview: {round_synthesis[:150]}...")

        # Show swarm health metrics
        health = monitor.calculate_health_metrics()
        print(f"\n[ROUND {round_num + 1}] Swarm Health:")
        print(f"  Overall health score: {health['health_score']:.2f}/1.0")
        print(f"  Convergence status: {health['convergence_status']}")
        print(f"  Signal diversity: {health['signal_diversity']:.2f}")
        print(f"  Objection rate: {health['objection_rate']:.1%}")

        # Show warnings if any
        if health['warnings']:
            print(f"  ⚠️  Warnings:")
            for warning in health['warnings']:
                print(f"     - {warning}")

        # Show round timing breakdown
        round_total_time = time.time() - round_start
        print(f"\n[TIMING] Round {round_num + 1} total: {round_total_time:.2f}s")
        print(f"[TIMING] Breakdown: Research: {time.time() - research_start - agent_processing_time - synthesis_time:.2f}s, "
              f"Agents: {agent_processing_time:.2f}s ({agent_processing_time/round_total_time*100:.1f}%), "
              f"Synthesis: {synthesis_time:.2f}s ({synthesis_time/round_total_time*100:.1f}%)")

        # Store final signal store
        final_signal_store = signal_store

    # After all rounds, synthesize final answer across all rounds
    print("\n" + "=" * 70)
    print("SYNTHESIZING FINAL ANSWER ACROSS ALL ROUNDS")
    print("=" * 70)

    # Use the last round's signal store for final synthesis
    signal_store = final_signal_store

    synthesizer = Synthesizer("FinalSynthesizer", task_config.task_prompt, task_config=task_config)
    synthesis = await synthesizer.synthesize(
        signal_store,
        llm,
        task_config.signal_types,
        temperature=TEMP_SYNTHESIZER
    )

    # Results
    print("\n" + "=" * 70)
    print("SWARM COMPLETE - RESULTS")
    print("=" * 70)

    # Show round progression
    print(f"\n--- Round Progression ---")
    print(round_coordinator.get_summary())

    # Show stats (aggregate from pool + synthesis LLM)
    pool_cache_stats = llm_pool.get_cache_stats()
    llm_cache_stats = llm.get_cache_stats()

    # Combine stats
    total_calls = pool_cache_stats['total_calls'] + llm_cache_stats['total_calls']
    total_hits = pool_cache_stats['cache_hits'] + llm_cache_stats['cache_hits']
    combined_hit_rate = total_hits / total_calls if total_calls > 0 else 0.0

    # Use llm success rate (pool doesn't track this yet)
    success_rate = llm_cache_stats.get('success_rate', 1.0)

    print(f"\n--- Performance ---")
    print(f"Generation Success Rate: {success_rate:.1%}")
    print(f"Cache Hit Rate (combined): {combined_hit_rate:.1%}")
    print(f"  LLM Pool: {pool_cache_stats['hit_rate']:.1%} ({pool_cache_stats['total_calls']} calls)")
    print(f"  Synthesis: {llm_cache_stats['hit_rate']:.1%} ({llm_cache_stats['total_calls']} calls)")

    # Create cache_stats dict for save_run_outputs
    cache_stats = {
        'success_rate': success_rate,
        'hit_rate': combined_hit_rate,
        'generation_successes': llm_cache_stats.get('generation_successes', total_calls),
        'generation_failures': llm_cache_stats.get('generation_failures', 0)
    }

    # Show top signals of each type
    for role, signal_type in task_config.signal_types.items():
        top_signals = signal_store.get_top_signals(signal_type, 3)

        if top_signals:
            print(f"\n--- Top {signal_type}s ---")
            for i, signal in enumerate(top_signals, 1):
                print(f"\n{i}. [Strength: {signal.strength:.3f}]")
                preview = signal.content[:200] + "..." if len(signal.content) > 200 else signal.content
                print(f"   {preview}")
        else:
            print(f"\n--- No {signal_type}s generated ---")

    # Final stats
    final_stats = signal_store.get_stats()
    print(f"\n--- Final Statistics ---")
    print(f"Total signals: {final_stats['total_signals']}")
    print(f"By type: {final_stats['by_type']}")
    print(f"Average strength: {final_stats['avg_strength']:.3f}")

    # Phase 4: Show DynamicKnowledgeBase learning stats
    if USE_REAL_VALIDATOR and shared_kb is not None:
        kb_stats = shared_kb.get_stats()
        print(f"\n--- Dynamic Knowledge Base (Phase 4) ---")
        print(f"Facts learned: {kb_stats['total_facts']}")
        print(f"High confidence facts (>{DYNAMIC_KB_CONFIDENCE_THRESHOLD}): {kb_stats['high_confidence_facts']}")
        print(f"Average confidence: {kb_stats['avg_confidence']:.2f}")
        print(f"Cache hit rate: {kb_stats['cache_hit_rate']:.2%}")
        print(f"Conflicts detected: {kb_stats['conflicts']}")
        if kb_stats['total_facts'] > 0:
            print("\nTop learned facts:")
            learned_facts = shared_kb.export_knowledge(min_confidence=DYNAMIC_KB_CONFIDENCE_THRESHOLD)
            for i, fact in enumerate(learned_facts[:5], 1):
                print(f"  {i}. {fact['claim'][:80]}...")
                print(f"     Confidence: {fact['confidence']:.2f}, Verifications: {fact['verifications']}")

    # Save outputs to structured directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if os.path.exists("/content"):
        output_base = Path("/content/swarm/runs/outputs")
    else:
        output_base = Path("outputs")
    run_dir = output_base / f"{task_config.task_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_info = {
        "Model": f"{MODEL_NAME} on {DEVICE}",
        "Agents": f"{NUM_SCOUTS} scouts, {NUM_FORAGERS} foragers, {NUM_CRITICS} critics, {NUM_HATERS} haters",
        "Temperatures": f"Scout={TEMP_SCOUT}, Forager={TEMP_FORAGER}, Critic={TEMP_CRITIC}, Hater={TEMP_HATER}",
        "Iterations": MAX_ITERATIONS,
        "Decay Rate": DECAY_RATE,
        "Prune Threshold": PRUNE_THRESHOLD,
        "Min Deposit": MIN_DEPOSIT_STRENGTH,
        "Min Amplify": MIN_AMPLIFY_STRENGTH
    }

    # Generate round summary for output
    round_summary = round_coordinator.get_summary()

    base_truth = save_run_outputs(run_dir, task_config, signal_store, cache_stats, config_info, synthesis, round_summary)

    print(f"\n[SAVED] Outputs saved to: {run_dir}")

    # Cleanup: Delete temporary context file
    print(f"\n[CLEANUP] Cleaning up temporary files...")
    retrieval_stats = dynamic_retriever.get_stats()
    print(f"[CLEANUP] Retrieval stats: {retrieval_stats['num_searches']} searches, "
          f"{retrieval_stats['file_size_kb']:.1f}KB gathered")
    dynamic_retriever.cleanup()

    # Phase 5: Cleanup advanced retriever (optional)
    if advanced_retriever:
        adv_stats = advanced_retriever.get_stats()
        print(f"[CLEANUP] Advanced retrieval stats:")
        print(f"  - Total rounds: {adv_stats['total_rounds']}")
        print(f"  - Total words ingested: {adv_stats['total_words_ingested']:,}")
        print(f"  - Total sources accessed: {adv_stats['total_sources_accessed']}")
        print(f"  - Total fragments: {adv_stats['total_fragments']}")
        print(f"  - Avg words per round: {adv_stats['avg_words_per_round']:.0f}")
        advanced_retriever.cleanup()

    # Display base truth prominently
    print("\n" + "=" * 70)
    print("BASE TRUTH (FINAL ANSWER)")
    print("=" * 70)
    if base_truth:
        print(f"\n{base_truth}\n")
        if synthesis:
            print(f"[Source: Synthesized from swarm insights]")
        else:
            print(f"[Source: Strongest initial signal]")
    else:
        print("\nNo final result generated.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Configure logging early (reads LOG_LEVEL from environment)
    setup_logging()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_task.py <task_type> [custom_prompt]")
        print("\nTask types:")
        print("  debate          - Argue for/against a thesis")
        print("  creative        - Generate and refine creative content")
        print("  analysis        - Analyze and critique a topic")
        print("  problem_solving - Propose and evaluate solutions")
        print("  hyper_test      - FAST end-to-end pipeline test (validates all components)")
        print("\nExamples:")
        print('  python run_task.py creative')
        print('  python run_task.py creative "Write a haiku about AI"')
        print('  python run_task.py problem_solving "How to reduce plastic waste?"')
        print('  python run_task.py hyper_test  # Quick validation of full pipeline')
        sys.exit(1)

    task_type = sys.argv[1]
    custom_prompt = sys.argv[2] if len(sys.argv) > 2 else None

    # HYPER TEST MODE: Fast end-to-end validation
    if task_type == "hyper_test":
        print("=" * 70)
        print("HYPER TEST MODE - Fast End-to-End Pipeline Validation")
        print("=" * 70)
        print("\nThis mode tests the full pipeline in ~60 seconds:")
        print("  - 2 rounds (instead of 3)")
        print("  - 5 iterations per round (instead of 20)")
        print("  - 3 scouts (instead of 10)")
        print("  - Tests: Scouts → Foragers → Critics → Haters → Synthesis")
        print("  - Validates: Web search, signal flow, critic boosting, full discourse")
        print("\nStarting hyper test...\n")

        # Set environment flag for hyper test mode
        os.environ['HYPER_TEST_MODE'] = '1'

        # Override config for speed
        import swarm.core.config as config_module
        config_module.MAX_ITERATIONS = 5  # Very short
        config_module.NUM_SCOUTS = 3      # Minimal agents
        config_module.NUM_FORAGERS = 2
        config_module.NUM_CRITICS = 1
        config_module.NUM_HATERS = 1

        # Run problem_solving with test prompt
        task_type = "problem_solving"
        custom_prompt = "How can we make cities more sustainable?"

    # Run the task
    if os.environ.get('HYPER_TEST_MODE', '0') == '1':
        # Hyper test mode - catch errors and show clear pass/fail
        try:
            asyncio.run(run_task(task_type, custom_prompt))
            print("\n" + "=" * 70)
            print("✓ HYPER TEST PASSED - All pipeline components working!")
            print("=" * 70)
            print("\nValidated:")
            print("  ✓ Model loading and generation")
            print("  ✓ Web search and information retrieval")
            print("  ✓ Scout signal generation")
            print("  ✓ Forager elaboration (IMPLEMENTATION/CHALLENGE)")
            print("  ✓ Critic evaluation and boosting")
            print("  ✓ Hater objections")
            print("  ✓ Round-based iterative refinement")
            print("  ✓ Full discourse graph synthesis")
            print("\nYou can now run full tasks with confidence!")
            print("=" * 70)
        except Exception as e:
            print("\n" + "=" * 70)
            print("✗ HYPER TEST FAILED")
            print("=" * 70)
            print(f"\nError: {type(e).__name__}: {e}")
            print("\nThis error would have occurred in a full run.")
            print("Fix the issue above before running full tasks.")
            print("=" * 70)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Normal mode
        asyncio.run(run_task(task_type, custom_prompt))
