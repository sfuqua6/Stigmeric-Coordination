#!/usr/bin/env python3
"""Comprehensive evaluation for AI Swarm Mechanics.

Includes:
- Swarm-specific metrics (debate quality, signal quality, convergence)
- Performance metrics (throughput, latency, efficiency)
- Agent effectiveness metrics

For standard LLM benchmarks (MMLU, TruthfulQA, etc.), use lm-evaluation-harness:
    pip install lm-eval
    lm_eval --model hf --model_args pretrained=microsoft/phi-2 --tasks mmlu
"""

import asyncio
import json
import time
import argparse
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from swarm.core.signal_store import SignalStore, Signal


class SwarmEvaluator:
    """Evaluate swarm system on multiple dimensions."""

    def __init__(self, output_dir: str = "evaluation_results"):
        """Initialize evaluator.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.results = {
            'timestamp': datetime.now().isoformat(),
            'benchmarks': [],
            'summary': {}
        }

    def evaluate_debate_quality(self, signal_store: SignalStore) -> Dict:
        """Evaluate quality of debate/discussion.

        Metrics:
        - Evidence ratio: Evidence per claim
        - Critique coverage: Critiques per insight
        - Diversity score: Unique content ratio
        - Strength distribution: Average and extremes

        Args:
            signal_store: Completed signal store

        Returns:
            Dictionary of quality metrics
        """
        signals = signal_store.get_all_signals()

        if not signals:
            return {
                'error': 'No signals found',
                'total_signals': 0
            }

        # Count signal types
        by_type = {}
        for s in signals:
            by_type[s.type] = by_type.get(s.type, 0) + 1

        # Evidence ratio: EVIDENCE per (INSIGHT + DRAFT + REFINEMENT)
        claims = (by_type.get('INSIGHT', 0) +
                 by_type.get('DRAFT', 0) +
                 by_type.get('REFINEMENT', 0))
        evidence = by_type.get('EVIDENCE', 0)
        evidence_ratio = evidence / max(claims, 1)

        # Critique coverage: CRITIQUE per claim
        critiques = by_type.get('CRITIQUE', 0)
        critique_coverage = critiques / max(claims, 1)

        # Diversity: unique content ratio
        contents = [s.content[:100] for s in signals]  # First 100 chars
        unique_ratio = len(set(contents)) / max(len(contents), 1)

        # Strength statistics
        strengths = [s.strength for s in signals]
        avg_strength = sum(strengths) / len(strengths)
        max_strength = max(strengths)
        min_strength = min(strengths)

        # Quality tiers
        strong_signals = [s for s in signals if s.strength > 0.7]
        medium_signals = [s for s in signals if 0.3 < s.strength <= 0.7]
        weak_signals = [s for s in signals if s.strength <= 0.3]

        return {
            'total_signals': len(signals),
            'signal_types': by_type,
            'claims_count': claims,
            'evidence_ratio': round(evidence_ratio, 2),
            'critique_coverage': round(critique_coverage, 2),
            'diversity_score': round(unique_ratio, 2),
            'strength': {
                'average': round(avg_strength, 3),
                'max': round(max_strength, 3),
                'min': round(min_strength, 3),
                'strong_count': len(strong_signals),
                'medium_count': len(medium_signals),
                'weak_count': len(weak_signals)
            }
        }

    def evaluate_signal_quality(self, signal_store: SignalStore) -> Dict:
        """Evaluate signal ecosystem health.

        Metrics:
        - Provenance: Signals with parents
        - Depth: Maximum graph depth
        - Branching: Average children per signal
        - Validation: Signals with validation metadata

        Args:
            signal_store: Signal store

        Returns:
            Signal quality metrics
        """
        signals = signal_store.get_all_signals()

        if not signals:
            return {'error': 'No signals'}

        # Provenance tracking
        with_parents = [s for s in signals if s.parent]
        provenance_ratio = len(with_parents) / len(signals)

        # Graph depth
        def get_depth(signal_id: str, visited: set) -> int:
            if signal_id in visited:
                return 0
            visited.add(signal_id)
            signal = signal_store.get_signal(signal_id)
            if not signal or not signal.parent:
                return 1
            return 1 + get_depth(signal.parent, visited)

        depths = [get_depth(s.id, set()) for s in signals]
        max_depth = max(depths) if depths else 0
        avg_depth = sum(depths) / len(depths) if depths else 0

        # Branching factor (children per signal)
        parent_counts = {}
        for s in signals:
            if s.parent:
                parent_counts[s.parent] = parent_counts.get(s.parent, 0) + 1

        avg_children = sum(parent_counts.values()) / len(parent_counts) if parent_counts else 0

        # Validation coverage
        with_validation = [s for s in signals if s.metadata and 'validation' in s.metadata]
        validation_ratio = len(with_validation) / len(signals)

        return {
            'provenance_ratio': round(provenance_ratio, 2),
            'max_depth': max_depth,
            'avg_depth': round(avg_depth, 2),
            'avg_children_per_signal': round(avg_children, 2),
            'validation_coverage': round(validation_ratio, 2),
            'orphan_signals': len(signals) - len(with_parents)
        }

    def evaluate_convergence(self, signal_counts: List[int]) -> Dict:
        """Evaluate system convergence over time.

        Args:
            signal_counts: List of total signal counts per iteration/round

        Returns:
            Convergence metrics
        """
        if len(signal_counts) < 5:
            return {
                'converged': False,
                'note': 'Insufficient data (need >=5 iterations)'
            }

        # Consider converged when count varies <10% for last 5 iterations
        recent = signal_counts[-5:]
        if not recent:
            return {'converged': False, 'note': 'No recent data'}

        avg_recent = sum(recent) / len(recent)
        if avg_recent == 0:
            return {'converged': False, 'note': 'Zero signals'}

        max_var = max(abs(x - avg_recent) / avg_recent for x in recent if avg_recent > 0)
        converged = max_var < 0.1

        # Find convergence point
        convergence_iteration = None
        if converged and len(signal_counts) >= 10:
            for i in range(len(signal_counts) - 5):
                window = signal_counts[i:i+5]
                avg_window = sum(window) / len(window)
                if avg_window == 0:
                    continue
                window_var = max(abs(x - avg_window) / avg_window for x in window)
                if window_var < 0.1:
                    convergence_iteration = i
                    break

        # Growth rate
        growth_rates = [signal_counts[i+1] - signal_counts[i]
                       for i in range(len(signal_counts)-1)]
        avg_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 0

        return {
            'converged': converged,
            'convergence_iteration': convergence_iteration,
            'final_count': signal_counts[-1],
            'peak_count': max(signal_counts),
            'avg_growth_per_iteration': round(avg_growth, 2),
            'convergence_variance': round(max_var, 3)
        }

    def evaluate_performance(self, runtime: float, metrics: Dict) -> Dict:
        """Evaluate system performance.

        Args:
            runtime: Total runtime in seconds
            metrics: Runtime metrics (actions, generations, etc.)

        Returns:
            Performance metrics
        """
        total_actions = metrics.get('total_actions', 0)
        total_signals = metrics.get('total_signals', 0)
        total_generations = metrics.get('llm_generations', total_actions)

        return {
            'runtime_seconds': round(runtime, 2),
            'runtime_minutes': round(runtime / 60, 2),
            'total_actions': total_actions,
            'total_signals_created': total_signals,
            'actions_per_minute': round(total_actions / (runtime / 60), 2) if runtime > 0 else 0,
            'signals_per_minute': round(total_signals / (runtime / 60), 2) if runtime > 0 else 0,
            'llm_generations': total_generations,
            'generations_per_minute': round(total_generations / (runtime / 60), 2) if runtime > 0 else 0,
            'avg_seconds_per_action': round(runtime / total_actions, 3) if total_actions > 0 else 0
        }

    def evaluate_all(self, signal_store: SignalStore, runtime: float,
                    metrics: Dict, signal_counts: List[int]) -> Dict:
        """Run all evaluations.

        Args:
            signal_store: Final signal store
            runtime: Total runtime
            metrics: Runtime metrics
            signal_counts: Signal counts over time

        Returns:
            Complete evaluation results
        """
        return {
            'debate_quality': self.evaluate_debate_quality(signal_store),
            'signal_quality': self.evaluate_signal_quality(signal_store),
            'convergence': self.evaluate_convergence(signal_counts),
            'performance': self.evaluate_performance(runtime, metrics)
        }

    def print_summary(self, results: Dict):
        """Print human-readable summary.

        Args:
            results: Evaluation results
        """
        print("\n" + "="*70)
        print("SWARM EVALUATION RESULTS")
        print("="*70)

        # Debate Quality
        dq = results.get('debate_quality', {})
        print("\n📊 DEBATE QUALITY:")
        print(f"  Total signals: {dq.get('total_signals', 0)}")
        print(f"  Evidence ratio: {dq.get('evidence_ratio', 0):.2f} (target: 0.5-1.5)")
        print(f"  Critique coverage: {dq.get('critique_coverage', 0):.2f} (target: 0.3-0.7)")
        print(f"  Diversity: {dq.get('diversity_score', 0):.2f} (target: 0.7-0.9)")
        print(f"  Avg strength: {dq.get('strength', {}).get('average', 0):.3f}")

        # Signal Quality
        sq = results.get('signal_quality', {})
        print("\n🔗 SIGNAL QUALITY:")
        print(f"  Provenance ratio: {sq.get('provenance_ratio', 0):.2f}")
        print(f"  Max depth: {sq.get('max_depth', 0)}")
        print(f"  Avg children/signal: {sq.get('avg_children_per_signal', 0):.2f}")

        # Convergence
        conv = results.get('convergence', {})
        print("\n📈 CONVERGENCE:")
        print(f"  Converged: {'✓ Yes' if conv.get('converged') else '✗ No'}")
        if conv.get('convergence_iteration'):
            print(f"  Converged at iteration: {conv['convergence_iteration']}")
        print(f"  Final signal count: {conv.get('final_count', 0)}")

        # Performance
        perf = results.get('performance', {})
        print("\n⚡ PERFORMANCE:")
        print(f"  Runtime: {perf.get('runtime_minutes', 0):.2f} minutes")
        print(f"  Actions/min: {perf.get('actions_per_minute', 0):.1f}")
        print(f"  Signals/min: {perf.get('signals_per_minute', 0):.1f}")

        print("\n" + "="*70)

    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file.

        Args:
            results: Results to save
            filename: Output filename
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {filepath}")


async def main():
    """Run evaluation (example/testing)."""
    parser = argparse.ArgumentParser(description='Evaluate AI Swarm Mechanics')
    parser.add_argument('--output-dir', default='evaluation_results',
                       help='Output directory for results')
    args = parser.parse_args()

    evaluator = SwarmEvaluator(output_dir=args.output_dir)

    # Example: Create dummy signal store for testing
    signal_store = SignalStore()

    # Add some example signals
    signal_store.deposit('DRAFT', 'Climate change is accelerating', 0.8, 'scout_1')
    signal_store.deposit('EVIDENCE', 'CO2 levels at 420ppm', 0.9, 'validator_1',
                        parent='DRAFT_0000')
    signal_store.deposit('CRITIQUE', 'Need more recent data', 0.6, 'critic_1',
                        parent='DRAFT_0000')
    signal_store.deposit('REFINEMENT', 'Climate change accelerating based on 2023 data',
                        0.85, 'forager_1', parent='DRAFT_0000')

    # Example metrics
    metrics = {
        'total_actions': 50,
        'total_signals': 4,
        'llm_generations': 50
    }

    signal_counts = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4]  # Converged at iteration 5

    # Run evaluation
    results = evaluator.evaluate_all(
        signal_store=signal_store,
        runtime=120.0,  # 2 minutes
        metrics=metrics,
        signal_counts=signal_counts
    )

    # Print and save
    evaluator.print_summary(results)
    evaluator.save_results(results, 'example_evaluation.json')

    print("\n✅ Evaluation complete!")
    print(f"📁 Check {args.output_dir}/ for detailed results")


if __name__ == '__main__':
    asyncio.run(main())
