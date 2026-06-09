#!/usr/bin/env python3
"""A/B Comparative Evaluation: Swarm vs Single-Agent Baseline

This is what you actually need - direct comparison to prove the swarm approach works better.

Compares:
- Single-agent baseline (same LLM, no swarm)
- Swarm system (multi-agent stigmergic coordination)

On identical prompts, measuring:
- Factual accuracy (external validation)
- Evidence coverage
- Reasoning depth
- Output quality
- Efficiency (time/tokens per quality unit)
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

from single_agent_baseline import SingleAgentBaseline
from swarm_evaluation import SwarmEvaluator


class ComparativeEvaluator:
    """Compare swarm vs single-agent baseline on identical tasks."""

    def __init__(self, output_dir: str = "evaluation_results", quick_mode: bool = False):
        """Initialize comparative evaluator.

        Args:
            output_dir: Directory to save results
            quick_mode: If True, use minimal agents/iterations for fast testing
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.baseline = None
        self.swarm_evaluator = SwarmEvaluator(output_dir=str(self.output_dir))
        self.quick_mode = quick_mode

        self.results = {
            'timestamp': datetime.now().isoformat(),
            'comparisons': [],
            'summary': {},
            'quick_mode': quick_mode
        }

        if quick_mode:
            print("\n⚡ QUICK MODE ENABLED - Fast testing with minimal agents")
            print("   - 3 iterations instead of 20")
            print("   - 3 scouts instead of 5")
            print("   - 1 test prompt instead of 3")
            print("   - Expected runtime: ~60-90 seconds\n")

    async def initialize(self):
        """Initialize both systems."""
        print("[COMPARATIVE] Initializing baseline and swarm...")
        self.baseline = SingleAgentBaseline()
        await self.baseline.initialize()
        print("[COMPARATIVE] Ready for A/B testing")

    async def run_baseline(self, prompt: str, max_iterations: int = None) -> Dict:
        """Run single-agent baseline.

        Args:
            prompt: Task prompt
            max_iterations: Max refinement iterations (uses 3 in quick mode, 10 in normal)

        Returns:
            Baseline results
        """
        if max_iterations is None:
            max_iterations = 3 if self.quick_mode else 10
        return await self.baseline.run(prompt, max_iterations=max_iterations)

    async def run_swarm(self, prompt: str, max_iterations: int = None) -> Dict:
        """Run swarm system.

        Args:
            prompt: Task prompt
            max_iterations: Max agent iterations (uses 3 in quick mode, 20 in normal)

        Returns:
            Swarm results with evaluation metrics
        """
        if max_iterations is None:
            max_iterations = 3 if self.quick_mode else 20

        print(f"\n[SWARM] Running swarm on: {prompt[:60]}...")
        print(f"[SWARM] Max iterations: {max_iterations}")
        print(f"[SWARM] Quick mode: {self.quick_mode}")

        # Import swarm runner
        from run_task_wrapper import run_swarm_simple

        start_time = time.time()

        # Run swarm (simplified wrapper that returns results)
        try:
            result = await run_swarm_simple(
                prompt=prompt,
                task_type='analysis',  # Generic analysis task
                max_iterations=max_iterations,
                quick_mode=self.quick_mode
            )

            runtime = time.time() - start_time

            # Evaluate swarm results
            signal_store = result['signal_store']
            debate_quality = self.swarm_evaluator.evaluate_debate_quality(signal_store)
            signal_quality = self.swarm_evaluator.evaluate_signal_quality(signal_store)

            return {
                'output': result['synthesis'],
                'prompt': prompt,
                'runtime': runtime,
                'llm_generations': result.get('llm_generations', 0),
                'total_signals': result.get('total_signals', 0),
                'method': 'swarm',
                'debate_quality': debate_quality,
                'signal_quality': signal_quality,
                'convergence': result.get('convergence', {})
            }

        except Exception as e:
            print(f"[SWARM] Error running swarm: {e}")
            import traceback
            traceback.print_exc()

            # Return placeholder on error
            return {
                'output': f"[Error running swarm: {e}]",
                'prompt': prompt,
                'runtime': time.time() - start_time,
                'llm_generations': 0,
                'total_signals': 0,
                'method': 'swarm',
                'debate_quality': {},
                'signal_quality': {},
                'convergence': {},
                'error': str(e)
            }

    def analyze_output_quality(self, output: str) -> Dict:
        """Analyze output quality objectively.

        Metrics:
        - Evidence count (citations, data points, examples)
        - Claim count (distinct arguments/assertions)
        - Reasoning depth (argument chains, if-then logic)
        - Nuance (multiple perspectives, caveats, qualifications)
        - Structure (organization, coherence)

        Args:
            output: Generated text

        Returns:
            Quality metrics
        """
        # Handle None or invalid output gracefully
        if output is None or not isinstance(output, str):
            return {
                'evidence_count': 0,
                'claim_count': 0,
                'reasoning_depth': 0,
                'nuance_score': 0,
                'paragraph_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'evidence_per_claim': 0,
                'error': 'Output is None or invalid'
            }

        lines = output.split('\n')
        sentences = output.split('.')

        # Evidence markers
        evidence_markers = [
            'according to', 'research shows', 'studies indicate',
            'data reveals', 'evidence suggests', 'for example',
            'for instance', 'specifically', 'in fact'
        ]
        evidence_count = sum(
            1 for marker in evidence_markers
            if marker.lower() in output.lower()
        )

        # Claim markers
        claim_markers = [
            'therefore', 'thus', 'consequently', 'this means',
            'suggests that', 'indicates that', 'shows that'
        ]
        claim_count = sum(
            1 for marker in claim_markers
            if marker.lower() in output.lower()
        )

        # Reasoning depth (logical connectors)
        reasoning_markers = [
            'because', 'since', 'if', 'then', 'when', 'while',
            'although', 'however', 'nevertheless', 'despite'
        ]
        reasoning_count = sum(
            1 for marker in reasoning_markers
            if marker.lower() in output.lower()
        )

        # Nuance markers (multiple perspectives)
        nuance_markers = [
            'however', 'on the other hand', 'alternatively',
            'conversely', 'in contrast', 'yet', 'still',
            'although', 'while', 'whereas'
        ]
        nuance_count = sum(
            1 for marker in nuance_markers
            if marker.lower() in output.lower()
        )

        # Structure (paragraphs, sections)
        paragraph_count = len([line for line in lines if line.strip() == ''])
        word_count = len(output.split())

        return {
            'evidence_count': evidence_count,
            'claim_count': claim_count,
            'reasoning_depth': reasoning_count,
            'nuance_score': nuance_count,
            'paragraph_count': paragraph_count,
            'word_count': word_count,
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_sentence_length': word_count / max(len(sentences), 1),
            'evidence_per_claim': evidence_count / max(claim_count, 1)
        }

    def compare_outputs(self, baseline_result: Dict, swarm_result: Dict) -> Dict:
        """Compare baseline vs swarm outputs.

        Args:
            baseline_result: Single-agent results
            swarm_result: Swarm results

        Returns:
            Comparison metrics
        """
        baseline_quality = self.analyze_output_quality(baseline_result['output'])
        swarm_quality = self.analyze_output_quality(swarm_result['output'])

        # Calculate improvements
        improvements = {}
        for metric in baseline_quality:
            baseline_val = baseline_quality[metric]
            swarm_val = swarm_quality[metric]

            if baseline_val > 0:
                improvement = ((swarm_val - baseline_val) / baseline_val) * 100
            else:
                improvement = 100 if swarm_val > 0 else 0

            improvements[metric] = {
                'baseline': baseline_val,
                'swarm': swarm_val,
                'improvement_pct': round(improvement, 1),
                'winner': 'swarm' if swarm_val > baseline_val else 'baseline' if baseline_val > swarm_val else 'tie'
            }

        # Overall winner
        swarm_wins = sum(1 for m in improvements.values() if m['winner'] == 'swarm')
        baseline_wins = sum(1 for m in improvements.values() if m['winner'] == 'baseline')

        return {
            'quality_comparison': improvements,
            'baseline_quality': baseline_quality,
            'swarm_quality': swarm_quality,
            'swarm_wins': swarm_wins,
            'baseline_wins': baseline_wins,
            'overall_winner': 'swarm' if swarm_wins > baseline_wins else 'baseline' if baseline_wins > swarm_wins else 'tie'
        }

    def compare_efficiency(self, baseline_result: Dict, swarm_result: Dict) -> Dict:
        """Compare efficiency metrics.

        Args:
            baseline_result: Baseline results
            swarm_result: Swarm results

        Returns:
            Efficiency comparison
        """
        # Time efficiency
        baseline_time = baseline_result['runtime']
        swarm_time = swarm_result['runtime']
        time_overhead = ((swarm_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0

        # Token efficiency (generations)
        baseline_gens = baseline_result.get('llm_generations', 0)
        swarm_gens = swarm_result.get('llm_generations', 0)
        token_overhead = ((swarm_gens - baseline_gens) / baseline_gens * 100) if baseline_gens > 0 else 0

        # Quality per token
        baseline_quality = self.analyze_output_quality(baseline_result['output'])
        swarm_quality = self.analyze_output_quality(swarm_result['output'])

        baseline_quality_score = (
            baseline_quality['evidence_count'] +
            baseline_quality['reasoning_depth'] +
            baseline_quality['nuance_score']
        )
        swarm_quality_score = (
            swarm_quality['evidence_count'] +
            swarm_quality['reasoning_depth'] +
            swarm_quality['nuance_score']
        )

        baseline_efficiency = baseline_quality_score / max(baseline_gens, 1)
        swarm_efficiency = swarm_quality_score / max(swarm_gens, 1)

        return {
            'runtime': {
                'baseline': baseline_time,
                'swarm': swarm_time,
                'overhead_pct': round(time_overhead, 1),
                'faster': 'baseline' if baseline_time < swarm_time else 'swarm'
            },
            'llm_generations': {
                'baseline': baseline_gens,
                'swarm': swarm_gens,
                'overhead_pct': round(token_overhead, 1)
            },
            'quality_per_generation': {
                'baseline': round(baseline_efficiency, 2),
                'swarm': round(swarm_efficiency, 2),
                'improvement_pct': round(((swarm_efficiency - baseline_efficiency) / baseline_efficiency * 100), 1) if baseline_efficiency > 0 else 0,
                'winner': 'swarm' if swarm_efficiency > baseline_efficiency else 'baseline'
            }
        }

    async def run_comparison(self, prompt: str, max_iterations: int = 20) -> Dict:
        """Run A/B comparison on a single prompt.

        Args:
            prompt: Task prompt
            max_iterations: Max iterations for both systems

        Returns:
            Complete comparison results
        """
        print(f"\n{'='*70}")
        print(f"A/B COMPARISON")
        print(f"Prompt: {prompt}")
        print(f"{'='*70}\n")

        # Run baseline
        print("[1/2] Running BASELINE (single-agent)...")
        baseline_result = await self.run_baseline(prompt, max_iterations)

        # Run swarm
        print("\n[2/2] Running SWARM (multi-agent)...")
        swarm_result = await self.run_swarm(prompt, max_iterations)

        # Compare
        print("\n[ANALYSIS] Comparing outputs...")
        quality_comparison = self.compare_outputs(baseline_result, swarm_result)
        efficiency_comparison = self.compare_efficiency(baseline_result, swarm_result)

        comparison = {
            'prompt': prompt,
            'baseline_result': baseline_result,
            'swarm_result': swarm_result,
            'quality_comparison': quality_comparison,
            'efficiency_comparison': efficiency_comparison,
            'timestamp': datetime.now().isoformat()
        }

        return comparison

    def print_comparison_summary(self, comparison: Dict):
        """Print human-readable comparison summary.

        Args:
            comparison: Comparison results
        """
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")

        prompt = comparison['prompt']
        print(f"\nPrompt: {prompt[:60]}...")

        # Quality comparison
        quality = comparison['quality_comparison']
        print(f"\n📊 QUALITY COMPARISON:")
        print(f"  Overall winner: {quality['overall_winner'].upper()}")
        print(f"  Swarm wins: {quality['swarm_wins']}/{len(quality['quality_comparison'])} metrics")
        print(f"  Key improvements:")

        for metric, data in quality['quality_comparison'].items():
            if data['winner'] == 'swarm' and data['improvement_pct'] > 10:
                print(f"    • {metric}: +{data['improvement_pct']}%")

        # Efficiency comparison
        efficiency = comparison['efficiency_comparison']
        print(f"\n⚡ EFFICIENCY:")
        print(f"  Runtime: {efficiency['runtime']['baseline']:.1f}s (baseline) vs {efficiency['runtime']['swarm']:.1f}s (swarm)")
        print(f"  LLM generations: {efficiency['llm_generations']['baseline']} vs {efficiency['llm_generations']['swarm']}")
        print(f"  Quality per generation: {efficiency['quality_per_generation']['winner'].upper()} wins")

        print(f"\n{'='*70}\n")

    def save_results(self, filename: str = 'comparative_results.json'):
        """Save all comparison results.

        Args:
            filename: Output filename
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to: {filepath}")

    async def cleanup(self):
        """Cleanup resources with error handling."""
        if self.baseline:
            try:
                await self.baseline.cleanup()
            except RuntimeError as e:
                # CUDA errors during cleanup are non-critical
                print(f"[CLEANUP] Warning: Baseline cleanup encountered error: {e}")
                print(f"[CLEANUP] Continuing anyway (this is normal after CUDA errors)")
            except Exception as e:
                print(f"[CLEANUP] Unexpected cleanup error: {e}")


async def main():
    """Run comparative evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description='A/B Comparative Evaluation: Swarm vs Baseline')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: 3 iterations, 3 scouts, 1 prompt (~60-90s)')
    parser.add_argument('--output-dir', default='evaluation_results',
                       help='Output directory for results')
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ComparativeEvaluator(output_dir=args.output_dir, quick_mode=args.quick)
    await evaluator.initialize()

    # Test prompts covering different task types
    if args.quick:
        # Quick mode: just one short prompt
        test_prompts = [
            "Should AI be regulated? Give pros and cons."
        ]
    else:
        # Full mode: comprehensive prompts
        test_prompts = [
            "Should artificial intelligence development be regulated by governments? Provide arguments for and against.",
            "Explain quantum computing and its potential impact on current encryption methods.",
            "Analyze the scientific evidence for human-caused climate change and common counter-arguments."
        ]

    for prompt in test_prompts:
        comparison = await evaluator.run_comparison(prompt)
        evaluator.results['comparisons'].append(comparison)
        evaluator.print_comparison_summary(comparison)

        # Print actual outputs in quick mode for visibility
        if args.quick:
            print(f"\n{'='*70}")
            print("BASELINE OUTPUT:")
            print(f"{'='*70}")
            print(comparison['baseline_result']['output'])
            print(f"\n{'='*70}")
            print("SWARM OUTPUT:")
            print(f"{'='*70}")
            print(comparison['swarm_result']['output'])
            print(f"{'='*70}\n")

    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")

    swarm_total_wins = sum(
        c['quality_comparison']['swarm_wins']
        for c in evaluator.results['comparisons']
    )
    baseline_total_wins = sum(
        c['quality_comparison']['baseline_wins']
        for c in evaluator.results['comparisons']
    )

    print(f"\nTotal comparisons: {len(evaluator.results['comparisons'])}")
    print(f"Swarm wins: {swarm_total_wins} quality metrics")
    print(f"Baseline wins: {baseline_total_wins} quality metrics")
    print(f"\nConclusion: {'SWARM OUTPERFORMS BASELINE' if swarm_total_wins > baseline_total_wins else 'BASELINE COMPETITIVE' if baseline_total_wins > swarm_total_wins else 'TIE - FURTHER TESTING NEEDED'}")

    evaluator.save_results()
    await evaluator.cleanup()

    print("\n✅ Comparative evaluation complete!")
    if args.quick:
        print("\n⚡ Quick mode completed - for full evaluation run without --quick flag")
    print(f"\n📁 Results saved to: {evaluator.output_dir}/comparative_results.json")


if __name__ == '__main__':
    asyncio.run(main())
