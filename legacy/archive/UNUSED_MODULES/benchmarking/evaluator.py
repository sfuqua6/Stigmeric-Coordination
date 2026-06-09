"""Evaluation framework for benchmarking swarm vs baseline."""

import time
import asyncio
from typing import List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .datasets import BenchmarkQuestion


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    dataset: str
    num_questions: int

    # Accuracy metrics
    swarm_accuracy: float
    baseline_accuracy: float
    improvement: float  # swarm - baseline

    # Token metrics
    swarm_tokens: int
    baseline_tokens: int
    token_overhead: float  # swarm / baseline

    # Efficiency
    efficiency_ratio: float  # accuracy per token

    # Statistical
    p_value: float = None
    significant: bool = False

    # Timing
    swarm_time: float = 0.0
    baseline_time: float = 0.0

    # Individual results
    swarm_results: List[bool] = field(default_factory=list)
    baseline_results: List[bool] = field(default_factory=list)

    def __str__(self) -> str:
        """Generate readable report."""
        improvement_symbol = '✓' if self.improvement > 0 else '✗'
        sig_symbol = '✓' if self.significant else '✗'

        return f"""
BENCHMARK RESULTS: {self.dataset}
{'=' * 70}

Accuracy:
  Swarm:    {self.swarm_accuracy:.3f} ({sum(self.swarm_results)}/{self.num_questions})
  Baseline: {self.baseline_accuracy:.3f} ({sum(self.baseline_results)}/{self.num_questions})
  Delta:    {self.improvement:+.3f} {improvement_symbol}

Statistical Significance:
  p-value:  {self.p_value:.4f if self.p_value else 'N/A'}
  Significant (p < 0.05): {sig_symbol}

Token Usage:
  Swarm:    {self.swarm_tokens:,} tokens
  Baseline: {self.baseline_tokens:,} tokens
  Overhead: {self.token_overhead:.1f}x

Timing:
  Swarm:    {self.swarm_time:.1f}s
  Baseline: {self.baseline_time:.1f}s

Efficiency (accuracy per 1000 tokens):
  Swarm:    {(self.swarm_accuracy * 1000 / self.swarm_tokens):.3f}
  Baseline: {(self.baseline_accuracy * 1000 / self.baseline_tokens):.3f}
"""


class SwarmEvaluator:
    """Evaluate swarm system against benchmarks."""

    def __init__(self, similarity_threshold: float = 0.7):
        """Initialize evaluator.

        Args:
            similarity_threshold: Minimum similarity for answer matching (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.token_counts = {"swarm": 0, "baseline": 0}

    def check_answer(self, generated: str, ground_truth: str) -> bool:
        """Check if generated answer matches ground truth.

        Uses fuzzy string matching to handle variation in phrasing.

        Args:
            generated: Generated answer
            ground_truth: Ground truth answer

        Returns:
            True if answers match (similarity >= threshold)
        """
        if not generated or not ground_truth:
            return False

        # Normalize
        gen_lower = generated.lower().strip()
        truth_lower = ground_truth.lower().strip()

        # Exact match
        if gen_lower == truth_lower:
            return True

        # Substring match (generated contains ground truth or vice versa)
        if truth_lower in gen_lower or gen_lower in truth_lower:
            return True

        # Fuzzy match using sequence matcher
        similarity = SequenceMatcher(None, gen_lower, truth_lower).ratio()
        return similarity >= self.similarity_threshold

    async def evaluate_swarm(
        self,
        questions: List[BenchmarkQuestion],
        swarm_fn: Callable[[str], Awaitable[str]],
        baseline_fn: Callable[[str], Awaitable[str]],
        dataset_name: str = "benchmark"
    ) -> BenchmarkResult:
        """Run full benchmark evaluation.

        Args:
            questions: List of benchmark questions
            swarm_fn: Async function that runs swarm on a question
            baseline_fn: Async function that runs baseline on a question
            dataset_name: Name of dataset for reporting

        Returns:
            Benchmark results with statistical analysis
        """
        print(f"\n{'=' * 70}")
        print(f"BENCHMARKING: {dataset_name}")
        print(f"{'=' * 70}\n")
        print(f"Questions: {len(questions)}")
        print(f"Similarity threshold: {self.similarity_threshold}\n")

        # Run swarm
        print("Running swarm system...")
        swarm_start = time.time()
        swarm_results = []
        swarm_tokens_total = 0

        for i, q in enumerate(questions):
            print(f"  [{i+1}/{len(questions)}] {q.id}: {q.question[:60]}...")

            # Reset token counter
            self.token_counts["swarm"] = 0

            # Run swarm
            try:
                answer = await swarm_fn(q.question)
                correct = self.check_answer(answer, q.answer)
                swarm_results.append(correct)
                swarm_tokens_total += self.token_counts["swarm"]

                symbol = '✓' if correct else '✗'
                print(f"    {symbol} Swarm: {answer[:80]}...")
                print(f"    Ground truth: {q.answer[:80]}...")
            except Exception as e:
                print(f"    ✗ Error: {e}")
                swarm_results.append(False)

        swarm_time = time.time() - swarm_start

        # Run baseline
        print("\nRunning baseline system...")
        baseline_start = time.time()
        baseline_results = []
        baseline_tokens_total = 0

        for i, q in enumerate(questions):
            print(f"  [{i+1}/{len(questions)}] {q.id}: {q.question[:60]}...")

            # Reset token counter
            self.token_counts["baseline"] = 0

            # Run baseline
            try:
                answer = await baseline_fn(q.question)
                correct = self.check_answer(answer, q.answer)
                baseline_results.append(correct)
                baseline_tokens_total += self.token_counts["baseline"]

                symbol = '✓' if correct else '✗'
                print(f"    {symbol} Baseline: {answer[:80]}...")
            except Exception as e:
                print(f"    ✗ Error: {e}")
                baseline_results.append(False)

        baseline_time = time.time() - baseline_start

        # Calculate metrics
        swarm_acc = sum(swarm_results) / len(swarm_results) if swarm_results else 0.0
        baseline_acc = sum(baseline_results) / len(baseline_results) if baseline_results else 0.0
        improvement = swarm_acc - baseline_acc

        token_overhead = swarm_tokens_total / baseline_tokens_total if baseline_tokens_total > 0 else 0.0

        # Statistical significance (t-test)
        p_value = None
        significant = False

        if len(swarm_results) >= 5:  # Need enough samples
            try:
                from scipy.stats import ttest_ind
                # Convert bool to int for t-test
                swarm_int = [int(x) for x in swarm_results]
                baseline_int = [int(x) for x in baseline_results]
                t_stat, p_value = ttest_ind(swarm_int, baseline_int)
                significant = p_value < 0.05
            except ImportError:
                print("[WARNING] scipy not available, skipping statistical test")
                print("Install with: pip install scipy")

        result = BenchmarkResult(
            dataset=dataset_name,
            num_questions=len(questions),
            swarm_accuracy=swarm_acc,
            baseline_accuracy=baseline_acc,
            improvement=improvement,
            swarm_tokens=swarm_tokens_total,
            baseline_tokens=baseline_tokens_total,
            token_overhead=token_overhead,
            efficiency_ratio=swarm_acc / (swarm_tokens_total / 1000) if swarm_tokens_total > 0 else 0.0,
            p_value=p_value,
            significant=significant,
            swarm_time=swarm_time,
            baseline_time=baseline_time,
            swarm_results=swarm_results,
            baseline_results=baseline_results,
        )

        print(result)
        return result

    def record_tokens(self, system: str, count: int):
        """Record token usage for tracking.

        Args:
            system: "swarm" or "baseline"
            count: Number of tokens used
        """
        if system in self.token_counts:
            self.token_counts[system] += count
