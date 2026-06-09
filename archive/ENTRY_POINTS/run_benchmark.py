"""Run benchmarks to compare swarm vs baseline."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarking.datasets import get_dataset
from benchmarking.evaluator import SwarmEvaluator
from swarm.llm.simple_llm import SimpleLLM
from swarm.core.config import MODEL_NAME, DEVICE


async def run_baseline_single_llm(question: str, llm: SimpleLLM, evaluator: SwarmEvaluator) -> str:
    """Run baseline: single LLM call with direct question.

    Args:
        question: Benchmark question
        llm: Language model
        evaluator: Evaluator to track tokens

    Returns:
        Generated answer
    """
    prompt = f"""Answer this question directly and concisely:

Question: {question}

Answer:"""

    # Generate answer
    result = await llm.generate(prompt, max_tokens=100, temperature=0.6)

    # Track tokens (prompt + generation)
    # Rough estimate: prompt length + 100
    token_count = len(prompt.split()) + 100
    evaluator.record_tokens("baseline", token_count)

    if result:
        return result.strip()
    else:
        return "[No answer generated]"


async def run_swarm_system(question: str, llm: SimpleLLM, evaluator: SwarmEvaluator) -> str:
    """Run current swarm system on question.

    For now, this is a placeholder that simulates the swarm.
    TODO: Integrate with actual run_task.py swarm execution.

    Args:
        question: Benchmark question
        llm: Language model
        evaluator: Evaluator to track tokens

    Returns:
        Generated answer
    """
    # PLACEHOLDER: Simulate swarm with multiple LLM calls
    # In reality, this should call the full swarm pipeline from run_task.py

    # Simulate scouts (10 scouts × 70 tokens = 700)
    num_scouts = 10
    scout_tokens = 70
    scout_answers = []

    for i in range(num_scouts):
        prompt = f"Generate one specific idea to answer: {question}\nIdea:"
        answer = await llm.generate(prompt, max_tokens=scout_tokens, temperature=0.9, use_cache=False)
        if answer:
            scout_answers.append(answer.strip())

    evaluator.record_tokens("swarm", num_scouts * scout_tokens)

    # Simulate synthesis (200 tokens)
    synthesis_prompt = f"""Based on these ideas:
{chr(10).join(f"- {a[:100]}" for a in scout_answers[:5])}

Provide a direct answer to: {question}

Answer:"""

    synthesis = await llm.generate(synthesis_prompt, max_tokens=200, temperature=0.6)
    evaluator.record_tokens("swarm", 200)

    if synthesis:
        return synthesis.strip()
    else:
        return "[No answer generated]"


async def main():
    """Run benchmark suite."""
    print("=" * 70)
    print("SWARM INTELLIGENCE BENCHMARKING")
    print("=" * 70)

    # Load datasets
    print("\nLoading datasets...")
    datasets = {
        'truthfulqa': get_dataset('truthfulqa', limit=5),
        'mmlu': get_dataset('mmlu', limit=3),
        'gsm8k': get_dataset('gsm8k', limit=3),
    }

    # Initialize LLM
    print("\nInitializing language model...")
    llm = SimpleLLM(MODEL_NAME, DEVICE, enable_cache=False)  # Disable cache for fair comparison
    await llm.load()
    print(f"Model loaded: {MODEL_NAME} on {DEVICE}")

    # Run benchmarks
    evaluator = SwarmEvaluator(similarity_threshold=0.7)
    results = {}

    for dataset_name, questions in datasets.items():
        result = await evaluator.evaluate_swarm(
            questions=questions,
            swarm_fn=lambda q: run_swarm_system(q, llm, evaluator),
            baseline_fn=lambda q: run_baseline_single_llm(q, llm, evaluator),
            dataset_name=dataset_name
        )
        results[dataset_name] = result

    # Summary report
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for dataset_name, result in results.items():
        improvement_symbol = '✓' if result.improvement > 0 else '✗'
        print(f"\n{dataset_name.upper()}:")
        print(f"  Swarm:    {result.swarm_accuracy:.3f}")
        print(f"  Baseline: {result.baseline_accuracy:.3f}")
        print(f"  Delta:    {result.improvement:+.3f} {improvement_symbol}")
        print(f"  Tokens:   {result.token_overhead:.1f}x overhead")

    # Overall assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    avg_improvement = sum(r.improvement for r in results.values()) / len(results)
    avg_overhead = sum(r.token_overhead for r in results.values()) / len(results)

    print(f"\nAverage improvement: {avg_improvement:+.3f}")
    print(f"Average token overhead: {avg_overhead:.1f}x")

    if avg_improvement > 0.05 and avg_improvement > (avg_overhead - 1) * 0.02:
        print("\n✓ VERDICT: Swarm shows meaningful improvement that justifies overhead")
    elif avg_improvement > 0:
        print("\n⚠ VERDICT: Swarm shows slight improvement but overhead may not be justified")
    else:
        print("\n✗ VERDICT: Swarm does not beat baseline - optimization needed")

    print("\nNEXT STEPS:")
    if avg_improvement <= 0:
        print("  1. Investigate why swarm isn't beating baseline")
        print("  2. Check if scouts are generating diverse/useful signals")
        print("  3. Verify synthesis is actually using scout outputs")
    else:
        print("  1. Proceed with Phase 2: Simple scout redesign")
        print("  2. Re-benchmark after each major change")
        print("  3. Track improvement vs token cost tradeoff")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
