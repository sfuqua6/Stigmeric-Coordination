#!/usr/bin/env python3
"""Single-agent baseline for A/B comparison against swarm system.

This script runs the SAME LLM (Phi-2) without any swarm mechanics,
providing an apples-to-apples comparison baseline.

The swarm should outperform this baseline through:
- Multi-perspective analysis
- Evidence-based validation
- Critical review and refinement
- Emergent consensus building
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, List

from swarm.llm.simple_llm import SimpleLLM
from swarm.core.config import MAX_ITERATIONS, MODEL_NAME, DEVICE, ENABLE_LLM_CACHE, LLM_CACHE_SIZE


class SingleAgentBaseline:
    """Single-agent baseline using same LLM as swarm."""

    def __init__(self):
        """Initialize baseline agent."""
        self.llm = None

    async def initialize(self):
        """Initialize LLM (same as swarm uses)."""
        print(f"[BASELINE] Initializing {MODEL_NAME} on {DEVICE}...")
        self.llm = SimpleLLM(MODEL_NAME, DEVICE, enable_cache=ENABLE_LLM_CACHE, cache_size=LLM_CACHE_SIZE)
        await self.llm.load()  # Changed from initialize() to load() to match swarm
        print("[BASELINE] Ready")

    async def run(self, prompt: str, max_iterations: int = None) -> Dict:
        """Run single-agent on a prompt.

        Args:
            prompt: Task prompt
            max_iterations: Maximum refinement iterations (for fair comparison)

        Returns:
            Results dictionary with output and metadata
        """
        if max_iterations is None:
            max_iterations = MAX_ITERATIONS

        print(f"\n{'='*70}")
        print(f"SINGLE-AGENT BASELINE")
        print(f"Prompt: {prompt}")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*70}\n")

        start_time = time.time()
        llm_generations = 0

        # Initial generation
        initial_prompt = f"""Task: {prompt}

Provide a comprehensive, well-reasoned response. Include:
- Clear claims and arguments
- Supporting evidence
- Multiple perspectives
- Critical analysis

Response:"""

        print("[BASELINE] Generating initial response...")
        initial_response = await self.llm.generate(initial_prompt)
        llm_generations += 1

        # Check if initial generation failed
        if initial_response is None:
            print("[BASELINE] ERROR: Initial generation failed")
            return {
                'output': "[Baseline generation failed]",
                'prompt': prompt,
                'runtime': time.time() - start_time,
                'llm_generations': llm_generations,
                'iterations': 0,
                'response_length': 0,
                'method': 'single_agent',
                'error': 'LLM generation failed'
            }

        # Optional: Self-refinement iterations (to match swarm iterations)
        current_response = initial_response

        for i in range(1, max_iterations):
            refinement_prompt = f"""Previous response:
{current_response}

Task: {prompt}

Review the above response and improve it by:
1. Adding more evidence
2. Addressing potential critiques
3. Clarifying arguments
4. Correcting any errors

Improved response:"""

            print(f"[BASELINE] Refinement iteration {i}/{max_iterations-1}...")
            refined_response = await self.llm.generate(refinement_prompt)
            llm_generations += 1

            # Check if refinement failed
            if refined_response is None:
                print(f"[BASELINE] Refinement {i} failed, using previous response")
                break

            # Check if response is converging (no significant changes)
            if len(refined_response) < len(current_response) * 1.1:
                print(f"[BASELINE] Converged at iteration {i}")
                current_response = refined_response
                break

            current_response = refined_response

        runtime = time.time() - start_time

        print(f"\n[BASELINE] Complete in {runtime:.2f}s")
        print(f"[BASELINE] LLM generations: {llm_generations}")
        print(f"[BASELINE] Response length: {len(current_response)} chars")

        return {
            'output': current_response,
            'prompt': prompt,
            'runtime': runtime,
            'llm_generations': llm_generations,
            'iterations': llm_generations,
            'response_length': len(current_response),
            'method': 'single_agent'
        }

    async def cleanup(self):
        """Cleanup resources."""
        if self.llm:
            await self.llm.cleanup()


async def main():
    """Run baseline on example prompts."""
    baseline = SingleAgentBaseline()
    await baseline.initialize()

    # Example prompts for testing
    test_prompts = [
        "Analyze the potential benefits and risks of artificial intelligence regulation",
        "Explain quantum computing and its potential impact on cryptography",
        "Discuss the evidence for and against human-caused climate change"
    ]

    results = []

    for prompt in test_prompts:
        result = await baseline.run(prompt, max_iterations=5)
        results.append(result)

        print(f"\n{'='*70}")
        print("OUTPUT:")
        print(result['output'])
        print(f"{'='*70}\n")

    # Save results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "baseline_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Baseline results saved to: {output_file}")

    await baseline.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
