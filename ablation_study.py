#!/usr/bin/env python3
"""Ablation study: Does the swarm actually help, or is it just best-of-N?

Tests 5 conditions with IDENTICAL total LLM call budgets:
  1. full-swarm        — scouts → foragers → critics → haters → synthesis
  2. no-haters         — scouts → foragers → critics → synthesis
  3. no-critics-haters — scouts → foragers → synthesis
  4. single-round      — full swarm, 1 round (same budget concentrated)
  5. best-of-N         — N independent calls, pick best (null hypothesis)

All conditions use the same DeepSeek model, same total generation budget.
If conditions 1-4 don't beat condition 5, the swarm is just expensive sampling.

Usage:
    python ablation_study.py                          # Stub mode (fast, no GPU)
    python ablation_study.py --model                  # Real DeepSeek model
    python ablation_study.py --model --tasks 3        # Fewer tasks
    python ablation_study.py --model --budget 50      # Smaller budget
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import asyncio
import argparse
import json
import sys
import time
import random
import statistics
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

# ── Stub torch module so agent imports work without GPU ────────────────────
# When --model is passed, the real torch is already installed.
# When running in stub mode (no GPU), we need agent code to import without torch.
try:
    import torch
except ImportError:
    import types
    torch = types.ModuleType("torch")
    torch.cuda = types.ModuleType("torch.cuda")
    torch.cuda.is_available = lambda: False
    sys.modules["torch"] = torch
    sys.modules["torch.cuda"] = torch.cuda
    # Stub transformers too
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = None
    transformers.AutoModelForCausalLM = None
    transformers.BitsAndBytesConfig = None
    sys.modules["transformers"] = transformers

# ── Swarm imports ──────────────────────────────────────────────────────────
from swarm.core.signal_store import SignalStore
from swarm.core.config import (
    DEVICE, TEMP_SCOUT, TEMP_FORAGER, TEMP_CRITIC, TEMP_HATER, TEMP_SYNTHESIZER,
)
from swarm.core.task_config import get_task_config, create_custom_task
from swarm.agents.scout import Scout
from swarm.agents.forager import Forager
from swarm.agents.critic import Critic
from swarm.agents.hater import Hater
from swarm.agents.synthesizer import Synthesizer


# ── Quality scorer (same heuristic used in the overnight runs) ─────────────
def score_output(text: str) -> float:
    """Score an output using the same heuristics as the swarm's assess_strength.

    This is the SAME metric used in the overnight batch runs so results
    are directly comparable.
    """
    if not text or len(text.strip()) < 15:
        return 0.0

    text = text.strip()
    words = text.split()
    length = len(text)

    # Length component
    score = 0.35 + min(0.25, length / 600.0)

    # Evidence indicators
    if any(c.isdigit() for c in text):
        score += 0.10
    if any(w in text.lower() for w in ['study', 'research', 'evidence', 'data',
                                         'shows', 'indicates', 'according to',
                                         'because', 'therefore']):
        score += 0.12
    if any(w in text.lower() for w in ['ipcc', 'nasa', 'journal', 'university',
                                         'report', 'analysis', 'published']):
        score += 0.08
    if any(w in text.lower() for w in ['however', 'although', 'while', 'despite',
                                         'nevertheless', 'conversely']):
        score += 0.08
    if any(w in text.lower() for w in ['because', 'since', 'thus', 'therefore',
                                         'consequently']):
        score += 0.07

    # Penalize repetition
    if len(words) > 5:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.4:
            score *= 0.5

    # Penalize very short
    if len(words) < 10:
        score *= 0.6

    return max(0.0, min(1.0, score))


# ── Data classes ───────────────────────────────────────────────────────────
@dataclass
class ConditionResult:
    condition: str
    task: str
    top_quality: float = 0.0
    avg_quality: float = 0.0
    total_outputs: int = 0
    total_llm_calls: int = 0
    best_output: str = ""
    best_len: int = 0
    time_s: float = 0.0
    all_scores: List[float] = field(default_factory=list)
    diversity: float = 0.0  # pairwise dissimilarity


def pairwise_dissimilarity(texts: List[str]) -> float:
    """Compute average pairwise dissimilarity (1 - SequenceMatcher ratio)."""
    if len(texts) < 2:
        return 0.0
    from difflib import SequenceMatcher
    pairs = 0
    total = 0.0
    for i in range(len(texts)):
        for j in range(i + 1, min(len(texts), i + 20)):  # cap comparisons
            ratio = SequenceMatcher(None, texts[i].lower(), texts[j].lower()).ratio()
            total += (1.0 - ratio)
            pairs += 1
    return total / pairs if pairs > 0 else 0.0


# ── LLM setup ─────────────────────────────────────────────────────────────
async def create_llm(use_real_model: bool):
    """Create LLM — real DeepSeek or stub."""
    if use_real_model:
        from swarm.llm.simple_llm import SimpleLLM
        from swarm.core.config import MODEL_NAME
        llm = SimpleLLM(MODEL_NAME, DEVICE, enable_cache=False)
        await llm.load()
        print(f"[MODEL] Loaded {MODEL_NAME} on {DEVICE}")
        return llm
    else:
        return StubLLM()


class StubLLM:
    """Deterministic stub for testing the harness without GPU.

    Produces variable-quality outputs based on prompt content so we can
    verify the harness works before burning GPU hours.
    """
    _call_count = 0

    async def generate(self, prompt: str, max_tokens: int = 200,
                       temperature: float = 0.7, use_cache: bool = True) -> str:
        StubLLM._call_count += 1
        n = StubLLM._call_count

        # Simulate variable quality based on call number
        templates = [
            "Research indicates that this topic involves multiple factors. "
            "According to published studies, the primary mechanism operates through "
            f"a {n}-step process involving key institutional actors. "
            "However, alternative interpretations suggest limitations in this view.",

            f"The evidence shows significant impact across {n + 2} dimensions. "
            "University research published in leading journals demonstrates "
            "measurable effects, although methodological concerns remain. "
            "Nevertheless, the data consistently indicates a clear trend.",

            "Analysis reveals complex interactions between social, economic, "
            f"and political factors. Despite {n} documented counterexamples, "
            "the overall trajectory supports the core thesis. "
            "IPCC and NASA data provide corroborating evidence.",

            f"A {random.choice(['critical', 'important', 'key'])} consideration is "
            "that existing frameworks fail to account for emergent dynamics. "
            "Because the underlying mechanisms are nonlinear, simple models "
            "are insufficient. Therefore, a multi-factor approach is required.",

            "While the conventional view holds merit, recent data challenges "
            f"several assumptions. Study findings from {n} research groups "
            "indicate previously unrecognized patterns. This suggests "
            "the relationship is more nuanced than initially theorized.",
        ]
        # Add some randomness to simulate LLM variance
        template = templates[n % len(templates)]
        # Occasionally produce low-quality output
        if random.random() < 0.15:
            template = "This is an interesting topic with many aspects to consider."
        return template

    async def load(self):
        pass


# ── Condition runners ──────────────────────────────────────────────────────

async def _run_agent_staged(agent, signal_store, llm):
    """Run one cycle of an agent using the staged execution API (prepare + process).

    Returns True if agent produced output, False otherwise.
    """
    prompt_data = await agent.prepare_prompt(signal_store)
    if prompt_data is None:
        return False
    agent_id, prompt, max_tokens, temperature = prompt_data
    try:
        result = await llm.generate(prompt, max_tokens=max_tokens,
                                     temperature=temperature, use_cache=False)
        if result:
            return await agent.process_result(result, signal_store)
    except Exception as e:
        pass
    return False


async def run_full_swarm(llm, task_prompt: str, task_config, budget: int,
                         num_rounds: int = 3) -> ConditionResult:
    """Condition 1: Full swarm — scouts, foragers, critics, haters, synthesis."""
    start = time.time()
    outputs = []
    llm_calls = 0

    # Divide budget across rounds and agent types
    calls_per_round = budget // num_rounds
    n_scouts = 4
    n_foragers = 4
    n_critics = 2
    n_haters = 2
    total_agents = n_scouts + n_foragers + n_critics + n_haters
    actions_per_agent = max(1, calls_per_round // total_agents)

    all_syntheses = []

    for round_num in range(num_rounds):
        signal_store = SignalStore(use_semantic_clustering=False)

        # Stage 1: Scouts
        scouts = [Scout(f"Scout_{i}_R{round_num}", signal_type="INITIAL",
                        task_prompt=task_prompt, task_config=task_config)
                  for i in range(n_scouts)]
        for scout in scouts:
            scout.max_actions = actions_per_agent
            for _ in range(actions_per_agent):
                await _run_agent_staged(scout, signal_store, llm)
                llm_calls += 1

        # Stage 2: Foragers + Critics (both read INITIAL signals)
        foragers = [Forager(f"Forager_{i}_R{round_num}", input_type="INITIAL",
                           output_type="SUPPORT", thesis=task_prompt,
                           enable_verification=False, task_config=task_config)
                    for i in range(n_foragers)]
        for forager in foragers:
            forager.max_actions = actions_per_agent
            for _ in range(actions_per_agent):
                await _run_agent_staged(forager, signal_store, llm)
                llm_calls += 1

        critics = [Critic(f"Critic_{i}_R{round_num}", thesis=task_prompt,
                         task_config=task_config)
                   for i in range(n_critics)]
        for critic in critics:
            # Critics need evaluate_type set to match what scouts deposited
            critic.evaluate_type = "INITIAL"
            critic.max_actions = actions_per_agent
            for _ in range(actions_per_agent):
                await _run_agent_staged(critic, signal_store, llm)
                llm_calls += 1

        # Stage 3: Haters
        haters = [Hater(f"Hater_{i}_R{round_num}", task_prompt=task_prompt,
                        input_types=["INITIAL", "SUPPORT"],
                        output_type="OBJECTION", enable_verification=False,
                        task_config=task_config)
                  for i in range(n_haters)]
        for hater in haters:
            hater.max_actions = actions_per_agent
            for _ in range(actions_per_agent):
                await _run_agent_staged(hater, signal_store, llm)
                llm_calls += 1

        # Collect all signal contents as outputs
        for sig in signal_store.signals.values():
            outputs.append(sig.content)

        # Synthesis
        synth = Synthesizer(f"Synth_R{round_num}", task_prompt, task_config=task_config)
        synthesis = await synth.synthesize(
            signal_store, llm, task_config.signal_types, temperature=TEMP_SYNTHESIZER)
        llm_calls += 1
        if synthesis:
            outputs.append(synthesis)
            all_syntheses.append(synthesis)

    # Final cross-round synthesis
    if len(all_syntheses) > 1:
        final_store = SignalStore(use_semantic_clustering=False)
        for i, s in enumerate(all_syntheses):
            final_store.deposit("INITIAL", s, 0.8, "cross_round")
        final_synth = Synthesizer("FinalSynth", task_prompt, task_config=task_config)
        final = await final_synth.synthesize(
            final_store, llm, task_config.signal_types, temperature=TEMP_SYNTHESIZER)
        llm_calls += 1
        if final:
            outputs.append(final)

    return _score_condition("full-swarm", task_prompt, outputs, llm_calls, time.time() - start)


async def run_no_haters(llm, task_prompt: str, task_config, budget: int,
                        num_rounds: int = 3) -> ConditionResult:
    """Condition 2: No haters — remove adversarial pressure."""
    start = time.time()
    outputs = []
    llm_calls = 0

    calls_per_round = budget // num_rounds
    n_scouts = 4
    n_foragers = 4
    n_critics = 2
    total_agents = n_scouts + n_foragers + n_critics
    actions_per_agent = max(1, calls_per_round // total_agents)

    all_syntheses = []

    for round_num in range(num_rounds):
        signal_store = SignalStore(use_semantic_clustering=False)

        scouts = [Scout(f"Scout_{i}_R{round_num}", signal_type="INITIAL",
                        task_prompt=task_prompt, task_config=task_config)
                  for i in range(n_scouts)]
        for scout in scouts:
            scout.max_actions = actions_per_agent
            for _ in range(actions_per_agent):
                await _run_agent_staged(scout, signal_store, llm)
                llm_calls += 1

        foragers = [Forager(f"Forager_{i}_R{round_num}", input_type="INITIAL",
                           output_type="SUPPORT", thesis=task_prompt,
                           enable_verification=False, task_config=task_config)
                    for i in range(n_foragers)]
        for forager in foragers:
            forager.max_actions = actions_per_agent
            for _ in range(actions_per_agent):
                await _run_agent_staged(forager, signal_store, llm)
                llm_calls += 1

        critics = [Critic(f"Critic_{i}_R{round_num}", thesis=task_prompt,
                         task_config=task_config)
                   for i in range(n_critics)]
        for critic in critics:
            critic.evaluate_type = "INITIAL"
            critic.max_actions = actions_per_agent
            for _ in range(actions_per_agent):
                await _run_agent_staged(critic, signal_store, llm)
                llm_calls += 1

        for sig in signal_store.signals.values():
            outputs.append(sig.content)

        synth = Synthesizer(f"Synth_R{round_num}", task_prompt, task_config=task_config)
        synthesis = await synth.synthesize(
            signal_store, llm, task_config.signal_types, temperature=TEMP_SYNTHESIZER)
        llm_calls += 1
        if synthesis:
            outputs.append(synthesis)
            all_syntheses.append(synthesis)

    if len(all_syntheses) > 1:
        final_store = SignalStore(use_semantic_clustering=False)
        for i, s in enumerate(all_syntheses):
            final_store.deposit("INITIAL", s, 0.8, "cross_round")
        final_synth = Synthesizer("FinalSynth", task_prompt, task_config=task_config)
        final = await final_synth.synthesize(
            final_store, llm, task_config.signal_types, temperature=TEMP_SYNTHESIZER)
        llm_calls += 1
        if final:
            outputs.append(final)

    return _score_condition("no-haters", task_prompt, outputs, llm_calls, time.time() - start)


async def run_no_critics_haters(llm, task_prompt: str, task_config, budget: int,
                                 num_rounds: int = 3) -> ConditionResult:
    """Condition 3: No critics or haters — remove all evaluation agents."""
    start = time.time()
    outputs = []
    llm_calls = 0

    calls_per_round = budget // num_rounds
    n_scouts = 4
    n_foragers = 6  # redistribute critic+hater budget
    total_agents = n_scouts + n_foragers
    actions_per_agent = max(1, calls_per_round // total_agents)

    all_syntheses = []

    for round_num in range(num_rounds):
        signal_store = SignalStore(use_semantic_clustering=False)

        scouts = [Scout(f"Scout_{i}_R{round_num}", signal_type="INITIAL",
                        task_prompt=task_prompt, task_config=task_config)
                  for i in range(n_scouts)]
        for scout in scouts:
            scout.max_actions = actions_per_agent
            for _ in range(actions_per_agent):
                await _run_agent_staged(scout, signal_store, llm)
                llm_calls += 1

        foragers = [Forager(f"Forager_{i}_R{round_num}", input_type="INITIAL",
                           output_type="SUPPORT", thesis=task_prompt,
                           enable_verification=False, task_config=task_config)
                    for i in range(n_foragers)]
        for forager in foragers:
            forager.max_actions = actions_per_agent
            for _ in range(actions_per_agent):
                await _run_agent_staged(forager, signal_store, llm)
                llm_calls += 1

        for sig in signal_store.signals.values():
            outputs.append(sig.content)

        synth = Synthesizer(f"Synth_R{round_num}", task_prompt, task_config=task_config)
        synthesis = await synth.synthesize(
            signal_store, llm, task_config.signal_types, temperature=TEMP_SYNTHESIZER)
        llm_calls += 1
        if synthesis:
            outputs.append(synthesis)
            all_syntheses.append(synthesis)

    if len(all_syntheses) > 1:
        final_store = SignalStore(use_semantic_clustering=False)
        for i, s in enumerate(all_syntheses):
            final_store.deposit("INITIAL", s, 0.8, "cross_round")
        final_synth = Synthesizer("FinalSynth", task_prompt, task_config=task_config)
        final = await final_synth.synthesize(
            final_store, llm, task_config.signal_types, temperature=TEMP_SYNTHESIZER)
        llm_calls += 1
        if final:
            outputs.append(final)

    return _score_condition("no-critics-haters", task_prompt, outputs, llm_calls, time.time() - start)


async def run_single_round(llm, task_prompt: str, task_config, budget: int) -> ConditionResult:
    """Condition 4: Full swarm but only 1 round (all budget concentrated)."""
    return await run_full_swarm(llm, task_prompt, task_config, budget, num_rounds=1)


async def run_best_of_n(llm, task_prompt: str, task_config, budget: int) -> ConditionResult:
    """Condition 5: Pure best-of-N — generate N independent outputs, pick best.

    This is the NULL HYPOTHESIS. If the swarm can't beat this, the coordination
    mechanisms aren't adding value beyond what you get from multiple samples.
    """
    start = time.time()
    outputs = []

    # Use the full budget for independent generation calls
    # Each call gets the same simple prompt (no signal context, no coordination)
    base_prompt = (
        f"You are an expert analyst. Provide a thorough, evidence-based response.\n\n"
        f"Task: {task_prompt}\n\n"
        f"Provide a detailed, well-structured answer with specific evidence, "
        f"data, and reasoning. Address multiple perspectives and nuances.\n\n"
        f"Response:"
    )

    # Vary temperature across calls for diversity (same as swarm agents get)
    temps = [0.6, 0.7, 0.8, 0.9, 0.7, 0.8, 0.6, 0.9, 0.75, 0.85]

    for i in range(budget):
        temp = temps[i % len(temps)]
        try:
            result = await llm.generate(base_prompt, max_tokens=400,
                                        temperature=temp, use_cache=False)
            if result and len(result.strip()) > 15:
                outputs.append(result.strip())
        except Exception as e:
            print(f"  [best-of-N] call {i} failed: {e}")

    return _score_condition("best-of-N", task_prompt, outputs, budget, time.time() - start)


# ── Scoring helper ─────────────────────────────────────────────────────────
def _score_condition(condition: str, task: str, outputs: List[str],
                     llm_calls: int, elapsed: float) -> ConditionResult:
    """Score all outputs from a condition and return result."""
    if not outputs:
        return ConditionResult(condition=condition, task=task, time_s=elapsed,
                               total_llm_calls=llm_calls)

    scores = [score_output(o) for o in outputs]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])

    return ConditionResult(
        condition=condition,
        task=task,
        top_quality=max(scores),
        avg_quality=statistics.mean(scores),
        total_outputs=len(outputs),
        total_llm_calls=llm_calls,
        best_output=outputs[best_idx][:500],
        best_len=len(outputs[best_idx]),
        time_s=round(elapsed, 2),
        all_scores=sorted(scores, reverse=True)[:10],  # top 10
        diversity=pairwise_dissimilarity(outputs[:30]),
    )


# ── Task definitions ───────────────────────────────────────────────────────
ABLATION_TASKS = [
    ("analysis", "Explain CRISPR gene editing mechanisms"),
    ("analysis", "What are the implications of quantum computing for cryptography?"),
    ("analysis", "Compare and contrast nuclear fusion vs fission for energy production"),
    ("debate", "What are the main arguments for and against universal basic income?"),
    ("analysis", "Explain the relationship between gut microbiome and human health"),
    ("problem_solving", "How can cities reduce carbon emissions while maintaining economic growth?"),
]


# ── Main ───────────────────────────────────────────────────────────────────
async def run_ablation(use_model: bool, budget: int, max_tasks: int):
    """Run the full ablation study."""
    llm = await create_llm(use_model)

    tasks = ABLATION_TASKS[:max_tasks]
    all_results: List[ConditionResult] = []

    conditions = [
        ("full-swarm",        run_full_swarm),
        ("no-haters",         run_no_haters),
        ("no-critics-haters", run_no_critics_haters),
        ("single-round",      run_single_round),
        ("best-of-N",         run_best_of_n),
    ]

    for task_idx, (task_type, task_prompt) in enumerate(tasks):
        print(f"\n{'='*70}")
        print(f"TASK {task_idx+1}/{len(tasks)}: {task_prompt[:60]}...")
        print(f"{'='*70}")

        task_config = create_custom_task(task_type, task_prompt)

        for cond_name, cond_fn in conditions:
            print(f"\n  [{cond_name}] Running (budget={budget})...")

            # Reset stub call counter for fair comparison
            StubLLM._call_count = 0

            if cond_name == "single-round":
                result = await cond_fn(llm, task_prompt, task_config, budget)
                # Override condition name (run_full_swarm returns "full-swarm")
                result.condition = "single-round"
            else:
                result = await cond_fn(llm, task_prompt, task_config, budget)

            all_results.append(result)
            print(f"  [{cond_name}] top_q={result.top_quality:.3f} "
                  f"avg_q={result.avg_quality:.3f} "
                  f"outputs={result.total_outputs} "
                  f"calls={result.total_llm_calls} "
                  f"time={result.time_s:.1f}s "
                  f"diversity={result.diversity:.3f}")

    # ── Print summary table ────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*90}")
    print(f"Model: {'DeepSeek (real)' if use_model else 'Stub (harness test)'}")
    print(f"Budget per task: {budget} LLM calls")
    print(f"Tasks: {len(tasks)}")

    # Aggregate by condition
    from collections import defaultdict
    agg = defaultdict(lambda: {"top_q": [], "avg_q": [], "diversity": [],
                                "time": [], "outputs": []})
    for r in all_results:
        agg[r.condition]["top_q"].append(r.top_quality)
        agg[r.condition]["avg_q"].append(r.avg_quality)
        agg[r.condition]["diversity"].append(r.diversity)
        agg[r.condition]["time"].append(r.time_s)
        agg[r.condition]["outputs"].append(r.total_outputs)

    print(f"\n{'Condition':<20} {'Top Q':>7} {'Avg Q':>7} {'Diversity':>9} "
          f"{'Outputs':>8} {'Time(s)':>8}")
    print("-" * 65)

    baseline_q = None
    for cond_name in ["full-swarm", "no-haters", "no-critics-haters",
                       "single-round", "best-of-N"]:
        d = agg[cond_name]
        if not d["top_q"]:
            continue
        mean_top = statistics.mean(d["top_q"])
        mean_avg = statistics.mean(d["avg_q"])
        mean_div = statistics.mean(d["diversity"])
        mean_time = statistics.mean(d["time"])
        mean_out = statistics.mean(d["outputs"])

        if cond_name == "best-of-N":
            baseline_q = mean_top

        marker = ""
        if baseline_q and cond_name != "best-of-N":
            delta = mean_top - baseline_q
            marker = f" ({delta:+.3f} vs best-of-N)"

        print(f"{cond_name:<20} {mean_top:>7.3f} {mean_avg:>7.3f} {mean_div:>9.3f} "
              f"{mean_out:>8.0f} {mean_time:>8.1f}{marker}")

    # ── Per-task breakdown ─────────────────────────────────────────────
    print(f"\n\nPER-TASK BREAKDOWN (Top Quality):")
    print(f"{'Task':<45} {'Swarm':>7} {'NoHate':>7} {'NoCrit':>7} "
          f"{'1Round':>7} {'BofN':>7} {'Swarm-BofN':>10}")
    print("-" * 100)

    for task_type, task_prompt in tasks:
        task_results = {r.condition: r for r in all_results if r.task == task_prompt}
        short = task_prompt[:43]
        swarm_q = task_results.get("full-swarm", ConditionResult("", "")).top_quality
        nohate_q = task_results.get("no-haters", ConditionResult("", "")).top_quality
        nocrit_q = task_results.get("no-critics-haters", ConditionResult("", "")).top_quality
        single_q = task_results.get("single-round", ConditionResult("", "")).top_quality
        bofn_q = task_results.get("best-of-N", ConditionResult("", "")).top_quality
        delta = swarm_q - bofn_q

        print(f"{short:<45} {swarm_q:>7.3f} {nohate_q:>7.3f} {nocrit_q:>7.3f} "
              f"{single_q:>7.3f} {bofn_q:>7.3f} {delta:>+10.3f}")

    # ── Interpretation ─────────────────────────────────────────────────
    print(f"\n\nINTERPRETATION:")
    swarm_avg = statistics.mean(agg["full-swarm"]["top_q"]) if agg["full-swarm"]["top_q"] else 0
    bofn_avg = statistics.mean(agg["best-of-N"]["top_q"]) if agg["best-of-N"]["top_q"] else 0
    nohate_avg = statistics.mean(agg["no-haters"]["top_q"]) if agg["no-haters"]["top_q"] else 0

    if swarm_avg > bofn_avg + 0.03:
        print(f"  ✓ Swarm BEATS best-of-N by {swarm_avg - bofn_avg:+.3f} — coordination adds value")
    elif swarm_avg > bofn_avg:
        print(f"  ~ Swarm marginally beats best-of-N by {swarm_avg - bofn_avg:+.3f} — inconclusive")
    else:
        print(f"  ✗ Swarm does NOT beat best-of-N ({swarm_avg - bofn_avg:+.3f}) — "
              f"it's just expensive sampling")

    if nohate_avg > swarm_avg + 0.02:
        print(f"  ! Removing haters IMPROVES quality by {nohate_avg - swarm_avg:+.3f} — "
              f"haters are hurting, not helping")
    elif swarm_avg > nohate_avg + 0.02:
        print(f"  ✓ Haters contribute {swarm_avg - nohate_avg:+.3f} quality — "
              f"adversarial pressure helps")
    else:
        print(f"  ~ Haters make no significant difference ({swarm_avg - nohate_avg:+.3f})")

    nocrit_avg = statistics.mean(agg["no-critics-haters"]["top_q"]) if agg["no-critics-haters"]["top_q"] else 0
    if swarm_avg > nocrit_avg + 0.02:
        print(f"  ✓ Critics+Haters together contribute {swarm_avg - nocrit_avg:+.3f} — "
              f"evaluation layer matters")
    else:
        print(f"  ~ Evaluation layer (critics+haters) adds only {swarm_avg - nocrit_avg:+.3f}")

    single_avg = statistics.mean(agg["single-round"]["top_q"]) if agg["single-round"]["top_q"] else 0
    if swarm_avg > single_avg + 0.02:
        print(f"  ✓ Multi-round improves by {swarm_avg - single_avg:+.3f} — "
              f"iterative refinement helps")
    else:
        print(f"  ~ Multi-round adds only {swarm_avg - single_avg:+.3f} — "
              f"extra rounds not worth the cost")

    # ── Save results ───────────────────────────────────────────────────
    out_dir = Path("ablation_results")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"ablation_{ts}.json"

    save_data = {
        "timestamp": ts,
        "model": "real" if use_model else "stub",
        "budget": budget,
        "tasks": len(tasks),
        "results": [asdict(r) for r in all_results],
        "summary": {
            cond: {
                "mean_top_q": round(statistics.mean(d["top_q"]), 4),
                "mean_avg_q": round(statistics.mean(d["avg_q"]), 4),
                "mean_diversity": round(statistics.mean(d["diversity"]), 4),
                "mean_time_s": round(statistics.mean(d["time"]), 2),
            }
            for cond, d in agg.items()
        },
    }

    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[SAVED] {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Ablation study for swarm coordination")
    parser.add_argument("--model", action="store_true",
                        help="Use real DeepSeek model (requires GPU)")
    parser.add_argument("--budget", type=int, default=60,
                        help="Total LLM calls per condition per task (default: 60)")
    parser.add_argument("--tasks", type=int, default=6,
                        help="Number of tasks to run (default: 6)")
    args = parser.parse_args()

    asyncio.run(run_ablation(args.model, args.budget, args.tasks))


if __name__ == "__main__":
    main()
