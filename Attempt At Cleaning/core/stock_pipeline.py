"""Standalone Stock Swarm orchestrator (round-based).

Drives the stock agents over a real `SignalStore` through the documented round
shape, then the `EquityBriefSynthesizer`:

    Phase A: scouts + validators   (validators deposit VERIFICATION first, so
                                    Phase B deposits pick up the provenance boost)
    Phase B: developers + critics + haters
    -> decay_all() + prune_weak()
    (repeat NUM_ROUNDS)
    -> synthesizer -> answer.txt + prediction.json

This is the round-based contract from run_swarm.run_pipeline, replicated here so
the stock POC does NOT depend on run_swarm.py's multi-path machinery (continuous
pool / phase-isolated). Used by run_stock.py (manual) and eval/backtest.py
(grader, in-process).

No network, no look-ahead: scouts/developers see only `snapshot` facts (which
are point-in-time by construction); the live web-search path is disabled in
ThesisDeveloper.sample().
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable, Optional

from core.signal_store import SignalStore
from core.stock_data import Snapshot
from agents.stock_roles import build_stock_agents

# Round-based defaults (kept small; the bounded data + numeric verification make
# stock converge fast). Override per call.
DEFAULT_NUM_ROUNDS = 3
DEFAULT_ITERATIONS_PER_ROUND = 6

_TASK_PROMPT = (
    "Assess whether to buy {ticker} over a {horizon}-trading-day horizon and "
    "project its return, grounded ONLY in the provided facts. Cite numbers."
)


async def run_stock_pipeline(
    snapshot: Snapshot,
    llm_for: Callable[[str], object],
    *,
    horizon_days: int = 21,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    iterations_per_round: int = DEFAULT_ITERATIONS_PER_ROUND,
    output_dir: Optional[str | Path] = None,
    task_prompt: Optional[str] = None,
    embedder=None,
    verbose: bool = True,
) -> dict:
    """Run the full stock swarm and return {prediction, answer, store}.

    `llm_for(role) -> llm` returns the engine for a role. Pass
    `router.engine_for` for the Groq path or `lambda _r: llm` for a single
    model. `embedder=None` uses the string-similarity fallback (no GPU).
    """
    task_prompt = task_prompt or _TASK_PROMPT.format(
        ticker=snapshot.ticker, horizon=horizon_days)
    store = SignalStore(embedder=embedder)
    agents = build_stock_agents(llm_for, snapshot, task_prompt,
                                horizon_days=horizon_days)

    async def _run(group):
        return await asyncio.gather(
            *(a.run(store, iterations_per_round) for a in group))

    for rnd in range(1, num_rounds + 1):
        # Phase A — scouts + validators (verification before downstream boost).
        await _run(list(agents["scout"]) + list(agents["validator"]))
        # Phase B — developers + critics + haters.
        await _run(list(agents["developer"]) + list(agents["critic"])
                   + list(agents["hater"]))
        store.decay_all()
        pruned = store.prune_weak()
        if verbose:
            print(f"[stock] round {rnd}/{num_rounds}: {store.stats()} pruned={pruned}")

    # Persist signals FIRST so a synthesizer crash can't lose the run.
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out.joinpath("signals.json").write_text(json.dumps([
            {"id": s.id, "type": s.type, "strength": round(s.strength, 4),
             "depositor": s.depositor, "parent_id": s.parent_id,
             "metadata": s.metadata, "content": s.content}
            for s in store.all()], indent=2), encoding="utf-8")

    synth = agents["synthesizer"]
    answer, citations, lineage = await synth.synthesize(
        store, has_validators=True, output_dir=output_dir, task_type="stock")

    if output_dir is not None:
        Path(output_dir, "answer.txt").write_text(answer, encoding="utf-8")

    prediction = getattr(synth, "_last_prediction", None)
    return {"prediction": prediction, "answer": answer, "store": store,
            "citations": citations}


# ---------------------------------------------------------------------------
# Convenience: build an llm_for from the environment (MOCK / Groq / single model)
# ---------------------------------------------------------------------------

def make_llm_for():
    """Return (llm_for, teardown_or_None) based on the environment.

    Honors GROQ_API_KEY (heterogeneous Groq routing) and MOCK_LLM. Falls back to
    the project's single-model factory. Kept here so both run_stock.py and the
    backtest share one resolution path.
    """
    import os
    if os.environ.get("GROQ_API_KEY", "").strip():
        from core.llm_groq import GroqRouter
        router = GroqRouter()
        return router.engine_for, getattr(router, "teardown", None)
    # Single-model (MOCK-aware) path. core/llm.py exposes the project factory.
    from core import llm as _llm_mod
    make = getattr(_llm_mod, "make_llm", None) or getattr(_llm_mod, "build_llm", None)
    if make is None:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "No LLM factory found in core.llm (expected make_llm/build_llm). "
            "Set GROQ_API_KEY or MOCK_LLM=1, or wire the factory.")
    llm = make()
    return (lambda _role: llm), None
