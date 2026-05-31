"""Standalone re-synthesis from a saved signals.json.

Run this when you want to re-synthesize without re-running the swarm —
either because the original synthesis crashed, the answer was bad, or you
want to try a different model on the same signal field.

Usage:
    python synthesize.py outputs/debate_20260512_010203/
    python synthesize.py outputs/debate_20260512_010203/ --ignore-kb
    python synthesize.py outputs/debate_20260512_010203/ --task-prompt "custom override"

Outputs written to <dir>/answer_resynth_TIMESTAMP.txt (original answer.txt preserved).
Also writes citations_resynth_TIMESTAMP.json and lineage_resynth_TIMESTAMP.dot.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.signal_store import SignalStore, Signal
from core.llm import make_llm
from core.knowledge_base import KnowledgeBase
from agents.synthesizer import Synthesizer


def _load_store(signals_path: Path) -> SignalStore:
    """Replay a saved signals.json into a fresh SignalStore.

    Writes directly into store internals to preserve IDs, strengths,
    and timestamps without triggering dedup.
    """
    data = json.loads(signals_path.read_text(encoding="utf-8"))
    store = SignalStore()
    for entry in data:
        sig = Signal(
            id=entry["id"],
            type=entry["type"],
            content=entry["content"],
            strength=float(entry["strength"]),
            timestamp=float(entry.get("timestamp", 0.0)),
            depositor=entry.get("depositor", "unknown"),
            parent_id=entry.get("parent_id"),
            cluster_id=entry.get("cluster_id"),  # restored from v6+ dumps
            visits=int(entry.get("visits", 0)),
            metadata=entry.get("metadata", {}) or {},
        )
        store._signals[sig.id] = sig
        store._by_type.setdefault(sig.type, set()).add(sig.id)
        if sig.parent_id:
            store._by_parent.setdefault(sig.parent_id, set()).add(sig.id)
    # Re-encode embeddings so the projection can cluster properly.
    # This is best-effort; if the embedder is unavailable, projection falls
    # back to one-cluster-per-INITIAL.
    if store._embedder is not None:
        for sig in store._signals.values():
            emb = store._encode(sig.content)
            if emb is not None:
                store._embeddings[sig.id] = emb
    return store


async def _resynthesize(
    run_dir: Path,
    task_prompt: str | None,
    ignore_kb: bool,
) -> None:
    signals_path = run_dir / "signals.json"
    if not signals_path.exists():
        raise SystemExit(f"no signals.json in {run_dir}")

    meta_path = run_dir / "run_meta.json"
    has_validators = True  # conservative default

    if task_prompt is None:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            task_type = meta.get("task_type", "analysis")
            user_prompt = meta.get("user_prompt", "")
            from run_swarm import TASK_PROMPTS
            task_prompt = TASK_PROMPTS[task_type].replace("{prompt}", user_prompt)
            active_roles = meta.get("active_roles", [])
            has_validators = "validator" in active_roles
        else:
            raise SystemExit(
                "no run_meta.json found; pass --task-prompt explicitly"
            )

    print(f"[synthesize] run dir:     {run_dir}")
    print(f"[synthesize] signals:     {signals_path}")
    print(f"[synthesize] task prompt: {task_prompt!r}")
    print(f"[synthesize] has_validators: {has_validators}")

    store = _load_store(signals_path)
    print(f"[synthesize] loaded {len(store.all())} signals")

    kb = KnowledgeBase()
    if not ignore_kb:
        kb.load()
        if not kb.is_empty():
            print(f"[synthesize] kb: {len(kb.prior_consensus())} prior survivors, "
                  f"{len(kb.prior_rejections())} prior rejections")

    llm = make_llm()
    print(f"[synthesize] llm backend: {llm.name}")

    synth = Synthesizer(llm, task_prompt)
    # Pass task_type so the genome FitnessCompositor uses the correct per-task
    # weight table and the synthesizer's task-aware rendering paths activate.
    answer, citations, lineage_dot = await synth.synthesize(
        store,
        has_validators=has_validators,
        prior_rejections=kb.prior_rejections() if not ignore_kb else None,
        prior_consensus=kb.prior_consensus() if not ignore_kb else None,
        task_type=task_type,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    answer_path = run_dir / f"answer_resynth_{timestamp}.txt"
    answer_path.write_text(answer, encoding="utf-8")
    print(f"[synthesize] wrote: {answer_path}")

    if citations:
        cit_path = run_dir / f"citations_resynth_{timestamp}.json"
        cit_path.write_text(json.dumps(citations, indent=2), encoding="utf-8")
        print(f"[synthesize] wrote: {cit_path} ({len(citations)} entries)")

    if lineage_dot:
        dot_path = run_dir / f"lineage_resynth_{timestamp}.dot"
        dot_path.write_text(lineage_dot, encoding="utf-8")
        print(f"[synthesize] wrote: {dot_path}")

    print()
    print("=" * 60)
    print(answer)
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="path to outputs/<run>/")
    p.add_argument("--task-prompt", default=None,
                   help="override the task prompt (else read from run_meta.json)")
    p.add_argument("--ignore-kb", action="store_true",
                   help="run as if no prior knowledge base exists")
    args = p.parse_args()
    asyncio.run(_resynthesize(Path(args.run_dir), args.task_prompt, args.ignore_kb))


if __name__ == "__main__":
    main()
