"""Per-role LLM router with sequential per-phase model loading.

Pattern 1 of heterogeneous routing: load each unique model exactly once
per phase, run all calls from that phase that need it, then unload before
switching to the next model. Adds 30-60 s of load overhead per swap on
this hardware (six swaps per 3-round run = ~3-6 min overhead on a
~90 min baseline).

Key contract:
    router.acquire(model_path) -> async context manager yielding an
        LLM-like object with .generate(). Loads on enter; does NOT
        unload on exit (next caller may reuse). Explicit teardown()
        unloads at end-of-run.
    router.group_agents(agents) -> groups agents by their assigned
        model so a single phase can be served by loading each unique
        model once in sequence.
"""

from __future__ import annotations

import asyncio
import gc
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from .config import (
    MODELS_DIR, MODEL_NAME, USE_MOCK_LLM,
    DEFAULT_HETEROGENEOUS_ASSIGNMENT,
)


def _load_assignment() -> dict:
    """Resolve the role->model-path map at runtime.

    Reads configs/heterogeneous.json if present; falls back to the
    Python default. Returns paths joined with MODELS_DIR.
    """
    cfg_path = Path(__file__).parent.parent / "configs" / "heterogeneous.json"
    if cfg_path.exists():
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[router] WARNING: could not parse {cfg_path}: {exc}; using defaults")
            raw = DEFAULT_HETEROGENEOUS_ASSIGNMENT
    else:
        raw = DEFAULT_HETEROGENEOUS_ASSIGNMENT
    return {role: str((MODELS_DIR / fname).resolve()) for role, fname in raw.items()}


class HeterogeneousRouter:
    """Sequential per-phase model loader.

    Usage:
        router = HeterogeneousRouter()
        groups = router.group_agents(agents)
        for model_path, agent_subset in groups.items():
            async with router.acquire(model_path) as llm:
                # rebind each agent's .llm to the loaded model, then gather
                ...
        await router.teardown()
    """

    def __init__(self):
        self.assignment = _load_assignment()
        self._current_model_path: Optional[str] = None
        self._current_llm = None
        self._load_count: int = 0

    def model_for_role(self, role: str) -> str:
        return self.assignment.get(
            role,
            str((MODELS_DIR / Path(MODEL_NAME).name).resolve()),
        )

    def group_agents(self, agents) -> dict:
        """Return {model_path: [agents]} grouped by their role's assigned model."""
        groups: dict = {}
        for a in agents:
            mp = self.model_for_role(a.ROLE)
            groups.setdefault(mp, []).append(a)
        return groups

    @asynccontextmanager
    async def acquire(self, model_path: str):
        """Acquire an LLM for the given model path, loading and unloading as needed.

        Reuses the in-place LLM if model_path matches what's already loaded.
        Otherwise unloads the previous one, loads the new one.
        """
        if model_path != self._current_model_path:
            await self._unload_current()
            await self._load(model_path)
        try:
            yield self._current_llm
        finally:
            pass  # do NOT unload on exit — next caller may reuse

    async def _load(self, model_path: str) -> None:
        if USE_MOCK_LLM:
            from .llm import MockLLM
            self._current_llm = MockLLM()
            self._current_model_path = model_path
            self._load_count += 1
            print(f"[router] mock-load #{self._load_count}: {Path(model_path).name}")
            return

        print(f"[router] loading #{self._load_count + 1}: {Path(model_path).name}")
        try:
            from .llm_gguf import LlamaCppLLM
            self._current_llm = LlamaCppLLM(model_path=model_path)
        except Exception as exc:
            print(
                f"[router] WARNING: could not load {model_path} ({exc}); "
                f"using MockLLM for this slot"
            )
            from .llm import MockLLM
            self._current_llm = MockLLM()
        self._current_model_path = model_path
        self._load_count += 1

    async def _unload_current(self) -> None:
        if self._current_llm is None:
            return
        print(f"[router] unloading: {Path(self._current_model_path).name}")
        # llama-cpp-python: dropping references + gc.collect() releases the
        # C-level handle. Calling .close() on the inner Llama object first
        # ensures the file handle is closed before gc runs.
        try:
            inner = getattr(self._current_llm, "_llm", None)
            if inner is not None and hasattr(inner, "close"):
                inner.close()
        except Exception:
            pass
        self._current_llm = None
        self._current_model_path = None
        gc.collect()
        # CUDA cache flush in case the bnb/HF path was used at any point
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    async def teardown(self) -> None:
        """Final cleanup at end of run."""
        await self._unload_current()
        print(f"[router] total model loads this run: {self._load_count}")

    def manifest(self) -> dict:
        """Return the resolved role->model map for run_meta.json."""
        return {role: Path(path).name for role, path in self.assignment.items()}
