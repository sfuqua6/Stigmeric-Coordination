"""Per-role LLM routers.

Two routers live here:

  HeterogeneousRouter (laptop GGUF — Pattern 1)
      Load each unique GGUF model exactly once per phase via llama-cpp-python,
      run all calls from that phase that need it, then unload before switching
      to the next model. Adds 30–60 s of load overhead per swap on a 6 GB
      laptop GPU (six swaps per 3-round run = ~3–6 min overhead on a ~90 min
      baseline). Pattern is wrong for Colab — see LoRAHeterogeneousRouter.

  LoRAHeterogeneousRouter (Colab L4/A100 — single base + LoRA adapters)
      Load one base model once via vLLM with enable_lora=True. Per-role
      adapters in `loras/` are selected per-request via LoRARequest. Adapter
      swap is effectively zero-cost (vLLM keeps adapters resident up to
      max_loras and pages them as needed).

      If `loras/` is missing or empty, the router logs and behaves as a
      pass-through single-base-model — the same as USE_HETEROGENEOUS=False
      from the caller's perspective. LoRA training is future work; this
      class is the scaffold so the pipeline is ready when adapters land.

Multi-vLLM-process alternative (B3, deferred)
----------------------------------------------
For A100_80 specifically, an alternative to multi-LoRA is to spawn 2–3
vLLM HTTP servers on the same GPU at gpu_memory_utilization=0.35 each,
each loaded with a different specialist model, and route per-role via
HTTP. Trade-offs:

  + Real specialist models (not just adapters) per role.
  + Each engine schedules independently — no LoRA queue contention.
  - Triples the cold-start cost (3 model loads vs 1).
  - Uses more VRAM total (3 × ~25 GB > one ~70 GB).
  - Requires an HTTP indirection layer in agents/base.py (or a thin
    requests-based VLLMHTTPBackend).

Implementation deferred until the LoRA path is exercised. Sketch:

    from vllm.entrypoints.openai.api_server import run_server
    # Spawn three subprocesses with different --model args, distinct ports,
    # gpu_memory_utilization=0.33. Route agent calls by role -> port via
    # a small dict in core/llm_router.py.

Both routers expose the same contract:
    router.acquire(group_key) -> async context manager yielding an
        LLM-like object with .generate(). Loads on enter; does NOT
        unload on exit (next caller may reuse). Explicit teardown()
        unloads at end-of-run.
    router.group_agents(agents) -> groups agents by their assigned
        slot so a single phase can be served in sequence per group.
    router.model_for_role(role) -> the group_key for that role.
    router.manifest() -> {role: friendly_name} for run_meta.json.
    router.teardown() -> end-of-run cleanup.
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


# ---------------------------------------------------------------------------
# LoRAHeterogeneousRouter — Colab L4/A100 path
# ---------------------------------------------------------------------------
# Single vLLM base model loaded once; per-role LoRA adapters selected per
# request via LoRARequest. Adapter swap is effectively zero-cost (vLLM keeps
# adapters resident up to max_loras and pages them as needed).
#
# Scaffold: when configs/heterogeneous_lora.json is missing OR the listed
# adapter files don't exist on disk, the router logs and degrades to a
# single-base-model pass-through. LoRA training is future work.

_LORA_CONFIG_FILENAME = "heterogeneous_lora.json"
_LORAS_DIRNAME = "loras"


def _load_lora_assignment() -> dict:
    """Return {role: absolute_adapter_path} for adapters that actually exist.

    Reads configs/heterogeneous_lora.json. Entries whose adapter file is
    missing on disk are dropped with a per-role log line. Returns {} when
    the config doesn't exist or no adapters are found.
    """
    project_root = Path(__file__).parent.parent
    cfg_path = project_root / "configs" / _LORA_CONFIG_FILENAME
    if not cfg_path.exists():
        return {}
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[router] could not parse {cfg_path}: {exc}; treating as no adapters")
        return {}
    resolved: dict = {}
    for role, rel in raw.items():
        # Skip non-role metadata keys (convention: leading underscore) and
        # any non-string values (e.g. the _doc array of strings).
        if role.startswith("_") or not isinstance(rel, str):
            continue
        path = Path(rel)
        if not path.is_absolute():
            path = project_root / rel
        if path.exists():
            resolved[role] = str(path.resolve())
        else:
            print(f"[router] LoRA adapter missing for role={role!r}: {path}")
    return resolved


def _stable_lora_id(role: str) -> int:
    """Deterministic int ID for vLLM's LoRARequest from a role name.

    vLLM tracks adapters by integer ID for cache eviction. Using a stable
    hash means the same role gets the same slot across the run, letting
    vLLM keep frequently-used adapters resident.
    """
    h = 0
    for c in role:
        h = (h * 131 + ord(c)) & 0x7FFFFFFF
    # Keep above 0 — vLLM treats 0 as "no LoRA" in some versions.
    return h if h > 0 else 1


class _RoleScopedLoRALLM:
    """LLM-shaped wrapper that injects a fixed lora_request on each generate.

    All role-scoped wrappers share the same underlying VLLMBackend, so the
    base model is loaded exactly once across all roles. vLLM's async engine
    queues requests with different lora_requests together.
    """

    def __init__(self, base_llm, lora_request, role_tag: str):
        self._base = base_llm
        self._lora_request = lora_request
        # Surface the same attributes BaseAgent / pipeline-level wrappers
        # inspect on a real backend.
        self.name = f"{getattr(base_llm, 'name', 'vLLM')}+lora:{role_tag}"
        self._uses_internal_batching = getattr(
            base_llm, "_uses_internal_batching", True,
        )

    async def generate(self, prompt: str, role: str = "agent",
                       max_tokens: int = 120, temperature: float = 0.7) -> str:
        return await self._base.generate(
            prompt, role=role, max_tokens=max_tokens,
            temperature=temperature, lora_request=self._lora_request,
        )


class LoRAHeterogeneousRouter:
    """vLLM single-base + per-role LoRA adapters.

    See module docstring for the rationale and B3 multi-vLLM alternative.

    Usage (same contract as HeterogeneousRouter):
        router = LoRAHeterogeneousRouter()
        groups = router.group_agents(agents)
        for group_key, agent_subset in groups.items():
            async with router.acquire(group_key) as llm:
                for a in agent_subset:
                    a.llm = llm
                await asyncio.gather(*(a.run(...) for a in agent_subset))
        await router.teardown()
    """

    def __init__(self, base_model_name: Optional[str] = None,
                 max_loras: int = 8, max_lora_rank: int = 32):
        from .config import MODEL_NAME, VLLM_DTYPE, LLM_CONCURRENCY

        self.assignment = _load_lora_assignment()  # role -> adapter path
        self.has_adapters = bool(self.assignment)
        self.base_model_name = base_model_name or MODEL_NAME
        self.base_dtype = VLLM_DTYPE
        self._max_loras = max_loras
        self._max_lora_rank = max_lora_rank
        self._base_concurrency = LLM_CONCURRENCY

        self._base_llm = None
        self._role_llms: dict = {}  # role -> _RoleScopedLoRALLM

        if not self.has_adapters:
            print(
                "[router] no LoRA adapters found "
                f"(checked configs/{_LORA_CONFIG_FILENAME} and {_LORAS_DIRNAME}/); "
                "falling back to single-base-model. Pipeline runs identically "
                "to USE_HETEROGENEOUS=False; scaffold stays ready for when "
                "adapters land."
            )
        else:
            print(
                f"[router] LoRA adapters resolved for roles: "
                f"{sorted(self.assignment.keys())} "
                f"(base={self.base_model_name})"
            )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _ensure_base_loaded(self) -> None:
        if self._base_llm is not None:
            return
        from .llm_vllm import VLLMBackend
        from .config import USE_MOCK_LLM
        if USE_MOCK_LLM:
            from .llm import MockLLM
            self._base_llm = MockLLM()
            return
        # enable_lora only when adapters exist — avoids paying any LoRA
        # init cost when the scaffold is degenerate.
        self._base_llm = VLLMBackend(
            model_name=self.base_model_name,
            dtype=self.base_dtype,
            max_num_seqs=max(8, self._base_concurrency),
            enable_lora=self.has_adapters,
            max_loras=self._max_loras,
            max_lora_rank=self._max_lora_rank,
        )

    def _llm_for_role(self, role: str):
        """Return the (possibly LoRA-scoped) LLM for a role."""
        self._ensure_base_loaded()
        if not self.has_adapters or role not in self.assignment:
            return self._base_llm
        cached = self._role_llms.get(role)
        if cached is not None:
            return cached
        try:
            from vllm.lora.request import LoRARequest  # type: ignore
        except Exception as exc:
            # vllm version too old or import failed — silently use base.
            print(f"[router] LoRARequest import failed ({type(exc).__name__}: "
                  f"{exc}); using base model for role={role!r}")
            return self._base_llm
        lora_request = LoRARequest(
            lora_name=role,
            lora_int_id=_stable_lora_id(role),
            lora_path=self.assignment[role],
        )
        scoped = _RoleScopedLoRALLM(self._base_llm, lora_request, role)
        self._role_llms[role] = scoped
        return scoped

    # ------------------------------------------------------------------
    # Public contract — mirrors HeterogeneousRouter
    # ------------------------------------------------------------------

    def model_for_role(self, role: str) -> str:
        """Return a stable group_key for the role.

        Used by group_agents() and by run_swarm.py to look up the synth
        slot. For LoRA, the key is `lora:<role>` when an adapter exists
        and `lora:base` otherwise — distinct keys force agents with
        different adapters into different groups, even though they all
        share the same underlying base model.
        """
        if self.has_adapters and role in self.assignment:
            return f"lora:{role}"
        return "lora:base"

    def group_agents(self, agents) -> dict:
        """Return {group_key: [agents]}.

        Each role with its own adapter gets its own group; roles without
        adapters share the "lora:base" group. Because the base model is
        shared, switching groups is free — but routing per-group keeps
        agents whose LoRA differs from the base from accidentally
        clobbering each other's adapter selection.
        """
        groups: dict = {}
        for a in agents:
            key = self.model_for_role(getattr(a, "ROLE", "agent"))
            groups.setdefault(key, []).append(a)
        return groups

    @asynccontextmanager
    async def acquire(self, group_key: str):
        """Yield the LLM for this group. No load/unload overhead between groups."""
        # group_key is "lora:<role>" or "lora:base"
        if group_key.startswith("lora:"):
            role = group_key[len("lora:"):]
        else:
            role = "base"
        llm = self._llm_for_role(role)
        try:
            yield llm
        finally:
            pass  # base stays loaded for the rest of the run

    async def teardown(self) -> None:
        """End-of-run cleanup. Drops references so vLLM's engine can GC."""
        self._role_llms.clear()
        self._base_llm = None
        try:
            import gc
            gc.collect()
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def manifest(self) -> dict:
        """For run_meta.json: which roles got which adapter (or base)."""
        out = {}
        for role in ("scout", "developer", "forager", "critic",
                     "hater", "validator", "synthesizer"):
            if self.has_adapters and role in self.assignment:
                out[role] = f"{self.base_model_name}+lora:{Path(self.assignment[role]).name}"
            else:
                out[role] = self.base_model_name
        return out


def make_router(tier):
    """Pick the right router for the current tier.

    A100_40 / A100_80 / L4 → LoRAHeterogeneousRouter (vLLM + LoRA).
    T4 → raises (USE_HETEROGENEOUS should be False on T4 per config.py B1).
    None (laptop) → HeterogeneousRouter (GGUF sequential).
    """
    if tier in ("a100_80", "a100_40", "l4"):
        return LoRAHeterogeneousRouter()
    if tier == "t4":
        raise RuntimeError(
            "Heterogeneous routing requested on T4 but VRAM does not fit "
            "a base model + LoRA adapter cache. Set USE_HETEROGENEOUS=False."
        )
    return HeterogeneousRouter()
