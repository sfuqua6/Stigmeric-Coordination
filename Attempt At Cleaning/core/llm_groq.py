"""Groq API backend — heterogeneous model routing without VRAM.

Provides real model-family diversity on T4 (and any environment with a Groq
API key) without loading multiple models into GPU memory. Different roles get
genuinely different model families: Llama for scouts/synthesizers, Mixtral
for foragers, Gemma2 for haters.

Activation
----------
Set GROQ_API_KEY in the environment. run_swarm.py's continuous pool detects
this key and constructs a GroqRouter instead of calling make_llm().

Dependencies
------------
Uses the `openai` package with base_url override (always available in Colab
via `pip install openai`). If `groq` package is installed it is preferred
(slightly simpler error messages). If neither is installed, raises ImportError
with a clear install hint.

Rate limits (Groq free tier, 2026)
-----------------------------------
Tokens per minute: 6000 on 70B, 20000 on 8B/Mixtral
Requests per minute: 30 on 70B, 30 on Mixtral, 30 on Gemma2

With 8 workers running parallel and 120-token average response, 70B RPM limit
is hit within ~60 concurrent calls. The semaphore below bounds per-model
in-flight requests to stay well under the per-minute cap.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional


# Per-model semaphore caps so the 30 RPM ceiling is respected under the
# asyncio scheduler. 70B models get a tighter cap (slower per token);
# smaller/MoE models get a wider one.
_SEM_LIMITS = {
    "llama-3.1-70b-versatile":  8,
    "llama-3.3-70b-versatile":  8,
    "mixtral-8x7b-32768":       16,
    "gemma2-9b-it":             16,
    "llama-3.1-8b-instant":     16,
    "llama3-8b-8192":           16,
    "qwen-qwq-32b":             8,
}
_DEFAULT_SEM_LIMIT = 12


def _get_groq_client(api_key: str):
    """Return an async client for the Groq API (OpenAI-compatible endpoint)."""
    try:
        from groq import AsyncGroq  # type: ignore
        return AsyncGroq(api_key=api_key)
    except ImportError:
        pass
    try:
        from openai import AsyncOpenAI  # type: ignore
        return AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    except ImportError:
        raise ImportError(
            "Groq backend requires the `openai` or `groq` package. "
            "Install with: pip install openai  (or pip install groq)"
        )


class GroqBackend:
    """Async Groq API client implementing the same contract as VLLMBackend.

    Shared across roles that use the same model; distinct per model name.
    """

    _uses_internal_batching = True  # Groq handles concurrency server-side

    def __init__(self, model: str, api_key: str):
        self.name = f"groq:{model}"
        self._model = model
        self._api_key = api_key
        self._client = _get_groq_client(api_key)
        limit = _SEM_LIMITS.get(model, _DEFAULT_SEM_LIMIT)
        self._sem = asyncio.Semaphore(limit)
        self._call_count = 0
        self._total_ms = 0.0

    async def generate(
        self,
        prompt: str,
        role: str = "agent",
        max_tokens: int = 120,
        temperature: float = 0.7,
        **_extra,
    ) -> str:
        """Send prompt to Groq and return the completion text.

        Retries once on rate-limit (429) with a 5-second back-off.
        """
        async with self._sem:
            return await self._call_with_retry(prompt, max_tokens, temperature)

    async def _call_with_retry(
        self, prompt: str, max_tokens: int, temperature: float, _attempt: int = 0
    ) -> str:
        t0 = time.monotonic()
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=max(0.0, min(2.0, temperature)),
            )
            text = resp.choices[0].message.content or ""
            self._call_count += 1
            self._total_ms += (time.monotonic() - t0) * 1000
            return text.strip()
        except Exception as exc:
            exc_str = str(exc)
            # Groq / OpenAI SDK surface rate limit as status 429.
            if "429" in exc_str and _attempt == 0:
                await asyncio.sleep(5.0)
                return await self._call_with_retry(prompt, max_tokens, temperature, 1)
            # Connection / timeout on first attempt → one more try.
            if _attempt == 0 and any(
                k in type(exc).__name__ for k in ("Timeout", "Connection", "Network")
            ):
                await asyncio.sleep(2.0)
                return await self._call_with_retry(prompt, max_tokens, temperature, 1)
            # Give up — surface empty string so the worker deposits nothing rather
            # than crashing the pool.
            print(f"[groq] {self._model} generate() failed: "
                  f"{type(exc).__name__}: {str(exc)[:200]}")
            return ""

    def stats(self) -> dict:
        return {
            "model": self._model,
            "calls": self._call_count,
            "avg_latency_ms": round(self._total_ms / max(1, self._call_count), 1),
        }


# ---------------------------------------------------------------------------
# Default role → Groq model assignment
# ---------------------------------------------------------------------------
# Choose models for genuine architectural diversity:
#   - Scout + Synthesizer: 70B Llama (depth, broad reasoning)
#   - Forager/Developer:   Mixtral MoE (wide context, efficient generation)
#   - Critic:              8B Llama (fast structured scoring)
#   - Hater:               Gemma2 (different architecture = real adversarial diversity)
#   - Validator:           8B Llama (fast, structured output)
#
# Override via GROQ_ROLE_* env vars, e.g. GROQ_ROLE_HATER=qwen-qwq-32b
# Override the whole assignment via GROQ_ROLE_MODELS_JSON env var (JSON dict).

_DEFAULT_GROQ_ROLE_MODELS = {
    "scout":       "llama-3.3-70b-versatile",
    "forager":     "mixtral-8x7b-32768",
    "developer":   "mixtral-8x7b-32768",
    "critic":      "llama-3.1-8b-instant",
    "hater":       "gemma2-9b-it",
    "validator":   "llama-3.1-8b-instant",
    "synthesizer": "llama-3.3-70b-versatile",
}


def _resolve_role_models() -> dict:
    """Merge defaults with any per-role env var overrides."""
    import json as _json
    raw_json = os.environ.get("GROQ_ROLE_MODELS_JSON", "").strip()
    if raw_json:
        try:
            overrides = _json.loads(raw_json)
            return {**_DEFAULT_GROQ_ROLE_MODELS, **overrides}
        except Exception as exc:
            print(f"[groq] WARNING: could not parse GROQ_ROLE_MODELS_JSON: {exc}; "
                  "using defaults")
    models = dict(_DEFAULT_GROQ_ROLE_MODELS)
    for role in models:
        env_key = f"GROQ_ROLE_{role.upper()}"
        override = os.environ.get(env_key, "").strip()
        if override:
            models[role] = override
    return models


# ---------------------------------------------------------------------------
# GroqRouter — implements MultiEngineRouter's engine_for() contract
# ---------------------------------------------------------------------------

class GroqRouter:
    """Routes per-role generate() calls to different Groq-hosted models.

    Implements the same contract as MultiEngineRouter so run_swarm.py can
    use it identically. No model loading — all calls are async HTTP.

    Usage in run_swarm.py::

        from core.llm_groq import GroqRouter
        router = GroqRouter(api_key=os.environ["GROQ_API_KEY"])
        # ... pass router= to run_pool()
    """

    # Attributes that run_swarm.py inspects on MultiEngineRouter:
    speculative_enabled = False
    disabled_roles: set

    def __init__(
        self,
        api_key: Optional[str] = None,
        role_models: Optional[dict] = None,
    ):
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "GroqRouter requires GROQ_API_KEY to be set in the environment."
            )
        self._role_models = role_models or _resolve_role_models()
        self.disabled_roles = set()

        # Pre-build one backend per unique model (shared across roles).
        models_needed = set(self._role_models.values())
        self._backends: dict[str, GroqBackend] = {
            m: GroqBackend(model=m, api_key=self._api_key)
            for m in models_needed
        }
        # Expose engines dict so _print_bundle_banner doesn't crash on
        # router.engines (it checks len()). Values are GroqBackend instances.
        self.engines = dict(self._backends)
        self.bundle_name = "groq"

        print(f"[groq] GroqRouter initialised ({len(self._backends)} distinct models):")
        for role, model in sorted(self._role_models.items()):
            print(f"[groq]   {role:12s} → {model}")

    # ------------------------------------------------------------------
    # MultiEngineRouter contract
    # ------------------------------------------------------------------

    def engine_for(self, role: str) -> GroqBackend:
        """Return the GroqBackend for this role."""
        model = self._role_models.get(role)
        if model is None:
            # Unknown role: fall back to scout model.
            model = self._role_models.get("scout", next(iter(self._backends)))
        return self._backends[model]

    def role_disabled(self, role: str) -> bool:
        return role in self.disabled_roles

    def action_disabled(self, action: str) -> bool:
        try:
            from .config import ACTION_TO_ROLE
            role = ACTION_TO_ROLE.get(action)
            return role in self.disabled_roles if role else False
        except Exception:
            return False

    def manifest(self) -> dict:
        return {role: f"groq:{model}" for role, model in self._role_models.items()}

    def engines_summary(self) -> dict:
        return {m: f"groq:{m}" for m in self._backends}

    def stats(self) -> list:
        return [b.stats() for b in self._backends.values()]

    async def teardown(self) -> None:
        total = sum(b._call_count for b in self._backends.values())
        print(f"[groq] total calls this run: {total}")
        for b in self._backends.values():
            s = b.stats()
            print(f"[groq]   {s['model']}: {s['calls']} calls, "
                  f"avg {s['avg_latency_ms']} ms/call")
        self._backends.clear()
        self.engines.clear()
