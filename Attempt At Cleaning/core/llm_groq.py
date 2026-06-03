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
import random
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Per-model RPM budgets (Groq free tier, 2026)
# We target 80% of the published limit so one other concurrent user of the
# same key doesn't immediately tip us into 429s.
# ---------------------------------------------------------------------------
_RPM_LIMITS: dict[str, int] = {
    "llama-3.3-70b-versatile":                       24,   # pub 30
    "llama-3.3-70b-specdec":                         24,
    "llama-3.1-8b-instant":                          24,   # pub 30
    "llama3-8b-8192":                                24,
    "deepseek-r1-distill-llama-70b":                 24,
    "qwen-qwq-32b":                                  24,
    "meta-llama/llama-4-scout-17b-16e-instruct":     24,
    "meta-llama/llama-4-maverick-17b-128e-instruct": 24,
}
_DEFAULT_RPM = 24

# ---------------------------------------------------------------------------
# Token-bucket rate limiter (one shared instance per model, across all
# GroqBackend instances that use the same model name).
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Async token bucket: smooths request bursts to stay within RPM quota.

    Workers call `await bucket.acquire()` before sending to Groq. If the
    bucket is empty they sleep just long enough for one token to refill —
    so throughput is capped at RPM/60 req/s without wasted 429 round-trips.
    """

    def __init__(self, rpm: int):
        self._rate = rpm / 60.0          # tokens added per second
        self._burst = max(3, rpm // 6)  # allow short burst before throttling
        self._tokens = float(self._burst)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._burst,
                self._tokens + (now - self._last) * self._rate,
            )
            self._last = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


# Global registry: model_name -> shared _TokenBucket
_RATE_LIMITERS: dict[str, _TokenBucket] = {}


def _rate_limiter_for(model: str) -> _TokenBucket:
    if model not in _RATE_LIMITERS:
        rpm = _RPM_LIMITS.get(model, _DEFAULT_RPM)
        _RATE_LIMITERS[model] = _TokenBucket(rpm)
    return _RATE_LIMITERS[model]


# Per-model semaphore caps on concurrent in-flight requests (orthogonal to
# RPM: this prevents >N simultaneous open HTTP connections per model).
_SEM_LIMITS = {
    "llama-3.3-70b-versatile":                    6,
    "llama-3.3-70b-specdec":                      6,
    "llama-3.1-8b-instant":                       12,
    "llama3-8b-8192":                             12,
    "deepseek-r1-distill-llama-70b":              4,
    "qwen-qwq-32b":                               4,
    "meta-llama/llama-4-scout-17b-16e-instruct":  8,
    "meta-llama/llama-4-maverick-17b-128e-instruct": 6,
}
_DEFAULT_SEM_LIMIT = 8

# Fallback used when a model returns a decommissioned-400. Must be a model
# that is reliably active on the free tier. llama-3.1-8b-instant has been
# the most stable across Groq deprecation cycles.
_FALLBACK_MODEL = "llama-3.1-8b-instant"

# Set of model names already confirmed decommissioned this session (logged
# once per model, not once per call).
_DECOMMISSIONED: set = set()


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
        self._sem = asyncio.Semaphore(_SEM_LIMITS.get(model, _DEFAULT_SEM_LIMIT))
        self._rl = _rate_limiter_for(model)   # shared across backends for same model
        self._call_count = 0
        self._total_ms = 0.0

    def _is_decommissioned_error(self, exc_str: str) -> bool:
        return "decommissioned" in exc_str or "no longer supported" in exc_str

    async def generate(
        self,
        prompt: str,
        role: str = "agent",
        max_tokens: int = 120,
        temperature: float = 0.7,
        **_extra,
    ) -> str:
        """Send prompt to Groq and return the completion text.

        Flow:
          1. Token-bucket acquire — smoothly blocks if we're at the RPM cap.
          2. Semaphore acquire — caps concurrent open connections per model.
          3. HTTP call with exponential-backoff retry on 429 / transient errors.
          4. Auto-heal on decommissioned-400: swap model to fallback, retry.
        """
        await self._rl.acquire()
        async with self._sem:
            return await self._call_with_retry(prompt, max_tokens, temperature)

    async def _call_with_retry(
        self, prompt: str, max_tokens: int, temperature: float, _attempt: int = 0
    ) -> str:
        _MAX_RETRIES = 5
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

            # Decommissioned model (400) — swap to fallback and retry once.
            if self._is_decommissioned_error(exc_str) and _attempt == 0:
                if self._model not in _DECOMMISSIONED:
                    _DECOMMISSIONED.add(self._model)
                    print(f"[groq] {self._model} decommissioned — "
                          f"auto-switching to {_FALLBACK_MODEL}")
                old = self._model
                self._model = _FALLBACK_MODEL
                self.name = f"groq:{_FALLBACK_MODEL}"
                # Adopt the fallback's rate limiter so we don't bypass its quota.
                self._rl = _rate_limiter_for(_FALLBACK_MODEL)
                await self._rl.acquire()
                return await self._call_with_retry(prompt, max_tokens, temperature, 1)

            # Rate-limit (429) — exponential backoff with jitter, up to MAX_RETRIES.
            if "429" in exc_str and _attempt < _MAX_RETRIES:
                delay = min(60.0, 5.0 * (2 ** _attempt)) + random.uniform(0, 2)
                if _attempt == 0:
                    print(f"[groq] {self._model} rate-limited (429), "
                          f"backing off {delay:.1f}s (attempt {_attempt+1}/{_MAX_RETRIES})")
                await asyncio.sleep(delay)
                await self._rl.acquire()   # re-acquire token for the retry
                return await self._call_with_retry(prompt, max_tokens, temperature, _attempt + 1)

            # Transient network error — up to 2 retries with short back-off.
            if _attempt < 2 and any(
                k in type(exc).__name__ for k in ("Timeout", "Connection", "Network")
            ):
                await asyncio.sleep(2.0 * (1 + _attempt))
                await self._rl.acquire()
                return await self._call_with_retry(prompt, max_tokens, temperature, _attempt + 1)

            # Give up — return empty string so the worker skips this iteration
            # rather than crashing the pool.
            print(f"[groq] {self._model} generate() failed after {_attempt} retries: "
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
    # Scout: llama-4-scout-17b (fast, confirmed working at 1695ms avg).
    # 70B was burning the daily token quota (TPD) on scout alone.
    # Scout fires ~15-20% of iterations; 70B is reserved for synthesis only.
    "scout":       "meta-llama/llama-4-scout-17b-16e-instruct",
    # Synthesizer: 70B for the single end-of-run cluster rendering pass.
    # One call per surviving cluster — TPD burn is bounded and justified.
    "synthesizer": "llama-3.3-70b-versatile",
    # Development: llama-4-scout — newer generation, different family from 70B.
    "forager":     "meta-llama/llama-4-scout-17b-16e-instruct",
    "developer":   "meta-llama/llama-4-scout-17b-16e-instruct",
    # Adversarial: DeepSeek R1 distill — reasoning model gives genuine
    # adversarial pressure. Auto-falls back to 8B if decommissioned.
    "hater":       "deepseek-r1-distill-llama-70b",
    # Fast structured roles: 8B instant is the most stable free-tier model.
    "critic":      "llama-3.1-8b-instant",
    "validator":   "llama-3.1-8b-instant",
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
