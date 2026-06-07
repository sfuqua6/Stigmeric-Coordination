"""HybridRouter — local HF/GPU model for high-volume roles + Groq for the rest.

Best-of-both for FREE Groq (scarce RPM/TPD quota) + a Colab GPU:

  * HIGH-VOLUME roles (scout, developer, critic, validator) → ONE local model on
    the GPU. Unlimited throughput, ZERO Groq quota burn. This is where most of
    the swarm's calls go.
  * LOW-VOLUME, HIGH-VALUE role (hater) → a Groq big/diverse model. A different
    model FAMILY for genuine adversarial diversity, at only ~1-2 calls/round, so
    the free quota lasts.

The SYNTHESIZER runs LOCAL by default. It is the token-heavy phase (per-cluster
renders + revision + edge composition = dozens of multi-thousand-token calls in
one burst) and it runs at end-of-run when the GPU is otherwise idle. On Groq's
free tier this burst exhausts the daily token limit (TPD) mid-render — see
outputs/lastgroqrun.txt: "revision skipped (prompt budget insufficient)" + 3
faithfulness flags + half-rendered clusters. Local has no TPD limit, so a 14B
that *completes* beats a 70B that fails halfway. Put it back on Groq with
SWARM_HYBRID_GROQ_ROLES="synthesizer,hater" if you have paid Groq quota.

Same contract as core.llm_groq.GroqRouter (engine_for / role_disabled / manifest /
teardown / bundle_name), so run_pool() and run_swarm.py use it identically — the
continuous pool already routes every role through router.engine_for(role)
(worker_pool.py:684) and the synthesizer through router.engine_for("synthesizer")
(run_swarm.py:1222).

Activate:  SWARM_BACKEND=hybrid  +  GROQ_API_KEY  (and a GPU for the local model).
Tune:      SWARM_HYBRID_GROQ_ROLES="hater"   (default; which roles hit Groq)
           GROQ_ROLE_SYNTHESIZER=...  / GROQ_ROLE_HATER=...   (per-role Groq model)
           SWARM_MODEL=...            (the local HF model; tier-auto otherwise)
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

# Roles routed to Groq by default. Only the hater (low-volume, ~1-2 calls/round)
# — adversarial pressure from a distinct model family at negligible quota burn.
# The synthesizer is deliberately NOT here: its end-of-run token burst blows the
# free-tier daily limit (see module docstring + lastgroqrun.txt), so it runs on
# the idle local GPU. Re-add it via SWARM_HYBRID_GROQ_ROLES if you have quota.
_DEFAULT_GROQ_ROLES = {"hater"}

# Per-role Groq model for the Groq-routed roles. 70B for synthesis depth; a
# 70B versatile for the hater so adversarial pressure comes from a model family
# (large Llama) distinct from the local model. The previous Llama-4 Maverick
# default 404'd on this tier and left the hater with 0 calls for whole runs
# (see lastgroqrun.txt); preflight + the unavailable-error fallback now guard
# against that. Override via GROQ_ROLE_<ROLE> (e.g. a local Colab model).
_DEFAULT_HYBRID_GROQ_MODELS = {
    "synthesizer": "llama-3.3-70b-versatile",
    "hater":       "llama-3.3-70b-versatile",
}
_GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"


class HybridRouter:
    """Routes per-role generate() calls to a local model OR a Groq model.

    Implements the MultiEngineRouter/GroqRouter contract used by run_pool and
    run_swarm.py. No vLLM `acquire()` is needed — the continuous pool calls
    engine_for(role).generate() directly.
    """

    speculative_enabled = False
    _uses_internal_batching = True   # Groq batches server-side; local llm self-throttles
    bundle_name = "hybrid"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        local_llm=None,
        groq_roles: Optional[set] = None,
        groq_models: Optional[dict] = None,
        groq_backends: Optional[dict] = None,   # injectable for tests
    ):
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self._api_key and groq_backends is None:
            raise ValueError("HybridRouter requires GROQ_API_KEY (or injected groq_backends).")

        # Which roles go to Groq (the rest go local).
        env_roles = os.environ.get("SWARM_HYBRID_GROQ_ROLES", "").strip()
        if groq_roles is not None:
            self._groq_roles = set(groq_roles)
        elif env_roles:
            self._groq_roles = {r.strip() for r in env_roles.split(",") if r.strip()}
        else:
            self._groq_roles = set(_DEFAULT_GROQ_ROLES)

        # role -> Groq model (env GROQ_ROLE_<R> overrides; safe fallback model).
        models = dict(_DEFAULT_HYBRID_GROQ_MODELS)
        if groq_models:
            models.update(groq_models)
        self._groq_models: dict[str, str] = {}
        for r in self._groq_roles:
            ov = os.environ.get(f"GROQ_ROLE_{r.upper()}", "").strip()
            self._groq_models[r] = ov or models.get(r, _GROQ_FALLBACK_MODEL)

        # Local model (HF/vLLM) for every non-Groq role. Injectable for tests.
        if local_llm is not None:
            self._local = local_llm
        else:
            from core.llm import make_llm
            self._local = make_llm()

        # Groq backends (one per unique Groq model). Injectable for tests.
        if groq_backends is not None:
            self._groq_backends = dict(groq_backends)
        else:
            from core.llm_groq import GroqBackend
            self._groq_backends = {
                m: GroqBackend(model=m, api_key=self._api_key)
                for m in set(self._groq_models.values())
            }

        self.disabled_roles: set = set()
        # engines dict (local + groq) so banners / len() work like other routers.
        self.engines = {"local": self._local, **self._groq_backends}

        print(f"[hybrid] local={self._local_name()} for "
              f"{sorted(set(('scout','developer','critic','validator')) - self._groq_roles)} "
              f"+ others; groq {self._groq_models}")

    # ---- MultiEngineRouter / GroqRouter contract ------------------------
    def engine_for(self, role: str):
        if role in self._groq_roles:
            return self._groq_backends[self._groq_models[role]]
        return self._local

    def role_disabled(self, role: str) -> bool:
        return role in self.disabled_roles

    def action_disabled(self, action: str) -> bool:
        try:
            from core.config import ACTION_TO_ROLE
            role = ACTION_TO_ROLE.get(action)
            return role in self.disabled_roles if role else False
        except Exception:
            return False

    def manifest(self) -> dict:
        roles = ("scout", "developer", "forager", "critic", "validator",
                 "hater", "synthesizer")
        out = {}
        for r in roles:
            if r in self._groq_roles:
                # Report the backend's *live* model so a preflight/runtime swap
                # (404/decommissioned -> fallback) is visible, not the stale
                # requested name.
                be = self._groq_backends.get(self._groq_models[r])
                model = getattr(be, "_model", self._groq_models[r]) if be else self._groq_models[r]
                out[r] = f"groq:{model}"
            else:
                out[r] = f"local:{self._local_name()}"
        return out

    async def preflight(self) -> None:
        """Ping each Groq model once so a bad/gated model name swaps to the
        fallback before the run instead of zeroing a role mid-run. The hater
        404'ing for a whole run (lastgroqrun.txt) is exactly what this prevents."""
        await asyncio.gather(
            *[b.preflight() for b in self._groq_backends.values()],
            return_exceptions=True,
        )

    def silent_roles(self) -> list:
        """Active Groq roles whose backend made 0 calls all run — a degraded
        run (e.g. the hater that 404'd and produced nothing). Caller surfaces
        this loudly. Note: preflight pings each model, so an unreachable model
        whose fallback is also down is what shows up here."""
        out = []
        for role in self._groq_roles:
            if role in self.disabled_roles:
                continue
            be = self._groq_backends.get(self._groq_models[role])
            if be is not None and getattr(be, "_call_count", 0) == 0:
                out.append(role)
        return sorted(out)

    def engines_summary(self) -> dict:
        return self.manifest()

    def stats(self) -> list:
        return [b.stats() for b in self._groq_backends.values() if hasattr(b, "stats")]

    async def load(self) -> None:   # no-op (local llm already loaded; Groq is HTTP)
        return None

    async def warmup(self, n: int = 3) -> None:
        wu = getattr(self._local, "warmup", None)
        if callable(wu):
            try:
                await wu(n=n)
            except Exception as exc:
                print(f"[hybrid] local warmup raised {type(exc).__name__}: {exc}")

    async def teardown(self) -> None:
        for b in self._groq_backends.values():
            try:
                s = b.stats()
                print(f"[hybrid] groq {s['model']}: {s['calls']} calls, "
                      f"avg {s['avg_latency_ms']} ms")
            except Exception:
                pass
        td = getattr(self._local, "teardown", None)
        if callable(td):
            try:
                await td()
            except Exception:
                pass

    # ---- helpers --------------------------------------------------------
    def _local_name(self) -> str:
        return getattr(self._local, "name", None) or getattr(self._local, "_model", "local")
