"""LLM wrapper.

Single instance of the configured model (default DeepSeek-R1-Distill-Qwen-7B,
override via SWARM_MODEL env var), 4-bit NF4 quantization, asyncio semaphore.
Includes MockLLM (MOCK_LLM=1) for development without GPU.

6GB GPU note: a 7B model in 4-bit NF4 fits in roughly 5 GB of VRAM (weights
+ activations + KV cache for short prompts). On a 6GB laptop card with
0.5-1 GB taken by the desktop session, you are right at the edge. This
wrapper tells transformers to:
  - cap GPU usage explicitly so the device_map planner doesn't over-commit
  - permit CPU offload for whatever doesn't fit
  - route inputs to the model's actual first-parameter device (not a fixed
    self._device), because accelerate dispatch may split the model across
    GPU and CPU
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import re as _re_llm

from .config import MODEL_NAME, LLM_CONCURRENCY, USE_MOCK_LLM


# Detect if on Colab and set cache to local SSD (CRITICAL for performance).
# On Colab, HF_HOME may be set to Google Drive (~46 MB/s). Override it to
# /content/hf_cache (local SSD, ~500 MB/s). This is done here at import time
# to ensure all HF operations use the fast path.
if "content/drive" in os.environ.get("HF_HOME", "").lower():
    os.environ["HF_HOME"] = "/content/hf_cache"

# GPU memory budget for HF 4-bit backend.
# Note: vLLM uses gpu_memory_utilization parameter instead, not this budget.
# For HF 4-bit, a conservative budget ensures device_map uses CPU offload for activations.
# Override with SWARM_GPU_MEM env var, e.g. "5500MiB" for tighter Colab constraints.
#
# Tier-aware default: laptop (6 GB VRAM) caps at 5000 MiB to leave headroom.
# Colab T4 (16 GB) and bigger get a budget that actually fits Qwen-7B at 4-bit
# (~4.5 GB weights + ~3 GB activations + KV cache). Without this, RealLLM as
# the HF-fallback after a vLLM cascade failure would still under-utilize the
# Colab GPU and crawl through CPU offload.
def _default_gpu_budget() -> str:
    try:
        from .config import _TIER
    except Exception:
        _TIER = None
    return {
        "t4":        "13GiB",
        "l4":        "20GiB",
        "a100_40":   "36GiB",
        "a100_80":   "76GiB",
        "h100":      "76GiB",
        "blackwell": "76GiB",
        "unknown":   "13GiB",
    }.get(_TIER, "5000MiB")


_GPU_MEM_BUDGET = os.environ.get("SWARM_GPU_MEM") or _default_gpu_budget()
_CPU_MEM_BUDGET = os.environ.get("SWARM_CPU_MEM") or "30GiB"

# Tokenizer truncation. Lower = smaller KV cache during generation = more
# VRAM headroom. 1024 is comfortable for the swarm's short prompts.
_PROMPT_MAX_LEN = int(os.environ.get("SWARM_PROMPT_MAX_LEN", "1024"))


def _gguf_available() -> bool:
    """Detect whether llama-cpp-python is importable, so GGUF can be the default."""
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# vLLM load cascade (Colab-friendly forced-to-work fallback ladder)
# ---------------------------------------------------------------------------
# Colab T4 (16 GB VRAM) is right at the edge for Qwen-7B-Instruct at fp16:
# weights are ~14 GB and vLLM needs headroom for KV cache + activations +
# the profile_run on init. Many failure modes look like:
#   - CUDA OOM during model.from_pretrained or profile_run
#   - kv_cache_dtype=fp8_e5m2 unsupported on this CUDA/torch combo
#   - vllm API mismatch (max_model_len argument names change across versions)
#   - HF Hub download rate-limited mid-load
#
# Strategy: try a sequence of progressively more conservative configs.
# First-attempt is the user's configured (potentially aggressive) settings;
# if that fails, drop to AWQ 4-bit (fits T4 with room for real concurrency);
# then 3B fp16 (fits anywhere); then bail to HF transformers + bnb 4-bit.


def _log_vram_state(label: str = "") -> None:
    """Print free / total VRAM so cascade failures are diagnosable."""
    try:
        import torch
        if not torch.cuda.is_available():
            print(f"[llm-cascade] {label}: no CUDA device")
            return
        free, total = torch.cuda.mem_get_info()
        name = torch.cuda.get_device_name(0)
        print(f"[llm-cascade] {label}: {free/1e9:.2f}/{total/1e9:.2f} GB free "
              f"on {name}")
    except Exception as exc:
        print(f"[llm-cascade] {label}: VRAM probe failed ({type(exc).__name__})")


def _clear_cuda_cache() -> None:
    """Free anything vLLM left behind between cascade attempts."""
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except Exception:
        return "not-installed"


def _build_cascade(base_model: str, base_dtype: str,
                    base_concurrency: int, tier) -> list:
    """Return list of (label, model_name, kwargs) to try in order."""
    attempts: list = []

    # Attempt 1: configured (user-tuned settings, possibly aggressive T4 ones).
    primary_kwargs = {
        "dtype": base_dtype,
        "max_num_seqs": max(32, base_concurrency),
    }
    if tier == "t4":
        primary_kwargs.update({
            "gpu_memory_utilization": 0.91,
            "max_num_seqs": 2,
            "max_model_len": 768,
            "enforce_eager": True,
            "kv_cache_dtype": "fp8_e5m2",
            "enable_chunked_prefill": False,
        })
    elif tier in ("a100_40", "a100_80", "h100", "blackwell"):
        # max_model_len bumped 4096 -> 8192 (a100_40) / 16384 (a100_80) so
        # longer prompts fit without truncation. vLLM uses paged KV cache —
        # blocks are allocated dynamically, so a higher per-request ceiling
        # doesn't pre-reserve the full max × seqs worst-case. Most pipeline
        # prompts run 2-4 K tokens; the headroom matters for:
        #   - synthesizer per-cluster calls when the cluster has many
        #     SUPPORT excerpts plus external context
        #   - a future global integration pass that reads all surviving
        #     paragraphs at once
        #   - retrieval-rich SCOUT prompts with multiple chunks
        # max_num_seqs=32 keeps the continuous batcher saturated.
        # enforce_eager=False unlocks Inductor compile + CUDA graphs —
        # ~25-40% throughput gain on Ampere, costs ~30s extra warm-up.
        primary_kwargs.update({
            "gpu_memory_utilization": 0.90,
            "max_num_seqs": 32,
            "max_model_len": 16384 if tier in ("a100_80", "h100", "blackwell") else 8192,
            # Blackwell + older Colab CUDA must keep eager mode and skip
            # chunked prefill (Inductor compile triggers FlashInfer JIT
            # which needs CUDA >= 12.9 to build sm_120 kernels).
            "enforce_eager": tier == "blackwell",
            "enable_chunked_prefill": tier != "blackwell",
        })
    elif tier == "l4":
        primary_kwargs.update({
            "max_model_len": 8192,
            "enforce_eager": False,
        })
    attempts.append(("configured", base_model, primary_kwargs))

    # Attempt 2: AWQ 4-bit variant of the configured model. Fits 16 GB T4
    # with room to spare; max_num_seqs can rise back to ~8.
    if "AWQ" not in base_model.upper() and "GPTQ" not in base_model.upper():
        awq_model = base_model.rstrip("/")
        # Heuristic: append -AWQ if the base is a *-Instruct model.
        # Users can override SWARM_AWQ_MODEL to point elsewhere.
        custom_awq = os.environ.get("SWARM_AWQ_MODEL")
        if custom_awq:
            awq_model = custom_awq
        elif awq_model.endswith("-Instruct"):
            awq_model = awq_model + "-AWQ"
        else:
            awq_model = awq_model + "-AWQ"
        awq_kwargs = {
            "dtype": "float16",  # AWQ models always pair with fp16 inference
            "quantization": "awq",
            "max_num_seqs": 8,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.88,
            "enforce_eager": True,
        }
        attempts.append(("awq-4bit", awq_model, awq_kwargs))

    # Attempt 3: smaller fp16 model. Qwen2.5-3B-Instruct is ~6 GB fp16 —
    # fits any GPU including T4 with full max_num_seqs.
    if "3B" not in base_model and "1.5B" not in base_model:
        small_kwargs = {
            "dtype": base_dtype,
            "max_num_seqs": 16,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
        }
        attempts.append(("smaller-3B", "Qwen/Qwen2.5-3B-Instruct", small_kwargs))

    return attempts


_PARAM_COUNT_RE = _re_llm.compile(r"(\d+(?:\.\d+)?)[bB]\b")


def _estimate_weights_gb(model_name: str, dtype: str = "float16",
                         quantization: str = "") -> float | None:
    """Rough weight footprint in GB from the parameter count in the model
    name (e.g. '...-14B-Instruct' → 14e9 params). Returns None when no
    count is parseable. fp16/bf16 ≈ 2 bytes/param; AWQ/GPTQ 4-bit ≈ 0.6
    bytes/param including scales/zeros overhead."""
    last = None
    for m in _PARAM_COUNT_RE.finditer(model_name):
        last = m
    if last is None:
        return None
    params_b = float(last.group(1))
    name_l = model_name.lower()
    if quantization or "awq" in name_l or "gptq" in name_l:
        bytes_per = 0.6
    elif dtype in ("float32", "fp32"):
        bytes_per = 4.0
    else:
        bytes_per = 2.0
    return params_b * bytes_per


def _should_skip_attempt(model_name: str, kwargs: dict,
                         free_gb: float | None) -> str:
    """Return a skip reason when the attempt's weights obviously cannot fit
    the available VRAM (e.g. 14B fp16 ≈ 28 GB on a 22 GB L4 — the doomed
    load wastes 1-2 min of download/OOM/cleanup per session). Empty string
    = attempt is plausible. Conservative: unknown sizes never skip."""
    if free_gb is None:
        return ""
    est = _estimate_weights_gb(
        model_name, kwargs.get("dtype", "float16"),
        kwargs.get("quantization", ""),
    )
    if est is None:
        return ""
    budget = free_gb * float(kwargs.get("gpu_memory_utilization", 0.90))
    if est > budget:
        return (f"weights ≈{est:.0f} GB exceed VRAM budget "
                f"{budget:.0f} GB (free {free_gb:.0f} GB)")
    return ""


def _free_vram_gb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / 1e9
    except Exception:
        return None


def _try_vllm_cascade(base_model: str, base_dtype: str,
                       base_concurrency: int, tier):
    """Walk the cascade until one model loads. Returns None if all fail."""
    from .llm_vllm import VLLMBackend
    print(f"[llm-cascade] starting (vllm version: {_vllm_version()}, "
          f"tier: {tier}, base_model: {base_model})")
    _log_vram_state("before cascade")

    attempts = _build_cascade(base_model, base_dtype, base_concurrency, tier)
    last_exc: Optional[Exception] = None  # noqa: F821
    free_gb = _free_vram_gb()
    for i, (label, model_name, kwargs) in enumerate(attempts, 1):
        skip_reason = _should_skip_attempt(model_name, kwargs, free_gb)
        if skip_reason:
            print(f"[llm-cascade] [{i}/{len(attempts)}] SKIP {label!r} "
                  f"model={model_name}: {skip_reason}")
            continue
        print(f"[llm-cascade] [{i}/{len(attempts)}] attempt {label!r} "
              f"model={model_name}")
        for k, v in kwargs.items():
            print(f"[llm-cascade]   {k} = {v}")
        try:
            llm = VLLMBackend(model_name=model_name, **kwargs)
            print(f"[llm-cascade] SUCCESS at attempt {i}/{len(attempts)} ({label})")
            return llm
        except Exception as exc:
            last_exc = exc
            short = str(exc)[:240].replace("\n", " ")
            print(f"[llm-cascade] FAILED {label}: {type(exc).__name__}: {short}")
            _clear_cuda_cache()
            _log_vram_state(f"after {label} cleanup")

    print(f"[llm-cascade] all {len(attempts)} attempts failed; "
          f"last error: {type(last_exc).__name__}: {last_exc}" if last_exc
          else f"[llm-cascade] all {len(attempts)} attempts failed.")
    return None


# Late import to avoid circulars in tests; real code paths import lazily.
from typing import Optional  # noqa: E402


def make_llm(force_mock: bool = False):
    """Return an LLM instance. Selects backend via SWARM_BACKEND env var.

    SWARM_BACKEND values:
        "vllm" (auto-selected on Colab if vllm is importable)
               — one non-quantized model in VRAM; vLLM batches internally
                 via AsyncLLMEngine. Best for T4 / L4 / A100. Set COLAB=1
                 to force this path on a non-Colab GPU host.
        "gguf" (default if llama-cpp-python is installed)
               — llama-cpp-python with GGUF 4-bit weights; bypasses
                 bitsandbytes entirely. Best for 16 GB RAM / 6 GB VRAM.
        "hf"   — HuggingFace transformers + bitsandbytes 4-bit. Requires
                 ~14 GB system RAM transient for fp16->4bit conversion.
        anything else — treated as "hf"

    Falls back to MockLLM on any load failure or when MOCK_LLM=1.
    """
    if force_mock or USE_MOCK_LLM:
        return MockLLM()

    # vLLM path: explicit SWARM_BACKEND=vllm, or COLAB=1, or auto-detect
    # when a Colab-tier GPU is detected by config._TIER.
    explicit_backend = os.environ.get("SWARM_BACKEND", "").lower()
    colab_flag = os.environ.get("COLAB", "").strip() not in ("", "0", "false", "False")
    try:
        from .config import _TIER as _detected_tier
    except Exception:
        _detected_tier = None
    # SWARM_BACKEND is overloaded: run_swarm.py uses 'hybrid'/'local' to choose
    # the ROUTER, while make_llm() reads the same var to choose the local engine.
    # Treat 'hybrid'/'local' as auto so the local model still gets vLLM on a
    # Colab GPU. (Bug: 'hybrid' was neither 'vllm' nor '' -> want_vllm=False ->
    # the slow HF cascade -> 16 cold workers stalled at iter=0.)
    want_vllm = (
        explicit_backend == "vllm"
        or (explicit_backend in ("", "hybrid", "local")
            and (colab_flag or _detected_tier is not None))
    )
    # Diagnostic banner — makes it obvious in Colab logs what ladder is
    # about to run and what env vars control it.
    print("=" * 72)
    print(f"[llm] make_llm() routing:")
    print(f"[llm]   MODEL_NAME       = {MODEL_NAME}")
    print(f"[llm]   _TIER            = {_detected_tier}")
    print(f"[llm]   SWARM_BACKEND    = {explicit_backend or '(unset)'}")
    print(f"[llm]   COLAB env        = {os.environ.get('COLAB', '(unset)')}")
    print(f"[llm]   want_vllm        = {want_vllm}")
    print(f"[llm]   load order       = "
          f"{'vllm-cascade -> hf-cascade -> mock' if want_vllm else 'gguf -> hf-cascade -> mock'}")
    print("=" * 72)
    if want_vllm:
        try:
            from .llm_vllm import VLLMBackend, _VLLM_AVAILABLE
            if _VLLM_AVAILABLE:
                from .config import VLLM_DTYPE, LLM_CONCURRENCY, _TIER as tier_detected
                cascaded = _try_vllm_cascade(
                    MODEL_NAME, VLLM_DTYPE, LLM_CONCURRENCY, tier_detected,
                )
                if cascaded is not None:
                    return cascaded
                print("[llm] vLLM cascade exhausted all candidates; falling "
                      "back to HF transformers path.")
            else:
                print("[llm] vllm not importable; falling back to GGUF/HF path")
        except Exception as exc:
            print(f"[llm] vLLM cascade unexpected failure "
                  f"({type(exc).__name__}: {exc}); falling back to GGUF/HF.")
        if explicit_backend == "vllm":
            # User explicitly asked for vllm and it failed; don't silently
            # pretend the laptop path is equivalent.
            print("[llm] WARNING: SWARM_BACKEND=vllm requested but unavailable.")

    # Laptop / GGUF / HF path
    if explicit_backend:
        backend = explicit_backend
    elif _gguf_available():
        backend = "gguf"
        print("[llm] Auto-selected GGUF backend (llama-cpp-python detected).")
        print("[llm] To force HF/bnb instead: $env:SWARM_BACKEND = 'hf'")
    else:
        backend = "hf"

    if backend == "gguf":
        try:
            from .llm_gguf import LlamaCppLLM
            return LlamaCppLLM()
        except Exception as exc:
            print(f"[llm] GGUF backend unavailable ({type(exc).__name__}: {exc}); falling back to HF.")
            # fall through to HF attempt

    # HF transformers + bnb 4-bit cascade. First try the configured MODEL_NAME;
    # if it fails (typically: OOM or version mismatch), drop to a smaller
    # model that always fits on a 16 GB GPU.
    hf_attempts = [MODEL_NAME]
    if "3B" not in MODEL_NAME and "1.5B" not in MODEL_NAME:
        hf_attempts.append("Qwen/Qwen2.5-3B-Instruct")
    if "1.5B" not in MODEL_NAME:
        hf_attempts.append("Qwen/Qwen2.5-1.5B-Instruct")

    last_hf_exc = None
    for i, hf_model in enumerate(hf_attempts, 1):
        try:
            print(f"[llm-hf] [{i}/{len(hf_attempts)}] trying {hf_model}")
            return RealLLM(model_name=hf_model)
        except TypeError:
            # Older RealLLM signature without model_name kw — only the first
            # attempt (configured model) works in that path.
            try:
                return RealLLM()
            except Exception as exc:
                last_hf_exc = exc
                print(f"[llm-hf] {hf_model} failed: {type(exc).__name__}: {exc}")
                _clear_cuda_cache()
        except Exception as exc:
            last_hf_exc = exc
            print(f"[llm-hf] {hf_model} failed: {type(exc).__name__}: {exc}")
            _clear_cuda_cache()

    print(f"[llm] all HF attempts failed (last: {type(last_hf_exc).__name__}: "
          f"{last_hf_exc}); falling back to MockLLM. The pipeline will "
          f"complete but outputs are not empirical — they prove plumbing only.")
    return MockLLM()


class MockLLM:
    """Deterministic pseudo-LLM for development without GPU."""

    name = "MockLLM"
    _uses_internal_batching = False  # legacy semaphore is the gate

    def __init__(self):
        self._sem = asyncio.Semaphore(LLM_CONCURRENCY)

    async def generate(self, prompt, role="agent", max_tokens=100, temperature=0.7):
        async with self._sem:
            await asyncio.sleep(0)
            digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:6]
            seed_phrases = _MOCK_PHRASES.get(role, _MOCK_PHRASES["agent"])
            rng = random.Random(int(digest, 16) ^ random.randint(0, 1 << 16))
            n_sentences = max(1, min(4, max_tokens // 30))
            picked = rng.sample(seed_phrases, k=min(n_sentences, len(seed_phrases)))
            return " ".join(p.format(tag=digest) for p in picked)


_MOCK_PHRASES = {
    "scout": [
        "The evidence in chunk {tag} suggests an initial pattern worth tracking.",
        "Reading partition {tag}, one observation stands out as a candidate claim.",
        "From this slice ({tag}), a tentative hypothesis emerges.",
    ],
    "forager": [
        "Building on the prior trace, signal {tag} extends the claim with supporting detail.",
        "The development from artifact {tag} introduces an additional angle.",
        "Following from the deposited signal, {tag} adds context.",
    ],
    "critic": [
        "Evaluating artifact {tag}: the claim has plausibility but limited corroboration. SCORE: 0.5",
        "Critique of {tag}: structurally coherent but unsupported. SCORE: 0.4",
        "Quality assessment of {tag}: medium; revisit after further evidence. SCORE: 0.55",
    ],
    "hater": [
        "Consensus cluster {tag} appears to overgeneralize from a narrow base.",
        "Objection to the prevailing pattern around {tag}: it ignores counter-cases.",
        "The signal cluster anchored at {tag} relies on shared but unexamined priors.",
    ],
    "validator": [
        "External check {tag}: claim partially supported by snippet. SCORE: 0.6",
        "Verification {tag}: source agrees on core fact, disputes peripheral detail. SCORE: 0.55",
    ],
    "synthesizer": [
        "Synthesis {tag}: the surviving signals converge on a narrow set of claims with caveats.",
        "Final read-out {tag}: behavior-consensus across independent agents supports a partial answer.",
    ],
    "agent": ["Generic response derived from prompt fingerprint {tag}."],
}


class RealLLM:
    """4-bit quantized HuggingFace model behind an asyncio semaphore.

    Configured for a 6GB consumer GPU: bounded VRAM budget, CPU offload
    permitted, dispatch-aware input routing.
    """

    name = MODEL_NAME
    _uses_internal_batching = False  # legacy semaphore is the gate

    def __init__(self, model_name: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Allow the HF cascade to step down to a smaller model on OOM.
        resolved_model = model_name or MODEL_NAME
        self.name = resolved_model

        cuda_ok = torch.cuda.is_available()
        cache_dir = (
            os.environ.get("HF_HOME")
            or os.environ.get("TRANSFORMERS_CACHE")
            or os.path.expanduser("~/.cache/huggingface")
        )
        print(f"[llm] Loading {resolved_model} (4-bit NF4)...")
        print(f"[llm]   cache:        {cache_dir}")
        print(f"[llm]   cuda:         {cuda_ok}")
        if cuda_ok:
            free_b, total_b = torch.cuda.mem_get_info()
            print(f"[llm]   GPU mem:      {free_b/1e9:.2f} GB free / {total_b/1e9:.2f} GB total")
            print(f"[llm]   GPU budget:   {_GPU_MEM_BUDGET}  (override with $env:SWARM_GPU_MEM)")
            print(f"[llm]   CPU budget:   {_CPU_MEM_BUDGET}  (override with $env:SWARM_CPU_MEM)")
        print(f"[llm]   prompt cap:   {_PROMPT_MAX_LEN} tokens  (override with $env:SWARM_PROMPT_MAX_LEN)")

        # KEY FIX 1: enable fp32 CPU offload so 4-bit dispatch allows spillover.
        # The flag name says int8 but it gates 4-bit CPU offload as well.
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        # KEY FIX 2: cap GPU usage so device_map="auto" doesn't try to
        # cram everything onto the card and then OOM during generation.
        max_memory = None
        if cuda_ok:
            max_memory = {0: _GPU_MEM_BUDGET, "cpu": _CPU_MEM_BUDGET}

        self._tokenizer = AutoTokenizer.from_pretrained(resolved_model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        kwargs = dict(
            quantization_config=bnb,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        if max_memory is not None:
            kwargs["max_memory"] = max_memory

        self._model = AutoModelForCausalLM.from_pretrained(resolved_model, **kwargs)
        self._model.eval()

        # KEY FIX 3: route inputs to wherever the embedding layer actually
        # ended up. With CPU offload, self._model may not have a single
        # `.device` — accelerate dispatch routes per-layer.
        try:
            self._input_device = next(self._model.parameters()).device
        except StopIteration:
            self._input_device = torch.device("cpu")

        # Free any allocator slack now that the model is in place.
        if cuda_ok:
            torch.cuda.empty_cache()
            free_b, total_b = torch.cuda.mem_get_info()
            print(f"[llm]   post-load:    {free_b/1e9:.2f} GB free / {total_b/1e9:.2f} GB total")

        # Show where layers actually landed — useful for diagnosing offload.
        device_map = getattr(self._model, "hf_device_map", None)
        if device_map:
            on_gpu = sum(1 for d in device_map.values() if str(d).startswith("cuda") or d == 0)
            on_cpu = sum(1 for d in device_map.values() if str(d) == "cpu")
            on_disk = sum(1 for d in device_map.values() if str(d) == "disk")
            total = len(device_map)
            print(f"[llm]   dispatch:     {on_gpu}/{total} on GPU, {on_cpu}/{total} on CPU, {on_disk}/{total} on disk")

        self._sem = asyncio.Semaphore(LLM_CONCURRENCY)
        print(f"[llm] Loaded. Input device: {self._input_device}. Concurrency cap: {LLM_CONCURRENCY}")

    async def generate(self, prompt, role="agent", max_tokens=120, temperature=0.7):
        async with self._sem:
            return await asyncio.to_thread(
                self._generate_sync, prompt, max_tokens, temperature
            )

    def _generate_sync(self, prompt, max_tokens, temperature):
        import torch

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=_PROMPT_MAX_LEN,
        )
        # Route to the model's first-parameter device — works for both
        # all-on-GPU and dispatched configurations.
        inputs = {k: v.to(self._input_device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(0.05, temperature),
                do_sample=temperature > 0,
                top_p=0.92,
                repetition_penalty=1.15,
                pad_token_id=self._tokenizer.pad_token_id,
                use_cache=True,
            )

        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Aggressively free per-call to avoid creep across many generations.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return text.strip()
