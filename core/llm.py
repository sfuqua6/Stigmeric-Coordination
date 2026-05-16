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

from .config import MODEL_NAME, LLM_CONCURRENCY, USE_MOCK_LLM


# How much VRAM to reserve for weights only. The remaining VRAM (~1 GB on a
# 6GB card) is used by activations and the KV cache during generation.
# Override with SWARM_GPU_MEM env var, e.g. "4500MiB" for a tighter cap.
_GPU_MEM_BUDGET = os.environ.get("SWARM_GPU_MEM") or "5000MiB"
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
    want_vllm = (
        explicit_backend == "vllm"
        or (explicit_backend == "" and (colab_flag or _detected_tier is not None))
    )
    if want_vllm:
        try:
            from .llm_vllm import VLLMBackend, _VLLM_AVAILABLE
            if _VLLM_AVAILABLE:
                from .config import VLLM_DTYPE, LLM_CONCURRENCY
                return VLLMBackend(
                    model_name=MODEL_NAME,
                    dtype=VLLM_DTYPE,
                    max_num_seqs=max(32, LLM_CONCURRENCY),
                )
            print("[llm] vllm not importable; falling back to GGUF/HF path")
        except Exception as exc:
            print(f"[llm] vLLM backend init failed "
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

    try:
        return RealLLM()
    except Exception as exc:
        print(f"[llm] Real model unavailable ({type(exc).__name__}: {exc}); falling back to MockLLM.")
        return MockLLM()


class MockLLM:
    """Deterministic pseudo-LLM for development without GPU."""

    name = "MockLLM"

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

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        cuda_ok = torch.cuda.is_available()
        cache_dir = (
            os.environ.get("HF_HOME")
            or os.environ.get("TRANSFORMERS_CACHE")
            or os.path.expanduser("~/.cache/huggingface")
        )
        print(f"[llm] Loading {MODEL_NAME} (4-bit NF4)...")
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

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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

        self._model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
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
