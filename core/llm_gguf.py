"""GGUF / llama-cpp-python backend.

Alternative to RealLLM for hardware where bitsandbytes is unreliable
(Windows + specific torch/CUDA combos that SEGV during 4-bit quantization).

Advantages on 16 GB RAM / 6 GB VRAM hardware:
  - Loads pre-quantized 4-bit weights directly from disk (no fp16 transient).
  - ~5 GB RAM peak instead of ~14 GB for the HF + bnb path.
  - llama.cpp handles its own CUDA offload — no bitsandbytes.

Pick the GGUF file with env vars (defaults shown):

    $env:SWARM_BACKEND   = "gguf"
    $env:SWARM_GGUF_REPO = "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"
    $env:SWARM_GGUF_FILE = "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
    $env:SWARM_GPU_LAYERS = "-1"   # -1 = offload everything that fits; or set a number

Install (CPU-only wheel works but is slow; CUDA wheel is what you want):

    # CUDA 12.1 prebuilt wheels
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from .config import LLM_CONCURRENCY


_DEFAULT_REPO = "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"
_DEFAULT_FILE = "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"


class LlamaCppLLM:
    """llama-cpp-python wrapper matching the RealLLM async interface."""

    _uses_internal_batching = False  # llama-cpp-python is single-threaded per Llama

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        # Bypass xet BEFORE huggingface_hub is imported — it reads env at import.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

        from llama_cpp import Llama

        repo = os.environ.get("SWARM_GGUF_REPO") or _DEFAULT_REPO
        filename = os.environ.get("SWARM_GGUF_FILE") or _DEFAULT_FILE
        # Explicit model_path argument wins over env-based resolution.
        # The heterogeneous router passes a fully-resolved path here per role.
        local_path = model_path or os.environ.get("SWARM_GGUF_PATH")
        n_gpu_layers = int(os.environ.get("SWARM_GPU_LAYERS", "-1"))
        # n_ctx must cover the LARGEST prompt the pipeline ever sends + its
        # generation budget. The synthesizer is the long-prompt outlier
        # (~1100 tokens with 16 truncated signals + 256-token generation).
        # 2048 is comfortable margin; override with SWARM_PROMPT_MAX_LEN if you
        # know what you are doing.
        _prompt_cap = int(os.environ.get("SWARM_PROMPT_MAX_LEN", "1024"))
        n_ctx = max(2048, _prompt_cap + 512)
        n_threads = int(os.environ.get("SWARM_CPU_THREADS", "0")) or None

        print(f"[llm-gguf] backend:    llama-cpp-python")
        print(f"[llm-gguf] n_gpu_layers: {n_gpu_layers}  (override $env:SWARM_GPU_LAYERS)")
        print(f"[llm-gguf] n_ctx:      {n_ctx}")

        # Resolve the GGUF file. Priority:
        #   1. SWARM_GGUF_PATH env var (explicit override)
        #   2. ~/.cache/swarm_gguf/{filename} (auto-probe; this is where curl
        #      writes the file in the earlier troubleshooting recipe)
        #   3. hf_hub_download with xet disabled
        if local_path:
            if not os.path.isfile(local_path):
                raise FileNotFoundError(
                    f"SWARM_GGUF_PATH={local_path!r} but no file exists there"
                )
            model_path = local_path
            print(f"[llm-gguf] explicit local file: {model_path}")
        else:
            probe = os.path.join(os.path.expanduser("~"), ".cache", "swarm_gguf", filename)
            if os.path.isfile(probe):
                model_path = probe
                print(f"[llm-gguf] auto-detected local file: {model_path}")
            else:
                from huggingface_hub import hf_hub_download
                print(f"[llm-gguf] repo:       {repo}")
                print(f"[llm-gguf] file:       {filename}")
                print(f"[llm-gguf] no local cache; downloading via HF (xet disabled)...")
                try:
                    model_path = hf_hub_download(repo_id=repo, filename=filename)
                except Exception as exc:
                    raise RuntimeError(
                        f"hf_hub_download failed ({type(exc).__name__}: {exc}). "
                        f"Workaround: download the file manually and set "
                        f"$env:SWARM_GGUF_PATH to its full path."
                    ) from exc
                print(f"[llm-gguf] downloaded to: {model_path}")

        self.name = f"GGUF:{filename}"
        self._llm = Llama(
            # repeat_penalty=1.15 kills the paragraph-multiplication pathology
            # observed with DeepSeek-R1 + short prompts. Higher (>1.3) drops pronouns.
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
            seed=-1,
            last_n_tokens_size=64,
        )
        self._sem = asyncio.Semaphore(LLM_CONCURRENCY)
        print(f"[llm-gguf] loaded ok. concurrency cap: {LLM_CONCURRENCY}")

    async def generate(self, prompt, role="agent", max_tokens=120, temperature=0.7):
        async with self._sem:
            return await asyncio.to_thread(
                self._generate_sync, prompt, max_tokens, temperature
            )

    def _generate_sync(self, prompt, max_tokens, temperature):
        # repeat_penalty applied per call (some llama-cpp versions ignore it at
        # construction time; passing it here is the reliable knob).
        result = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=max(0.05, temperature),
            top_p=0.92,
            repeat_penalty=1.15,
            echo=False,
        )
        try:
            text = result["choices"][0]["text"]
        except (KeyError, IndexError, TypeError):
            text = str(result)
        return text.strip()
