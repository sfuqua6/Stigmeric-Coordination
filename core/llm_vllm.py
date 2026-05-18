"""vLLM backend (Colab / high-VRAM GPUs).

One non-quantized model held in VRAM; multiple agent calls are served
concurrently via vLLM's AsyncLLMEngine, which batches internally.
``_uses_internal_batching = True`` signals to callers that external
semaphore gating is unnecessary — vLLM's continuous batcher handles
contention.

Chat templates are applied via the model's HF tokenizer
(``apply_chat_template``) rather than hand-built format strings — the
single biggest source of "Let me know if you need anything adjusted!"
chat artifacts in the laptop GGUF path's outputs.

Selected when COLAB=1 is set or SWARM_BACKEND=vllm. Falls back to
GGUF/HF if vllm isn't importable.
"""

from __future__ import annotations

import asyncio
import atexit
import os
from typing import Optional

# Blackwell-only env vars. On Blackwell sm_120 FlashInfer needs CUDA 12.9, so
# we force the Triton sampler + attention path there. A100 (sm_80) runs the
# default FlashInfer build cleanly and benefits from it — applying the Triton
# overrides indiscriminately costs ~15% throughput on Ampere. Gate by tier.
try:
    from .config import _TIER as _CONFIG_TIER
except Exception:
    _CONFIG_TIER = None
if _CONFIG_TIER == "blackwell":
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams  # type: ignore
    _VLLM_AVAILABLE = True
except Exception:
    AsyncLLMEngine = None  # type: ignore
    AsyncEngineArgs = None  # type: ignore
    SamplingParams = None  # type: ignore
    _VLLM_AVAILABLE = False


def _safe_shutdown(engine) -> None:
    """Best-effort vLLM engine shutdown for atexit. Quiets NCCL/ZMQ noise.

    vLLM exposes shutdown() in recent versions; older ones expose stop_remote()
    or just the engine_core. Tolerate every shape, swallow exceptions — the
    interpreter is exiting anyway.
    """
    for attr in ("shutdown", "stop_remote_worker_execution_loop"):
        fn = getattr(engine, attr, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
    try:
        import torch.distributed as dist  # type: ignore
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


# Set HF cache to local SSD if on Colab (critical for ~10× performance improvement).
# HF_HOME might be pre-set to Google Drive (~46 MB/s). Override to /content/hf_cache (~500 MB/s).
if "content/drive" in os.environ.get("HF_HOME", "").lower():
    os.environ["HF_HOME"] = "/content/hf_cache"


# Stop tokens covering Qwen / Llama / Mistral chat closers plus a triple-newline
# guard that catches the "multiple paragraph retries in one generation"
# pathology seen on the free-will laptop run.
_STOP_TOKENS = [
    "<|im_end|>",       # Qwen
    "<|endoftext|>",    # generic
    "</s>",             # Llama / Mistral
    "[/INST]",          # Mistral instruct
    "\n\n\n",           # drift detector
]


class VLLMBackend:
    """Async vLLM wrapper matching the .generate(prompt, role, max_tokens, temperature) API.

    Important attributes:
      - name: human-readable backend tag for run_meta.json
      - _uses_internal_batching: True. BaseAgent / pipeline-level
        Semaphore(LLM_CONCURRENCY) gates should be skipped on this backend
        because vLLM's continuous batcher already schedules requests.
    """

    _uses_internal_batching = True

    def __init__(self, model_name: str, dtype: str = "float16",
                 gpu_memory_utilization: float = 0.92,
                 max_num_seqs: int = 4,
                 max_model_len: int = 1024,
                 enforce_eager: bool = True,
                 kv_cache_dtype: Optional[str] = None,
                 enable_chunked_prefill: bool = False,
                 quantization: Optional[str] = None,
                 trust_remote_code: bool = True,
                 enable_lora: bool = False,
                 max_loras: int = 8,
                 max_lora_rank: int = 32,
                 **extra_engine_args):
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "vllm is not installed. `pip install vllm` to use VLLMBackend."
            )
        print(f"[llm-vllm] loading {model_name} "
              f"(dtype={dtype}, max_num_seqs={max_num_seqs}, "
              f"max_model_len={max_model_len}, "
              f"quant={quantization or 'none'}, "
              f"kv_cache={kv_cache_dtype or 'auto'}, "
              f"lora={'on' if enable_lora else 'off'})")
        engine_kwargs = dict(
            model=model_name,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            enable_chunked_prefill=enable_chunked_prefill,
            trust_remote_code=trust_remote_code,
        )
        # Only include kv_cache_dtype / quantization if explicitly set —
        # passing None or an unsupported value through to AsyncEngineArgs
        # can fail outright on older vllm versions (e.g. fp8_e5m2 on Turing).
        if kv_cache_dtype:
            engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
        if quantization:
            engine_kwargs["quantization"] = quantization
        # LoRA support — only enable when requested. Loading with enable_lora=False
        # avoids any extra setup on T4 and laptop paths.
        if enable_lora:
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_loras"] = max_loras
            engine_kwargs["max_lora_rank"] = max_lora_rank
        # Forward any other knobs the caller passed (e.g. tensor_parallel_size).
        for k, v in extra_engine_args.items():
            engine_kwargs[k] = v
        engine_args = AsyncEngineArgs(**engine_kwargs)
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._lora_enabled = enable_lora
        # Load tokenizer separately. vLLM's engine.get_tokenizer() is
        # awaitable in recent versions; the HF AutoTokenizer is version-stable
        # and only loads vocab/template files (cheap relative to the model).
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
        )
        self._model_name = model_name
        self._max_num_seqs = max_num_seqs
        self._request_counter = 0
        self.name = f"vLLM:{model_name}"
        # Acceptance-criterion banner: makes the live config trivially greppable
        # in Colab logs ("did the A100 path really raise max_model_len?").
        print(f"[llm] backend: vLLM:{model_name} "
              f"max_model_len={max_model_len} enforce_eager={enforce_eager}")
        print(f"[llm-vllm] loaded ok. internal batching cap: {max_num_seqs}")

        # Phase 0: clean shutdown. vLLM's AsyncLLMEngine doesn't tear down
        # cleanly on interpreter exit — NCCL workers and ZMQ sockets log a
        # stack of warnings the user can't act on. Register an atexit that
        # calls engine.shutdown() and tears down the distributed group.
        atexit.register(_safe_shutdown, self._engine)

    def _apply_chat_template(self, prompt: str, role: str) -> str:
        """Wrap the agent's prompt as a single user-message turn.

        The agent prompts in this codebase are self-contained instructions
        (task framing + signals/corpus + format directives). Wrapping them
        in the model's chat template prevents the model from defaulting to
        completion-style behavior, which is where Qwen/Llama emit chat
        artifacts.
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            return formatted if isinstance(formatted, str) else prompt
        except Exception:
            return prompt

    async def generate(self, prompt: str, role: str = "agent",
                       max_tokens: int = 120,
                       temperature: float = 0.7,
                       lora_request=None) -> str:
        """Generate one completion. Pass lora_request to select a LoRA adapter.

        lora_request: an instance of vllm.lora.request.LoRARequest or None.
        Ignored if the engine wasn't loaded with enable_lora=True.
        """
        formatted = self._apply_chat_template(prompt, role)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=max(0.05, temperature),
            top_p=0.92,
            repetition_penalty=1.15,
            stop=_STOP_TOKENS,
        )
        self._request_counter += 1
        req_id = f"req-{self._request_counter}"
        final_output = None
        # Only pass lora_request when LoRA is enabled and an adapter was
        # supplied; the AsyncLLMEngine.generate signature accepts it as a
        # keyword in vllm 0.4+, and rejecting it cleanly via a try/except
        # would mask real schedule errors. Branching at the call site is
        # the cleaner shape.
        if self._lora_enabled and lora_request is not None:
            stream = self._engine.generate(formatted, params, req_id,
                                            lora_request=lora_request)
        else:
            stream = self._engine.generate(formatted, params, req_id)
        async for output in stream:
            final_output = output
        if final_output is None or not getattr(final_output, "outputs", None):
            return ""
        text = final_output.outputs[0].text
        return text.strip() if isinstance(text, str) else ""

    async def warmup(self, n: int = 5) -> None:
        """Phase 0: fire `n` throwaway prompts to compile Triton JIT kernels.

        The first real iteration would otherwise stall ~30s on the
        slot-mapping kernel compile, blocking the convergence loop's quick
        startup feedback. After warmup the engine is hot.
        """
        if n <= 0:
            return
        try:
            print(f"[llm-vllm] JIT warmup: {n} throwaway prompts")
            for i in range(n):
                await self.generate(
                    f"warmup {i}: respond with 'ok'.",
                    role="agent", max_tokens=8, temperature=0.1,
                )
            print("[llm-vllm] JIT warmup complete")
        except Exception as exc:
            print(f"[llm-vllm] warmup skipped ({type(exc).__name__}: {exc})")
