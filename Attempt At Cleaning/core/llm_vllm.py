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
    # Blackwell sm_100 / sm_120 needs CUDA 12.9 for FlashInfer JIT to build
    # sampling kernels. Colab's CUDA stack is still on 12.x < 12.9 as of
    # 2026-Q2 → force the Triton sampler + Triton attention so vLLM never
    # touches FlashInfer. Without this the engine-core subprocess dies at
    # profile_run with "FlashInfer requires GPUs with sm75 or higher" (a
    # misleading message — the real issue is FlashInfer can't query a
    # newer arch with an older toolkit).
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
    print("[llm-vllm] Blackwell detected: forcing VLLM_USE_FLASHINFER_SAMPLER=0 "
          "and VLLM_ATTENTION_BACKEND=TRITON_ATTN")

def _import_vllm():
    """Resolve vLLM symbols across the API changes introduced in 0.7–0.22.

    Version history:
      0.4–0.6  from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
      0.7–0.21 AsyncEngineArgs → EngineArgs (vllm.engine.arg_utils);
               from vllm import AsyncLLMEngine still worked
      0.22+    AsyncLLMEngine no longer re-exported at top-level;
               canonical location: vllm.engine.async_llm_engine.AsyncLLMEngine
               (which is itself an alias for vllm.v1.engine.async_llm.AsyncLLM)

    Returns (AsyncLLMEngine, AsyncEngineArgs, SamplingParams) or raises ImportError.
    """
    import importlib

    # SamplingParams has been at vllm top-level across all versions.
    from vllm import SamplingParams  # type: ignore  # noqa: F401

    # --- AsyncLLMEngine: try locations from newest to oldest ---
    _engine_class = None
    for _mod_path, _cls_name in (
        ("vllm.engine.async_llm_engine", "AsyncLLMEngine"),  # 0.7–0.22 submodule
        ("vllm.v1.engine.async_llm",     "AsyncLLM"),         # 0.22+ V1 canonical
        ("vllm",                          "AsyncLLMEngine"),   # 0.4–0.6 top-level
    ):
        try:
            _mod = importlib.import_module(_mod_path)
            _engine_class = getattr(_mod, _cls_name)
            print(f"[llm-vllm] AsyncLLMEngine resolved from {_mod_path}.{_cls_name}")
            break
        except (ImportError, AttributeError):
            continue
    if _engine_class is None:
        raise ImportError(
            "AsyncLLMEngine not found in any known vLLM location. "
            "Tried: vllm.engine.async_llm_engine, vllm.v1.engine.async_llm, vllm"
        )

    # --- EngineArgs (was AsyncEngineArgs): try locations from newest to oldest ---
    _engine_args_class = None
    for _mod_path, _cls_name in (
        ("vllm.engine.arg_utils", "EngineArgs"),       # 0.7–0.22
        ("vllm.engine.arg_utils", "AsyncEngineArgs"),  # transitional
        ("vllm",                  "EngineArgs"),        # some builds re-export
        ("vllm",                  "AsyncEngineArgs"),   # 0.4–0.6 top-level
    ):
        try:
            _mod = importlib.import_module(_mod_path)
            _engine_args_class = getattr(_mod, _cls_name)
            print(f"[llm-vllm] EngineArgs resolved from {_mod_path}.{_cls_name}")
            break
        except (ImportError, AttributeError):
            continue
    if _engine_args_class is None:
        raise ImportError(
            "EngineArgs/AsyncEngineArgs not found in any known vLLM location."
        )

    return _engine_class, _engine_args_class, SamplingParams


try:
    AsyncLLMEngine, AsyncEngineArgs, SamplingParams = _import_vllm()  # type: ignore
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
                 enable_prefix_caching: bool = True,
                 quantization: Optional[str] = None,
                 trust_remote_code: bool = True,
                 enable_lora: bool = False,
                 max_loras: int = 8,
                 max_lora_rank: int = 32,
                 speculative_config: Optional[dict] = None,
                 engine_tag: Optional[str] = None,
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
            enable_prefix_caching=enable_prefix_caching,
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
        # Speculative decoding: vLLM verifies a small draft model's
        # tokens with the primary engine, getting ~1.8-2.2× throughput on
        # long generations (synthesizer cluster calls). Bundle config wires
        # this only onto the synthesizer-targeted engine.
        if speculative_config:
            engine_kwargs["speculative_config"] = speculative_config
        # Forward any other knobs the caller passed (e.g. tensor_parallel_size).
        for k, v in extra_engine_args.items():
            engine_kwargs[k] = v
        # AsyncEngineArgs may not accept certain kwargs depending on vLLM version.
        # Drop unknowns gracefully so a stale install or version bump still loads.
        # Known version-sensitive keys:
        #   enable_prefix_caching  — added in vLLM 0.4, removed/renamed later
        #   speculative_config     — dict form added in vLLM 0.5
        #   disable_sliding_window — vLLM 0.4-0.6 workaround for Phi-3.5 SWA;
        #                            absorbed into engine internals in later versions
        _OPTIONAL_KWARGS = (
            "enable_prefix_caching",
            "speculative_config",
            "disable_sliding_window",
        )
        try:
            engine_args = AsyncEngineArgs(**engine_kwargs)
        except TypeError as exc:
            msg = str(exc)
            dropped = []
            for opt_key in _OPTIONAL_KWARGS:
                if opt_key in msg and opt_key in engine_kwargs:
                    engine_kwargs.pop(opt_key, None)
                    dropped.append(opt_key)
            if not dropped:
                raise
            print(f"[llm-vllm] warning: AsyncEngineArgs rejected {dropped}; "
                  f"dropping and retrying (likely older vllm version)")
            engine_args = AsyncEngineArgs(**engine_kwargs)
        # Bind back whatever finally stuck so introspection logs the truth.
        self._prefix_caching_enabled = bool(engine_kwargs.get("enable_prefix_caching"))
        self._speculative_config = engine_kwargs.get("speculative_config")
        try:
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        except Exception as exc:
            # Most common cause: speculative_config validation failure
            # (target vs draft vocab mismatch, unsupported pair, etc).
            # Disable speculative and retry once. Vocab-mismatch surfaces
            # as a pydantic ValidationError; we can't import pydantic just
            # to type-check it, so match on the message instead.
            msg = str(exc)
            spec_signals = ("speculative", "vocab_size", "draft model",
                            "SpeculativeConfig")
            if (
                "speculative_config" in engine_kwargs
                and any(sig in msg for sig in spec_signals)
            ):
                print(f"[llm-vllm] speculative_config rejected "
                      f"({type(exc).__name__}: {msg.splitlines()[0][:200]}); "
                      f"retrying without speculative decoding")
                engine_kwargs.pop("speculative_config", None)
                self._speculative_config = None
                engine_args = AsyncEngineArgs(**engine_kwargs)
                self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            else:
                raise
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
        # Engine tag drives per-engine SamplingParams selection. Default
        # ("primary") matches the Qwen sampling profile if no MultiEngineRouter
        # is wired (homogeneous-vLLM path).
        self.engine_tag = engine_tag or "primary"
        self.name = f"vLLM:{model_name}"
        # Acceptance-criterion banner: makes the live config trivially greppable
        # in Colab logs ("did the A100 path really raise max_model_len?").
        print(f"[llm] backend: vLLM:{model_name} "
              f"max_model_len={max_model_len} enforce_eager={enforce_eager}")
        print(f"[llm-vllm] prefix_caching={'enabled' if self._prefix_caching_enabled else 'disabled'}")
        if self._speculative_config:
            print(f"[llm-vllm] speculative: draft={self._speculative_config.get('model')} "
                  f"num_speculative_tokens={self._speculative_config.get('num_speculative_tokens')}")
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

        Sampling params are pulled from SAMPLING_PER_ENGINE[engine_tag],
        with per-call temperature/max_tokens overriding the engine default
        (the caller's `temperature` argument always wins for that field).
        """
        formatted = self._apply_chat_template(prompt, role)
        try:
            from .config import SAMPLING_PER_ENGINE
            engine_defaults = dict(SAMPLING_PER_ENGINE.get(self.engine_tag, {}))
        except Exception:
            engine_defaults = {}
        # Per-call overrides (caller-supplied temperature / max_tokens always
        # take precedence over the engine default).
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=max(0.05, temperature),
            top_p=engine_defaults.get("top_p", 0.92),
            repetition_penalty=engine_defaults.get("repetition_penalty", 1.15),
            stop=_STOP_TOKENS,
        )
        self._request_counter += 1
        req_id = f"req-{self._request_counter}"
        final_output = None
        # vLLM V1 (0.8+/0.22+) made request_id keyword-only in generate().
        # Try positional first (0.4–0.7 shape); fall back to keyword.
        if self._lora_enabled and lora_request is not None:
            try:
                stream = self._engine.generate(formatted, params, req_id,
                                                lora_request=lora_request)
            except TypeError:
                stream = self._engine.generate(formatted, params,
                                                request_id=req_id,
                                                lora_request=lora_request)
        else:
            try:
                stream = self._engine.generate(formatted, params, req_id)
            except TypeError:
                stream = self._engine.generate(formatted, params, request_id=req_id)
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
