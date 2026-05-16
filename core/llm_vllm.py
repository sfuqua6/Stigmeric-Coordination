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
import os
from typing import Optional

try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams  # type: ignore
    _VLLM_AVAILABLE = True
except Exception:
    AsyncLLMEngine = None  # type: ignore
    AsyncEngineArgs = None  # type: ignore
    SamplingParams = None  # type: ignore
    _VLLM_AVAILABLE = False


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
                 gpu_memory_utilization: float = 0.95,
                 max_num_seqs: int = 8,
                 max_model_len: int = 2048,
                 enforce_eager: bool = True,
                 kv_cache_dtype: str = "fp8_e5m2",
                 enable_chunked_prefill: bool = True,
                 trust_remote_code: bool = True):
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "vllm is not installed. `pip install vllm` to use VLLMBackend."
            )
        print(f"[llm-vllm] loading {model_name} "
              f"(dtype={dtype}, max_num_seqs={max_num_seqs}, "
              f"max_model_len={max_model_len})")
        engine_args = AsyncEngineArgs(
            model=model_name,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            kv_cache_dtype=kv_cache_dtype,
            enable_chunked_prefill=enable_chunked_prefill,
            trust_remote_code=trust_remote_code,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
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
        print(f"[llm-vllm] loaded ok. internal batching cap: {max_num_seqs}")

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
                       temperature: float = 0.7) -> str:
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
        async for output in self._engine.generate(formatted, params, req_id):
            final_output = output
        if final_output is None or not getattr(final_output, "outputs", None):
            return ""
        text = final_output.outputs[0].text
        return text.strip() if isinstance(text, str) else ""
