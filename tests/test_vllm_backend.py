"""Tests for VLLMBackend.

Real vLLM is not installed locally (and a real model load would be too
expensive even if it were). We test:

  - Module imports cleanly even when vllm is missing (laptop-friendly).
  - VLLMBackend.__init__ raises a clear error when vllm is unavailable.
  - With vllm mocked, the chat template is applied via the tokenizer.
  - SamplingParams carries the expected fields (repetition_penalty=1.15,
    stop tokens, top_p, etc.).
  - _uses_internal_batching is True.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_fake_transformers(tokenizer):
    """Build a minimal fake `transformers` module that returns ``tokenizer``
    from AutoTokenizer.from_pretrained. Avoids the real transformers import
    (which trips a huggingface-hub version compatibility check on some hosts).
    """
    fake_mod = types.ModuleType("transformers")
    fake_auto = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: tokenizer,
    )
    fake_mod.AutoTokenizer = fake_auto
    return fake_mod


class TestImportsCleanly(unittest.TestCase):
    def test_module_imports_without_vllm(self):
        # The module is already imported by other tests; just check the
        # availability flag and that _STOP_TOKENS includes the drift guard.
        from core.llm_vllm import _VLLM_AVAILABLE, _STOP_TOKENS, VLLMBackend
        self.assertIsInstance(_VLLM_AVAILABLE, bool)
        self.assertIn("\n\n\n", _STOP_TOKENS)
        self.assertIn("<|im_end|>", _STOP_TOKENS)
        self.assertTrue(VLLMBackend._uses_internal_batching)

    def test_init_raises_when_vllm_missing(self):
        from core import llm_vllm
        if llm_vllm._VLLM_AVAILABLE:
            self.skipTest("vllm is actually installed; cannot test the absence path")
        with self.assertRaises(RuntimeError):
            llm_vllm.VLLMBackend("any/model")


class TestChatTemplate(unittest.TestCase):
    def _make_backend_with_mocks(self):
        """Construct a VLLMBackend with vllm + transformers patched."""
        fake_engine_cls = MagicMock()
        fake_engine_args_cls = MagicMock(return_value=MagicMock())
        fake_sampling_params_cls = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.apply_chat_template.return_value = "<TEMPLATED PROMPT>"
        with patch("core.llm_vllm._VLLM_AVAILABLE", True), \
             patch("core.llm_vllm.AsyncLLMEngine", fake_engine_cls), \
             patch("core.llm_vllm.AsyncEngineArgs", fake_engine_args_cls), \
             patch("core.llm_vllm.SamplingParams", fake_sampling_params_cls), \
             patch.dict(sys.modules, {"transformers": _make_fake_transformers(fake_tokenizer)}):
            from core.llm_vllm import VLLMBackend
            backend = VLLMBackend("test/model", dtype="float16",
                                   gpu_memory_utilization=0.85,
                                   max_num_seqs=16, max_model_len=2048)
        return backend, fake_engine_cls, fake_sampling_params_cls, fake_tokenizer

    def test_chat_template_applied_in_generate(self):
        backend, engine_cls, sp_cls, tok = self._make_backend_with_mocks()
        # Backend setup recorded the expected init params
        self.assertEqual(backend._max_num_seqs, 16)
        self.assertTrue(backend.name.startswith("vLLM:"))
        # apply_chat_template should produce the templated prompt
        formatted = backend._apply_chat_template("raw user prompt", "scout")
        self.assertEqual(formatted, "<TEMPLATED PROMPT>")
        tok.apply_chat_template.assert_called()
        # The call should use add_generation_prompt=True (so the model
        # produces an assistant turn, not just continue the user message)
        call_kwargs = tok.apply_chat_template.call_args.kwargs
        self.assertTrue(call_kwargs.get("add_generation_prompt"))


class TestSamplingParams(unittest.TestCase):
    def test_sampling_params_carries_expected_fields(self):
        # Patch all vllm bits so we can capture the SamplingParams init call
        captured = {}

        def fake_sp(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        fake_tok = MagicMock()
        fake_tok.apply_chat_template.return_value = "tmpl"

        with patch("core.llm_vllm._VLLM_AVAILABLE", True), \
             patch("core.llm_vllm.AsyncLLMEngine") as engine_cls, \
             patch("core.llm_vllm.AsyncEngineArgs"), \
             patch("core.llm_vllm.SamplingParams", side_effect=fake_sp), \
             patch.dict(sys.modules, {"transformers": _make_fake_transformers(fake_tok)}):

            # The async generate path needs the engine's generate() to be
            # an async generator. Mock it.
            async def fake_gen(*a, **k):
                # yield a single result then stop
                result = MagicMock()
                out = MagicMock()
                out.text = "  generated answer  "
                result.outputs = [out]
                yield result

            engine_cls.from_engine_args.return_value.generate = fake_gen

            from core.llm_vllm import VLLMBackend
            backend = VLLMBackend("test/model")

            import asyncio
            result = asyncio.run(backend.generate("hi", role="scout",
                                                    max_tokens=120,
                                                    temperature=0.7))
            self.assertEqual(result, "generated answer")

        # Verify SamplingParams was constructed with the expected fields
        self.assertEqual(captured.get("repetition_penalty"), 1.15)
        self.assertEqual(captured.get("max_tokens"), 120)
        self.assertEqual(captured.get("top_p"), 0.92)
        self.assertEqual(captured.get("temperature"), 0.7)
        stops = captured.get("stop") or []
        self.assertIn("\n\n\n", stops)
        self.assertIn("<|im_end|>", stops)


if __name__ == "__main__":
    unittest.main()
