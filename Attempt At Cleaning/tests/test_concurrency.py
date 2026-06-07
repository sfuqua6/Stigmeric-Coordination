"""Tests for the concurrency / batching contract.

Phase D: backends with ``_uses_internal_batching = True`` (currently
VLLMBackend) handle scheduling themselves and must NOT be wrapped in
an external asyncio.Semaphore. Backends with the attribute False
(MockLLM / RealLLM / LlamaCppLLM) still use their per-instance
Semaphore(LLM_CONCURRENCY) gate.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestInternalBatchingAttribute(unittest.TestCase):
    def test_vllm_backend_advertises_internal_batching(self):
        from core.llm_vllm import VLLMBackend
        self.assertTrue(VLLMBackend._uses_internal_batching)

    def test_mock_llm_does_not(self):
        from core.llm import MockLLM
        self.assertFalse(MockLLM._uses_internal_batching)

    def test_real_llm_does_not(self):
        from core.llm import RealLLM
        self.assertFalse(RealLLM._uses_internal_batching)

    def test_llama_cpp_does_not(self):
        from core.llm_gguf import LlamaCppLLM
        self.assertFalse(LlamaCppLLM._uses_internal_batching)


class TestConcurrencyTierAware(unittest.TestCase):
    def test_laptop_concurrency_is_one(self):
        import os
        import importlib
        if os.environ.get("COLAB"):
            self.skipTest("COLAB env set; can't test the laptop path")
        import core.config as _cfg
        importlib.reload(_cfg)
        if _cfg._TIER is not None:
            self.skipTest(f"detected tier {_cfg._TIER}; can't test the laptop path")
        self.assertEqual(_cfg.LLM_CONCURRENCY, 1)

    def test_colab_forced_concurrency_is_32(self):
        import os
        import importlib
        from unittest.mock import patch
        import core.config as _cfg
        with patch.dict(os.environ, {"COLAB": "1"}, clear=False):
            importlib.reload(_cfg)
            self.assertEqual(_cfg.LLM_CONCURRENCY, 32)
        # Restore: reload without COLAB so subsequent tests see the laptop value
        importlib.reload(_cfg)


class TestSemaphoreSkipContract(unittest.TestCase):
    """A consumer that does external gating should consult the attribute
    and skip the semaphore on internal-batching backends. Future code that
    adds an external semaphore (e.g., a future per-phase rate limiter)
    must respect this. This test documents the contract."""

    def test_attribute_present_on_all_backends(self):
        from core.llm import MockLLM, RealLLM
        from core.llm_gguf import LlamaCppLLM
        from core.llm_vllm import VLLMBackend
        for cls in (MockLLM, RealLLM, LlamaCppLLM, VLLMBackend):
            self.assertTrue(
                hasattr(cls, "_uses_internal_batching"),
                f"{cls.__name__} missing _uses_internal_batching attribute"
            )
            self.assertIsInstance(getattr(cls, "_uses_internal_batching"), bool)


if __name__ == "__main__":
    unittest.main()
