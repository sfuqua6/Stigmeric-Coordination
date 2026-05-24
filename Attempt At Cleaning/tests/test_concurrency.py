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


def _purge_core_modules():
    """Force a fresh re-import of core.config.

    `del sys.modules['core.config']` alone is not enough: the parent `core`
    package still has a `config` attribute pointing at the old module, and
    `from core import config` resolves via that attribute (skipping the
    re-load). Pop the parent too.
    """
    for m in list(sys.modules):
        if m == "core" or m.startswith("core."):
            del sys.modules[m]


class TestConcurrencyTierAware(unittest.TestCase):
    def test_laptop_concurrency_is_one(self):
        import os
        if os.environ.get("COLAB"):
            self.skipTest("COLAB env set; can't test the laptop path")
        _purge_core_modules()
        from core import config
        if config._TIER is not None:
            self.skipTest(f"detected tier {config._TIER}; can't test the laptop path")
        self.assertEqual(config.LLM_CONCURRENCY, 1)

    def test_colab_forced_concurrency_is_32(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"COLAB": "1"}, clear=False):
            _purge_core_modules()
            from core import config
            self.assertEqual(config.LLM_CONCURRENCY, 32)
        # Restore so subsequent tests see the laptop value
        _purge_core_modules()


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
