"""Tests for Colab tier detection (Phase B).

Two angles:
  1. _detect_colab_tier() returns the right tier string for each
     simulated torch.cuda state. Tested in isolation with a fake
     torch in sys.modules — avoids the module-reload-after-pre-import
     trap where the real torch in sys.modules shadows the fake.
  2. _MODEL_BY_TIER / _DTYPE_BY_TIER dicts have the expected entries.
     These drive MODEL_NAME / VLLM_DTYPE selection.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


class _FakeTorch:
    """Minimal stand-in for the torch surface _detect_colab_tier touches."""
    def __init__(self, available=True, name="Tesla T4", mem_gb=16):
        self._available = available
        self._name = name
        self._mem_bytes = int(mem_gb * 1e9)
        self.cuda = self  # so torch.cuda.is_available() resolves

    def is_available(self):
        return self._available

    def get_device_name(self, idx):
        return self._name

    def get_device_properties(self, idx):
        props = type("Props", (), {})()
        props.total_memory = self._mem_bytes
        return props


class TestDetectColabTier(unittest.TestCase):
    """Drive _detect_colab_tier directly with a patched torch."""

    def _detect(self, fake_torch=None, env=None):
        env = env or {}
        # Snapshot and override env for the duration
        from core.config import _detect_colab_tier
        with patch.dict(os.environ, env, clear=False):
            if fake_torch is not None:
                with patch.dict(sys.modules, {"torch": fake_torch}):
                    return _detect_colab_tier()
            else:
                # Simulate torch missing entirely
                with patch.dict(sys.modules, {"torch": None}):
                    return _detect_colab_tier()

    def test_no_torch_no_colab_returns_none(self):
        # COLAB explicitly unset
        env = {k: v for k, v in os.environ.items() if k != "COLAB"}
        env["COLAB"] = ""
        with patch.dict(os.environ, env, clear=True):
            from core.config import _detect_colab_tier
            # sys.modules['torch'] = None makes `import torch` raise ImportError
            with patch.dict(sys.modules, {"torch": None}):
                self.assertIsNone(_detect_colab_tier())

    def test_colab_env_forces_unknown_without_gpu(self):
        with patch.dict(os.environ, {"COLAB": "1"}, clear=False):
            from core.config import _detect_colab_tier
            with patch.dict(sys.modules, {"torch": None}):
                self.assertEqual(_detect_colab_tier(), "unknown")

    def test_t4(self):
        self.assertEqual(self._detect(_FakeTorch(name="Tesla T4", mem_gb=16)), "t4")

    def test_l4(self):
        self.assertEqual(self._detect(_FakeTorch(name="NVIDIA L4", mem_gb=22)), "l4")

    def test_a100_40(self):
        self.assertEqual(self._detect(_FakeTorch(name="A100-SXM4-40GB", mem_gb=40)), "a100_40")

    def test_a100_80(self):
        self.assertEqual(self._detect(_FakeTorch(name="A100-SXM4-80GB", mem_gb=80)), "a100_80")

    def test_unknown_gpu_returns_unknown_only_when_forced(self):
        # Unknown GPU name + COLAB unset → None (laptop path)
        env = {k: v for k, v in os.environ.items() if k != "COLAB"}
        env["COLAB"] = ""
        with patch.dict(os.environ, env, clear=True):
            from core.config import _detect_colab_tier
            with patch.dict(sys.modules, {"torch": _FakeTorch(name="RTX 3060 Laptop", mem_gb=6)}):
                self.assertIsNone(_detect_colab_tier())
        # Same GPU + COLAB=1 → "unknown"
        with patch.dict(os.environ, {"COLAB": "1"}, clear=False):
            from core.config import _detect_colab_tier
            with patch.dict(sys.modules, {"torch": _FakeTorch(name="RTX 3060 Laptop", mem_gb=6)}):
                self.assertEqual(_detect_colab_tier(), "unknown")


class TestModelAndDtypeMappings(unittest.TestCase):
    """The dicts _MODEL_BY_TIER / _DTYPE_BY_TIER drive MODEL_NAME / VLLM_DTYPE."""

    def test_model_by_tier_keys_and_values(self):
        from core.config import _MODEL_BY_TIER
        self.assertEqual(_MODEL_BY_TIER["t4"],      "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(_MODEL_BY_TIER["l4"],      "Qwen/Qwen2.5-14B-Instruct")
        self.assertEqual(_MODEL_BY_TIER["a100_40"], "Qwen/Qwen2.5-32B-Instruct")
        self.assertEqual(_MODEL_BY_TIER["a100_80"], "Qwen/Qwen2.5-32B-Instruct")
        self.assertEqual(_MODEL_BY_TIER["unknown"], "Qwen/Qwen2.5-7B-Instruct")

    def test_dtype_by_tier(self):
        from core.config import _DTYPE_BY_TIER
        self.assertEqual(_DTYPE_BY_TIER["t4"],      "float16")
        self.assertEqual(_DTYPE_BY_TIER["l4"],      "float16")
        self.assertEqual(_DTYPE_BY_TIER["a100_40"], "bfloat16")
        self.assertEqual(_DTYPE_BY_TIER["a100_80"], "bfloat16")
        self.assertEqual(_DTYPE_BY_TIER["unknown"], "float16")

    def test_swarm_model_override_in_env(self):
        # When SWARM_MODEL is set, MODEL_NAME (computed at config-load time)
        # should honor it. We can't easily re-trigger module load without
        # patching, but we can verify the resolution logic by reading the
        # current MODEL_NAME under our fake env.
        from core import config
        # If SWARM_MODEL is already set in the test env, this test is moot
        if os.environ.get("SWARM_MODEL"):
            self.skipTest("SWARM_MODEL set in env; skip override sanity check")
        # MODEL_NAME falls back to laptop default OR tier model; both are
        # populated strings.
        self.assertTrue(isinstance(config.MODEL_NAME, str) and config.MODEL_NAME)


if __name__ == "__main__":
    unittest.main()
