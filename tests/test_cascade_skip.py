"""Tests for the vLLM cascade VRAM pre-check.

The 'configured' rung tries the full fp16 model even when its weights
obviously cannot fit (14B fp16 ≈ 28 GB on a 22 GB L4), wasting 1-2 min of
download/OOM/cleanup per Colab session. _should_skip_attempt() estimates
the weight footprint from the parameter count in the model name and skips
physically-impossible rungs. Conservative: anything unparseable never skips.

No GPU required.

Run with:
    pytest tests/test_cascade_skip.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm import _estimate_weights_gb, _should_skip_attempt


class TestEstimateWeights(unittest.TestCase):
    def test_14b_fp16(self):
        est = _estimate_weights_gb("Qwen/Qwen2.5-14B-Instruct", "float16")
        self.assertAlmostEqual(est, 28.0, delta=0.1)

    def test_decimal_params(self):
        est = _estimate_weights_gb("Qwen/Qwen2.5-1.5B-Instruct", "float16")
        self.assertAlmostEqual(est, 3.0, delta=0.1)

    def test_awq_quantized_by_name(self):
        est = _estimate_weights_gb("Qwen/Qwen2.5-14B-Instruct-AWQ", "float16")
        self.assertAlmostEqual(est, 8.4, delta=0.1)

    def test_quantization_kwarg(self):
        est = _estimate_weights_gb("Some/Model-14B", "float16",
                                   quantization="awq")
        self.assertAlmostEqual(est, 8.4, delta=0.1)

    def test_unparseable_returns_none(self):
        self.assertIsNone(_estimate_weights_gb("deepseek-ai/DeepSeek-V4-Flash",
                                               "float16"))

    def test_last_param_count_wins(self):
        # 'R1' must not be parsed; '7B' must be.
        est = _estimate_weights_gb(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "float16")
        self.assertAlmostEqual(est, 14.0, delta=0.1)


class TestShouldSkip(unittest.TestCase):
    def test_14b_fp16_skipped_on_l4(self):
        reason = _should_skip_attempt(
            "Qwen/Qwen2.5-14B-Instruct",
            {"dtype": "float16", "gpu_memory_utilization": 0.90},
            free_gb=23.5,
        )
        self.assertTrue(reason)
        self.assertIn("exceed", reason)

    def test_14b_awq_allowed_on_l4(self):
        reason = _should_skip_attempt(
            "Qwen/Qwen2.5-14B-Instruct-AWQ",
            {"dtype": "float16", "quantization": "awq",
             "gpu_memory_utilization": 0.88},
            free_gb=23.5,
        )
        self.assertEqual(reason, "")

    def test_14b_fp16_allowed_on_a100_40(self):
        reason = _should_skip_attempt(
            "Qwen/Qwen2.5-14B-Instruct",
            {"dtype": "float16", "gpu_memory_utilization": 0.90},
            free_gb=39.5,
        )
        self.assertEqual(reason, "")

    def test_no_vram_info_never_skips(self):
        reason = _should_skip_attempt(
            "Qwen/Qwen2.5-14B-Instruct", {"dtype": "float16"}, free_gb=None)
        self.assertEqual(reason, "")

    def test_unparseable_model_never_skips(self):
        reason = _should_skip_attempt(
            "org/UnknownModel", {"dtype": "float16"}, free_gb=8.0)
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
