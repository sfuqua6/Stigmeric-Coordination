"""Regression tests for output_diversity JSON-serializability.

The Colab heterogeneous run crashed at end-of-round with
`TypeError: Object of type float32 is not JSON serializable` because
sentence-transformers embedders return numpy.ndarray, and the previous
implementation kept numpy scalars all the way through to the round_log
dict. These tests freeze that round-trip with a fake embedder so the
regression can't return.
"""

import json
import math
import unittest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.output_diversity import (
    output_diversity_by_model, _agent_centroid, _cosine_centroids,
)
from core.diversity import AgentContextRecord


class _FakeNumpyArray:
    """Mimics numpy.ndarray's tolist() + iteration shape, with scalars that
    are NOT json-serializable (mimicking numpy.float32 behaviour)."""

    class _NotJsonable(float):
        def __repr__(self):
            return f"NotJsonable({float(self)!r})"

    def __init__(self, values):
        self._values = [self._NotJsonable(v) for v in values]

    def tolist(self):
        # numpy.ndarray.tolist() converts scalars to native Python floats.
        return [float(v) for v in self._values]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class _FakeEmbedder:
    """Returns _FakeNumpyArray for each encode call, deterministically."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, text: str):
        # Hash-based deterministic vector
        h = hash(text) & 0xFFFFFFFF
        return _FakeNumpyArray([
            math.sin(h * (i + 1) * 0.0001) for i in range(self.dim)
        ])


class TestOutputDiversityJsonSafety(unittest.TestCase):
    def _records(self):
        r1 = AgentContextRecord(agent_id="scout_R1_0", role="scout")
        r1.deposit_contents = [
            "scout one observed pattern A in chunk X",
            "scout one observed pattern B in chunk X",
        ]
        r2 = AgentContextRecord(agent_id="forager_R1_0_recent", role="forager")
        r2.deposit_contents = [
            "forager developed claim A with support",
            "forager developed claim B with support",
        ]
        r3 = AgentContextRecord(agent_id="critic_R1_0_stratified", role="critic")
        r3.deposit_contents = [
            "critic flagged claim B as weakly grounded",
        ]
        return [r1, r2, r3]

    def test_returns_native_floats_for_heterogeneous(self):
        manifest = {"scout": "model-A.gguf", "forager": "model-B.gguf",
                     "critic": "model-C.gguf"}
        out = output_diversity_by_model(self._records(), _FakeEmbedder(), manifest)
        # All values must be native Python floats so json.dumps works
        self.assertIsInstance(out["within_model"], dict)
        for v in out["within_model"].values():
            self.assertIs(type(v), float, f"within_model value is {type(v).__name__}")
        if out["between_model"] is not None:
            self.assertIs(type(out["between_model"]), float)
        if out["delta"] is not None:
            self.assertIs(type(out["delta"]), float)
        # Final sanity: the dict must json.dumps without raising
        json.dumps(out)

    def test_returns_native_floats_for_homogeneous(self):
        manifest = {"all": "MockLLM"}
        out = output_diversity_by_model(self._records(), _FakeEmbedder(), manifest)
        for v in out["within_model"].values():
            self.assertIs(type(v), float)
        # Homogeneous: between and delta should be None
        self.assertIsNone(out["between_model"])
        self.assertIsNone(out["delta"])
        json.dumps(out)

    def test_cosine_centroids_returns_native_float(self):
        a = _agent_centroid(["text one"], _FakeEmbedder())
        b = _agent_centroid(["text two"], _FakeEmbedder())
        sim = _cosine_centroids(a, b)
        self.assertIsNotNone(sim)
        self.assertIs(type(sim), float)
        json.dumps(sim)


if __name__ == "__main__":
    unittest.main()
