"""Tests for phase-isolated execution (subprocess-per-phase).

Two angles:
1. SignalStore round-trip — save/load preserves signals + _logit + counters.
2. End-to-end orchestration — running each phase as a separate process
   produces a final run dir with run_meta.json marked execution_mode=phase_isolated.
"""

import os
os.environ["MOCK_LLM"] = "1"

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.signal_store import SignalStore, _NULL_EMBEDDER


class TestStoreRoundTrip(unittest.TestCase):
    def test_save_load_preserves_signals_and_logit(self):
        s = SignalStore(embedder=_NULL_EMBEDDER)
        a = s.deposit("INITIAL", "first claim", 0.5, "scout",
                      metadata={"partition_id": "partition_0"})
        b = s.deposit("INITIAL", "second different claim entirely", 0.7, "scout",
                      metadata={"partition_id": "partition_1"})
        c = s.deposit("SUPPORT", "a developmental support", 0.4, "forager",
                       parent_id=a, metadata={"partition_id": "partition_0"})
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertIsNotNone(c)
        before = s.stats()
        original_logit = s.get(a)._logit

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            s.save_state(path)
            s2 = SignalStore(embedder=_NULL_EMBEDDER)
            s2.load_state(path)
        finally:
            os.unlink(path)

        self.assertEqual(s2.stats(), before)
        self.assertEqual(s2.get(a)._logit, original_logit)
        self.assertEqual(s2.get(a).strength, s.get(a).strength)
        self.assertEqual(s2.get(c).parent_id, a)
        self.assertEqual(s2._next_id, s._next_id)

    def test_empty_store_round_trip(self):
        s = SignalStore(embedder=_NULL_EMBEDDER)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            s.save_state(path)
            s2 = SignalStore(embedder=_NULL_EMBEDDER)
            s2.load_state(path)
        finally:
            os.unlink(path)
        self.assertEqual(s2.stats()["total"], 0)
        self.assertEqual(s2._next_id, 0)


class TestIsolatedOrchestration(unittest.TestCase):
    """End-to-end: orchestrator runs 19 subprocesses, produces a complete run dir."""

    def test_orchestrator_completes_end_to_end(self):
        # Pick a unique run-id so this test doesn't collide with parallel runs.
        run_id = f"_test_iso_{os.getpid()}"
        repo = Path(__file__).parent.parent
        # Clean up any prior leftover from a failed previous test
        target = repo / "outputs_mock" / run_id
        if target.exists():
            import shutil
            shutil.rmtree(target)

        result = subprocess.run(
            [sys.executable, "tools/run_isolated.py",
             "debate", "phase isolation test",
             f"--run-id={run_id}", "--corpus=placeholder", "--ignore-kb"],
            cwd=str(repo),
            capture_output=True, text=True, timeout=300,
            env={
                **os.environ,
                "MOCK_LLM": "1",
                "PYTHONIOENCODING": "utf-8",
                "SWARM_MIN_TIME_S": "0",
                "SWARM_MIN_ITERATIONS": "5",
                "SWARM_MIN_INITIALS_FOR_HALT": "0",
                "SWARM_MIN_INTER_CLUSTER_EDGES": "0",
                "SWARM_SAT_NO_NEW_SURVIVING": "5",
                "SWARM_MAX_ITERATIONS": "20",
            },
        )
        self.assertEqual(result.returncode, 0,
                         f"orchestrator failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

        # Verify the final run dir has the expected artifacts
        self.assertTrue(target.exists(), f"run dir not created: {target}")
        meta = json.loads((target / "run_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta.get("execution_mode"), "phase_isolated")
        self.assertIn("model_assignment", meta)

        manifest = json.loads(
            (target / "phase_manifest.json").read_text(encoding="utf-8"))
        # 6 phases × 3 rounds + synth = 19 for task=debate (all roles active)
        self.assertEqual(len(manifest["completed"]), 19)

        # signals.json should exist and parse
        signals = json.loads((target / "signals.json").read_text(encoding="utf-8"))
        self.assertIsInstance(signals, list)
        self.assertGreater(len(signals), 0)

        # Clean up the test artifact
        import shutil
        shutil.rmtree(target)


if __name__ == "__main__":
    unittest.main()
