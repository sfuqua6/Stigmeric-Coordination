"""End-to-end MOCK subprocess test for the Stage 2 ledger pipeline entry
point (run_swarm.py --ledger). Mirrors tests/test_phase_isolation.py's
subprocess-test conventions: MOCK_LLM=1, fast-halt env vars, capture_output,
assert returncode 0, then check the artifacts a real run must produce.

This proves plumbing only (MOCK_LLM=1 output is SHA1-seeded, not behavioral
evidence) — see CLAUDE.md "Mock-mode and real-model runs" / P0.1.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLedgerPipelineEndToEnd(unittest.TestCase):
    def test_ledger_pipeline_runs_end_to_end_mock(self):
        repo = Path(__file__).parent.parent
        run_id = f"_test_ledger_{os.getpid()}"
        target = repo / "outputs_mock" / f"analysis_{run_id}"
        pack = repo / "eval" / "packs" / "synth_1_task_1_1x.jsonl"
        self.assertTrue(pack.exists(), f"fixture pack missing: {pack}")

        if target.exists():
            import shutil
            shutil.rmtree(target)

        result = subprocess.run(
            [sys.executable, "run_swarm.py", "analysis",
             "What drives adoption of the new transit policy?",
             "--ledger", "--ledger-workers=3", "--ledger-max-time=20",
             f"--corpus=pack:{pack}",
             f"--run-id=analysis_{run_id}"],
            cwd=str(repo),
            capture_output=True, text=True, timeout=180,
            env={
                **os.environ,
                "MOCK_LLM": "1",
                "PYTHONIOENCODING": "utf-8",
                "SWARM_LEDGER_MAX_TIME_S": "20",
                "SWARM_MAX_VERIFY_ATTEMPTS": "2",
                "SWARM_LEDGER_VERIFY_FLOOR": "2",
            },
        )
        self.assertEqual(
            result.returncode, 0,
            f"ledger pipeline failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        try:
            self.assertTrue(target.exists(), f"run dir not created: {target}")

            answer = (target / "answer.txt").read_text(encoding="utf-8")
            self.assertTrue(answer.strip())

            summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["pipeline"], "ledger")
            self.assertIn("halt_reason", summary)
            self.assertIn("claims_by_status", summary)
            self.assertIn("cost_accounting", summary)

            ledger_doc = json.loads((target / "ledger.json").read_text(encoding="utf-8"))
            self.assertIn("claims", ledger_doc)
            self.assertIn("verifications", ledger_doc)
        finally:
            if target.exists():
                import shutil
                shutil.rmtree(target)


if __name__ == "__main__":
    unittest.main()
