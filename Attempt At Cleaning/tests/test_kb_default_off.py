"""KB is OFF by default; --use-kb is the opt-in.

Pre-populates a temp KB with one surviving entry, runs the pipeline
both without and with --use-kb, and verifies the prior_consensus
empty-vs-loaded contract via stdout + a downstream behavioural check.

Uses MOCK_LLM=1 so no model loads.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# One pre-populated KB entry. schema_version=2 matches the live KB format
# (see knowledge_base/surviving_clusters.json on main).
_FAKE_SURVIVING = [
    {
        "schema_version": 2,
        "cluster_hash": "deadbeefdeadbeef",
        "representative_content":
            "Pre-populated prior claim asserting that traffic is bad.",
        "representative_embedding": None,
        "member_signal_ids": ["INITIAL_00000"],
        "lineage_signal_ids": ["INITIAL_00000"],
        "partition_origins": ["partition_0"],
        "support_diversity": 3,
        "dissent_pressure": 0.0,
        "verification_score": 0.0,
        "unverified": True,
        "run_count": 1,
        "first_seen": "2026-01-01T00:00:00",
        "last_seen": "2026-01-01T00:00:00",
        "topic_hash": "any",
    }
]


def _run_pipeline(extra_args: list[str], kb_dir: Path, outputs_dir: Path,
                  retrieval_cache_dir: Path) -> subprocess.CompletedProcess:
    """Run run_swarm.py with MOCK_LLM=1 and isolated dirs. Returns the
    completed process so the caller can inspect stdout/stderr/returncode."""
    env = {
        **os.environ,
        "MOCK_LLM": "1",
        "SWARM_KB_DIR": str(kb_dir),
        "SWARM_OUTPUTS_BASE_DIR": str(outputs_dir),
        "SWARM_RETRIEVAL_CACHE_DIR": str(retrieval_cache_dir),
        # Force laptop tier for deterministic behavior — we're not testing
        # tier overrides, and Colab tier detection on a CI box that has no
        # GPU would still return None, but be explicit.
        "COLAB": "0",
    }
    cmd = [
        "python", "run_swarm.py", "debate",
        "Test thesis for KB default", "--corpus=placeholder",
    ] + extra_args
    return subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=120,
    )


class TestKBDefaultOff(unittest.TestCase):
    """Default: --use-kb is NOT passed, so the pipeline must skip KB load."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.kb_dir = self.tmpdir / "kb"
        self.kb_dir.mkdir(parents=True)
        (self.kb_dir / "surviving_clusters.json").write_text(
            json.dumps(_FAKE_SURVIVING, indent=2), encoding="utf-8",
        )
        (self.kb_dir / "rejected_clusters.json").write_text("[]", encoding="utf-8")
        (self.kb_dir / "contested_clusters.json").write_text("[]", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_logs_disabled_and_skips_priors(self):
        """Without --use-kb, pipeline prints `[kb] disabled` and never
        loads the pre-populated prior. This means prior_consensus passed
        into build_projection is [], even though the KB file on disk has
        an entry."""
        result = _run_pipeline(
            extra_args=[],
            kb_dir=self.kb_dir,
            outputs_dir=self.tmpdir / "outputs",
            retrieval_cache_dir=self.tmpdir / "rcache",
        )
        self.assertEqual(
            result.returncode, 0,
            f"pipeline crashed:\nSTDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}",
        )
        # Hard assertion on the new log line.
        self.assertIn("[kb] disabled", result.stdout,
                      "default startup must print the disabled banner")
        # And the loaded-priors line MUST NOT appear, otherwise something
        # silently loaded the KB.
        self.assertNotIn("[kb] loaded", result.stdout,
                         "loaded-priors line must not appear when KB is off")
        # Behavioural check: kb_diff.matched in summary.json should be 0
        # since no prior was consulted (matched counts apply only when the
        # KB was active). This proves prior_consensus was [] at projection.
        outputs_root = self.tmpdir / "outputs" / "outputs_mock"
        runs = sorted(outputs_root.glob("debate_*"),
                      key=lambda p: p.stat().st_mtime)
        self.assertTrue(runs, f"no run produced under {outputs_root}")
        summary = json.loads((runs[-1] / "summary.json").read_text())
        self.assertEqual(
            summary.get("kb_diff", {}).get("matched", -1), 0,
            f"kb_diff.matched must be 0 when KB is off; got summary={summary}",
        )


class TestKBOptInOn(unittest.TestCase):
    """With --use-kb, pipeline DOES load the prior."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.kb_dir = self.tmpdir / "kb"
        self.kb_dir.mkdir(parents=True)
        (self.kb_dir / "surviving_clusters.json").write_text(
            json.dumps(_FAKE_SURVIVING, indent=2), encoding="utf-8",
        )
        (self.kb_dir / "rejected_clusters.json").write_text("[]", encoding="utf-8")
        (self.kb_dir / "contested_clusters.json").write_text("[]", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_use_kb_logs_loaded(self):
        result = _run_pipeline(
            extra_args=["--use-kb"],
            kb_dir=self.kb_dir,
            outputs_dir=self.tmpdir / "outputs",
            retrieval_cache_dir=self.tmpdir / "rcache",
        )
        self.assertEqual(
            result.returncode, 0,
            f"pipeline crashed:\nSTDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}",
        )
        self.assertIn("[kb] loaded", result.stdout,
                      "--use-kb must print the loaded-priors banner")
        # The disabled banner MUST NOT appear when KB is on.
        self.assertNotIn("[kb] disabled", result.stdout,
                         "disabled banner must not appear when --use-kb is set")


if __name__ == "__main__":
    unittest.main()
