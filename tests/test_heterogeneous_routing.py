"""Tests for heterogeneous model routing.

Use MOCK_LLM=1 so no real model loads. Verifies the routing logic itself,
not the underlying LLM behavior.
"""

import os
os.environ["MOCK_LLM"] = "1"

import asyncio
import json
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_router import HeterogeneousRouter


class _FakeAgent:
    def __init__(self, role: str, agent_id: str = "a"):
        self.ROLE = role
        self.agent_id = agent_id


class TestRouterGrouping(unittest.TestCase):
    def test_group_by_model(self):
        r = HeterogeneousRouter()
        agents = [
            _FakeAgent("scout"), _FakeAgent("scout"),
            _FakeAgent("forager"), _FakeAgent("critic"),
        ]
        groups = r.group_agents(agents)
        # Two scouts should be in the same group
        scout_model = r.model_for_role("scout")
        self.assertIn(scout_model, groups)
        self.assertEqual(len(groups[scout_model]), 2)

    def test_manifest_contains_all_roles(self):
        r = HeterogeneousRouter()
        manifest = r.manifest()
        for role in ("scout", "forager", "critic", "hater", "validator", "synthesizer"):
            self.assertIn(role, manifest)


class TestRouterLoadCount(unittest.TestCase):
    def test_load_count_increments(self):
        async def go():
            r = HeterogeneousRouter()
            self.assertEqual(r._load_count, 0)
            async with r.acquire(r.model_for_role("scout")) as _:
                pass
            self.assertEqual(r._load_count, 1)
            # reacquire same model — should NOT reload
            async with r.acquire(r.model_for_role("scout")) as _:
                pass
            self.assertEqual(r._load_count, 1)
            # different model — should reload
            async with r.acquire(r.model_for_role("synthesizer")) as _:
                pass
            self.assertEqual(r._load_count, 2)
            await r.teardown()
        asyncio.run(go())


class TestPipelineSmoke(unittest.TestCase):
    def test_heterogeneous_mock_run_succeeds(self):
        """End-to-end with MOCK_LLM=1 and --heterogeneous; verify run_meta has model_assignment."""
        import subprocess
        result = subprocess.run(
            ["python", "run_swarm.py", "debate", "Test thesis",
             "--heterogeneous", "--corpus=placeholder"],
            cwd=str(Path(__file__).parent.parent),
            capture_output=True, text=True, timeout=180,
            env={**os.environ, "MOCK_LLM": "1"},
        )
        self.assertEqual(result.returncode, 0,
                         f"pipeline failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        # Find latest run_meta
        outputs_mock = Path(__file__).parent.parent / "outputs_mock"
        runs = sorted(outputs_mock.glob("debate_*"), key=lambda p: p.stat().st_mtime)
        self.assertTrue(runs, "no mock output produced")
        meta = json.loads((runs[-1] / "run_meta.json").read_text(encoding="utf-8"))
        self.assertIn("model_assignment", meta)


if __name__ == "__main__":
    unittest.main()
