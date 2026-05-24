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

from core.llm_router import (
    HeterogeneousRouter, LoRAHeterogeneousRouter, make_router,
    _load_lora_assignment,
)


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


class TestLoRARouterFallback(unittest.TestCase):
    """Without adapter files on disk, LoRAHeterogeneousRouter logs and
    behaves as a single-base-model pass-through. Validates the scaffold
    is safe to land before any LoRAs are trained.

    Runs under MOCK_LLM=1 so VLLMBackend is never touched — the test
    proves the *routing logic* degrades cleanly, not that vLLM works.
    """

    def test_no_adapters_means_no_lora(self):
        # Heuristic: the scaffold config exists but the actual .safetensors
        # files do not (LoRA training is future work). Router must fall
        # back to single-base-model and set has_adapters=False.
        r = LoRAHeterogeneousRouter()
        self.assertFalse(r.has_adapters,
                         "scaffold should produce no resolved adapters until "
                         "LoRA files actually exist in loras/")
        self.assertEqual(r.assignment, {})

    def test_loads_config_strips_doc_metadata(self):
        """_load_lora_assignment must skip the _doc array and only return
        role->path entries (the JSON config carries documentation alongside
        the mapping)."""
        assignment = _load_lora_assignment()
        # _doc is the only non-role key in the scaffold config
        self.assertNotIn("_doc", assignment)
        # Adapter files don't exist yet, so this should be empty
        self.assertEqual(assignment, {})

    def test_model_for_role_returns_lora_base_key(self):
        r = LoRAHeterogeneousRouter()
        # No adapters → every role maps to "lora:base"
        self.assertEqual(r.model_for_role("scout"), "lora:base")
        self.assertEqual(r.model_for_role("synthesizer"), "lora:base")

    def test_group_agents_single_group_when_no_adapters(self):
        r = LoRAHeterogeneousRouter()
        agents = [_FakeAgent("scout"), _FakeAgent("critic"), _FakeAgent("hater")]
        groups = r.group_agents(agents)
        # All agents land in the same "lora:base" group since no adapters
        self.assertEqual(list(groups.keys()), ["lora:base"])
        self.assertEqual(len(groups["lora:base"]), 3)

    def test_manifest_lists_every_role(self):
        r = LoRAHeterogeneousRouter()
        m = r.manifest()
        for role in ("scout", "developer", "forager", "critic",
                     "hater", "validator", "synthesizer"):
            self.assertIn(role, m)
        # With no adapters, every entry should be the base model only
        for v in m.values():
            self.assertNotIn("lora:", v)


class TestMakeRouter(unittest.TestCase):
    def test_laptop_tier_returns_homogeneous(self):
        self.assertIsInstance(make_router(None), HeterogeneousRouter)

    def test_l4_returns_lora(self):
        self.assertIsInstance(make_router("l4"), LoRAHeterogeneousRouter)

    def test_a100_returns_lora(self):
        self.assertIsInstance(make_router("a100_40"), LoRAHeterogeneousRouter)
        self.assertIsInstance(make_router("a100_80"), LoRAHeterogeneousRouter)

    def test_t4_raises(self):
        with self.assertRaises(RuntimeError):
            make_router("t4")


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
