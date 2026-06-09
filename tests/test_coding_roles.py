"""Tests for coding-task roles (§5 directive).

Run with:
    MOCK_LLM=1 python -m unittest tests.test_coding_roles -v

All tests use MOCK_LLM=1 so no GPU is needed.
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("MOCK_LLM", "1")

from core.signal_store import SignalStore, _NULL_EMBEDDER
from core.signal_types import INITIAL, SUPPORT, CRITIQUE_POSITIVE, CRITIQUE_NEGATIVE, VERIFICATION
from agents.coding_roles import StaticCritic, _extract_code, _has_unbound_self_refs
from core.role_registry import get_role_classes
from core.sampling import strategy_for_forager, strategy_for_critic, strategy_for_validator


def _make_store() -> SignalStore:
    return SignalStore(embedder=_NULL_EMBEDDER)


class TestExtractCode(unittest.TestCase):
    def test_extracts_fenced_python_block(self):
        text = "Some prose.\n```python\ndef foo():\n    return 1\n```\nMore prose."
        code = _extract_code(text)
        self.assertIn("def foo():", code)

    def test_returns_empty_when_no_block(self):
        self.assertEqual(_extract_code("No code here."), "")

    def test_fenced_without_python_annotation(self):
        text = "```\ndef bar(): pass\n```"
        code = _extract_code(text)
        self.assertIn("def bar", code)

    def test_detects_top_level_method_fragment(self):
        code = "def get(self, key: int) -> int:\n    return self.map[key]"
        self.assertTrue(_has_unbound_self_refs(code))


class TestStaticCritic(unittest.TestCase):
    def setUp(self):
        from core.llm import make_llm
        self.store = _make_store()
        self.llm = make_llm()

    def _deposit_support(self, code: str) -> str:
        content = f"```python\n{code}\n```"
        sid = self.store.deposit(
            signal_type=SUPPORT, content=content, strength=0.6,
            depositor="developer",
            metadata={
                "depositor_agent_id": "developer_R1_0_stratified_extremes",
                "partition_id": "test_partition_0",
            },
        )
        return sid

    def test_good_code_deposits_critique_positive(self):
        """Valid Python AST → CRITIQUE_POSITIVE deposited."""
        sid = self._deposit_support("def fizzbuzz(n):\n    return str(n)")
        self.assertIsNotNone(sid)

        name, strat = strategy_for_critic(0)
        critic = StaticCritic(
            agent_id="static_critic_0",
            llm=self.llm,
            strategy=strat, strategy_name=name,
            task_prompt="Implement fizzbuzz",
        )

        asyncio.run(critic.run(self.store, 2))
        pos_crits = self.store.by_type(CRITIQUE_POSITIVE)
        self.assertGreater(len(pos_crits), 0, "Valid code should produce CRITIQUE_POSITIVE")

    def test_bad_code_deposits_critique_negative(self):
        """Syntax error → CRITIQUE_NEGATIVE deposited."""
        sid = self._deposit_support("def broken(:\n    pass")  # bad syntax
        self.assertIsNotNone(sid)

        name, strat = strategy_for_critic(0)
        critic = StaticCritic(
            agent_id="static_critic_1",
            llm=self.llm,
            strategy=strat, strategy_name=name,
            task_prompt="Implement fizzbuzz",
        )
        asyncio.run(critic.run(self.store, 2))
        neg_crits = self.store.by_type(CRITIQUE_NEGATIVE)
        self.assertGreater(len(neg_crits), 0, "Bad syntax should produce CRITIQUE_NEGATIVE")


class TestRoleRegistry(unittest.TestCase):
    def test_coding_roles_available(self):
        classes = get_role_classes("coding")
        self.assertIn("scout", classes)
        self.assertIn("developer", classes)
        self.assertIn("critic", classes)
        self.assertIn("hater", classes)
        self.assertIn("validator", classes)
        self.assertIn("synthesizer", classes)

    def test_coding_scout_is_requirements_scout(self):
        from agents.coding_roles import RequirementsScout
        classes = get_role_classes("coding")
        self.assertIs(classes["scout"], RequirementsScout)

    def test_default_roles_unchanged_for_debate(self):
        from agents.scout import Scout
        classes = get_role_classes("debate")
        self.assertIs(classes["scout"], Scout)


class TestFizzbuzzEndToEnd(unittest.TestCase):
    """End-to-end test using MOCK_LLM — proves plumbing, not prose quality."""

    def setUp(self):
        from core.llm import make_llm
        self.llm = make_llm()
        self.store = _make_store()

    def test_requirements_scout_deposits_initials(self):
        from agents.coding_roles import RequirementsScout
        scout = RequirementsScout(
            agent_id="scout_R1_0", llm=self.llm,
            task_prompt="Implement fizzbuzz for numbers 1-100",
        )
        asyncio.run(scout.run(self.store, 4))
        initials = self.store.by_type(INITIAL)
        # MockLLM may produce duplicates; at least one should get through
        self.assertGreater(len(initials), 0, "RequirementsScout must deposit at least one INITIAL")


if __name__ == "__main__":
    unittest.main()
