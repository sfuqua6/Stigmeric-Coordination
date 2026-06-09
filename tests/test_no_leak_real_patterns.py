"""Tests for the no-leak rule with real pattern inputs (§7 directive).

Verifies that _assert_no_leak() catches all forbidden tokens when they appear
in a prompt, and does NOT fire on clean prompts. Uses BaseAgent directly so
the check is testable without running a full LLM pipeline.

Run with:
    python -m unittest tests.test_no_leak_real_patterns -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import BaseAgent
from core.signal_store import SignalStore, _NULL_EMBEDDER
from core.signal_types import INITIAL, SUPPORT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> SignalStore:
    return SignalStore(embedder=_NULL_EMBEDDER)


class _ConcreteAgent(BaseAgent):
    """Minimal concrete agent for testing _assert_no_leak."""
    ROLE = "test_agent"
    OUTPUT_TYPE = SUPPORT
    INPUT_TYPE = INITIAL
    MAX_TOKENS = 50
    TEMPERATURE = 0.7
    DEFAULT_DEPOSIT_STRENGTH = 0.5

    def sample(self, store):
        return store.by_type(INITIAL)[:1]

    def build_prompt(self, samples, *, store_count=0, own_ids=()):
        return "clean prompt with no forbidden tokens"

    def __init__(self, agent_id: str):
        super().__init__(agent_id, llm=MagicMock())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoLeakRuleEnforcement(unittest.TestCase):
    def setUp(self):
        self.agent = _ConcreteAgent("test_agent_0")
        self.store = _make_store()
        # Deposit one INITIAL so sample() returns something
        self.store.deposit(
            signal_type=INITIAL, content="A claim about climate action.",
            strength=0.6, depositor="scout",
            metadata={"partition_id": "test_partition_0"},
        )

    def _samples(self):
        return self.store.by_type(INITIAL)[:1]

    # -- Forbidden tokens that MUST trigger the rule

    def test_parent_content_forbidden(self):
        prompt = "Here is the parent_content of the signal: x"
        with self.assertRaises(AssertionError) as ctx:
            self.agent._assert_no_leak(prompt, self._samples())
        self.assertIn("parent_content", str(ctx.exception))

    def test_provenance_chain_forbidden(self):
        prompt = "The provenance_chain shows a long lineage."
        with self.assertRaises(AssertionError):
            self.agent._assert_no_leak(prompt, self._samples())

    def test_chain_of_thought_forbidden(self):
        prompt = "Based on chain_of_thought analysis we conclude."
        with self.assertRaises(AssertionError):
            self.agent._assert_no_leak(prompt, self._samples())

    def test_previous_reasoning_forbidden(self):
        prompt = "Given the previous reasoning provided by agent X."
        with self.assertRaises(AssertionError):
            self.agent._assert_no_leak(prompt, self._samples())

    def test_prior_reasoning_forbidden(self):
        prompt = "The prior reasoning suggests the following."
        with self.assertRaises(AssertionError):
            self.agent._assert_no_leak(prompt, self._samples())

    def test_agent_reasoning_forbidden(self):
        prompt = "This is the agent reasoning block."
        with self.assertRaises(AssertionError):
            self.agent._assert_no_leak(prompt, self._samples())

    def test_responses_colon_forbidden(self):
        prompt = "responses: [A, B, C]"
        with self.assertRaises(AssertionError):
            self.agent._assert_no_leak(prompt, self._samples())

    def test_dialogue_thread_forbidden(self):
        prompt = "Refer to the dialogue thread above."
        with self.assertRaises(AssertionError):
            self.agent._assert_no_leak(prompt, self._samples())

    # -- Clean prompts that MUST NOT trigger the rule

    def test_clean_content_passes(self):
        prompt = (
            "Given the following claim: [INITIAL_00001] Climate change is real.\n"
            "Provide one supporting argument."
        )
        # Should not raise
        self.agent._assert_no_leak(prompt, self._samples())

    def test_partition_info_passes(self):
        prompt = "Partition: partition_2\nClaim: [INITIAL_00002] Renewable energy."
        self.agent._assert_no_leak(prompt, self._samples())

    def test_support_diversity_mention_passes(self):
        prompt = "support_diversity=3 verification_score=0.80 — synthesize this claim."
        self.agent._assert_no_leak(prompt, self._samples())

    def test_task_prompt_passes(self):
        prompt = (
            "TASK: Argue both sides of: Climate action is necessary.\n\n"
            "Write one supporting claim based on the evidence below."
        )
        self.agent._assert_no_leak(prompt, self._samples())

    def test_external_context_tag_passes(self):
        prompt = (
            "[External context, not agent-derived]: Solar power grew 25% in 2023.\n"
            "PARAGRAPH:"
        )
        self.agent._assert_no_leak(prompt, self._samples())

    def test_empty_samples_does_not_crash(self):
        prompt = "No samples to check against."
        self.agent._assert_no_leak(prompt, [])


class TestNoLeakCaseInsensitivity(unittest.TestCase):
    """The check uses .lower() so it must catch mixed-case versions too."""

    def test_upper_case_parent_content(self):
        agent = _ConcreteAgent("agent_case_0")
        prompt = "PARENT_CONTENT of the signal is visible here."
        with self.assertRaises(AssertionError):
            agent._assert_no_leak(prompt, [])

    def test_mixed_case_chain_of_thought(self):
        agent = _ConcreteAgent("agent_case_1")
        prompt = "Chain_Of_Thought reasoning is present here."
        with self.assertRaises(AssertionError):
            agent._assert_no_leak(prompt, [])


if __name__ == "__main__":
    unittest.main()
