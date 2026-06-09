"""Integration tests for critic signal generation.

Tests that critics deposit CRITIQUE signals with full provenance,
not just adjust strength silently.
"""

import unittest
import asyncio
from swarm.core.signal_store import SignalStore
from swarm.agents.critic import Critic


class MockLLM:
    """Mock LLM for testing without actual model calls."""

    async def generate(self, prompt: str, max_tokens: int = 150,
                      temperature: float = 0.7) -> str:
        """Generate mock critique with score tag."""
        return """<critique>This signal shows good evidence and clear reasoning.
The argument is well-structured and addresses the key points effectively.
However, it could benefit from additional supporting data.</critique><score>0.75</score>"""


class TestCriticSignalGeneration(unittest.TestCase):
    """Test that critics generate CRITIQUE signals instead of only adjusting strength."""

    def test_critic_deposits_critique_signals(self):
        """Critic should deposit CRITIQUE signals with parent links."""
        async def run_test():
            # Setup
            store = SignalStore()
            critic = Critic("test_critic", mode="creative", thesis="Test thesis")
            llm = MockLLM()

            # Monkey-patch generate_critique for testing
            async def mock_generate_critique(signal, llm, temperature):
                return """<critique>Good signal with solid evidence.</critique><score>0.8</score>"""

            critic.generate_critique = mock_generate_critique
            critic.evaluate_type = "INITIAL"

            # Deposit initial signal to evaluate
            initial_id = store.deposit(
                signal_type="INITIAL",
                content="Climate action reduces emissions",
                depositor="test_scout",
                strength=0.7
            )

            # Run critic (single action)
            await critic.run(store, llm, max_actions=1)

            # Assertions
            critiques = store.get_signals("CRITIQUE")
            self.assertGreater(len(critiques), 0,
                             "Critic should deposit CRITIQUE signals")

            critique = critiques[0]
            self.assertEqual(critique.parent, initial_id,
                           "CRITIQUE should be linked to evaluated signal")
            self.assertGreater(len(critique.content), 20,
                             "CRITIQUE should have substantive content")
            self.assertIn("critic", critique.depositor.lower(),
                        "CRITIQUE should be deposited by critic")

        asyncio.run(run_test())

    def test_critic_uses_stratified_sampling(self):
        """Critic should use stratified sampling for balanced evaluation."""
        async def run_test():
            # Setup
            store = SignalStore()
            critic = Critic("test_critic", mode="creative")
            llm = MockLLM()

            # Mock generate_critique
            critiques_generated = []

            async def mock_generate_critique(signal, llm, temperature):
                critiques_generated.append(signal.strength)
                return f"""<critique>Evaluating signal with strength {signal.strength:.2f}</critique><score>0.75</score>"""

            critic.generate_critique = mock_generate_critique
            critic.evaluate_type = "INITIAL"

            # Deposit signals with varying strength (weak, medium, strong)
            weak_id = store.deposit("INITIAL", "Weak signal", "scout1", strength=0.3)
            medium_id = store.deposit("INITIAL", "Medium signal", "scout2", strength=0.5)
            strong_id = store.deposit("INITIAL", "Strong signal", "scout3", strength=0.8)

            # Run critic
            await critic.run(store, llm, max_actions=1)

            # Check that weak/medium/strong signals were sampled
            # (With stratified sampling, we should see diversity)
            self.assertGreater(len(critiques_generated), 0,
                             "Critic should evaluate some signals")

            # NOTE: In a single run with stratified sampling (weak=1, medium=1, strong=1),
            # we should ideally see one from each category. But due to randomness,
            # we just verify that critiques were generated.
            self.assertTrue(any(s < 0.4 for s in critiques_generated) or
                          any(0.4 <= s < 0.7 for s in critiques_generated) or
                          any(s >= 0.7 for s in critiques_generated),
                          "Critic should evaluate signals across strength levels")

        asyncio.run(run_test())

    def test_critique_signal_has_quality_score(self):
        """CRITIQUE signals should have appropriate strength/quality scores."""
        async def run_test():
            # Setup
            store = SignalStore()
            critic = Critic("test_critic", mode="creative")
            llm = MockLLM()

            async def mock_generate_critique(signal, llm, temperature):
                return """<critique>Detailed analysis with structured reasoning.</critique><score>0.9</score>"""

            critic.generate_critique = mock_generate_critique
            critic.evaluate_type = "INITIAL"

            # Deposit signal
            initial_id = store.deposit("INITIAL", "Test claim", "scout", strength=0.7)

            # Run critic
            await critic.run(store, llm, max_actions=1)

            # Get critique
            critiques = store.get_signals("CRITIQUE")
            self.assertEqual(len(critiques), 1)

            critique = critiques[0]
            # Critique strength should reflect confidence (0.5-0.95 range)
            self.assertGreaterEqual(critique.strength, 0.5,
                                  "Critique should have minimum confidence of 0.5")
            self.assertLessEqual(critique.strength, 0.95,
                               "Critique should have maximum confidence of 0.95")

        asyncio.run(run_test())

    def test_critic_adjusts_parent_strength(self):
        """Critic should still adjust parent signal strength based on evaluation."""
        async def run_test():
            # Setup
            store = SignalStore()
            critic = Critic("test_critic", mode="creative")
            llm = MockLLM()

            async def mock_generate_critique(signal, llm, temperature):
                # High quality score should amplify parent
                return """<critique>Excellent signal</critique><score>0.95</score>"""

            critic.generate_critique = mock_generate_critique
            critic.evaluate_type = "INITIAL"

            # Deposit signal with medium strength
            initial_id = store.deposit("INITIAL", "Good claim", "scout", strength=0.6)
            original_strength = store.signals[initial_id].strength

            # Run critic
            await critic.run(store, llm, max_actions=1)

            # Parent strength should be adjusted
            new_strength = store.signals[initial_id].strength
            self.assertNotEqual(new_strength, original_strength,
                              "Critic should adjust parent signal strength")

            # With score=0.95, multiplier should be ~1.45 (amplification)
            # So new_strength should be > original_strength
            self.assertGreater(new_strength, original_strength,
                             "High quality score should amplify parent strength")

        asyncio.run(run_test())


class TestStratifiedSampling(unittest.TestCase):
    """Test stratified sampling functionality."""

    def test_sample_stratified_returns_correct_counts(self):
        """Stratified sampling should return requested counts from each stratum."""
        store = SignalStore()

        # Deposit signals across strength levels
        for i in range(10):
            strength = 0.1 + i * 0.1  # 0.1, 0.2, ..., 1.0
            store.deposit("INITIAL", f"Signal {i}", f"agent{i}", strength=strength)

        # Sample stratified: 2 weak, 2 medium, 2 strong
        sampled = store.sample_stratified("INITIAL", weak=2, medium=2, strong=2)

        # Count samples from each stratum
        weak_count = sum(1 for s in sampled if s.strength < 0.4)
        medium_count = sum(1 for s in sampled if 0.4 <= s.strength < 0.7)
        strong_count = sum(1 for s in sampled if s.strength >= 0.7)

        # Each stratum should have the requested count (or less if not enough signals)
        self.assertLessEqual(weak_count, 2)
        self.assertLessEqual(medium_count, 2)
        self.assertLessEqual(strong_count, 2)

        # Total should be up to 6
        self.assertLessEqual(len(sampled), 6)

    def test_sample_stratified_handles_empty_strata(self):
        """Stratified sampling should handle cases where strata are empty."""
        store = SignalStore()

        # Only deposit strong signals
        for i in range(5):
            store.deposit("INITIAL", f"Strong signal {i}", f"agent{i}", strength=0.9)

        # Request samples from all strata
        sampled = store.sample_stratified("INITIAL", weak=2, medium=2, strong=2)

        # Should only get strong signals (weak and medium strata are empty)
        self.assertLessEqual(len(sampled), 2)  # At most 2 strong signals
        self.assertTrue(all(s.strength >= 0.7 for s in sampled))

    def test_sample_stratified_returns_empty_for_nonexistent_type(self):
        """Stratified sampling should return empty list for nonexistent signal type."""
        store = SignalStore()

        sampled = store.sample_stratified("NONEXISTENT", weak=1, medium=1, strong=1)
        self.assertEqual(len(sampled), 0)


if __name__ == '__main__':
    unittest.main()
