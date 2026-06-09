#!/usr/bin/env python3
"""Pipeline Sanity Test - Fast validation without model loading.

Tests core swarm components with hardcoded expected outputs.
This validates the pipeline is functional before running the full system.
"""

import asyncio
import sys
import time
from typing import List
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    expected: str
    got: str
    error: str = ""

    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        result = f"{status} | {self.name}\n"
        result += f"  Expected: {self.expected}\n"
        result += f"  Got: {self.got}"
        if self.error:
            result += f"\n  Error: {self.error}"
        return result


class PipelineTests:
    """Pipeline sanity tests."""

    def __init__(self):
        self.results: List[TestResult] = []

    def add_result(self, name: str, passed: bool, expected: str, got: str, error: str = ""):
        """Add test result."""
        self.results.append(TestResult(name, passed, expected, got, error))

    async def test_signal_creation(self):
        """Test signal creation."""
        from swarm.core.signal_store import Signal

        try:
            signal = Signal(
                id="test_001",
                type="HYPOTHESIS",
                content="Test hypothesis",
                strength=0.8,
                timestamp=time.time(),
                depositor="scout_1"
            )

            checks = [
                (signal.id == "test_001", "id"),
                (signal.type == "HYPOTHESIS", "type"),
                (signal.content == "Test hypothesis", "content"),
                (signal.strength == 0.8, "strength"),
                (signal.depositor == "scout_1", "depositor"),
                (hasattr(signal, 'visits'), "has visits"),
                (hasattr(signal, 'metadata'), "has metadata"),
            ]

            passed = all(c[0] for c in checks)
            failed = [c[1] for c in checks if not c[0]]

            self.add_result(
                "Signal Creation",
                passed,
                "Signal(id, type, content, strength, timestamp, depositor, visits, metadata)",
                "All attributes present" if passed else f"Failed: {failed}",
                "" if passed else f"Missing: {failed}"
            )

        except Exception as e:
            self.add_result("Signal Creation", False, "Signal created", "Exception", str(e))

    async def test_signal_store_operations(self):
        """Test signal store operations."""
        from swarm.core.signal_store import SignalStore

        try:
            store = SignalStore()

            # Add signals
            id1 = store.deposit("HYPOTHESIS", "Claim 1", 0.9, "scout_1")
            id2 = store.deposit("EVIDENCE", "Evidence 1", 0.8, "forager_1")
            id3 = store.deposit("CRITIQUE", "Critique 1", 0.7, "critic_1")

            # Get all signals
            all_signals = store.get_all_signals()

            # Count by type
            hypotheses = [s for s in all_signals if s.type == "HYPOTHESIS"]
            evidence = [s for s in all_signals if s.type == "EVIDENCE"]
            critiques = [s for s in all_signals if s.type == "CRITIQUE"]

            checks = [
                (len(all_signals) == 3, "3 total"),
                (len(hypotheses) == 1, "1 hypothesis"),
                (len(evidence) == 1, "1 evidence"),
                (len(critiques) == 1, "1 critique"),
            ]

            passed = all(c[0] for c in checks)
            failed = [c[1] for c in checks if not c[0]]

            self.add_result(
                "Signal Store Operations",
                passed,
                "Add 3 signals, retrieve all and by type",
                f"{len(all_signals)} total: {len(hypotheses)}H, {len(evidence)}E, {len(critiques)}C",
                "" if passed else f"Failed: {failed}"
            )

        except Exception as e:
            self.add_result("Signal Store Operations", False, "Operations work", "Exception", str(e))

    async def test_signal_strength_decay(self):
        """Test signal decay."""
        from swarm.core.signal_store import SignalStore

        try:
            store = SignalStore(decay_rate=0.1)

            # Add signal
            sig_id = store.deposit("HYPOTHESIS", "Test", 1.0, "agent_1")

            # Get initial
            all_sigs = store.get_all_signals()
            initial_strength = all_sigs[0].strength

            # Decay
            store.decay_all()

            # Get after decay
            all_sigs_after = store.get_all_signals()
            final_strength = all_sigs_after[0].strength

            checks = [
                (initial_strength == 1.0, "initial = 1.0"),
                (final_strength < initial_strength, "strength decreased"),
                (final_strength >= 0.0, "strength >= 0"),
                (abs(final_strength - 0.9) < 0.05, "~10% decay"),
            ]

            passed = all(c[0] for c in checks)
            failed = [c[1] for c in checks if not c[0]]

            self.add_result(
                "Signal Strength Decay",
                passed,
                "Decay reduces strength (1.0 → ~0.9)",
                f"{initial_strength:.2f} → {final_strength:.2f}",
                "" if passed else f"Failed: {failed}"
            )

        except Exception as e:
            self.add_result("Signal Strength Decay", False, "Decay works", "Exception", str(e))

    async def test_signal_amplification(self):
        """Test signal amplification."""
        from swarm.core.signal_store import SignalStore

        try:
            store = SignalStore()

            # Add signal
            sig_id = store.deposit("HYPOTHESIS", "Test", 0.5, "agent_1")

            # Get initial
            initial = store.get_all_signals()[0].strength

            # Amplify
            store.amplify(sig_id, factor=1.5)

            # Get final
            final = store.get_all_signals()[0].strength

            checks = [
                (initial == 0.5, "initial = 0.5"),
                (final > initial, "strength increased"),
                (final <= 1.0, "strength capped at 1.0"),
            ]

            passed = all(c[0] for c in checks)
            failed = [c[1] for c in checks if not c[0]]

            self.add_result(
                "Signal Amplification",
                passed,
                "Amplify increases strength (0.5 → >0.5, ≤1.0)",
                f"{initial:.2f} → {final:.2f}",
                "" if passed else f"Failed: {failed}"
            )

        except Exception as e:
            self.add_result("Signal Amplification", False, "Amplify works", "Exception", str(e))

    async def test_signal_hierarchy(self):
        """Test parent-child relationships."""
        from swarm.core.signal_store import SignalStore

        try:
            store = SignalStore()

            # Add parent
            parent_id = store.deposit("HYPOTHESIS", "Parent", 0.8, "scout_1")

            # Add child with parent
            child_id = store.deposit("EVIDENCE", "Child", 0.7, "forager_1", parent=parent_id)

            # Get signals
            all_sigs = store.get_all_signals()
            parent = next(s for s in all_sigs if s.id == parent_id)
            child = next(s for s in all_sigs if s.id == child_id)

            checks = [
                (parent.id == parent_id, "parent id"),
                (child.id == child_id, "child id"),
                (child.parent == parent_id, "child has parent ref"),
                (parent.parent is None, "parent has no parent"),
            ]

            passed = all(c[0] for c in checks)
            failed = [c[1] for c in checks if not c[0]]

            self.add_result(
                "Signal Hierarchy",
                passed,
                "Parent-child relationships work",
                f"Parent: {parent_id[:8]}, Child: {child_id[:8]} → parent={child.parent[:8] if child.parent else None}",
                "" if passed else f"Failed: {failed}"
            )

        except Exception as e:
            self.add_result("Signal Hierarchy", False, "Hierarchy works", "Exception", str(e))

    async def test_signal_pruning(self):
        """Test pruning weak signals."""
        from swarm.core.signal_store import SignalStore

        try:
            store = SignalStore(prune_threshold=0.3)

            # Add signals with different strengths
            strong = store.deposit("HYPOTHESIS", "Strong", 0.8, "agent_1")
            medium = store.deposit("HYPOTHESIS", "Medium", 0.4, "agent_2")
            weak = store.deposit("HYPOTHESIS", "Weak", 0.1, "agent_3")

            before = len(store.get_all_signals())

            # Prune
            pruned = store.prune_weak()

            after_sigs = store.get_all_signals()
            after = len(after_sigs)
            remaining_ids = [s.id for s in after_sigs]

            checks = [
                (before == 3, "started with 3"),
                (after == 2, "2 remain"),
                (strong in remaining_ids, "strong kept"),
                (medium in remaining_ids, "medium kept"),
                (weak not in remaining_ids, "weak removed"),
                (pruned == 1, "1 pruned"),
            ]

            passed = all(c[0] for c in checks)
            failed = [c[1] for c in checks if not c[0]]

            self.add_result(
                "Signal Pruning",
                passed,
                "Prune removes weak signals (threshold=0.3)",
                f"{before} → {after}, pruned {pruned}",
                "" if passed else f"Failed: {failed}"
            )

        except Exception as e:
            self.add_result("Signal Pruning", False, "Pruning works", "Exception", str(e))

    async def test_spatial_signal_store(self):
        """Test spatial queries (optional)."""
        try:
            from swarm.core.spatial_signal_store import SpatialSignalStore

            store = SpatialSignalStore(dimensions=100)

            # Add signals at positions (2D tuples)
            nearby1 = store.deposit("HYPOTHESIS", "Near 1", 0.8, "agent_1", position=(0.0, 0.0))
            nearby2 = store.deposit("HYPOTHESIS", "Near 2", 0.7, "agent_2", position=(0.1, 0.1))
            far = store.deposit("HYPOTHESIS", "Far", 0.9, "agent_3", position=(50.0, 50.0))

            # Get nearby
            nearby = store.get_nearby(position=(0.0, 0.0), radius=1.0)
            nearby_ids = [s.id for s in nearby]

            checks = [
                (len(nearby) == 2, "found 2 nearby"),
                (nearby1 in nearby_ids, "nearby1 found"),
                (nearby2 in nearby_ids, "nearby2 found"),
                (far not in nearby_ids, "far not found"),
            ]

            passed = all(c[0] for c in checks)
            failed = [c[1] for c in checks if not c[0]]

            self.add_result(
                "Spatial Signal Store",
                passed,
                "Spatial queries work (radius=1.0)",
                f"Found {len(nearby)}/3 nearby",
                "" if passed else f"Failed: {failed}"
            )

        except ImportError:
            self.add_result(
                "Spatial Signal Store",
                True,  # Optional - pass if not available
                "Spatial store (optional)",
                "Not available (skipped)",
                ""
            )
        except Exception as e:
            self.add_result("Spatial Signal Store", False, "Spatial works", "Exception", str(e))

    async def test_validation_symbolic(self):
        """Test symbolic validation."""
        try:
            from swarm.validation.real_validator import RealValidator
            from swarm.core.signal_store import Signal
            import time

            validator = RealValidator(agent_id="test_validator")

            # Create test signal
            signal = Signal(
                id="test_val_001",
                type="HYPOTHESIS",
                content="2 + 2 = 4",
                strength=0.8,
                timestamp=time.time(),
                depositor="test_agent"
            )

            # Validate signal
            result = await validator.validate_signal(signal)

            checks = [
                (result is not None, "result returned"),
                (hasattr(result, 'accuracy'), "has accuracy"),
                (hasattr(result, 'confidence'), "has confidence"),
                (hasattr(result, 'verified_claims'), "has verified_claims"),
                (result.accuracy >= 0.0 and result.accuracy <= 1.0, "accuracy in range"),
            ]

            passed = all(c[0] for c in checks)
            failed = [c[1] for c in checks if not c[0]]

            self.add_result(
                "Validation Integration",
                passed,
                "Validation returns ValidationResult with accuracy/confidence",
                f"Accuracy: {result.accuracy:.2f}, Confidence: {result.confidence:.2f}" if result else "None",
                "" if passed else f"Failed: {failed}"
            )

        except Exception as e:
            self.add_result("Validation Integration", False, "Validation works", "Exception", str(e))

    async def test_task_config(self):
        """Test task configuration."""
        from swarm.core.task_config import TaskConfig

        try:
            config = TaskConfig(
                task_type="test",
                task_prompt="Test prompt",
                signal_types={"initial": "CLAIM"},
                scout_prompt_template="Scout: {task_prompt}",
                forager_evidence_prompt_template="Evidence: {parent_content}",
                forager_critique_prompt_template="Critique: {parent_content}",
                critic_prompt_template="Critic: {parent_content}",
                hater_prompt_template="Hater: {parent_content}"
            )

            checks = [
                (config.task_type == "test", "task_type"),
                (config.task_prompt == "Test prompt", "task_prompt"),
                (hasattr(config, 'signal_types'), "has signal_types"),
                (hasattr(config, 'scout_prompt_template'), "has scout template"),
            ]

            passed = all(c[0] for c in checks)
            failed = [c[1] for c in checks if not c[0]]

            self.add_result(
                "Task Configuration",
                passed,
                "TaskConfig(task_type, task_prompt, signal_types, templates)",
                f"Type: {config.task_type}, Prompt: {len(config.task_prompt)} chars",
                "" if passed else f"Failed: {failed}"
            )

        except Exception as e:
            self.add_result("Task Configuration", False, "Config works", "Exception", str(e))

    async def test_round_coordinator(self):
        """Test round coordination."""
        from swarm.core.round_coordinator import RoundCoordinator

        try:
            coord = RoundCoordinator(num_rounds=3, iterations_per_round=10)

            # Test keyword extraction
            keywords = coord.extract_keywords_from_task("How can sustainable energy reduce climate change?")

            checks = [
                (coord.num_rounds == 3, "num_rounds = 3"),
                (coord.iterations_per_round == 10, "iterations = 10"),
                (isinstance(keywords, list), "keywords is list"),
                (len(keywords) > 0, "keywords not empty"),
                (all(isinstance(k, str) for k in keywords), "keywords are strings"),
            ]

            passed = all(c[0] for c in checks)
            failed = [c[1] for c in checks if not c[0]]

            self.add_result(
                "Round Coordinator",
                passed,
                "RoundCoordinator(num_rounds, iterations_per_round)",
                f"Rounds: {coord.num_rounds}, Keywords: {len(keywords)}",
                "" if passed else f"Failed: {failed}"
            )

        except Exception as e:
            self.add_result("Round Coordinator", False, "Coordinator works", "Exception", str(e))

    async def test_agent_metrics(self):
        """Test agent metrics tracking."""
        from swarm.core.agent_metrics import AgentPerformanceMetrics, AgentMetricsTracker
        from swarm.core.signal_store import SignalStore

        try:
            # Test dataclass
            metrics = AgentPerformanceMetrics(agent_id="test_agent", agent_type="Scout")

            # Test tracker
            tracker = AgentMetricsTracker()
            tracker.record_signal_deposit("agent_1", "Scout", "sig_001", 0.8, 0.9)
            tracker.record_signal_deposit("agent_1", "Scout", "sig_002", 0.7, 0.85)
            tracker.record_signal_rejection("agent_1", "Scout", "too weak", 0.3)

            agent_metrics = tracker.get_metrics("agent_1")

            checks = [
                (metrics.agent_id == "test_agent", "metrics.agent_id"),
                (metrics.agent_type == "Scout", "metrics.agent_type"),
                (agent_metrics is not None, "tracker found agent"),
                (agent_metrics.signals_deposited == 2, "2 deposited"),
                (agent_metrics.signals_rejected == 1, "1 rejected"),
            ]

            passed = all(c[0] for c in checks)
            failed = [c[1] for c in checks if not c[0]]

            self.add_result(
                "Agent Metrics",
                passed,
                "AgentPerformanceMetrics dataclass and AgentMetricsTracker work",
                f"Deposited: {agent_metrics.signals_deposited if agent_metrics else 0}, Rejected: {agent_metrics.signals_rejected if agent_metrics else 0}",
                "" if passed else f"Failed: {failed}"
            )

        except Exception as e:
            self.add_result("Agent Metrics", False, "Metrics work", "Exception", str(e))

    async def run_all_tests(self):
        """Run all tests."""
        tests = [
            self.test_signal_creation,
            self.test_signal_store_operations,
            self.test_signal_strength_decay,
            self.test_signal_amplification,
            self.test_signal_hierarchy,
            self.test_signal_pruning,
            self.test_spatial_signal_store,
            self.test_validation_symbolic,
            self.test_task_config,
            self.test_round_coordinator,
            self.test_agent_metrics,
        ]

        for test in tests:
            try:
                await test()
            except Exception as e:
                import traceback
                name = test.__name__.replace("test_", "").replace("_", " ").title()
                self.add_result(name, False, "Test completes", "Uncaught exception", f"{e}\n{traceback.format_exc()}")

    def print_results(self):
        """Print test results."""
        print("\nPipeline Sanity Test")
        print("=" * 60)
        print()

        for result in self.results:
            print(result)
            print()

        print("=" * 60)
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print(f"Results: {passed} passed, {total - passed} failed")
        print(f"Success rate: {100.0 * passed / total:.1f}%")
        print()

        return passed == total


async def main():
    """Run all tests."""
    tester = PipelineTests()

    print("\n🧪 Pipeline Sanity Test")
    print("Fast validation of core components (no model loading required)\n")

    await tester.run_all_tests()
    all_passed = tester.print_results()

    if all_passed:
        print("✅ ALL TESTS PASSED - Pipeline ready!")
        print("\nCore systems validated:")
        print("  • Signal store operations")
        print("  • Signal strength mechanics")
        print("  • Parent-child relationships")
        print("  • Validation integration")
        print("  • Configuration system")
        print("  • Coordination and metrics")
        print("\nReady to run full swarm with LLM!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
