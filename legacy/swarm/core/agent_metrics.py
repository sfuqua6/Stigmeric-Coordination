"""Agent effectiveness metrics for swarm optimization.

This module tracks individual agent performance and calculates role-specific
effectiveness metrics to identify top/bottom performers and optimize swarm composition.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from ..core.signal_store import SignalStore, Signal


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for an individual agent."""
    agent_id: str
    agent_type: str

    # Activity metrics
    signals_deposited: int = 0
    signals_rejected: int = 0  # Failed verification
    responses_generated: int = 0

    # Quality metrics
    avg_signal_strength: float = 0.0
    avg_signal_visits: float = 0.0  # How often others interact with this agent's signals
    avg_verification_score: float = 0.0

    # Impact metrics
    signals_amplified_by_others: int = 0  # Others corroborated
    signals_decayed_by_others: int = 0  # Others challenged
    dialogue_threads_started: int = 0
    dialogue_depth: float = 0.0  # Avg responses per signal

    # Role-specific metrics
    role_effectiveness: float = 0.0  # 0.0-1.0

    # Computed score
    overall_effectiveness: float = 0.0


class AgentMetricsTracker:
    """Track agent effectiveness metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}

    def record_signal_deposit(self, agent_id: str, agent_type: str,
                             signal_id: str, strength: float,
                             verification_score: Optional[float] = None):
        """Record that an agent deposited a signal.

        Args:
            agent_id: Agent ID
            agent_type: Agent type (Scout, Forager, Gatherer, Critic, Hater)
            signal_id: Signal ID that was deposited
            strength: Signal strength
            verification_score: Optional verification quality score
        """
        metrics = self._get_or_create_metrics(agent_id, agent_type)
        metrics.signals_deposited += 1

        # Update averages
        total = metrics.signals_deposited + metrics.signals_rejected
        if total > 0:
            metrics.avg_signal_strength = (
                (metrics.avg_signal_strength * (total - 1) + strength) / total
            )

            if verification_score is not None:
                metrics.avg_verification_score = (
                    (metrics.avg_verification_score * (total - 1) + verification_score) / total
                )

    def record_signal_rejection(self, agent_id: str, agent_type: str,
                               reason: str, verification_score: float):
        """Record that an agent's signal was rejected.

        Args:
            agent_id: Agent ID
            agent_type: Agent type
            reason: Rejection reason
            verification_score: Verification quality score
        """
        metrics = self._get_or_create_metrics(agent_id, agent_type)
        metrics.signals_rejected += 1

        # Update averages
        total = metrics.signals_deposited + metrics.signals_rejected
        if total > 0:
            metrics.avg_verification_score = (
                (metrics.avg_verification_score * (total - 1) + verification_score) / total
            )

    def record_response(self, agent_id: str, agent_type: str,
                       response_to: str):
        """Record that an agent responded to another signal.

        Args:
            agent_id: Agent ID
            agent_type: Agent type
            response_to: Signal ID that was responded to
        """
        metrics = self._get_or_create_metrics(agent_id, agent_type)
        metrics.responses_generated += 1

    def calculate_effectiveness(self, agent_id: str, agent_type: str,
                               signal_store: SignalStore) -> float:
        """Calculate role-specific effectiveness.

        Args:
            agent_id: Agent ID
            agent_type: Agent type
            signal_store: Signal store for analysis

        Returns:
            Overall effectiveness score (0.0-1.0)
        """
        metrics = self._get_or_create_metrics(agent_id, agent_type)

        # Get agent's signals
        agent_signals = [s for s in signal_store.get_all_signals()
                        if s.depositor == agent_id]

        if not agent_signals:
            return 0.0

        # Update visit metrics
        total_visits = sum(s.visits for s in agent_signals)
        metrics.avg_signal_visits = total_visits / len(agent_signals)

        # Update dialogue metrics
        signals_with_responses = [s for s in agent_signals
                                 if signal_store.get_responses(s.id)]
        metrics.dialogue_threads_started = len(signals_with_responses)

        if signals_with_responses:
            total_responses = sum(len(signal_store.get_responses(s.id))
                                for s in signals_with_responses)
            metrics.dialogue_depth = total_responses / len(signals_with_responses)

        # Calculate role-specific effectiveness
        if agent_type == "Scout":
            # Scouts: effectiveness = observation quality + diversity
            effectiveness = self._calculate_scout_effectiveness(agent_signals, signal_store)
        elif agent_type == "Forager":
            # Foragers: effectiveness = insight quality + novelty + validation
            effectiveness = self._calculate_forager_effectiveness(agent_signals, signal_store)
        elif agent_type == "Gatherer":
            # Gatherers: effectiveness = evidence quality + validation rate
            effectiveness = self._calculate_gatherer_effectiveness(agent_signals, signal_store)
        elif agent_type == "Critic":
            # Critics: effectiveness = accuracy + calibration
            effectiveness = self._calculate_critic_effectiveness(agent_signals, signal_store)
        elif agent_type == "Hater":
            # Haters: effectiveness = objection impact + dialogue engagement
            effectiveness = self._calculate_hater_effectiveness(agent_signals, signal_store)
        else:
            effectiveness = 0.5

        metrics.role_effectiveness = effectiveness

        # Overall effectiveness
        metrics.overall_effectiveness = (
            metrics.role_effectiveness * 0.5 +
            metrics.avg_verification_score * 0.3 +
            min(metrics.dialogue_depth / 2.0, 1.0) * 0.2
        )

        return metrics.overall_effectiveness

    def _calculate_scout_effectiveness(self, signals: List[Signal],
                                      signal_store: SignalStore) -> float:
        """Scout effectiveness = observation quality + contribution to insights.

        Args:
            signals: Agent's signals
            signal_store: Signal store

        Returns:
            Effectiveness score (0.0-1.0)
        """
        if not signals:
            return 0.0

        # Check how many insights reference these observations
        observation_ids = [s.id for s in signals]
        insights = signal_store.get_all_signals()
        insights = [i for i in insights if i.type == "INSIGHT"]

        referenced_count = 0
        for obs_id in observation_ids:
            for insight in insights:
                obs_refs = insight.metadata.get('observation_ids', [])
                if obs_id in obs_refs:
                    referenced_count += 1
                    break  # Count once per observation

        reference_rate = referenced_count / len(observation_ids) if observation_ids else 0.0

        # Scout effectiveness = verification score + reference rate
        avg_score = sum(s.strength for s in signals) / len(signals)

        return avg_score * 0.5 + reference_rate * 0.5

    def _calculate_forager_effectiveness(self, signals: List[Signal],
                                        signal_store: SignalStore) -> float:
        """Forager effectiveness = insight quality + validation + novelty.

        Args:
            signals: Agent's signals
            signal_store: Signal store

        Returns:
            Effectiveness score (0.0-1.0)
        """
        insights = [s for s in signals if s.type == "INSIGHT"]
        if not insights:
            return 0.0

        # Quality = avg strength
        avg_strength = sum(i.strength for i in insights) / len(insights)

        # Validation = % with evidence
        validated_count = 0
        for insight in insights:
            evidence = signal_store.get_descendants(insight.id, "EVIDENCE")
            if len(evidence) >= 2:
                validated_count += 1
        validation_rate = validated_count / len(insights)

        # Novelty = diversity score (how different from other insights)
        # Higher visits = more novel/interesting
        avg_visits = sum(i.visits for i in insights) / len(insights)
        novelty_score = min(avg_visits / 5.0, 1.0)

        return avg_strength * 0.4 + validation_rate * 0.4 + novelty_score * 0.2

    def _calculate_gatherer_effectiveness(self, signals: List[Signal],
                                         signal_store: SignalStore) -> float:
        """Gatherer effectiveness = evidence quality + relevance.

        Args:
            signals: Agent's signals
            signal_store: Signal store

        Returns:
            Effectiveness score (0.0-1.0)
        """
        evidence = [s for s in signals if s.type == "EVIDENCE"]
        if not evidence:
            return 0.0

        # Quality = avg strength
        avg_strength = sum(e.strength for e in evidence) / len(evidence)

        # Relevance = % that actually helped validate insights
        # (Check if parent insights have high strength)
        relevant_count = 0
        for ev in evidence:
            if ev.parent:
                parent = signal_store.get_signal(ev.parent)
                if parent and parent.strength >= 0.7:
                    relevant_count += 1
        relevance_rate = relevant_count / len(evidence)

        return avg_strength * 0.5 + relevance_rate * 0.5

    def _calculate_critic_effectiveness(self, signals: List[Signal],
                                       signal_store: SignalStore) -> float:
        """Critic effectiveness = calibration (how accurate were strength adjustments).

        Args:
            signals: Agent's signals
            signal_store: Signal store

        Returns:
            Effectiveness score (0.0-1.0)
        """
        # This is tricky - need to track if critic's adjustments were correct
        # For now, use dialogue engagement as proxy
        critiques = [s for s in signals if s.type == "CRITIQUE"]
        if not critiques:
            return 0.0

        # Check if critiques sparked dialogue
        engaged_count = 0
        for critique in critiques:
            responses = signal_store.get_responses(critique.id)
            if responses:
                engaged_count += 1

        engagement_rate = engaged_count / len(critiques)

        # Check avg strength (higher = more confident)
        avg_strength = sum(c.strength for c in critiques) / len(critiques)

        return engagement_rate * 0.6 + avg_strength * 0.4

    def _calculate_hater_effectiveness(self, signals: List[Signal],
                                      signal_store: SignalStore) -> float:
        """Hater effectiveness = objection impact + dialogue engagement.

        Args:
            signals: Agent's signals
            signal_store: Signal store

        Returns:
            Effectiveness score (0.0-1.0)
        """
        objections = [s for s in signals if s.type in ["OBJECTION", "COUNTER_EVIDENCE"]]
        if not objections:
            return 0.0

        # Impact = did target's strength decrease?
        impact_count = 0
        for objection in objections:
            if objection.parent:
                target = signal_store.get_signal(objection.parent)
                if target:
                    # Check if target's strength is below average
                    # (hard to track historical strength without metadata)
                    if target.strength < 0.6:
                        impact_count += 1

        impact_rate = impact_count / len(objections)

        # Engagement = did anyone respond?
        engaged_count = 0
        for objection in objections:
            responses = signal_store.get_responses(objection.id)
            if responses:
                engaged_count += 1

        engagement_rate = engaged_count / len(objections)

        # Hater effectiveness = impact + engagement
        return impact_rate * 0.6 + engagement_rate * 0.4

    def _get_or_create_metrics(self, agent_id: str,
                               agent_type: str) -> AgentPerformanceMetrics:
        """Get or create metrics for an agent.

        Args:
            agent_id: Agent ID
            agent_type: Agent type

        Returns:
            Agent metrics object
        """
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentPerformanceMetrics(
                agent_id=agent_id,
                agent_type=agent_type
            )
        return self.agent_metrics[agent_id]

    def get_metrics(self, agent_id: str) -> Optional[AgentPerformanceMetrics]:
        """Get metrics for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Agent metrics or None if not found
        """
        return self.agent_metrics.get(agent_id)

    def get_all_metrics(self) -> List[AgentPerformanceMetrics]:
        """Get all agent metrics.

        Returns:
            List of all agent metrics
        """
        return list(self.agent_metrics.values())

    def print_report(self):
        """Print effectiveness report for all agents."""
        print("\n" + "=" * 80)
        print("AGENT EFFECTIVENESS REPORT")
        print("=" * 80)

        # Group by agent type
        by_type: Dict[str, List[AgentPerformanceMetrics]] = {}
        for metrics in self.agent_metrics.values():
            if metrics.agent_type not in by_type:
                by_type[metrics.agent_type] = []
            by_type[metrics.agent_type].append(metrics)

        for agent_type, agents in by_type.items():
            print(f"\n--- {agent_type}s ---")

            # Sort by effectiveness
            agents.sort(key=lambda a: a.overall_effectiveness, reverse=True)

            # Print top 3 and bottom 3
            print("\nTop performers:")
            for metrics in agents[:3]:
                print(f"  {metrics.agent_id}:")
                print(f"    Overall effectiveness: {metrics.overall_effectiveness:.2f}")
                print(f"    Role effectiveness: {metrics.role_effectiveness:.2f}")
                print(f"    Signals deposited: {metrics.signals_deposited}")
                print(f"    Signals rejected: {metrics.signals_rejected}")
                print(f"    Avg strength: {metrics.avg_signal_strength:.2f}")
                print(f"    Dialogue depth: {metrics.dialogue_depth:.2f}")

            if len(agents) > 6:
                print("\nBottom performers:")
                for metrics in agents[-3:]:
                    print(f"  {metrics.agent_id}:")
                    print(f"    Overall effectiveness: {metrics.overall_effectiveness:.2f}")
                    print(f"    Role effectiveness: {metrics.role_effectiveness:.2f}")
                    print(f"    Signals deposited: {metrics.signals_deposited}")

            # Type averages
            avg_effectiveness = sum(a.overall_effectiveness for a in agents) / len(agents)
            print(f"\n{agent_type} average effectiveness: {avg_effectiveness:.2f}")

        print("=" * 80 + "\n")
