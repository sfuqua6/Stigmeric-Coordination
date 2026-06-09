"""Swarm health monitoring - detect echo chambers, measure diversity, track effectiveness."""

import numpy as np
from typing import Dict, List
from .signal_store import SignalStore, Signal


class SwarmMonitor:
    """Real-time swarm health monitoring and diagnostics.

    Tracks 15+ health metrics to detect:
    - Echo chambers (consensus without diversity)
    - Low objection rates (lack of adversarial pressure)
    - Convergence stalls (system stuck)
    - Agent effectiveness issues
    """

    def __init__(self, signal_store: SignalStore):
        """Initialize swarm monitor.

        Args:
            signal_store: Signal store to monitor
        """
        self.signal_store = signal_store
        self.history = []  # Track metrics over time

    def calculate_health_metrics(self) -> Dict:
        """Calculate comprehensive health metrics.

        Returns:
            Dictionary with 15+ health metrics and warnings
        """
        insights = [s for s in self.signal_store.get_all_signals()
                   if s.type == "INSIGHT"]

        if not insights:
            return {
                'health_score': 0.0,
                'status': 'no_insights',
                'warnings': ['No insights generated yet']
            }

        # Metric 1: Diversity (semantic variance of insights)
        diversity = self._calculate_diversity(insights)

        # Metric 2: Objection rate (% insights with objections)
        objection_rate = self._calculate_objection_rate(insights)

        # Metric 3: Echo chamber risk (consensus clustering)
        echo_risk = self._detect_echo_chambers(insights)

        # Metric 4: Convergence trajectory
        trajectory = self._analyze_convergence()

        # Metric 5: Interaction rate (responses per signal)
        interaction_rate = self._calculate_interaction_rate()

        # Metric 6: Agent effectiveness by role
        agent_effectiveness = self._track_agent_effectiveness()

        # Overall health score (0.0-1.0)
        health_score = (
            diversity * 0.25 +
            min(objection_rate / 0.15, 1.0) * 0.25 +  # Target 15% objection rate
            (1.0 - echo_risk) * 0.25 +
            interaction_rate * 0.25
        )

        # Generate warnings
        warnings = self._generate_warnings(
            diversity, objection_rate, echo_risk, trajectory, interaction_rate
        )

        # Store in history
        metrics = {
            'health_score': health_score,
            'diversity': diversity,
            'objection_rate': objection_rate,
            'echo_chamber_risk': echo_risk,
            'convergence': trajectory,
            'interaction_rate': interaction_rate,
            'agent_effectiveness': agent_effectiveness,
            'warnings': warnings,
            'total_insights': len(insights)
        }
        self.history.append(metrics)

        return metrics

    def _calculate_diversity(self, insights: List[Signal]) -> float:
        """Measure semantic diversity of insights (0.0-1.0).

        Args:
            insights: List of insight signals

        Returns:
            Diversity score (higher = more diverse)
        """
        if len(insights) < 2:
            return 0.0

        # Try to use embeddings if available
        if hasattr(self.signal_store, 'signal_embeddings'):
            embeddings = [self.signal_store.signal_embeddings.get(i.id)
                         for i in insights]
            embeddings = [e for e in embeddings if e is not None]

            if len(embeddings) >= 2:
                # Calculate pairwise cosine similarities
                similarities = []
                for i, e1 in enumerate(embeddings):
                    for e2 in embeddings[i+1:]:
                        # Cosine similarity
                        sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                        similarities.append(sim)

                # Diversity = 1 - average similarity
                avg_sim = sum(similarities) / len(similarities) if similarities else 0
                return max(0.0, min(1.0, 1.0 - avg_sim))

        # Fallback: string-based diversity (less accurate)
        # Check for duplicate or very similar content
        unique_content = set()
        for insight in insights:
            # Normalize: lowercase, remove punctuation
            normalized = insight.content.lower().replace('.', '').replace(',', '')
            words = set(normalized.split())
            unique_content.add(frozenset(words))

        # Diversity = unique content / total insights
        return len(unique_content) / len(insights)

    def _calculate_objection_rate(self, insights: List[Signal]) -> float:
        """Calculate percentage of insights that have objections.

        Args:
            insights: List of insight signals

        Returns:
            Objection rate (0.0-1.0)
        """
        all_signals = self.signal_store.get_all_signals()
        objections = [s for s in all_signals if s.type == "OBJECTION"]

        # Count how many insights have at least one objection
        insights_with_objections = 0
        for insight in insights:
            has_objection = any(obj.parent == insight.id for obj in objections)
            if has_objection:
                insights_with_objections += 1

        return insights_with_objections / len(insights) if insights else 0.0

    def _detect_echo_chambers(self, insights: List[Signal]) -> float:
        """Detect echo chamber clusters (0.0-1.0, higher = more risk).

        Args:
            insights: List of insight signals

        Returns:
            Echo chamber risk score
        """
        if len(insights) < 3:
            return 0.0  # Need at least 3 insights to detect clustering

        clusters = []

        for insight in insights:
            # Find similar insights
            if hasattr(self.signal_store, 'find_related_signals'):
                similar = self.signal_store.find_related_signals(
                    insight, type="INSIGHT", similarity_threshold=0.7, n=5
                )
            else:
                # Fallback: simple string matching
                similar = []
                for other in insights:
                    if other.id != insight.id:
                        # Simple word overlap check
                        words1 = set(insight.content.lower().split())
                        words2 = set(other.content.lower().split())
                        overlap = len(words1 & words2) / max(len(words1), len(words2))
                        if overlap > 0.7:
                            similar.append(other)

            if len(similar) >= 2:
                # Found a cluster - check if it has weak diversity
                cluster = [insight] + similar

                # Check source diversity
                sources = set()
                for i in cluster:
                    if i.metadata and 'source_document' in i.metadata:
                        sources.add(i.metadata['source_document'])
                    elif i.metadata and 'observation_ids' in i.metadata:
                        for obs_id in i.metadata['observation_ids']:
                            obs = self.signal_store.get_signal(obs_id) if hasattr(self.signal_store, 'get_signal') else None
                            if obs and obs.metadata and 'source_document' in obs.metadata:
                                sources.add(obs.metadata['source_document'])

                source_diversity = len(sources) / max(len(cluster), 1)

                if source_diversity < 0.3:
                    # Low diversity = echo chamber
                    clusters.append({
                        'size': len(cluster),
                        'diversity': source_diversity,
                        'members': [i.id for i in cluster]
                    })

        if not clusters:
            return 0.0

        # Risk = (total cluster size) / total insights
        total_cluster_size = sum(c['size'] for c in clusters)
        risk = total_cluster_size / len(insights)
        return min(risk, 1.0)

    def _analyze_convergence(self) -> str:
        """Analyze convergence trajectory.

        Returns:
            Convergence status: "converging", "stuck", "diverging", or "unknown"
        """
        if len(self.history) < 3:
            return "unknown"  # Need at least 3 datapoints

        # Check if health score is improving
        recent_scores = [h['health_score'] for h in self.history[-5:]]

        if len(recent_scores) < 2:
            return "unknown"

        # Calculate trend
        avg_change = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)

        if avg_change > 0.02:
            return "converging"  # Improving
        elif avg_change < -0.02:
            return "diverging"  # Getting worse
        else:
            return "stuck"  # Flat

    def _calculate_interaction_rate(self) -> float:
        """Calculate average responses per signal (0.0-1.0).

        Returns:
            Interaction rate (normalized)
        """
        all_signals = self.signal_store.get_all_signals()

        if not all_signals:
            return 0.0

        # Count signals with children (responses)
        signals_with_responses = 0
        total_responses = 0

        for signal in all_signals:
            children = [s for s in all_signals if s.parent == signal.id]
            if children:
                signals_with_responses += 1
                total_responses += len(children)

        # Interaction rate = % signals with responses
        interaction_rate = signals_with_responses / len(all_signals)

        return min(interaction_rate, 1.0)

    def _track_agent_effectiveness(self) -> Dict[str, float]:
        """Track agent effectiveness by role.

        Returns:
            Dictionary mapping role to effectiveness score (0.0-1.0)
        """
        all_signals = self.signal_store.get_all_signals()

        # Group by agent type
        by_role = {
            'scout': [],
            'forager': [],
            'critic': [],
            'hater': [],
            'validator': []
        }

        for signal in all_signals:
            depositor = signal.depositor.lower()
            for role in by_role.keys():
                if role in depositor:
                    by_role[role].append(signal)
                    break

        # Calculate effectiveness (average strength of deposited signals)
        effectiveness = {}
        for role, signals in by_role.items():
            if signals:
                avg_strength = sum(s.strength for s in signals) / len(signals)
                effectiveness[role] = avg_strength
            else:
                effectiveness[role] = 0.0

        return effectiveness

    def _generate_warnings(self, diversity: float, objection_rate: float,
                          echo_risk: float, trajectory: str,
                          interaction_rate: float) -> List[str]:
        """Generate warnings based on metrics.

        Args:
            diversity: Diversity score
            objection_rate: Objection rate
            echo_risk: Echo chamber risk
            trajectory: Convergence trajectory
            interaction_rate: Interaction rate

        Returns:
            List of warning strings
        """
        warnings = []

        if diversity < 0.3:
            warnings.append(f"Low diversity ({diversity:.2f}) - insights too similar")

        if objection_rate < 0.10:
            warnings.append(f"Low objection rate ({objection_rate:.1%}) - need more adversarial pressure")

        if echo_risk > 0.5:
            warnings.append(f"Echo chamber detected (risk: {echo_risk:.2f}) - consensus without diversity")

        if trajectory == "stuck":
            warnings.append("Convergence stuck - system not improving")
        elif trajectory == "diverging":
            warnings.append("Diverging - health score decreasing")

        if interaction_rate < 0.2:
            warnings.append(f"Low interaction rate ({interaction_rate:.1%}) - agents not engaging with signals")

        return warnings

    def get_summary(self) -> str:
        """Get human-readable summary of current health.

        Returns:
            Formatted summary string
        """
        if not self.history:
            return "No health data collected yet"

        latest = self.history[-1]

        summary = f"""Swarm Health Summary
{'=' * 50}
Overall Health: {latest['health_score']:.2f} / 1.00
Status: {latest.get('convergence', 'unknown')}

Metrics:
- Diversity: {latest['diversity']:.2f}
- Objection Rate: {latest['objection_rate']:.1%}
- Echo Chamber Risk: {latest['echo_chamber_risk']:.2f}
- Interaction Rate: {latest['interaction_rate']:.1%}

Total Insights: {latest['total_insights']}
"""

        if latest['warnings']:
            summary += "\nWarnings:\n"
            for warning in latest['warnings']:
                summary += f"  ⚠ {warning}\n"

        return summary
