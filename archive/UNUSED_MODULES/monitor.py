"""Meta-agent for swarm health monitoring and diagnostics."""

import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from ..core.signal_store import SignalStore, Signal


@dataclass
class SwarmHealthMetrics:
    """Comprehensive swarm health metrics."""
    # Signal statistics
    total_signals: int
    signals_by_type: Dict[str, int]
    avg_strength: float
    strength_gini: float

    # Diversity metrics
    insight_diversity: float  # 0.0-1.0, how diverse are insights?
    evidence_diversity: float
    source_diversity: float

    # Interaction metrics
    objection_rate: float  # objections per insight
    validation_rate: float  # evidence per insight
    response_rate: float  # descendants per signal

    # Convergence metrics
    convergence_score: float  # 0.0-1.0, based on strength stability
    trajectory: str  # "converging", "oscillating", "stuck", "diverging"

    # Echo chamber detection
    echo_chamber_risk: float  # 0.0-1.0, based on consensus clustering
    groupthink_clusters: List[List[str]]  # Lists of similar insight IDs

    # Health flags
    is_healthy: bool
    warnings: List[str]
    critical_issues: List[str]


class SwarmMonitor:
    """Meta-agent that monitors swarm health and provides diagnostics."""

    def __init__(self, agent_id: str = "SwarmMonitor"):
        """Initialize monitor.

        Args:
            agent_id: Agent identifier
        """
        self.agent_id = agent_id
        self.health_history: List[SwarmHealthMetrics] = []
        self.monitoring_interval: float = 10.0  # Check every 10 seconds
        self.active = True

    async def run(self, signal_store: SignalStore,
                  max_iterations: int = 100) -> None:
        """Monitor swarm health continuously.

        Args:
            signal_store: Signal store to monitor
            max_iterations: Maximum monitoring cycles
        """
        print(f"\n[{self.agent_id}] 🔍 Starting health monitoring")
        print(f"[{self.agent_id}] Monitoring interval: {self.monitoring_interval}s")

        for iteration in range(max_iterations):
            if not self.active:
                break

            # Collect metrics
            metrics = self.collect_metrics(signal_store)
            self.health_history.append(metrics)

            # Analyze health and trigger self-healing if needed
            if not metrics.is_healthy:
                print(f"\n[{self.agent_id}] ⚠️  SWARM HEALTH WARNING")
                for warning in metrics.warnings:
                    print(f"[{self.agent_id}]   - {warning}")
                for issue in metrics.critical_issues:
                    print(f"[{self.agent_id}]   ❌ CRITICAL: {issue}")

                # Trigger self-healing actions
                await self.trigger_self_healing(signal_store, metrics)
            else:
                if iteration % 5 == 0:  # Report every 5 checks
                    print(f"\n[{self.agent_id}] ✓ Swarm healthy (cycle {iteration})")
                    print(f"[{self.agent_id}]   Diversity: {metrics.insight_diversity:.2f}")
                    print(f"[{self.agent_id}]   Objection rate: {metrics.objection_rate:.2f}")
                    print(f"[{self.agent_id}]   Echo risk: {metrics.echo_chamber_risk:.2f}")
                    print(f"[{self.agent_id}]   Trajectory: {metrics.trajectory}")

            await asyncio.sleep(self.monitoring_interval)

        print(f"\n[{self.agent_id}] Monitoring complete. Collected {len(self.health_history)} snapshots.")

    def collect_metrics(self, signal_store: SignalStore) -> SwarmHealthMetrics:
        """Collect comprehensive swarm health metrics.

        Args:
            signal_store: Signal store to analyze

        Returns:
            SwarmHealthMetrics with all collected data
        """
        stats = signal_store.get_stats()
        all_signals = signal_store.get_all_signals()

        # Calculate diversity metrics
        insights = [s for s in all_signals if s.type == "INSIGHT"]
        insight_diversity = self.calculate_diversity(insights)

        evidence = [s for s in all_signals if s.type == "EVIDENCE"]
        evidence_diversity = self.calculate_diversity(evidence)

        source_diversity = self.calculate_source_diversity(insights, signal_store)

        # Calculate interaction metrics
        objections = [s for s in all_signals if s.type in ["OBJECTION", "COUNTER_EVIDENCE"]]
        objection_rate = len(objections) / max(len(insights), 1)

        validation_rate = len(evidence) / max(len(insights), 1)

        # Calculate response rate (avg descendants per signal)
        total_descendants = 0
        for signal in all_signals:
            descendants = signal_store.get_descendants(signal.id)
            total_descendants += len(descendants)
        response_rate = total_descendants / max(len(all_signals), 1)

        # Detect echo chambers
        groupthink_clusters = self.detect_echo_chambers(insights, signal_store)
        echo_chamber_risk = min(1.0, len(groupthink_clusters) / max(len(insights) / 5, 1))

        # Analyze convergence trajectory
        trajectory = self.analyze_trajectory()
        convergence_score = self.calculate_convergence_score()

        # Check health
        warnings = []
        critical_issues = []

        if insight_diversity < 0.3 and len(insights) > 5:
            warnings.append(f"Low insight diversity ({insight_diversity:.2f})")

        if objection_rate < 0.05 and len(insights) > 10:
            critical_issues.append(f"Very low objection rate ({objection_rate:.2f}) - insufficient adversarial challenge")

        if echo_chamber_risk > 0.5:
            critical_issues.append(f"High echo chamber risk ({echo_chamber_risk:.2f})")

        if validation_rate < 0.3 and len(insights) > 5:
            warnings.append(f"Low validation rate ({validation_rate:.2f})")

        if trajectory == "stuck":
            critical_issues.append("Swarm convergence stuck - no progress")

        is_healthy = len(critical_issues) == 0

        return SwarmHealthMetrics(
            total_signals=stats['total_signals'],
            signals_by_type=stats['by_type'],
            avg_strength=stats['avg_strength'],
            strength_gini=stats.get('gini', 0.0),
            insight_diversity=insight_diversity,
            evidence_diversity=evidence_diversity,
            source_diversity=source_diversity,
            objection_rate=objection_rate,
            validation_rate=validation_rate,
            response_rate=response_rate,
            convergence_score=convergence_score,
            trajectory=trajectory,
            echo_chamber_risk=echo_chamber_risk,
            groupthink_clusters=groupthink_clusters,
            is_healthy=is_healthy,
            warnings=warnings,
            critical_issues=critical_issues
        )

    def calculate_diversity(self, signals: List[Signal]) -> float:
        """Calculate semantic diversity of signal set.

        Args:
            signals: List of signals to analyze

        Returns:
            Diversity score 0.0-1.0 (higher = more diverse)
        """
        if len(signals) < 2:
            return 1.0

        # Calculate pairwise similarity
        from difflib import SequenceMatcher

        similarities = []
        sample_size = min(len(signals), 20)  # Sample for efficiency
        sampled = signals[:sample_size]

        for i, sig1 in enumerate(sampled):
            for sig2 in sampled[i+1:]:
                sim = SequenceMatcher(None, sig1.content.lower(), sig2.content.lower()).ratio()
                similarities.append(sim)

        if not similarities:
            return 1.0

        # Diversity = 1 - avg_similarity
        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity

    def calculate_source_diversity(self, insights: List[Signal],
                                   signal_store: SignalStore) -> float:
        """Calculate diversity of source documents.

        Args:
            insights: Insight signals to analyze
            signal_store: Signal store for provenance

        Returns:
            Source diversity score 0.0-1.0
        """
        if not insights:
            return 1.0

        source_docs = set()
        for insight in insights:
            obs_ids = insight.metadata.get('observation_ids', [])
            for obs_id in obs_ids:
                obs = signal_store.get_signal(obs_id)
                if obs:
                    source = obs.metadata.get('source_document', obs_id)
                    source_docs.add(source)

        # Diversity = unique sources / total insights
        return min(1.0, len(source_docs) / max(len(insights), 1))

    def detect_echo_chambers(self, insights: List[Signal],
                            signal_store: SignalStore) -> List[List[str]]:
        """Detect clusters of similar insights (potential groupthink).

        Args:
            insights: Insight signals to analyze
            signal_store: Signal store

        Returns:
            List of clusters (each cluster is a list of insight IDs)
        """
        clusters = []
        used = set()

        for insight in insights:
            if insight.id in used:
                continue

            # Find similar insights
            similar = signal_store.find_related_signals(
                insight,
                type="INSIGHT",
                similarity_threshold=0.7,
                n=5
            )

            if len(similar) >= 2:  # 3+ similar insights = potential echo chamber
                cluster_ids = [insight.id] + [s.id for s in similar]
                clusters.append(cluster_ids)
                used.add(insight.id)
                for s in similar:
                    used.add(s.id)

        return clusters

    def analyze_trajectory(self) -> str:
        """Analyze convergence trajectory from health history.

        Returns:
            Trajectory description: "converging", "stuck", "oscillating", "diverging", "initializing"
        """
        if len(self.health_history) < 5:
            return "initializing"

        # Look at last 5 measurements
        recent = self.health_history[-5:]

        # Check if Gini is increasing (converging)
        gini_trend = [m.strength_gini for m in recent]

        # All increasing
        if all(gini_trend[i] <= gini_trend[i+1] for i in range(len(gini_trend)-1)):
            return "converging"

        # Check if Gini is stable (converged or stuck)
        gini_variance = self._variance(gini_trend)
        if gini_variance < 0.01:
            if recent[-1].strength_gini > 0.7:
                return "converged"
            else:
                return "stuck"

        # Check if oscillating
        if self._is_oscillating(gini_trend):
            return "oscillating"

        return "diverging"

    def calculate_convergence_score(self) -> float:
        """Calculate convergence score based on trajectory.

        Returns:
            Convergence score 0.0-1.0
        """
        if len(self.health_history) < 2:
            return 0.0

        current = self.health_history[-1]

        # Score based on Gini coefficient and diversity
        gini_score = current.strength_gini
        diversity_score = current.insight_diversity

        # High Gini + moderate diversity = good convergence
        # High Gini + low diversity = echo chamber
        if diversity_score > 0.3:
            return gini_score
        else:
            return gini_score * 0.5  # Penalize low diversity

    def _variance(self, values: List[float]) -> float:
        """Calculate variance of values.

        Args:
            values: List of numbers

        Returns:
            Variance
        """
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _is_oscillating(self, values: List[float]) -> bool:
        """Check if values are oscillating.

        Args:
            values: List of numbers

        Returns:
            True if oscillating pattern detected
        """
        if len(values) < 4:
            return False

        # Check for up-down-up-down pattern
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        sign_changes = sum(1 for i in range(len(diffs)-1)
                          if diffs[i] * diffs[i+1] < 0)

        return sign_changes >= len(diffs) / 2

    def get_latest_metrics(self) -> Optional[SwarmHealthMetrics]:
        """Get the most recent health metrics.

        Returns:
            Latest SwarmHealthMetrics or None if no history
        """
        return self.health_history[-1] if self.health_history else None

    def print_summary(self):
        """Print a summary of swarm health monitoring."""
        if not self.health_history:
            print(f"\n[{self.agent_id}] No health data collected")
            return

        latest = self.health_history[-1]

        print(f"\n{'='*80}")
        print(f"SWARM HEALTH SUMMARY (Final Report)")
        print(f"{'='*80}")

        print(f"\n{'─'*80}")
        print(f"SIGNAL STATISTICS")
        print(f"{'─'*80}")
        print(f"Total signals: {latest.total_signals}")
        print(f"By type:")
        for sig_type, count in sorted(latest.signals_by_type.items()):
            print(f"  {sig_type}: {count}")
        print(f"Average strength: {latest.avg_strength:.3f}")
        print(f"Strength Gini: {latest.strength_gini:.3f}")

        print(f"\n{'─'*80}")
        print(f"DIVERSITY METRICS")
        print(f"{'─'*80}")
        print(f"Insight diversity: {latest.insight_diversity:.3f} {'✓' if latest.insight_diversity >= 0.4 else '⚠️'}")
        print(f"Evidence diversity: {latest.evidence_diversity:.3f}")
        print(f"Source diversity: {latest.source_diversity:.3f}")

        print(f"\n{'─'*80}")
        print(f"INTERACTION METRICS")
        print(f"{'─'*80}")
        print(f"Objection rate: {latest.objection_rate:.3f} {'✓' if latest.objection_rate >= 0.15 else '⚠️'}")
        print(f"Validation rate: {latest.validation_rate:.3f} {'✓' if latest.validation_rate >= 0.6 else '⚠️'}")
        print(f"Response rate: {latest.response_rate:.3f}")

        print(f"\n{'─'*80}")
        print(f"CONVERGENCE & ECHO CHAMBERS")
        print(f"{'─'*80}")
        print(f"Convergence score: {latest.convergence_score:.3f}")
        print(f"Trajectory: {latest.trajectory}")
        print(f"Echo chamber risk: {latest.echo_chamber_risk:.3f} {'✓' if latest.echo_chamber_risk < 0.3 else '⚠️'}")
        print(f"Groupthink clusters detected: {len(latest.groupthink_clusters)}")

        print(f"\n{'─'*80}")
        print(f"HEALTH STATUS")
        print(f"{'─'*80}")

        if latest.is_healthy:
            print(f"✓ SWARM IS HEALTHY")
        else:
            print(f"⚠️  SWARM HAS ISSUES")

        if latest.warnings:
            print(f"\nWarnings:")
            for warning in latest.warnings:
                print(f"  ⚠️  {warning}")

        if latest.critical_issues:
            print(f"\nCritical Issues:")
            for issue in latest.critical_issues:
                print(f"  ❌ {issue}")

        print(f"\n{'='*80}")

    async def trigger_self_healing(self, signal_store: SignalStore,
                                   metrics: SwarmHealthMetrics) -> None:
        """Trigger self-healing actions based on detected health issues.

        This method automatically fixes swarm health problems:
        - Low objection rate → decay strong signals to invite challenge
        - Echo chambers → decay consensus clusters
        - Low validation rate → boost weak signals awaiting validation
        - Stuck convergence → inject diversity
        - Low diversity → increase diversity threshold

        Args:
            signal_store: Signal store to modify
            metrics: Current health metrics
        """
        print(f"\n[{self.agent_id}] 🔧 Initiating self-healing...")

        healing_actions = []

        # Issue 1: Low objection rate (haters not challenging enough)
        if metrics.objection_rate < 0.05 and metrics.signals_by_type.get('INSIGHT', 0) > 10:
            print(f"[{self.agent_id}]   📉 Low objection rate ({metrics.objection_rate:.2f})")
            print(f"[{self.agent_id}]   🔧 Action: Decaying strong signals to invite challenge")

            # Decay strongest signals to create space for objections
            all_signals = signal_store.get_all_signals()
            strong_insights = [s for s in all_signals
                             if s.type == "INSIGHT" and s.strength > 0.7]

            for signal in strong_insights[:5]:  # Decay top 5 strong insights
                signal.strength *= 0.85  # 15% decay
                healing_actions.append(f"Decayed strong insight {signal.id[:8]}")

            print(f"[{self.agent_id}]   ✓ Decayed {len(strong_insights[:5])} strong insights")

        # Issue 2: Echo chambers detected
        if metrics.echo_chamber_risk > 0.5 and metrics.groupthink_clusters:
            print(f"[{self.agent_id}]   🔊 High echo chamber risk ({metrics.echo_chamber_risk:.2f})")
            print(f"[{self.agent_id}]   🔧 Action: Decaying consensus clusters")

            # Decay all signals in echo chamber clusters
            decayed_count = 0
            for cluster in metrics.groupthink_clusters:
                for insight_id in cluster:
                    insight = signal_store.get_signal(insight_id)
                    if insight:
                        insight.strength *= 0.75  # 25% decay
                        decayed_count += 1

            healing_actions.append(f"Decayed {decayed_count} echo chamber signals")
            print(f"[{self.agent_id}]   ✓ Decayed {decayed_count} signals in {len(metrics.groupthink_clusters)} clusters")

            # Also increase exploration bonus to encourage diversity
            if hasattr(signal_store, 'exploration_bonus'):
                old_bonus = signal_store.exploration_bonus
                signal_store.exploration_bonus = min(0.8, signal_store.exploration_bonus * 1.3)
                healing_actions.append(f"Increased exploration bonus: {old_bonus:.2f} → {signal_store.exploration_bonus:.2f}")
                print(f"[{self.agent_id}]   ✓ Increased exploration bonus: {old_bonus:.2f} → {signal_store.exploration_bonus:.2f}")

        # Issue 3: Low validation rate
        if metrics.validation_rate < 0.3 and metrics.signals_by_type.get('INSIGHT', 0) > 5:
            print(f"[{self.agent_id}]   📊 Low validation rate ({metrics.validation_rate:.2f})")
            print(f"[{self.agent_id}]   🔧 Action: Boosting unvalidated insights")

            # Boost insights with little/no evidence
            all_signals = signal_store.get_all_signals()
            insights = [s for s in all_signals if s.type == "INSIGHT"]

            boosted_count = 0
            for insight in insights:
                evidence = signal_store.get_descendants(insight.id, "EVIDENCE")
                if len(evidence) < 2:  # Needs validation
                    insight.strength = min(0.9, insight.strength * 1.2)  # 20% boost
                    boosted_count += 1

            healing_actions.append(f"Boosted {boosted_count} undervalidated insights")
            print(f"[{self.agent_id}]   ✓ Boosted {boosted_count} insights awaiting validation")

        # Issue 4: Convergence stuck
        if metrics.trajectory == "stuck":
            print(f"[{self.agent_id}]   🚫 Convergence stuck")
            print(f"[{self.agent_id}]   🔧 Action: Injecting diversity by boosting weak signals")

            # Boost weak signals to break stagnation
            all_signals = signal_store.get_all_signals()
            weak_signals = [s for s in all_signals if s.strength < 0.4]

            for signal in weak_signals[:10]:  # Boost top 10 weak signals
                signal.strength = min(0.7, signal.strength * 1.5)  # 50% boost, cap at 0.7
                healing_actions.append(f"Boosted weak signal {signal.id[:8]}")

            print(f"[{self.agent_id}]   ✓ Boosted {len(weak_signals[:10])} weak signals")

        # Issue 5: Low diversity
        if metrics.insight_diversity < 0.3 and metrics.signals_by_type.get('INSIGHT', 0) > 5:
            print(f"[{self.agent_id}]   📉 Low insight diversity ({metrics.insight_diversity:.2f})")
            print(f"[{self.agent_id}]   🔧 Action: Increasing diversity threshold")

            # Increase diversity threshold to reject similar signals
            if hasattr(signal_store, 'diversity_threshold'):
                old_threshold = signal_store.diversity_threshold
                signal_store.diversity_threshold = min(0.98, signal_store.diversity_threshold * 1.05)
                healing_actions.append(f"Increased diversity threshold: {old_threshold:.2f} → {signal_store.diversity_threshold:.2f}")
                print(f"[{self.agent_id}]   ✓ Increased diversity threshold: {old_threshold:.2f} → {signal_store.diversity_threshold:.2f}")

        # Log all actions taken
        if healing_actions:
            print(f"\n[{self.agent_id}] ✓ Self-healing complete: {len(healing_actions)} action(s) taken")
        else:
            print(f"\n[{self.agent_id}] ℹ️  No healing actions needed")

    def stop(self):
        """Stop monitoring."""
        self.active = False
