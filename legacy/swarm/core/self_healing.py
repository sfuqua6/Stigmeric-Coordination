"""Self-healing coordinator - automatic problem detection and recovery."""

import asyncio
import time
from typing import Dict, List
from .signal_store import SignalStore
from .swarm_monitor import SwarmMonitor
from ..agents.hater import Hater


class SelfHealingCoordinator:
    """Automatic problem detection and recovery for swarm systems.

    Monitors swarm health and applies interventions:
    - Spawn additional haters if objection rate too low
    - Break echo chambers through targeted decay
    - Boost weak signals if diversity too low
    - Increase exploration if convergence stuck
    """

    def __init__(self, signal_store: SignalStore, monitor: SwarmMonitor):
        """Initialize self-healing coordinator.

        Args:
            signal_store: Signal store to modify
            monitor: Swarm monitor for health metrics
        """
        self.signal_store = signal_store
        self.monitor = monitor
        self.interventions = []  # Track all interventions
        self.spawned_agents = []  # Track dynamically spawned agents

    async def check_and_heal(self, agents_by_type: Dict[str, List], llm):
        """Check system health and apply interventions if needed.

        Args:
            agents_by_type: Dictionary of agent lists by type
            llm: Language model for spawned agents
        """
        health = self.monitor.calculate_health_metrics()

        # Intervention 1: Low objection rate → spawn more haters
        if health['objection_rate'] < 0.10:
            await self._spawn_haters(agents_by_type, llm, count=2)
            self.interventions.append({
                'time': time.time(),
                'type': 'spawn_haters',
                'reason': f"Low objection rate: {health['objection_rate']:.1%}",
                'metric': health['objection_rate']
            })

        # Intervention 2: Echo chamber detected → decay cluster
        if health['echo_chamber_risk'] > 0.5:
            await self._break_echo_chamber()
            self.interventions.append({
                'time': time.time(),
                'type': 'decay_echo_chamber',
                'reason': f"Echo chamber risk: {health['echo_chamber_risk']:.2f}",
                'metric': health['echo_chamber_risk']
            })

        # Intervention 3: Low diversity → boost weak signals
        if health['diversity'] < 0.3:
            await self._boost_weak_signals()
            self.interventions.append({
                'time': time.time(),
                'type': 'boost_weak_signals',
                'reason': f"Low diversity: {health['diversity']:.2f}",
                'metric': health['diversity']
            })

        # Intervention 4: Convergence stuck → increase exploration
        if health['convergence'] == 'stuck':
            await self._increase_exploration(agents_by_type)
            self.interventions.append({
                'time': time.time(),
                'type': 'increase_exploration',
                'reason': "Convergence stuck",
                'metric': 0.0
            })

    async def _spawn_haters(self, agents_by_type: Dict[str, List], llm, count: int):
        """Spawn additional hater agents dynamically.

        Args:
            agents_by_type: Dictionary of agent lists
            llm: Language model
            count: Number of haters to spawn
        """
        print(f"[SELF_HEAL] Spawning {count} additional haters (low objection rate)")

        for i in range(count):
            hater_id = f"AutoHater_{int(time.time())}_{i}"
            hater = Hater(hater_id, "Challenge insights")

            self.spawned_agents.append(hater)

            # Add to agents_by_type if hater key exists
            if 'hater' in agents_by_type:
                agents_by_type['hater'].append(hater)

            # Launch hater task immediately
            # Note: This won't integrate perfectly with the main loop
            # but will add adversarial pressure
            asyncio.create_task(
                hater.run(self.signal_store, llm, max_actions=50, target_consensus=True)
            )

    async def _break_echo_chamber(self):
        """Apply decay to consensus clusters to break echo chambers."""
        print(f"[SELF_HEAL] Breaking echo chamber (applying cluster decay)")

        all_signals = self.signal_store.get_all_signals()
        insights = [s for s in all_signals if s.type == "INSIGHT"]

        clusters_broken = 0

        for insight in insights:
            # Find similar insights (potential cluster)
            if hasattr(self.signal_store, 'find_related_signals'):
                similar = self.signal_store.find_related_signals(
                    insight, type="INSIGHT", similarity_threshold=0.7, n=5
                )
            else:
                # Fallback: simple string similarity
                similar = []
                for other in insights:
                    if other.id != insight.id:
                        words1 = set(insight.content.lower().split())
                        words2 = set(other.content.lower().split())
                        overlap = len(words1 & words2) / max(len(words1), len(words2))
                        if overlap > 0.7:
                            similar.append(other)

            if len(similar) >= 3:
                # Decay the entire cluster
                for sig in [insight] + similar:
                    if hasattr(self.signal_store, 'signals'):
                        if sig.id in self.signal_store.signals:
                            self.signal_store.signals[sig.id].strength *= 0.7
                    else:
                        # Fallback: modify signal object directly
                        sig.strength *= 0.7

                clusters_broken += 1
                print(f"[SELF_HEAL] Decayed cluster around {insight.id} ({len(similar)+1} insights)")

        if clusters_broken == 0:
            print(f"[SELF_HEAL] No clusters found to break")

    async def _boost_weak_signals(self):
        """Amplify under-explored weak signals to increase diversity."""
        print(f"[SELF_HEAL] Boosting weak signals (low diversity)")

        all_signals = self.signal_store.get_all_signals()

        # Find weak signals with low visits
        weak_signals = [s for s in all_signals
                       if s.strength < 0.5 and s.visits < 2]

        # Sort by quality indicators
        # Prefer longer, more detailed signals
        weak_signals.sort(key=lambda s: len(s.content), reverse=True)

        # Boost top 10
        boosted = 0
        for signal in weak_signals[:10]:
            if hasattr(self.signal_store, 'amplify'):
                self.signal_store.amplify(signal.id, factor=1.4)
            else:
                # Fallback: modify directly
                if hasattr(self.signal_store, 'signals'):
                    if signal.id in self.signal_store.signals:
                        self.signal_store.signals[signal.id].strength *= 1.4
                else:
                    signal.strength *= 1.4

            boosted += 1

        print(f"[SELF_HEAL] Boosted {boosted} weak signals")

    async def _increase_exploration(self, agents_by_type: Dict[str, List]):
        """Increase exploration bonus to help stuck convergence.

        Args:
            agents_by_type: Dictionary of agent lists
        """
        print(f"[SELF_HEAL] Increasing exploration (convergence stuck)")

        # Increase exploration bonus in signal store
        if hasattr(self.signal_store, 'exploration_bonus'):
            old_bonus = self.signal_store.exploration_bonus
            self.signal_store.exploration_bonus *= 1.5
            print(f"[SELF_HEAL] Exploration bonus: {old_bonus:.2f} → {self.signal_store.exploration_bonus:.2f}")

        # Decay all high-strength signals to encourage re-exploration
        all_signals = self.signal_store.get_all_signals()
        strong_signals = [s for s in all_signals if s.strength > 0.7]

        for signal in strong_signals:
            if hasattr(self.signal_store, 'signals'):
                if signal.id in self.signal_store.signals:
                    self.signal_store.signals[signal.id].strength *= 0.9
            else:
                signal.strength *= 0.9

        print(f"[SELF_HEAL] Decayed {len(strong_signals)} strong signals to encourage exploration")

    def get_intervention_summary(self) -> str:
        """Get summary of all interventions applied.

        Returns:
            Formatted summary string
        """
        if not self.interventions:
            return "No interventions applied"

        summary = f"Self-Healing Interventions ({len(self.interventions)} total)\n"
        summary += "=" * 50 + "\n"

        # Group by type
        by_type = {}
        for intervention in self.interventions:
            itype = intervention['type']
            if itype not in by_type:
                by_type[itype] = []
            by_type[itype].append(intervention)

        for itype, interventions in by_type.items():
            summary += f"\n{itype}: {len(interventions)} times\n"
            for i in interventions[-3:]:  # Show last 3
                summary += f"  - {i['reason']}\n"

        return summary

    def get_stats(self) -> Dict:
        """Get statistics on self-healing activity.

        Returns:
            Dictionary with intervention stats
        """
        by_type = {}
        for intervention in self.interventions:
            itype = intervention['type']
            by_type[itype] = by_type.get(itype, 0) + 1

        return {
            'total_interventions': len(self.interventions),
            'by_type': by_type,
            'spawned_agents': len(self.spawned_agents)
        }
