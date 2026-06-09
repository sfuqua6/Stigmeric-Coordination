"""Dialogue coordinator - manages multi-turn agent dialogues."""

import asyncio
from typing import Dict, List
from .signal_store import SignalStore


class DialogueCoordinator:
    """Coordinates multi-turn agent dialogues for insight refinement.

    This enables forager-hater dialogues where:
    1. Hater objects to an insight
    2. Forager defends or refines the insight
    3. Hater counters or acknowledges
    4. Process continues for multiple turns
    """

    def __init__(self, signal_store: SignalStore, task_config=None):
        """Initialize dialogue coordinator.

        Args:
            signal_store: Signal store to monitor for dialogues
            task_config: Task configuration with signal type mappings (optional)
        """
        self.signal_store = signal_store
        self.active_threads = {}  # Track ongoing dialogues
        self.processed_objections = set()  # Track which objections we've handled

        # Signal type configuration (mode-aware)
        if task_config:
            self.insight_type = task_config.signal_types.get("initial", "INSIGHT")
            self.objection_type = task_config.signal_types.get("counter", "OBJECTION")
        else:
            # Legacy defaults
            self.insight_type = "INSIGHT"
            self.objection_type = "OBJECTION"

    async def check_and_trigger_responses(self, agents_by_type: Dict[str, List], llm):
        """Check for unanswered objections/responses and trigger agent reactions.

        Args:
            agents_by_type: Dict with keys like 'forager', 'hater' mapping to agent lists
            llm: Language model for generating responses
        """
        # Find unanswered objections to insights (mode-aware)
        insights = [s for s in self.signal_store.get_all_signals() if s.type == self.insight_type]

        for insight in insights:
            # Check for objections to this insight
            all_signals = self.signal_store.get_all_signals()
            objections = [s for s in all_signals
                         if s.parent == insight.id and s.type == self.objection_type]

            for objection in objections:
                # Skip if we already processed this objection
                if objection.id in self.processed_objections:
                    continue

                # Check if forager responded
                responses = self._get_responses(objection.id)
                forager_responses = [r for r in responses
                                    if r.depositor and "Forager" in r.depositor]

                if not forager_responses:
                    # UNANSWERED OBJECTION - trigger forager defense
                    await self._trigger_forager_defense(
                        insight, objection, agents_by_type.get('forager', []), llm
                    )
                    self.processed_objections.add(objection.id)

                # Check for forager responses that need hater counter
                elif forager_responses:
                    latest_response = forager_responses[-1]
                    counter_responses = self._get_responses(latest_response.id)
                    hater_counters = [r for r in counter_responses
                                     if r.depositor and "Hater" in r.depositor]

                    if not hater_counters:
                        # UNANSWERED DEFENSE - trigger hater counter
                        await self._trigger_hater_counter(
                            objection, latest_response, agents_by_type.get('hater', []), llm
                        )

    def _get_responses(self, signal_id: str) -> List:
        """Get all responses to a signal.

        Args:
            signal_id: Signal ID to find responses for

        Returns:
            List of response signals
        """
        # Check if signal_store has get_responses method
        if hasattr(self.signal_store, 'get_responses'):
            return self.signal_store.get_responses(signal_id)

        # Fallback: manually find responses by parent
        all_signals = self.signal_store.get_all_signals()
        return [s for s in all_signals if s.parent == signal_id]

    async def _trigger_forager_defense(self, insight, objection, foragers, llm):
        """Trigger forager to defend insight against objection.

        Args:
            insight: Original insight signal
            objection: Objection signal
            foragers: List of forager agents
            llm: Language model
        """
        # Find the forager who created this insight
        creator_id = insight.depositor

        # Find matching forager agent
        for forager in foragers:
            if forager.agent_id == creator_id:
                # Check if forager has defend_insights method
                if hasattr(forager, 'generate_defense'):
                    try:
                        defense = await forager.generate_defense(
                            insight, objection, self.signal_store, llm
                        )

                        if defense and len(defense.strip()) > 50:
                            # Deposit response
                            if hasattr(self.signal_store, 'deposit_response'):
                                # Use deposit_response if available
                                response_id = self.signal_store.deposit_response(
                                    signal_type=self.insight_type,  # Mode-aware
                                    content=defense,
                                    strength=0.6,
                                    depositor=forager.agent_id,
                                    responding_to=objection.id
                                )
                            else:
                                # Fallback: use regular deposit with parent
                                response_id = self.signal_store.deposit(
                                    signal_type=self.insight_type,  # Mode-aware
                                    content=defense,
                                    strength=0.6,
                                    depositor=forager.agent_id,
                                    parent=objection.id,
                                    metadata={'responding_to': objection.id}
                                )

                            if response_id:
                                print(f"[DIALOGUE] {forager.agent_id} defended {insight.id} "
                                      f"against {objection.id}")
                    except Exception as e:
                        print(f"[DIALOGUE] Error triggering forager defense: {e}")
                        import traceback
                        traceback.print_exc()
                break

    async def _trigger_hater_counter(self, objection, defense, haters, llm):
        """Trigger hater to counter forager defense.

        Args:
            objection: Original objection signal
            defense: Forager's defense signal
            haters: List of hater agents
            llm: Language model
        """
        # Find the hater who created this objection
        hater_id = objection.depositor

        # Find matching hater agent
        for hater in haters:
            if hater.agent_id == hater_id:
                # Check if hater has generate_counter_response method
                if hasattr(hater, 'generate_counter_response'):
                    try:
                        counter = await hater.generate_counter_response(
                            objection, defense, self.signal_store, llm
                        )

                        if counter and len(counter.strip()) > 50:
                            # Deposit counter-response
                            if hasattr(self.signal_store, 'deposit_response'):
                                response_id = self.signal_store.deposit_response(
                                    signal_type=self.objection_type,  # Mode-aware
                                    content=counter,
                                    strength=0.7,
                                    depositor=hater.agent_id,
                                    responding_to=defense.id
                                )
                            else:
                                response_id = self.signal_store.deposit(
                                    signal_type=self.objection_type,  # Mode-aware
                                    content=counter,
                                    strength=0.7,
                                    depositor=hater.agent_id,
                                    parent=defense.id,
                                    metadata={'responding_to': defense.id}
                                )

                            if response_id:
                                print(f"[DIALOGUE] {hater.agent_id} countered defense "
                                      f"on {objection.id}")
                    except Exception as e:
                        print(f"[DIALOGUE] Error triggering hater counter: {e}")
                        import traceback
                        traceback.print_exc()
                break

    def get_dialogue_stats(self) -> Dict:
        """Get statistics on dialogue activity.

        Returns:
            Dictionary with dialogue metrics
        """
        all_signals = self.signal_store.get_all_signals()

        # Count objections (mode-aware)
        objections = [s for s in all_signals if s.type == self.objection_type]

        # Count responses
        total_responses = 0
        max_depth = 0

        for objection in objections:
            # Count responses in this thread
            if hasattr(self.signal_store, 'get_dialogue_thread'):
                thread = self.signal_store.get_dialogue_thread(objection.id)
                depth = len(thread)
                total_responses += depth
                max_depth = max(max_depth, depth)
            else:
                # Fallback: count children
                responses = [s for s in all_signals if s.parent == objection.id]
                total_responses += len(responses)
                max_depth = max(max_depth, len(responses))

        avg_depth = total_responses / len(objections) if objections else 0

        return {
            'total_objections': len(objections),
            'total_responses': total_responses,
            'average_dialogue_depth': avg_depth,
            'max_dialogue_depth': max_depth,
            'objections_processed': len(self.processed_objections)
        }
