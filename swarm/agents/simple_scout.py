"""Simple scout agent with spatial movement for true swarm behavior.

Key differences from original Scout:
- Position in 2D space (not stateless)
- Simple movement rules (not complex reasoning)
- Local information only (radius-based perception)
- Tiny LLM queries (20 tokens, not 70)
- Emergent behavior through local interactions
"""

import asyncio
import random
import math
from typing import Optional, Tuple

from ..core.spatial_signal_store import SpatialSignalStore, SpatialSignal
from ..llm.simple_llm import SimpleLLM


class SimpleScout:
    """Simple scout agent with position and local rules.

    True swarm intelligence properties:
    - Simple agent (not full LLM reasoning)
    - Local information (only sees nearby signals)
    - Simple rules (move, observe, deposit)
    - Emergent behavior (clustering, consensus)
    """

    def __init__(self,
                 agent_id: str,
                 position: Tuple[float, float],
                 knowledge: str,
                 perception_radius: float = 10.0,
                 movement_speed: float = 2.0):
        """Initialize simple scout.

        Args:
            agent_id: Unique agent ID
            position: Starting (x, y) position
            knowledge: Small knowledge fragment (1-2 sentences)
            perception_radius: How far agent can see (default 10.0)
            movement_speed: Maximum movement per step (default 2.0)
        """
        self.agent_id = agent_id
        self.position = list(position)  # [x, y] mutable
        self.knowledge = knowledge
        self.perception_radius = perception_radius
        self.movement_speed = movement_speed

        # Agent state
        self.velocity = [0.0, 0.0]  # [vx, vy]
        self.confidence = 0.5  # 0.0-1.0
        self.actions_taken = 0
        self.active = True

        # History
        self.signals_deposited = 0
        self.last_deposit_time = 0

    async def step(self,
                   signal_store: SpatialSignalStore,
                   llm_oracle: SimpleLLM,
                   task_prompt: str,
                   current_iteration: int) -> Optional[str]:
        """Execute one timestep of agent behavior.

        Simple rule-based logic:
        1. Perceive local environment
        2. Decide action based on local state
        3. Execute action (move, observe, deposit)

        Args:
            signal_store: Spatial signal store
            llm_oracle: Language model for tiny queries
            task_prompt: Task description for context
            current_iteration: Current iteration number

        Returns:
            Signal ID if deposited, None otherwise
        """
        # 1. PERCEIVE: Get local signals
        local_signals = signal_store.get_nearby(
            tuple(self.position),
            self.perception_radius
        )

        # 2. DECIDE: Simple rules based on local state
        action = self._decide_action(local_signals, current_iteration)

        # 3. ACT: Execute decided action
        signal_id = None

        if action == "move_away":
            self._move_away_from_crowd(local_signals)

        elif action == "move_toward":
            self._move_toward_signals(signal_store)

        elif action == "explore":
            self._explore_randomly()

        elif action == "deposit":
            signal_id = await self._deposit_signal(
                signal_store, llm_oracle, task_prompt
            )

        elif action == "observe":
            self._observe_and_learn(local_signals)

        # Update state
        self.actions_taken += 1

        return signal_id

    def _decide_action(self, local_signals: list, iteration: int) -> str:
        """Decide action based on local state (simple rules).

        Decision tree:
        - Too crowded → move away
        - Isolated + low confidence → move toward signals
        - Confident + not recently deposited → deposit
        - Low confidence + signals nearby → observe
        - Default → explore

        Args:
            local_signals: Nearby signals
            iteration: Current iteration

        Returns:
            Action string: "move_away", "move_toward", "deposit", "observe", "explore"
        """
        # Count local signals
        num_local = len(local_signals)

        # Check crowding (too many signals nearby)
        if num_local > 10:
            return "move_away"

        # Check isolation (very few signals)
        if num_local < 2 and self.confidence < 0.6:
            return "move_toward"

        # Check if confident enough to deposit
        # and not recently deposited (avoid spam)
        if self.confidence > 0.65 and (iteration - self.last_deposit_time) > 5:
            return "deposit"

        # Observe if signals nearby and low confidence
        if num_local >= 2 and self.confidence < 0.7:
            return "observe"

        # Default: explore
        return "explore"

    def _move_away_from_crowd(self, local_signals: list):
        """Move away from crowded area.

        Calculates repulsion vector from nearby signals.

        Args:
            local_signals: Nearby signals
        """
        if not local_signals:
            return

        # Calculate repulsion vector
        repulsion_x = 0.0
        repulsion_y = 0.0

        for signal in local_signals:
            # Vector from signal to agent
            dx = self.position[0] - signal.position[0]
            dy = self.position[1] - signal.position[1]

            # Normalize and accumulate
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0:
                repulsion_x += dx / dist
                repulsion_y += dy / dist

        # Normalize to movement speed
        magnitude = math.sqrt(repulsion_x * repulsion_x + repulsion_y * repulsion_y)
        if magnitude > 0:
            self.velocity[0] = (repulsion_x / magnitude) * self.movement_speed
            self.velocity[1] = (repulsion_y / magnitude) * self.movement_speed

        # Apply movement
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

        # Clamp to bounds (assuming 100x100 grid)
        self.position[0] = max(0, min(99, self.position[0]))
        self.position[1] = max(0, min(99, self.position[1]))

    def _move_toward_signals(self, signal_store: SpatialSignalStore):
        """Move toward stronger signals using gradient.

        Args:
            signal_store: Spatial signal store
        """
        # Get gradient (points toward higher strength)
        gradient = signal_store.get_gradient(
            tuple(self.position),
            self.perception_radius * 2  # Wider perception for gradient
        )

        # Set velocity toward gradient
        self.velocity[0] = gradient[0] * self.movement_speed
        self.velocity[1] = gradient[1] * self.movement_speed

        # Apply movement
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

        # Clamp to bounds
        self.position[0] = max(0, min(99, self.position[0]))
        self.position[1] = max(0, min(99, self.position[1]))

    def _explore_randomly(self):
        """Random walk for exploration (Lévy flight pattern).

        90% local movement, 10% long jumps.
        """
        if random.random() < 0.9:
            # Local movement
            dx = random.uniform(-self.movement_speed, self.movement_speed)
            dy = random.uniform(-self.movement_speed, self.movement_speed)
        else:
            # Long jump (10× movement speed)
            dx = random.uniform(-self.movement_speed * 10, self.movement_speed * 10)
            dy = random.uniform(-self.movement_speed * 10, self.movement_speed * 10)

        self.position[0] += dx
        self.position[1] += dy

        # Clamp to bounds
        self.position[0] = max(0, min(99, self.position[0]))
        self.position[1] = max(0, min(99, self.position[1]))

    async def _deposit_signal(self,
                              signal_store: SpatialSignalStore,
                              llm_oracle: SimpleLLM,
                              task_prompt: str) -> Optional[str]:
        """Deposit signal using tiny LLM query.

        Key difference from original Scout:
        - 20 tokens (not 70)
        - Simple prompt (not complex reasoning)
        - Deposits at current position

        Args:
            signal_store: Spatial signal store
            llm_oracle: Language model
            task_prompt: Task description

        Returns:
            Signal ID if deposited
        """
        # Simple prompt: knowledge → fact
        prompt = f"Based on: {self.knowledge}\nTask: {task_prompt}\nOne specific fact (10 words):"

        try:
            # Tiny LLM query (20 tokens)
            result = await llm_oracle.generate(
                prompt,
                max_tokens=20,  # Reduced from 70!
                temperature=0.9,
                use_cache=False
            )

            if result and len(result.strip()) > 10:
                content = result.strip()

                # Deposit at current position
                signal_id = signal_store.deposit(
                    signal_type="SCOUT_SIGNAL",
                    content=content,
                    strength=self.confidence,
                    depositor=self.agent_id,
                    position=tuple(self.position)
                )

                if signal_id:
                    self.signals_deposited += 1
                    self.last_deposit_time = self.actions_taken
                    print(f"[{self.agent_id}] Deposited at ({self.position[0]:.1f}, {self.position[1]:.1f}): {content[:50]}...")
                    return signal_id

        except Exception as e:
            print(f"[{self.agent_id}] Deposit error: {e}")

        return None

    def _observe_and_learn(self, local_signals: list):
        """Observe nearby signals and adjust confidence.

        Learning rule:
        - Strong nearby signals → increase confidence
        - Weak nearby signals → decrease confidence
        - Similar content → increase confidence

        Args:
            local_signals: Nearby signals
        """
        if not local_signals:
            # No signals to learn from, slight confidence decrease
            self.confidence *= 0.98
            return

        # Calculate average strength of nearby signals
        avg_strength = sum(s.strength for s in local_signals) / len(local_signals)

        # Adjust confidence based on environment
        if avg_strength > 0.6:
            # Strong environment → increase confidence
            self.confidence = min(1.0, self.confidence * 1.05)
        elif avg_strength < 0.4:
            # Weak environment → decrease confidence
            self.confidence = max(0.1, self.confidence * 0.95)

        # Check for content similarity (simple heuristic)
        similar_count = 0
        for signal in local_signals[:5]:  # Check up to 5 nearest
            # Simple similarity: shared words
            knowledge_words = set(self.knowledge.lower().split())
            signal_words = set(signal.content.lower().split())
            overlap = len(knowledge_words & signal_words)

            if overlap >= 2:  # At least 2 shared words
                similar_count += 1

        if similar_count >= 2:
            # Found similar content → boost confidence
            self.confidence = min(1.0, self.confidence * 1.08)

    def get_state(self) -> dict:
        """Get agent state for debugging/visualization.

        Returns:
            Dictionary with agent state
        """
        return {
            "agent_id": self.agent_id,
            "position": tuple(self.position),
            "velocity": tuple(self.velocity),
            "confidence": self.confidence,
            "signals_deposited": self.signals_deposited,
            "actions_taken": self.actions_taken,
        }

    def stop(self):
        """Stop the agent."""
        self.active = False
