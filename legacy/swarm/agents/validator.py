"""Validator agent - verifies factual accuracy and source credibility."""

import asyncio
import random
import re
from typing import Optional, List
from ..core.signal_store import SignalStore, Signal
from ..llm.simple_llm import SimpleLLM
from ..core.signal_types import SignalType, INITIAL, SUPPORT
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class Validator:
    """Validator agent that fact-checks signals and verifies source credibility.

    Distinct role from Critic:
    - Critic evaluates ARGUMENT QUALITY (logic, coherence, strength)
    - Validator verifies FACTUAL ACCURACY (claims, sources, evidence)
    """

    def __init__(self, agent_id: str, task_prompt: str = "Validate claims"):
        """Initialize validator.

        Args:
            agent_id: Unique agent ID
            task_prompt: Task description (for context)
        """
        self.agent_id = agent_id
        self.task_prompt = task_prompt
        self.active = True
        self.actions_taken = 0

    async def run(self, signal_store: SignalStore, llm: SimpleLLM,
                  min_strength: float = 0.3, max_actions: int = 200,
                  temperature: float = 0.5):
        """Run validator behavior loop.

        Args:
            signal_store: Shared signal store
            llm: Language model
            min_strength: Minimum strength to deposit verification signal
            max_actions: Maximum actions before stopping
            temperature: Sampling temperature (low for factual checking)
        """
        while self.active and self.actions_taken < max_actions:
            # Sample signals that need fact-checking
            # Use universal types: INITIAL and SUPPORT
            initial_signals = signal_store.sample_weighted(INITIAL, n=2)
            support_signals_1 = signal_store.sample_weighted(SUPPORT, n=2)
            support_signals_2 = signal_store.sample_weighted(SUPPORT, n=1)

            targets = initial_signals + support_signals_1 + support_signals_2

            if not targets:
                self.actions_taken += 1
                # Event-driven: wait for new signals
                await signal_store.wait_for_signal(INITIAL, timeout=1.0)
                signal_store.clear_signal_event(INITIAL)
                continue

            # Prioritize signals with factual claims (numbers, proper nouns)
            targets_with_scores = []
            for t in targets:
                # Check if signal has existing verification
                # PERFORMANCE: Use indexed retrieval O(k) instead of O(n) scan
                children = signal_store.get_signals_by_parent(t.id)
                verifications = [s for s in children if s.type == "VERIFICATION"]

                # Score: has_numbers/names + lacks_verification
                has_claims = self._has_factual_claims(t.content)
                lacks_verification = len(verifications) == 0
                score = (1.0 if has_claims else 0.3) * (1.5 if lacks_verification else 0.5)

                targets_with_scores.append((score, t))

            if not targets_with_scores:
                self.actions_taken += 1
                # Event-driven: wait for new signals
                await signal_store.wait_for_signal(INITIAL, timeout=1.0)
                signal_store.clear_signal_event(INITIAL)
                continue

            targets_with_scores.sort(key=lambda x: x[0], reverse=True)
            target = targets_with_scores[0][1]

            # Verify the signal
            verification = await self.verify_signal(target, llm, temperature)

            if verification and len(verification['content'].strip()) > 30:
                signal_id = signal_store.deposit(
                    signal_type="VERIFICATION",
                    content=verification['content'],
                    strength=verification['strength'],
                    depositor=self.agent_id,
                    parent=target.id,
                    metadata={
                        'verification_score': verification['score'],
                        'factual_accuracy': verification['accurate'],
                        'has_sources': verification['has_sources']
                    }
                )

                # Adjust target signal strength based on verification
                if verification['accurate']:
                    # Boost verified signals
                    signal_store.amplify(target.id, factor=1.2)
                    logger.info(f"{self.agent_id} verified {target.id} "
                                f"(score={verification['score']:.2f}, boosted)")
                else:
                    # Decay unverified signals
                    current_strength = signal_store.signals[target.id].strength
                    new_strength = current_strength * 0.75
                    signal_store.signals[target.id].strength = max(0.1, new_strength)
                    logger.info(f"{self.agent_id} flagged {target.id} "
                                f"(score={verification['score']:.2f}, decayed to {new_strength:.2f})")

            self.actions_taken += 1
            # No sleep - pure stigmergic event-driven behavior

    def _has_factual_claims(self, content: str) -> bool:
        """Check if content has factual claims (numbers, proper nouns, dates).

        Args:
            content: Signal content

        Returns:
            True if has factual claims
        """
        # Check for numbers (statistics, percentages, dates)
        has_numbers = bool(re.search(r'\d+', content))

        # Check for proper nouns (organizations, places, people)
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content))

        # Check for citation indicators
        has_citations = any(word in content.lower() for word in
                          ['study', 'research', 'report', 'according to', 'found that',
                           'shows that', 'data', 'evidence', 'survey'])

        return has_numbers or has_proper_nouns or has_citations

    async def verify_signal(self, target: Signal, llm: SimpleLLM,
                           temperature: float) -> Optional[dict]:
        """Verify factual accuracy of a signal.

        Args:
            target: Signal to verify
            llm: Language model
            temperature: Sampling temperature

        Returns:
            Dictionary with verification content, strength, score, accuracy
        """
        prompt = self._make_verification_prompt(target)

        try:
            result = await llm.generate(prompt, max_tokens=120, temperature=temperature)

            if result and len(result.strip()) > 30:
                # Parse verification result
                verification_text = result.strip()

                # Extract verification score and accuracy assessment
                score, accurate, has_sources = self._parse_verification(verification_text)

                return {
                    'content': verification_text,
                    'strength': max(0.4, score),
                    'score': score,
                    'accurate': accurate,
                    'has_sources': has_sources
                }
        except Exception as e:
            logger.error(f"{self.agent_id} verification error: {e}", exc_info=True)

        return None

    def _make_verification_prompt(self, target: Signal) -> str:
        """Generate verification prompt.

        Args:
            target: Signal to verify

        Returns:
            Verification prompt
        """
        return f"""You are a fact-checker verifying the accuracy of claims.

Signal to verify:
"{target.content}"

Task: Assess the factual accuracy of this statement.

Check for:
1. Are specific claims verifiable? (numbers, statistics, facts)
2. Are sources credible if mentioned?
3. Are claims reasonable given general knowledge?
4. Are there red flags? (too specific, no sources, extraordinary claims)

Respond in this format:
ACCURACY: [HIGH/MEDIUM/LOW]
REASONING: [1-2 sentences explaining why]

If HIGH: Claims are specific, verifiable, and reasonable
If MEDIUM: Claims are plausible but lack specifics or sources
If LOW: Claims are vague, unverifiable, or contradict known facts

Verification:"""

    def _parse_verification(self, verification_text: str) -> tuple:
        """Parse verification result to extract score and accuracy.

        Args:
            verification_text: Verification text from LLM

        Returns:
            Tuple of (score, accurate, has_sources)
        """
        text_lower = verification_text.lower()

        # Check accuracy level
        if 'accuracy: high' in text_lower or 'high accuracy' in text_lower:
            score = 0.8
            accurate = True
        elif 'accuracy: medium' in text_lower or 'medium accuracy' in text_lower:
            score = 0.6
            accurate = True
        elif 'accuracy: low' in text_lower or 'low accuracy' in text_lower:
            score = 0.4
            accurate = False
        else:
            # Fallback heuristics
            if any(word in text_lower for word in ['verified', 'accurate', 'correct', 'reasonable']):
                score = 0.7
                accurate = True
            elif any(word in text_lower for word in ['questionable', 'unverified', 'unsupported']):
                score = 0.4
                accurate = False
            else:
                score = 0.5
                accurate = True  # Neutral default

        # Check if sources mentioned
        has_sources = any(word in text_lower for word in
                         ['source', 'study', 'research', 'reference', 'citation'])

        return score, accurate, has_sources

    def stop(self):
        """Stop the agent."""
        self.active = False
