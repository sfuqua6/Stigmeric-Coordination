"""Real Validator using external sources and dynamic learning.

This REPLACES the fake Validator that used LLM self-critique.

Key improvements:
- Uses DynamicKnowledgeBase that LEARNS during execution
- Verifies via external sources (Wikipedia, web search, symbolic math)
- NO LLM self-critique (no asking LLM to verify itself)
- Tracks confidence and updates beliefs over time

This is TRUE validation, not security theater.
"""

import re
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .dynamic_knowledge_base import DynamicKnowledgeBase
from .external_sources import MultiSourceVerifier, WikipediaSource, WebSearchSource, SymbolicMathSource
from ..core.signal_store import Signal


@dataclass
class ValidationResult:
    """Result of signal validation."""
    signal_id: str
    accuracy: float  # 0.0-1.0
    verified_claims: int
    total_claims: int
    method: str
    confidence: float
    learned_facts: int
    conflicts: int
    evidence: List[str]


class RealValidator:
    """Real fact-checking agent using external sources and learning.

    Unlike the old Validator that asked the LLM to verify itself,
    this agent:
    1. Extracts factual claims from signals
    2. Checks DynamicKnowledgeBase (may already know this)
    3. If unknown, queries external sources
    4. LEARNS verified facts into KB
    5. Updates confidence based on evidence
    6. Detects conflicts between claims

    This enables TRUE swarm intelligence with external grounding.
    """

    def __init__(self,
                 agent_id: str,
                 knowledge_base: Optional[DynamicKnowledgeBase] = None,
                 confidence_threshold: float = 0.6):
        """Initialize real validator.

        Args:
            agent_id: Unique agent ID
            knowledge_base: Shared knowledge base (creates if None)
            confidence_threshold: Minimum confidence for high accuracy
        """
        self.agent_id = agent_id
        self.kb = knowledge_base or DynamicKnowledgeBase(confidence_threshold=confidence_threshold)
        self.verifier = MultiSourceVerifier()
        self.confidence_threshold = confidence_threshold

        # Statistics
        self.signals_validated = 0
        self.claims_verified = 0
        self.facts_learned = 0
        self.conflicts_detected = 0

    async def validate_signal(self, signal: Signal) -> ValidationResult:
        """Validate signal using external sources and knowledge base.

        This is the main validation method - completely different from
        the old Validator that used LLM self-critique.

        Args:
            signal: Signal to validate

        Returns:
            Validation result with accuracy and learned facts
        """
        self.signals_validated += 1

        # Extract factual claims from signal content
        claims = self._extract_claims(signal.content)

        if not claims:
            # No factual claims to verify
            return ValidationResult(
                signal_id=signal.id,
                accuracy=0.5,  # Neutral
                verified_claims=0,
                total_claims=0,
                method='no_claims',
                confidence=0.5,
                learned_facts=0,
                conflicts=0,
                evidence=["No factual claims extracted"],
            )

        # Verify each claim
        verified_count = 0
        total_confidence = 0.0
        learned_this_signal = 0
        conflicts_this_signal = 0
        all_evidence = []

        for claim in claims:
            # 1. Check knowledge base first (may already know this)
            kb_result = self.kb.query(claim)

            if kb_result['known'] and kb_result['confidence'] >= self.confidence_threshold:
                # Already known with high confidence
                verified_count += 1
                total_confidence += kb_result['confidence']
                all_evidence.append(f"KB: {claim[:50]} (conf={kb_result['confidence']:.2f})")
                continue

            # 2. Unknown or low confidence - verify via external sources
            verification = await self.verifier.verify(claim)

            if verification['verified']:
                verified_count += 1
                total_confidence += verification['confidence']
                all_evidence.append(f"{verification.get('method', 'external')}: {claim[:50]}")

                # 3. LEARN verified fact into knowledge base
                if verification['confidence'] >= self.confidence_threshold:
                    # Detect conflicts before learning
                    conflict = self.kb.detect_conflict(claim, verification['confidence'])
                    if conflict:
                        conflicts_this_signal += 1
                        self.conflicts_detected += 1

                    # Learn fact
                    self.kb.learn(
                        claim=claim,
                        confidence=verification['confidence'],
                        source=verification.get('method', 'external'),
                        evidence=verification.get('evidence', '')
                    )
                    learned_this_signal += 1
                    self.facts_learned += 1

        self.claims_verified += verified_count

        # Calculate overall accuracy
        accuracy = verified_count / len(claims) if claims else 0.5
        avg_confidence = total_confidence / len(claims) if claims else 0.5

        return ValidationResult(
            signal_id=signal.id,
            accuracy=accuracy,
            verified_claims=verified_count,
            total_claims=len(claims),
            method='external_verification',
            confidence=avg_confidence,
            learned_facts=learned_this_signal,
            conflicts=conflicts_this_signal,
            evidence=all_evidence,
        )

    def _extract_claims(self, content: str) -> List[str]:
        """Extract factual claims from content.

        A factual claim is a statement that can be verified as true/false.

        Args:
            content: Signal content

        Returns:
            List of extracted claims
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)

        claims = []
        for sentence in sentences:
            sentence = sentence.strip()

            # Skip very short sentences
            if len(sentence) < 10:
                continue

            # Skip questions
            if sentence.endswith('?'):
                continue

            # Skip opinion markers
            opinion_markers = ['i think', 'i believe', 'in my opinion', 'perhaps', 'maybe',
                             'probably', 'might', 'could be', 'seems like']
            if any(marker in sentence.lower() for marker in opinion_markers):
                continue

            # Check for factual indicators
            factual_indicators = [
                r'\b(is|are|was|were|has|have)\b',  # State of being
                r'\b(because|therefore|thus|hence)\b',  # Causation
                r'\b\d+',  # Numbers (often factual)
                r'\b(percent|%|ratio|increase|decrease)\b',  # Quantities
                r'\b(research|study|studies|evidence|data)\b',  # Citations
            ]

            if any(re.search(pattern, sentence.lower()) for pattern in factual_indicators):
                claims.append(sentence)

        return claims[:5]  # Limit to top 5 claims per signal

    async def run(self, signal_store, llm, min_strength: float = 0.3,
                  max_actions: int = 200, temperature: float = 0.5):
        """Main agent loop for compatibility with existing system.

        This signature matches the old Validator for drop-in replacement.

        Args:
            signal_store: Signal store to validate from
            llm: Language model (not used - we use external sources)
            min_strength: Minimum strength to boost signal (compatibility param)
            max_actions: Maximum validation actions to take
            temperature: Temperature (not used, for compatibility)

        Returns:
            None (modifies signals in place)
        """
        actions = 0

        while self.active and actions < max_actions:
            # Sample signals to validate (prioritize by type)
            # For compatibility with old agents, use sample_weighted
            solution_signals = signal_store.sample_weighted("SOLUTION", n=2) if hasattr(signal_store, 'sample_weighted') else []
            impl_signals = signal_store.sample_weighted("IMPLEMENTATION", n=2) if hasattr(signal_store, 'sample_weighted') else []
            evidence_signals = signal_store.sample_weighted("EVIDENCE", n=1) if hasattr(signal_store, 'sample_weighted') else []

            targets = solution_signals + impl_signals + evidence_signals

            # If no typed signals, fall back to all signals
            if not targets:
                all_signals = signal_store.get_all_signals() if hasattr(signal_store, 'get_all_signals') else list(signal_store.signals.values())
                if not all_signals:
                    break
                # Sample up to 5 random signals
                import random
                targets = random.sample(all_signals, min(5, len(all_signals)))

            if not targets:
                break

            # Validate each target signal
            for target in targets:
                if actions >= max_actions:
                    break

                result = await self.validate_signal(target)

                # Boost/reduce signal strength based on validation
                if result.accuracy >= 0.7:
                    target.strength = min(1.0, target.strength * 1.2)
                elif result.accuracy < 0.5:
                    target.strength = max(0.1, target.strength * 0.8)

                actions += 1

            # Sleep briefly between validation batches
            await asyncio.sleep(0.5)

        return None

    def get_stats(self) -> Dict:
        """Get validator statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'agent_id': self.agent_id,
            'signals_validated': self.signals_validated,
            'claims_verified': self.claims_verified,
            'facts_learned': self.facts_learned,
            'conflicts_detected': self.conflicts_detected,
            'kb_stats': self.kb.get_stats(),
            'verifier_stats': self.verifier.get_stats(),
        }

    def stop(self):
        """Stop the validator agent."""
        pass  # Cleanup if needed
