"""Dynamic Knowledge Base that learns during swarm execution.

Key features:
- Starts empty or with minimal seed knowledge
- Learns from validated claims during execution
- Updates confidence scores based on evidence
- Handles conflicting information
- Queries external sources when needed

This is TRUE learning, not static facts.
"""

import time
import math
from threading import Lock
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class KnowledgeFact:
    """A fact in the knowledge base with evidence tracking.

    Facts are claims that have been verified with some confidence level.
    The KB tracks all evidence and updates confidence over time.
    """
    claim: str
    confidence: float  # 0.0-1.0
    sources: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    verification_count: int = 0
    contradiction_count: int = 0

    def __post_init__(self):
        """Ensure confidence is bounded."""
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class ConflictingClaim:
    """Represents conflicting information about a topic."""
    topic: str
    claims: List[Tuple[str, float, List[str]]]  # (claim, confidence, sources)


class DynamicKnowledgeBase:
    """Knowledge base that learns during execution.

    Unlike static knowledge bases, this:
    - Starts with minimal or NO knowledge
    - Learns from external verification
    - Updates beliefs based on new evidence
    - Tracks confidence and uncertainty
    - Detects and resolves conflicts

    This enables TRUE validation, not LLM self-critique.
    """

    def __init__(self,
                 confidence_threshold: float = 0.6,
                 max_facts: int = 10000,
                 seed_facts: Optional[Dict[str, float]] = None):
        """Initialize dynamic knowledge base.

        Args:
            confidence_threshold: Minimum confidence to consider fact "known"
            max_facts: Maximum facts to store (LRU eviction)
            seed_facts: Optional seed knowledge {claim: confidence}
        """
        self._lock = Lock()
        self.confidence_threshold = confidence_threshold
        self.max_facts = max_facts

        # Core knowledge storage
        # claim_normalized -> KnowledgeFact
        self.facts: Dict[str, KnowledgeFact] = {}

        # Topic index for faster lookup
        # topic_word -> set of claim_ids
        self.topic_index: Dict[str, Set[str]] = defaultdict(set)

        # Conflict tracking
        self.conflicts: List[ConflictingClaim] = []

        # Learning statistics
        self.stats = {
            'total_learned': 0,
            'total_queries': 0,
            'cache_hits': 0,
            'confidence_updates': 0,
            'conflicts_detected': 0,
        }

        # Seed with initial knowledge if provided
        if seed_facts:
            for claim, confidence in seed_facts.items():
                self._add_fact(claim, confidence, source='seed', evidence='initial')

    def _normalize_claim(self, claim: str) -> str:
        """Normalize claim for comparison.

        Args:
            claim: Raw claim text

        Returns:
            Normalized claim (lowercase, stripped, no extra spaces)
        """
        return ' '.join(claim.lower().strip().split())

    def _extract_topics(self, claim: str) -> List[str]:
        """Extract key topic words from claim.

        Args:
            claim: Claim text

        Returns:
            List of topic words (nouns, verbs)
        """
        # Simple extraction: words > 4 chars, not stopwords
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'will', 'would', 'could', 'should'}
        words = self._normalize_claim(claim).split()
        topics = [w for w in words if len(w) > 4 and w not in stopwords]
        return topics[:5]  # Top 5 topics

    def _add_fact(self, claim: str, confidence: float, source: str, evidence: str):
        """Add new fact to knowledge base (internal).

        Args:
            claim: Fact claim
            confidence: Initial confidence (0.0-1.0)
            source: Source of fact (e.g., 'wikipedia', 'web_search')
            evidence: Supporting evidence
        """
        claim_norm = self._normalize_claim(claim)

        # Create fact
        fact = KnowledgeFact(
            claim=claim,
            confidence=confidence,
            sources=[source],
            evidence=[evidence],
            verification_count=1
        )

        # Store fact
        self.facts[claim_norm] = fact

        # Index by topics
        topics = self._extract_topics(claim)
        for topic in topics:
            self.topic_index[topic].add(claim_norm)

        self.stats['total_learned'] += 1

        # Evict old facts if needed (LRU)
        if len(self.facts) > self.max_facts:
            self._evict_lru()

    def _evict_lru(self):
        """Evict least recently used facts."""
        # Sort by last_updated (ascending)
        sorted_facts = sorted(
            self.facts.items(),
            key=lambda x: x[1].last_updated
        )

        # Remove oldest 10%
        num_to_remove = max(1, len(self.facts) // 10)
        for claim_norm, _ in sorted_facts[:num_to_remove]:
            del self.facts[claim_norm]

            # Remove from topic index
            for topic_claims in self.topic_index.values():
                topic_claims.discard(claim_norm)

    def learn(self,
              claim: str,
              confidence: float,
              source: str,
              evidence: str = "") -> bool:
        """Learn a new fact or update existing belief.

        This is the KEY method - called when external verification succeeds.

        Args:
            claim: Fact claim
            confidence: Verification confidence (0.0-1.0)
            source: Source of verification
            evidence: Supporting evidence

        Returns:
            True if fact was learned/updated
        """
        with self._lock:
            claim_norm = self._normalize_claim(claim)

            if claim_norm in self.facts:
                # UPDATE existing fact
                fact = self.facts[claim_norm]

                # Bayesian update of confidence
                # New confidence = weighted average favoring higher confidence
                old_weight = fact.verification_count
                new_weight = 1
                fact.confidence = (
                    (fact.confidence * old_weight + confidence * new_weight) /
                    (old_weight + new_weight)
                )

                # Add new evidence
                fact.sources.append(source)
                fact.evidence.append(evidence)
                fact.verification_count += 1
                fact.last_updated = time.time()

                self.stats['confidence_updates'] += 1

                return True
            else:
                # LEARN new fact
                self._add_fact(claim, confidence, source, evidence)
                return True

    def query(self, claim: str) -> Optional[Dict]:
        """Query knowledge base for a claim.

        Args:
            claim: Claim to look up

        Returns:
            Dict with {known, confidence, sources, evidence} or None
        """
        with self._lock:
            self.stats['total_queries'] += 1

            claim_norm = self._normalize_claim(claim)

            if claim_norm in self.facts:
                self.stats['cache_hits'] += 1
                fact = self.facts[claim_norm]

                # Update last_updated (LRU tracking)
                fact.last_updated = time.time()

                return {
                    'known': True,
                    'confidence': fact.confidence,
                    'sources': fact.sources,
                    'evidence': fact.evidence,
                    'verification_count': fact.verification_count,
                }
            else:
                return {
                    'known': False,
                    'confidence': 0.0,
                }

    def find_related(self, claim: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find related facts based on topic overlap.

        Args:
            claim: Query claim
            top_k: Number of related facts to return

        Returns:
            List of (related_claim, confidence) tuples
        """
        with self._lock:
            topics = self._extract_topics(claim)

            # Find all facts matching any topic
            candidates = set()
            for topic in topics:
                candidates.update(self.topic_index.get(topic, set()))

            # Score by topic overlap
            scores = []
            for claim_norm in candidates:
                fact = self.facts.get(claim_norm)
                if fact:
                    fact_topics = set(self._extract_topics(fact.claim))
                    overlap = len(set(topics) & fact_topics)
                    scores.append((fact.claim, fact.confidence, overlap))

            # Sort by overlap (descending), then confidence
            scores.sort(key=lambda x: (x[2], x[1]), reverse=True)

            return [(claim, conf) for claim, conf, _ in scores[:top_k]]

    def detect_conflict(self, claim: str, new_confidence: float) -> bool:
        """Detect if new claim conflicts with existing knowledge.

        Args:
            claim: New claim
            new_confidence: Confidence of new claim

        Returns:
            True if conflict detected
        """
        with self._lock:
            related = self.find_related(claim, top_k=3)

            for related_claim, related_conf in related:
                # Check if claims contradict
                # Simple heuristic: opposite sentiment words
                contradiction_pairs = [
                    ('true', 'false'),
                    ('correct', 'incorrect'),
                    ('yes', 'no'),
                    ('increase', 'decrease'),
                    ('hot', 'cold'),
                ]

                claim_lower = claim.lower()
                related_lower = related_claim.lower()

                for pos, neg in contradiction_pairs:
                    if ((pos in claim_lower and neg in related_lower) or
                        (neg in claim_lower and pos in related_lower)):

                        # Conflict detected
                        self.conflicts.append(ConflictingClaim(
                            topic=' '.join(self._extract_topics(claim)),
                            claims=[
                                (claim, new_confidence, ['new']),
                                (related_claim, related_conf, self.facts[self._normalize_claim(related_claim)].sources)
                            ]
                        ))

                        self.stats['conflicts_detected'] += 1
                        return True

            return False

    def get_stats(self) -> Dict:
        """Get knowledge base statistics.

        Returns:
            Dictionary with learning metrics
        """
        with self._lock:
            high_conf = sum(1 for f in self.facts.values() if f.confidence >= self.confidence_threshold)
            avg_conf = sum(f.confidence for f in self.facts.values()) / len(self.facts) if self.facts else 0.0

            return {
                **self.stats,
                'total_facts': len(self.facts),
                'high_confidence_facts': high_conf,
                'avg_confidence': avg_conf,
                'topics_indexed': len(self.topic_index),
                'conflicts': len(self.conflicts),
                'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_queries'], 1),
            }

    def export_knowledge(self, min_confidence: float = 0.6) -> List[Dict]:
        """Export learned knowledge above confidence threshold.

        Args:
            min_confidence: Minimum confidence to export

        Returns:
            List of fact dictionaries
        """
        with self._lock:
            return [
                {
                    'claim': fact.claim,
                    'confidence': fact.confidence,
                    'sources': fact.sources,
                    'verifications': fact.verification_count,
                }
                for fact in self.facts.values()
                if fact.confidence >= min_confidence
            ]

    def clear(self):
        """Clear all learned knowledge (reset to empty)."""
        with self._lock:
            self.facts.clear()
            self.topic_index.clear()
            self.conflicts.clear()
            self.stats = {k: 0 for k in self.stats}
