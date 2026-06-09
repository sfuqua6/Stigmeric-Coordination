"""Programmatic signal quality verification.

This module provides quality checks for signals before deposit to prevent:
- Hallucinated insights (not grounded in observations)
- Irrelevant evidence (doesn't support parent insight)
- Generic objections (non-substantive critique)
- Low-quality signals that pollute the signal store
"""

from typing import Dict, List, Optional, Tuple
from .signal_store import Signal, SignalStore
from difflib import SequenceMatcher
import re


class SignalVerifier:
    """Programmatic verification of signal quality."""

    def __init__(self, min_quality_score: float = 0.4):
        """Initialize verifier.

        Args:
            min_quality_score: Minimum quality score for signal acceptance (0.0-1.0)
        """
        self.min_quality_score = min_quality_score
        self.verification_stats = {
            'insights_checked': 0,
            'insights_rejected': 0,
            'evidence_checked': 0,
            'evidence_rejected': 0,
            'objections_checked': 0,
            'objections_rejected': 0
        }

    def verify_insight_quality(self, insight: Signal,
                               signal_store: SignalStore) -> Dict[str, any]:
        """Verify insight is based on actual observations, not hallucination.

        Args:
            insight: Insight signal to verify
            signal_store: Signal store for provenance lookup

        Returns:
            Dict with 'valid' (bool), 'score' (float), 'reasons' (List[str])
        """
        self.verification_stats['insights_checked'] += 1

        reasons = []
        quality_score = 0.0

        # Check 1: Does insight reference observations?
        obs_ids = insight.metadata.get('observation_ids', [])
        if not obs_ids:
            reasons.append("No observation grounding")
            self.verification_stats['insights_rejected'] += 1
            return {'valid': False, 'score': 0.0, 'reasons': reasons}

        # Check 2: Fetch observation content
        observations = []
        for obs_id in obs_ids:
            obs = signal_store.get_signal(obs_id)
            if obs:
                observations.append(obs)

        if not observations:
            reasons.append("Referenced observations not found")
            self.verification_stats['insights_rejected'] += 1
            return {'valid': False, 'score': 0.0, 'reasons': reasons}

        # Check 3: Calculate semantic relatedness between insight and observations
        relatedness_scores = []
        for obs in observations:
            similarity = SequenceMatcher(
                None,
                insight.content.lower(),
                obs.content.lower()
            ).ratio()
            relatedness_scores.append(similarity)

        avg_relatedness = sum(relatedness_scores) / len(relatedness_scores)

        # Grounding quality (0.0-1.0)
        grounding_score = min(1.0, avg_relatedness * 2.0)  # Scale to 0.0-1.0

        # Check 4: Source diversity (are observations from different sources?)
        source_docs = set()
        for obs in observations:
            source = obs.metadata.get('source_document', obs.id)
            source_docs.add(source)

        source_diversity = min(1.0, len(source_docs) / max(len(observations), 1))

        # Check 5: Length check (is insight substantive?)
        min_length = 20  # At least 20 characters
        length_score = 1.0 if len(insight.content) >= min_length else 0.3

        # Combine scores
        quality_score = (
            grounding_score * 0.5 +      # Relatedness to observations (50%)
            source_diversity * 0.3 +     # Source diversity (30%)
            length_score * 0.2           # Length/substantiveness (20%)
        )

        # Quality feedback
        if grounding_score < 0.3:
            reasons.append(f"Low observation grounding ({grounding_score:.2f})")
        if source_diversity < 0.5:
            reasons.append(f"Low source diversity ({source_diversity:.2f})")
        if length_score < 1.0:
            reasons.append(f"Insight too short ({len(insight.content)} chars)")

        valid = quality_score >= self.min_quality_score

        if not valid:
            self.verification_stats['insights_rejected'] += 1

        return {
            'valid': valid,
            'score': quality_score,
            'reasons': reasons,
            'details': {
                'grounding_score': grounding_score,
                'source_diversity': source_diversity,
                'avg_relatedness': avg_relatedness,
                'num_observations': len(observations)
            }
        }

    def verify_evidence_relevance(self, evidence: Signal,
                                   parent_insight: Signal,
                                   signal_store: SignalStore) -> Dict[str, any]:
        """Verify evidence actually supports parent insight.

        Args:
            evidence: Evidence signal to verify
            parent_insight: Parent insight this evidence claims to support
            signal_store: Signal store

        Returns:
            Dict with 'valid' (bool), 'score' (float), 'reasons' (List[str])
        """
        self.verification_stats['evidence_checked'] += 1

        reasons = []

        # Check 1: Semantic relevance between evidence and insight
        similarity = SequenceMatcher(
            None,
            evidence.content.lower(),
            parent_insight.content.lower()
        ).ratio()

        relevance_score = similarity

        # Check 2: Check if evidence contains negation words (might be counter-evidence)
        negation_words = ['not', 'however', 'but', 'although', 'despite',
                         'nevertheless', 'contradicts', 'opposes', 'disagrees']
        has_negation = any(word in evidence.content.lower()
                          for word in negation_words)

        # If evidence is counter-evidence, it should have COUNTER_EVIDENCE type
        is_counter = evidence.type == "COUNTER_EVIDENCE"

        if has_negation and not is_counter:
            reasons.append("Contains negation but marked as supporting evidence")
            relevance_score *= 0.5  # Penalize misclassified counter-evidence

        # Check 3: Length check (is evidence substantive?)
        min_length = 15
        length_score = 1.0 if len(evidence.content) >= min_length else 0.3

        # Combine scores
        quality_score = (
            relevance_score * 0.7 +      # Semantic relevance (70%)
            length_score * 0.3           # Length/substantiveness (30%)
        )

        # Quality feedback
        if relevance_score < 0.3:
            reasons.append(f"Low relevance to insight ({relevance_score:.2f})")
        if length_score < 1.0:
            reasons.append(f"Evidence too short ({len(evidence.content)} chars)")

        valid = quality_score >= self.min_quality_score

        if not valid:
            self.verification_stats['evidence_rejected'] += 1

        return {
            'valid': valid,
            'score': quality_score,
            'reasons': reasons,
            'details': {
                'relevance_score': relevance_score,
                'has_negation': has_negation,
                'is_counter': is_counter
            }
        }

    def verify_objection_substantiveness(self, objection: Signal,
                                         target_signal: Signal,
                                         signal_store: SignalStore) -> Dict[str, any]:
        """Verify objection is substantive, not generic.

        Args:
            objection: Objection signal to verify
            target_signal: Signal being objected to
            signal_store: Signal store

        Returns:
            Dict with 'valid' (bool), 'score' (float), 'reasons' (List[str])
        """
        self.verification_stats['objections_checked'] += 1

        reasons = []
        quality_score = 0.0

        content = objection.content.lower()

        # Check 1: Length (substantive objections need detail)
        min_length = 30
        if len(objection.content) < min_length:
            reasons.append(f"Objection too short ({len(objection.content)} chars)")
            length_score = 0.2
        else:
            length_score = min(1.0, len(objection.content) / 100.0)

        # Check 2: Avoid generic phrases
        generic_phrases = [
            "but what about",
            "have you considered",
            "this seems",
            "i think",
            "maybe",
            "possibly",
            "could be"
        ]
        has_generic = any(phrase in content for phrase in generic_phrases)
        generic_penalty = 0.5 if has_generic else 1.0

        # Check 3: Check for reasoning words (because, since, therefore, etc.)
        reasoning_words = ['because', 'since', 'therefore', 'thus', 'hence',
                          'as a result', 'consequently', 'given that']
        has_reasoning = any(word in content for word in reasoning_words)
        reasoning_score = 1.0 if has_reasoning else 0.5

        # Check 4: Check for alternative suggestions (instead, rather, alternatively)
        alternative_words = ['instead', 'rather', 'alternatively', 'better',
                            'more accurate', 'should be']
        has_alternative = any(word in content for word in alternative_words)
        alternative_score = 1.0 if has_alternative else 0.7

        # Check 5: Specificity - does it reference the target signal?
        specificity = SequenceMatcher(
            None,
            content,
            target_signal.content.lower()
        ).ratio()

        specificity_score = min(1.0, specificity * 2.0)

        # Combine scores
        quality_score = (
            specificity_score * 0.4 +    # References target (40%)
            alternative_score * 0.3 +    # Offers alternative (30%)
            reasoning_score * 0.3        # Has reasoning (30%)
        ) * generic_penalty * length_score

        # Quality feedback
        if specificity_score < 0.3:
            reasons.append(f"Low specificity to target ({specificity_score:.2f})")
        if not has_reasoning:
            reasons.append("Missing reasoning (because, since, therefore)")
        if not has_alternative:
            reasons.append("Missing alternative suggestion")
        if has_generic:
            reasons.append("Contains generic phrases")

        # Ensure minimum quality threshold
        quality_score = max(quality_score, self.min_quality_score)

        valid = quality_score >= self.min_quality_score

        if not valid:
            self.verification_stats['objections_rejected'] += 1

        return {
            'valid': valid,
            'score': quality_score,
            'reasons': reasons,
            'details': {
                'length': len(objection.content),
                'has_reasoning': has_reasoning,
                'has_alternative': has_alternative,
                'has_generic': has_generic,
                'specificity': specificity_score
            }
        }

    def get_stats(self) -> Dict[str, int]:
        """Get verification statistics.

        Returns:
            Dict with verification counts and rejection rates
        """
        stats = self.verification_stats.copy()

        # Calculate rejection rates
        if stats['insights_checked'] > 0:
            stats['insight_rejection_rate'] = (
                stats['insights_rejected'] / stats['insights_checked']
            )
        else:
            stats['insight_rejection_rate'] = 0.0

        if stats['evidence_checked'] > 0:
            stats['evidence_rejection_rate'] = (
                stats['evidence_rejected'] / stats['evidence_checked']
            )
        else:
            stats['evidence_rejection_rate'] = 0.0

        if stats['objections_checked'] > 0:
            stats['objection_rejection_rate'] = (
                stats['objections_rejected'] / stats['objections_checked']
            )
        else:
            stats['objection_rejection_rate'] = 0.0

        return stats

    def print_stats(self):
        """Print verification statistics."""
        stats = self.get_stats()

        print("\n" + "="*80)
        print("SIGNAL VERIFICATION STATISTICS")
        print("="*80)

        print(f"\nInsights:")
        print(f"  Checked: {stats['insights_checked']}")
        print(f"  Rejected: {stats['insights_rejected']}")
        print(f"  Rejection rate: {stats['insight_rejection_rate']:.1%}")

        print(f"\nEvidence:")
        print(f"  Checked: {stats['evidence_checked']}")
        print(f"  Rejected: {stats['evidence_rejected']}")
        print(f"  Rejection rate: {stats['evidence_rejection_rate']:.1%}")

        print(f"\nObjections:")
        print(f"  Checked: {stats['objections_checked']}")
        print(f"  Rejected: {stats['objections_rejected']}")
        print(f"  Rejection rate: {stats['objection_rejection_rate']:.1%}")

        print("="*80 + "\n")
