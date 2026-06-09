"""Round-based coordinator for iterative information refinement.

Each round:
1. Search phase: Extract keywords → web search (scripting) → dump to temp file
2. Processing phase: Scouts read file → LLM generates signals → agents process
3. Synthesis phase: Synthesize round results
4. Refinement phase: Extract new keywords from synthesis → delete temp file → next round
"""

import asyncio
from typing import List, Optional
from pathlib import Path


class RoundCoordinator:
    """Coordinates iterative rounds of information refinement.

    Each round builds on the previous:
    - Round 1: Broad exploration from task keywords
    - Round 2+: Deep refinement from previous round's insights
    """

    def __init__(self, num_rounds: int = 3, iterations_per_round: int = 20):
        """Initialize round coordinator.

        Args:
            num_rounds: Number of refinement rounds
            iterations_per_round: LLM iterations per round
        """
        self.num_rounds = num_rounds
        self.iterations_per_round = iterations_per_round
        self.round_syntheses = []  # Track what was learned each round
        self.round_keywords = []  # Track keyword evolution

    def extract_keywords_from_task(self, task_prompt: str) -> List[str]:
        """Extract initial keywords from task prompt with improved quality.

        Args:
            task_prompt: The task/question

        Returns:
            List of keywords for Round 1 search
        """
        stop_words = {
            'how', 'can', 'we', 'what', 'when', 'where', 'why', 'who', 'which',
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'must',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'from', 'by', 'as',
            'make', 'made', 'makes', 'making', 'get', 'getting', 'way', 'ways'
        }

        # Improved tokenization - preserve case for proper nouns
        import re
        # Remove punctuation but preserve words
        words = re.findall(r'\b[a-zA-Z]+\b', task_prompt)

        # Score words by importance
        scored_keywords = []
        for word in words:
            word_lower = word.lower()

            # Skip stopwords and very short words
            if word_lower in stop_words or len(word_lower) < 3:
                continue

            score = 1.0

            # Boost capitalized words (likely important concepts)
            if word[0].isupper() and word_lower not in ['how', 'what', 'when']:
                score += 2.0

            # Boost longer words (more specific concepts)
            if len(word_lower) >= 6:
                score += 1.0

            # Boost domain-specific terms
            domain_terms = ['sustainable', 'climate', 'energy', 'technology',
                          'economic', 'social', 'environmental', 'policy', 'system']
            if any(term in word_lower for term in domain_terms):
                score += 1.5

            scored_keywords.append((word_lower, score))

        # Sort by score (descending) and deduplicate
        scored_keywords.sort(key=lambda x: x[1], reverse=True)

        # Return unique keywords
        seen = set()
        unique = []
        for keyword, score in scored_keywords:
            if keyword not in seen:
                unique.append(keyword)
                seen.add(keyword)

        return unique[:10]  # Top 10 keywords

    def extract_keywords_from_synthesis(self, synthesis: str, max_keywords: int = 10) -> List[str]:
        """Extract refined keywords from round synthesis with improved quality.

        Args:
            synthesis: Text synthesized from previous round
            max_keywords: Maximum keywords to extract

        Returns:
            List of refined keywords for next round
        """
        if not synthesis:
            return []

        # Extended stopwords for synthesis (includes more common words)
        stop_words = {
            'how', 'can', 'we', 'what', 'when', 'where', 'why', 'who', 'which',
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'must',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'from', 'by', 'as',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their',
            'also', 'such', 'more', 'most', 'some', 'many', 'much', 'very',
            'through', 'into', 'than', 'then', 'them', 'there', 'here'
        }

        # Better tokenization
        import re
        words = re.findall(r'\b[a-zA-Z]+\b', synthesis)

        # Count word frequency (prefer words that appear multiple times = important concepts)
        from collections import Counter
        word_freq = Counter([w.lower() for w in words if len(w) > 3])

        # Score words by importance
        scored_keywords = []
        for word in words:
            word_lower = word.lower()

            # Skip stopwords and very short words
            if word_lower in stop_words or len(word_lower) < 4:
                continue

            # Skip if already added
            if any(kw == word_lower for kw, _ in scored_keywords):
                continue

            score = 1.0

            # Boost by frequency (repeated = important)
            freq = word_freq[word_lower]
            score += freq * 0.5

            # Boost capitalized words (proper nouns, key concepts)
            if word[0].isupper():
                score += 1.5

            # Boost longer words (more specific)
            if len(word_lower) >= 7:
                score += 1.0

            # Boost technical/domain terms
            domain_terms = ['sustainable', 'climate', 'energy', 'technology', 'renewable',
                          'economic', 'social', 'environmental', 'policy', 'system', 'infrastructure',
                          'emissions', 'electric', 'solar', 'transportation', 'efficiency']
            if any(term in word_lower for term in domain_terms):
                score += 2.0

            scored_keywords.append((word_lower, score))

        # Sort by score (descending)
        scored_keywords.sort(key=lambda x: x[1], reverse=True)

        # Return top keywords
        return [kw for kw, score in scored_keywords[:max_keywords]]

    def get_round_info(self, round_num: int) -> dict:
        """Get information about a specific round.

        Args:
            round_num: Round number (0-indexed)

        Returns:
            Dictionary with round info
        """
        if round_num >= len(self.round_syntheses):
            return {}

        return {
            'round': round_num,
            'synthesis': self.round_syntheses[round_num],
            'keywords': self.round_keywords[round_num] if round_num < len(self.round_keywords) else []
        }

    def record_round(self, round_num: int, synthesis: str, keywords: List[str]):
        """Record results from a round.

        Args:
            round_num: Round number
            synthesis: What was learned this round
            keywords: Keywords that will drive next round
        """
        # Ensure lists are long enough
        while len(self.round_syntheses) <= round_num:
            self.round_syntheses.append("")
        while len(self.round_keywords) <= round_num:
            self.round_keywords.append([])

        # Handle None synthesis (generation failure)
        if synthesis is None:
            synthesis = f"[Round {round_num + 1} synthesis failed - no output generated]"
            print(f"\n[ROUND {round_num + 1}] WARNING: Synthesis failed (returned None)")
        else:
            print(f"\n[ROUND {round_num + 1}] Synthesis: {synthesis[:100]}...")

        self.round_syntheses[round_num] = synthesis
        self.round_keywords[round_num] = keywords

        print(f"[ROUND {round_num + 1}] Next keywords: {keywords[:5]}")

    def get_summary(self) -> str:
        """Get summary of all rounds.

        Returns:
            Summary text
        """
        summary = f"Completed {len(self.round_syntheses)} rounds of iterative refinement:\n\n"

        for i, synthesis in enumerate(self.round_syntheses):
            summary += f"Round {i + 1}:\n"
            summary += f"  Keywords: {self.round_keywords[i][:5]}\n"
            if synthesis and len(synthesis) > 0:
                preview = synthesis[:150] if len(synthesis) > 150 else synthesis
                summary += f"  Learned: {preview}\n\n"
            else:
                summary += f"  Learned: [No synthesis generated]\n\n"

        return summary
