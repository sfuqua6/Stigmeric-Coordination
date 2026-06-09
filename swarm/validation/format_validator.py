"""Format validation for creative outputs (haiku, poems, etc.)."""

import re
from typing import Optional, Tuple


class FormatValidator:
    """Validates creative outputs against format requirements."""

    @staticmethod
    def count_syllables(word: str) -> int:
        """Approximate syllable count using simple heuristic.

        Args:
            word: Word to count syllables in

        Returns:
            Estimated syllable count
        """
        word = word.lower().strip()
        if not word:
            return 0

        # Remove non-alphabetic characters
        word = re.sub(r'[^a-z]', '', word)

        # Count vowel groups
        vowel_groups = re.findall(r'[aeiouy]+', word)
        count = len(vowel_groups)

        # Adjust for silent 'e' at end
        if word.endswith('e') and count > 1:
            count -= 1

        # Ensure at least 1 syllable
        return max(1, count)

    @staticmethod
    def count_line_syllables(line: str) -> int:
        """Count syllables in a line.

        Args:
            line: Line of text

        Returns:
            Total syllable count
        """
        words = line.split()
        return sum(FormatValidator.count_syllables(word) for word in words)

    @staticmethod
    def validate_haiku(text: str) -> Tuple[bool, str]:
        """Validate text as haiku (3 lines, 5-7-5 syllables).

        Args:
            text: Text to validate

        Returns:
            (is_valid, message) tuple
        """
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]

        # Check line count
        if len(lines) != 3:
            return False, f"Haiku must have 3 lines, found {len(lines)}"

        # Count syllables per line
        syllable_counts = [FormatValidator.count_line_syllables(line) for line in lines]

        # Check 5-7-5 pattern (with some tolerance for approximation errors)
        target = [5, 7, 5]
        tolerance = 1  # Allow ±1 syllable per line

        for i, (actual, expected) in enumerate(zip(syllable_counts, target)):
            if abs(actual - expected) > tolerance:
                return False, f"Line {i+1}: expected ~{expected} syllables, found {actual} ('{lines[i]}')"

        return True, "Valid haiku structure (5-7-5 syllables across 3 lines)"

    @staticmethod
    def validate_poem_structure(text: str, min_lines: int = 2, max_lines: int = 20) -> Tuple[bool, str]:
        """Validate general poem structure.

        Args:
            text: Text to validate
            min_lines: Minimum line count
            max_lines: Maximum line count

        Returns:
            (is_valid, message) tuple
        """
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]

        if len(lines) < min_lines:
            return False, f"Poem must have at least {min_lines} lines, found {len(lines)}"

        if len(lines) > max_lines:
            return False, f"Poem must have at most {max_lines} lines, found {len(lines)}"

        # Check that lines have reasonable length
        for i, line in enumerate(lines):
            if len(line) < 3:
                return False, f"Line {i+1} too short: '{line}'"
            if len(line) > 200:
                return False, f"Line {i+1} too long (>{200} chars)"

        return True, f"Valid poem structure ({len(lines)} lines)"

    @staticmethod
    def is_off_topic(text: str, task_keywords: list) -> bool:
        """Check if text appears off-topic (e.g., exam rubrics, exercises).

        Args:
            text: Text to check
            task_keywords: Keywords that should appear in on-topic text

        Returns:
            True if text appears off-topic
        """
        text_lower = text.lower()

        # Check for common off-topic indicators
        off_topic_indicators = [
            'exercise', 'answer key', 'rubric', 'grading',
            'question 1', 'question 2', 'part a', 'part b',
            'instructions:', 'assignment:', 'homework:',
            'points:', 'score:', 'grade:', 'exam'
        ]

        # If contains multiple off-topic indicators, probably contaminated
        indicator_count = sum(1 for indicator in off_topic_indicators if indicator in text_lower)
        if indicator_count >= 2:
            return True

        # If no task keywords present and has indicators, likely off-topic
        has_keywords = any(keyword.lower() in text_lower for keyword in task_keywords)
        if not has_keywords and indicator_count >= 1:
            return True

        return False

    @staticmethod
    def validate_content_quality(text: str, min_length: int = 20) -> Tuple[bool, str]:
        """Validate basic content quality.

        Args:
            text: Text to validate
            min_length: Minimum character count

        Returns:
            (is_valid, message) tuple
        """
        text = text.strip()

        if not text or text == "_":
            return False, "Empty content"

        if len(text) < min_length:
            return False, f"Content too short ({len(text)} chars, min {min_length})"

        # Check for placeholder text
        placeholders = ['lorem ipsum', '...', 'todo', 'tbd', 'placeholder']
        text_lower = text.lower()
        if any(placeholder in text_lower for placeholder in placeholders):
            return False, "Contains placeholder text"

        return True, "Content quality acceptable"

    @staticmethod
    def detect_format_from_task(task_prompt: str) -> Optional[str]:
        """Detect expected format from task prompt.

        Args:
            task_prompt: The task prompt/question

        Returns:
            Format type ('haiku', 'poem', None)
        """
        prompt_lower = task_prompt.lower()

        if 'haiku' in prompt_lower:
            return 'haiku'
        elif any(word in prompt_lower for word in ['poem', 'poetry', 'verse']):
            return 'poem'

        return None

    @staticmethod
    def extract_task_keywords(task_prompt: str) -> list:
        """Extract key nouns/topics from task prompt for relevance checking.

        Args:
            task_prompt: The task prompt

        Returns:
            List of keywords
        """
        # Simple extraction: split on common words and take substantive terms
        words = task_prompt.lower().split()

        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'write', 'about', 'on', 'of', 'for', 'in', 'to', 'and', 'or'}
        keywords = [word.strip('.,!?:;') for word in words if word.lower() not in stop_words]

        # Return unique keywords longer than 2 chars
        return list(set(k for k in keywords if len(k) > 2))
