"""Dynamic information retrieval driven by swarm signal evolution.

The swarm's own discourse drives what information gets retrieved:
1. Initial keywords from task → search → dump to temp file
2. Scouts parse file, generate signals
3. New signals contain refined concepts → extract keywords → search again
4. Emergent refinement: swarm collectively discovers relevant information
5. After run: delete temp file
"""

import asyncio
from pathlib import Path
from typing import List, Set, Optional


class DynamicRetriever:
    """Swarm-driven dynamic information retrieval.

    Agents extract keywords from their signals, search for information,
    and incrementally build a context file. The swarm's evolving discourse
    refines what information is relevant.
    """

    def __init__(self, temp_file: str = "research/temp_context.txt", max_file_size_mb: float = 10.0):
        """Initialize dynamic retriever.

        Args:
            temp_file: Path to temporary context file
            max_file_size_mb: Maximum file size in MB (prevents excessive growth)
        """
        self.temp_file = Path(temp_file)
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.searched_queries: Set[str] = set()  # Track queries to avoid duplicates
        self._lock = asyncio.Lock()

        # Ensure parent directory exists
        self.temp_file.parent.mkdir(exist_ok=True, parents=True)

        # Clear existing temp file on initialization
        if self.temp_file.exists():
            self.temp_file.unlink()
            print(f"[RETRIEVER] Cleared existing temp file: {self.temp_file}")

    async def search_and_append(self, keywords: List[str], web_search_fn) -> bool:
        """Search for information using keywords and append to context file.

        Args:
            keywords: Keywords to search for
            web_search_fn: Async function that performs web search

        Returns:
            True if search successful, False if skipped/failed
        """
        if not keywords:
            return False

        # Create search query from keywords
        query = " ".join(keywords[:5])  # Use top 5 keywords

        async with self._lock:
            # Skip if already searched
            query_lower = query.lower()
            if query_lower in self.searched_queries:
                return False

            # Check file size limit
            if self.temp_file.exists() and self.temp_file.stat().st_size >= self.max_file_size:
                print(f"[RETRIEVER] File size limit reached ({self.max_file_size / 1024 / 1024:.1f}MB)")
                return False

            # Search for information
            try:
                print(f"[RETRIEVER] Searching: '{query}'")
                result = await web_search_fn(query)

                if result and len(result.strip()) > 50:
                    # Append to context file
                    with open(self.temp_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n\n=== Search: {query} ===\n")
                        f.write(result)
                        f.write("\n")

                    # Mark as searched
                    self.searched_queries.add(query_lower)

                    file_size = self.temp_file.stat().st_size / 1024  # KB
                    print(f"[RETRIEVER] Added {len(result)} chars (file now {file_size:.1f}KB)")
                    return True
                else:
                    print(f"[RETRIEVER] Search returned empty/short result")
                    return False

            except Exception as e:
                print(f"[RETRIEVER] Search error: {e}")
                return False

    def get_context(self) -> str:
        """Get full context from temp file.

        Returns:
            Context text (empty if file doesn't exist)
        """
        if not self.temp_file.exists():
            return ""

        try:
            return self.temp_file.read_text(encoding='utf-8')
        except Exception as e:
            print(f"[RETRIEVER] Error reading context: {e}")
            return ""

    def get_context_snippet(self, keywords: List[str], max_chars: int = 500) -> str:
        """Get relevant snippet from context based on keywords.

        Args:
            keywords: Keywords to find relevant sections
            max_chars: Maximum characters to return

        Returns:
            Relevant snippet
        """
        context = self.get_context()
        if not context or not keywords:
            return context[:max_chars] if context else ""

        # Find sections containing keywords
        sections = context.split("===")
        scored_sections = []

        for section in sections:
            section_lower = section.lower()
            score = sum(1 for kw in keywords if kw.lower() in section_lower)
            if score > 0:
                scored_sections.append((score, section))

        if not scored_sections:
            # No keyword matches, return start of context
            return context[:max_chars]

        # Return highest scoring section
        scored_sections.sort(key=lambda x: x[0], reverse=True)
        best_section = scored_sections[0][1]

        # Return snippet from best section
        return best_section[:max_chars]

    def cleanup(self):
        """Delete temporary context file."""
        if self.temp_file.exists():
            size_kb = self.temp_file.stat().st_size / 1024
            self.temp_file.unlink()
            print(f"[RETRIEVER] Deleted temp file: {self.temp_file} ({size_kb:.1f}KB)")

    def reset_for_next_round(self):
        """Reset retriever for next round: delete temp file, clear search history.

        This enables true iterative refinement where each round gets fresh information
        based on keywords extracted from the previous round's synthesis.
        """
        # Delete old temp file
        if self.temp_file.exists():
            size_kb = self.temp_file.stat().st_size / 1024
            self.temp_file.unlink()
            print(f"[ROUND] Deleted old context file ({size_kb:.1f}KB)")

        # Clear search history so same queries can be re-searched with new context
        num_searches = len(self.searched_queries)
        self.searched_queries.clear()
        print(f"[ROUND] Reset search history ({num_searches} previous queries cleared)")

    def get_stats(self) -> dict:
        """Get retrieval statistics.

        Returns:
            Dictionary with stats
        """
        file_size = 0
        if self.temp_file.exists():
            file_size = self.temp_file.stat().st_size

        return {
            'temp_file': str(self.temp_file),
            'file_exists': self.temp_file.exists(),
            'file_size_kb': file_size / 1024,
            'num_searches': len(self.searched_queries),
            'searches': list(self.searched_queries)
        }

    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from text for search.

        Args:
            text: Input text
            max_keywords: Maximum keywords to return

        Returns:
            List of keywords
        """
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'from', 'by', 'as',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their',
            'how', 'what', 'when', 'where', 'why', 'who', 'which'
        }

        # Tokenize and filter
        words = text.lower().replace('?', '').replace(',', '').replace('.', '').split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]

        # Return unique keywords
        unique_keywords = []
        seen = set()
        for kw in keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)
                if len(unique_keywords) >= max_keywords:
                    break

        return unique_keywords
