"""Knowledge retrieval from external sources for swarm agents.

Provides agents with access to external knowledge bases to support
deeper, evidence-based signal generation.
"""

import asyncio
import aiohttp
from typing import Optional, Dict, List
from collections import OrderedDict


class KnowledgeRetriever:
    """Retrieves knowledge from external sources (Wikipedia, web search)."""

    def __init__(self, cache_size: int = 100):
        """Initialize knowledge retriever.

        Args:
            cache_size: Maximum number of cached results
        """
        self.cache: OrderedDict = OrderedDict()
        self.cache_size = cache_size
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def search_wikipedia(self, query: str, sentences: int = 3) -> Optional[str]:
        """Search Wikipedia for information on a topic.

        Args:
            query: Search query
            sentences: Number of sentences to return from summary

        Returns:
            Wikipedia summary or None if not found
        """
        # Check cache
        cache_key = f"wiki:{query}:{sentences}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        await self._ensure_session()

        try:
            # Wikipedia API endpoint
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")

            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    summary = data.get("extract", "")

                    # Limit to requested sentences
                    if summary:
                        sentences_list = summary.split(". ")[:sentences]
                        result = ". ".join(sentences_list)
                        if not result.endswith("."):
                            result += "."

                        # Cache result
                        self.cache[cache_key] = result
                        if len(self.cache) > self.cache_size:
                            self.cache.popitem(last=False)

                        return result
                else:
                    return None

        except Exception as e:
            print(f"[KNOWLEDGE] Wikipedia search failed for '{query}': {e}")
            return None

    async def get_context(self, topic: str) -> Dict[str, str]:
        """Get contextual knowledge about a topic from multiple sources.

        Args:
            topic: Topic to research

        Returns:
            Dictionary with knowledge from different sources
        """
        context = {}

        # Try Wikipedia
        wiki_result = await self.search_wikipedia(topic)
        if wiki_result:
            context["wikipedia"] = wiki_result

        # Future: Add more sources (web search, academic papers, etc.)

        return context

    async def close(self):
        """Close the retriever and cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()

    def get_stats(self) -> Dict[str, int]:
        """Get retriever statistics.

        Returns:
            Dictionary with cache size and other stats
        """
        return {
            "cache_entries": len(self.cache),
            "cache_capacity": self.cache_size
        }
