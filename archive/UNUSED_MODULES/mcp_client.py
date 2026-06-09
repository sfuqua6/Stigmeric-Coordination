"""MCP Client for external knowledge retrieval.

Integrates with external knowledge sources to validate insights discovered
in documents. Supports web search, Wikipedia, and other sources.
"""

from typing import List, Dict, Optional
import asyncio
import re


class MCPClient:
    """Client for accessing external knowledge sources.

    Supports multiple search backends with automatic fallback:
    1. DuckDuckGo web search (no API key required)
    2. Wikipedia API
    3. Simulated results (fallback for testing)
    """

    def __init__(self, enable_web_search: bool = True, enable_wikipedia: bool = True):
        """Initialize MCP client.

        Args:
            enable_web_search: Enable DuckDuckGo web search
            enable_wikipedia: Enable Wikipedia search
        """
        self.enable_web_search = enable_web_search
        self.enable_wikipedia = enable_wikipedia
        self.tools_available = []

        # Try to import optional dependencies
        self.has_duckduckgo = False
        self.has_wikipedia = False

        if enable_web_search:
            try:
                from duckduckgo_search import DDGS
                self.ddgs = DDGS()
                self.has_duckduckgo = True
                self.tools_available.append("DuckDuckGo")
                print("[MCP] DuckDuckGo search enabled")
            except ImportError:
                print("[MCP] DuckDuckGo not available (install: pip install duckduckgo-search)")

        if enable_wikipedia:
            try:
                import wikipedia
                self.wikipedia = wikipedia
                self.has_wikipedia = True
                self.tools_available.append("Wikipedia")
                print("[MCP] Wikipedia search enabled")
            except ImportError:
                print("[MCP] Wikipedia not available (install: pip install wikipedia)")

        if not self.tools_available:
            print("[MCP] WARNING: No external search tools available - using simulated results")
            print("[MCP] Install: pip install duckduckgo-search wikipedia")

    async def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search external sources using available tools.

        Tries multiple sources and combines results.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results with summary, source, url, relevance
        """
        results = []

        # Try web search first (usually more comprehensive)
        if self.has_duckduckgo:
            web_results = await self._search_duckduckgo(query, max_results)
            results.extend(web_results)

        # Try Wikipedia (often high quality for factual queries)
        if self.has_wikipedia and len(results) < max_results:
            wiki_result = await self.search_wikipedia(query)
            if wiki_result:
                results.append(wiki_result)

        # Fallback to simulated if no real results
        if not results:
            await asyncio.sleep(0.1)
            results = [
                {
                    'summary': f"[SIMULATED] No external sources available for: {query}",
                    'source': "Simulated",
                    'url': "",
                    'type': 'simulated',
                    'relevance': 0.3
                }
            ]

        # Deduplicate and rank by relevance
        results = self._deduplicate_results(results)
        results = sorted(results, key=lambda x: x.get('relevance', 0.5), reverse=True)

        return results[:max_results]

    async def _search_duckduckgo(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search using DuckDuckGo.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of search results
        """
        if not self.has_duckduckgo:
            return []

        try:
            # Run in thread pool since duckduckgo-search is synchronous
            loop = asyncio.get_event_loop()
            raw_results = await loop.run_in_executor(
                None,
                lambda: list(self.ddgs.text(query, max_results=max_results))
            )

            results = []
            for item in raw_results:
                # Extract and clean summary
                summary = item.get('body', '')
                if not summary:
                    summary = item.get('title', '')

                # Calculate relevance based on query match
                relevance = self._calculate_relevance(query, summary)

                results.append({
                    'summary': self._clean_text(summary),
                    'source': item.get('title', 'Web'),
                    'url': item.get('href', ''),
                    'type': 'web',
                    'relevance': relevance
                })

            return results

        except Exception as e:
            print(f"[MCP] DuckDuckGo search error: {e}")
            return []

    async def search_wikipedia(self, query: str) -> Optional[Dict]:
        """Search Wikipedia for article.

        Args:
            query: Search query

        Returns:
            Wikipedia result or None
        """
        if not self.has_wikipedia:
            return None

        try:
            loop = asyncio.get_event_loop()

            # Search for matching pages
            search_results = await loop.run_in_executor(
                None,
                lambda: self.wikipedia.search(query, results=3)
            )

            if not search_results:
                return None

            # Get summary of first result
            page_title = search_results[0]
            summary = await loop.run_in_executor(
                None,
                lambda: self.wikipedia.summary(page_title, sentences=3, auto_suggest=False)
            )

            # Get page URL
            page = await loop.run_in_executor(
                None,
                lambda: self.wikipedia.page(page_title, auto_suggest=False)
            )

            relevance = self._calculate_relevance(query, summary)

            return {
                'summary': self._clean_text(summary),
                'source': f"Wikipedia: {page_title}",
                'url': page.url,
                'type': 'wikipedia',
                'relevance': min(relevance + 0.1, 1.0)  # Slight boost for Wikipedia
            }

        except self.wikipedia.exceptions.DisambiguationError as e:
            # Disambiguation page - try first option
            try:
                option = e.options[0]
                summary = await loop.run_in_executor(
                    None,
                    lambda: self.wikipedia.summary(option, sentences=3, auto_suggest=False)
                )
                page = await loop.run_in_executor(
                    None,
                    lambda: self.wikipedia.page(option, auto_suggest=False)
                )

                relevance = self._calculate_relevance(query, summary)

                return {
                    'summary': self._clean_text(summary),
                    'source': f"Wikipedia: {option}",
                    'url': page.url,
                    'type': 'wikipedia',
                    'relevance': relevance
                }
            except Exception:
                return None

        except self.wikipedia.exceptions.PageError:
            # Page not found
            return None

        except Exception as e:
            print(f"[MCP] Wikipedia search error: {e}")
            return None

    async def search_arxiv(self, query: str) -> List[Dict]:
        """Search arXiv for research papers.

        Args:
            query: Search query

        Returns:
            List of arXiv results
        """
        # TODO: Integrate actual arXiv API
        # For now, return empty to avoid simulated results
        return []

    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score between query and text.

        Uses simple keyword matching and query term presence.

        Args:
            query: Search query
            text: Result text

        Returns:
            Relevance score 0.0-1.0
        """
        if not text:
            return 0.0

        query_lower = query.lower()
        text_lower = text.lower()

        # Check for exact phrase match
        if query_lower in text_lower:
            return 0.9

        # Count query term matches
        query_terms = re.findall(r'\w+', query_lower)
        query_terms = [t for t in query_terms if len(t) > 3]  # Skip short words

        if not query_terms:
            return 0.5

        matches = sum(1 for term in query_terms if term in text_lower)
        match_ratio = matches / len(query_terms)

        # Score based on match ratio
        if match_ratio >= 0.8:
            return 0.8
        elif match_ratio >= 0.6:
            return 0.7
        elif match_ratio >= 0.4:
            return 0.6
        elif match_ratio >= 0.2:
            return 0.5
        else:
            return 0.4

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters that might cause issues
        text = text.replace('\n', ' ').replace('\r', '')

        # Limit length
        if len(text) > 500:
            text = text[:497] + "..."

        return text.strip()

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on content similarity.

        Args:
            results: List of search results

        Returns:
            Deduplicated list
        """
        if len(results) <= 1:
            return results

        unique_results = []
        seen_summaries = set()

        for result in results:
            summary = result.get('summary', '').lower()

            # Simple deduplication by checking if very similar summary exists
            is_duplicate = False
            for seen in seen_summaries:
                # Check overlap
                words_new = set(re.findall(r'\w+', summary))
                words_seen = set(re.findall(r'\w+', seen))

                if len(words_new) == 0 or len(words_seen) == 0:
                    continue

                overlap = len(words_new & words_seen) / len(words_new | words_seen)

                if overlap > 0.7:  # 70% similarity threshold
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_results.append(result)
                seen_summaries.add(summary)

        return unique_results
