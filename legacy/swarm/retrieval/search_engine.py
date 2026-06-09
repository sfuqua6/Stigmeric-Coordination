"""Professional search engine integration for knowledge retrieval.

Provides multiple search backends:
- Wikipedia API (official, free)
- DuckDuckGo Instant Answers (free, no API key)
- Web search with scraping

All content is temporary and transformed by agents.
"""

import asyncio
import time
import json
import requests
from typing import List, Dict, Optional
from urllib.parse import quote_plus
from .web_scraper import WebScraper
from swarm.core.logging_config import get_logger

logger = get_logger(__name__)


class WikipediaAPI:
    """Official Wikipedia API integration.

    Uses MediaWiki API for searching and extracting article content.
    Free and legal - no API key required.
    """

    def __init__(self, language: str = 'en'):
        """Initialize Wikipedia API client.

        Args:
            language: Wikipedia language code (default: en)
        """
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SwarmAI-Research/1.0 (Educational; +https://github.com/anthropics/swarm)'
        })

        # Rate limiting
        self.last_request = 0
        self.min_delay = 0.1  # 100ms between requests

        # Stats
        self.queries_made = 0
        self.articles_fetched = 0

    async def _rate_limit(self):
        """Enforce rate limiting (async, non-blocking)."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            await asyncio.sleep(self.min_delay - elapsed)  # Non-blocking async sleep
        self.last_request = time.time()

    async def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search Wikipedia for articles.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of search results with title, snippet
        """
        await self._rate_limit()
        self.queries_made += 1

        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': limit,
            'srprop': 'snippet|titlesnippet|timestamp'
        }

        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get('query', {}).get('search', []):
                results.append({
                    'title': item['title'],
                    'snippet': self._clean_html(item.get('snippet', '')),
                    'page_id': item['pageid'],
                    'timestamp': item.get('timestamp', '')
                })

            return results

        except requests.exceptions.RequestException as e:
            logger.warning(f"Wikipedia API request failed for query '{query}': {e}")
            return []  # Return empty but log the network error
        except Exception as e:
            logger.error(f"Unexpected error in Wikipedia search for '{query}': {e}", exc_info=True)
            return []  # Return empty but log unexpected errors with traceback

    async def get_article(self, title: str) -> Optional[Dict]:
        """Get full article content.

        Args:
            title: Article title

        Returns:
            Dict with title, content, summary, links
        """
        await self._rate_limit()
        self.articles_fetched += 1

        # Get article content and summary
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts|links',
            'exintro': True,  # Just intro section
            'explaintext': True,  # Plain text
            'pllimit': 'max'  # All links
        }

        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            pages = data.get('query', {}).get('pages', {})
            if not pages:
                return None

            # Get first (should be only) page
            page_data = list(pages.values())[0]

            if 'missing' in page_data:
                return None

            # Extract links for further exploration
            links = [link['title'] for link in page_data.get('links', [])[:10]]

            return {
                'title': page_data['title'],
                'content': page_data.get('extract', ''),
                'page_id': page_data['pageid'],
                'links': links,
                'source': 'wikipedia',
                'url': f"https://{self.language}.wikipedia.org/wiki/{quote_plus(title)}"
            }

        except requests.exceptions.RequestException as e:
            logger.warning(f"Wikipedia article fetch failed for '{title}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Wikipedia article '{title}': {e}", exc_info=True)
            return None

    async def get_full_article(self, title: str) -> Optional[Dict]:
        """Get complete article content (not just intro).

        Args:
            title: Article title

        Returns:
            Dict with full content
        """
        await self._rate_limit()

        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,  # Plain text
            'exintro': False  # Full article
        }

        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            pages = data.get('query', {}).get('pages', {})
            if not pages:
                return None

            page_data = list(pages.values())[0]

            if 'missing' in page_data:
                return None

            content = page_data.get('extract', '')

            return {
                'title': page_data['title'],
                'content': content,
                'word_count': len(content.split()),
                'source': 'wikipedia',
                'url': f"https://{self.language}.wikipedia.org/wiki/{quote_plus(title)}"
            }

        except requests.exceptions.RequestException as e:
            logger.warning(f"Wikipedia full article fetch failed for '{title}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching full Wikipedia article '{title}': {e}", exc_info=True)
            return None

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from snippet.

        Args:
            text: Text with HTML

        Returns:
            Clean text
        """
        import re
        return re.sub(r'<[^>]+>', '', text)

    def get_stats(self) -> Dict:
        """Get API usage stats.

        Returns:
            Statistics dict
        """
        return {
            'queries_made': self.queries_made,
            'articles_fetched': self.articles_fetched
        }


class DuckDuckGoSearch:
    """DuckDuckGo Instant Answer API.

    Free, no API key required. Provides instant answers and web results.
    """

    def __init__(self):
        """Initialize DuckDuckGo client."""
        self.base_url = "https://api.duckduckgo.com/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SwarmAI-Research/1.0'
        })

        self.queries_made = 0
        self.last_request = 0
        self.min_delay = 0.2

    async def _rate_limit(self):
        """Enforce rate limiting (async, non-blocking)."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            await asyncio.sleep(self.min_delay - elapsed)  # Non-blocking async sleep
        self.last_request = time.time()

    async def instant_answer(self, query: str) -> Optional[Dict]:
        """Get instant answer for query.

        Args:
            query: Search query

        Returns:
            Dict with answer and related info
        """
        await self._rate_limit()
        self.queries_made += 1

        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1
        }

        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract answer
            answer = data.get('AbstractText', '')
            if not answer:
                answer = data.get('Answer', '')

            if not answer:
                return None

            return {
                'query': query,
                'answer': answer,
                'source': data.get('AbstractSource', 'DuckDuckGo'),
                'url': data.get('AbstractURL', ''),
                'related_topics': [
                    topic.get('Text', '') for topic in data.get('RelatedTopics', [])
                    if isinstance(topic, dict) and 'Text' in topic
                ][:5],
                'type': data.get('Type', 'instant_answer')
            }

        except requests.exceptions.RequestException as e:
            logger.warning(f"DuckDuckGo instant answer failed for query '{query}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in DuckDuckGo instant answer for '{query}': {e}", exc_info=True)
            return None

    def get_stats(self) -> Dict:
        """Get usage stats.

        Returns:
            Statistics dict
        """
        return {
            'queries_made': self.queries_made
        }


class MultiSourceSearch:
    """Combines multiple search sources for comprehensive results."""

    def __init__(self):
        """Initialize multi-source search."""
        self.wikipedia = WikipediaAPI()
        self.ddg = DuckDuckGoSearch()
        self.scraper = WebScraper()

    async def search_all(self, query: str, depth: str = 'shallow') -> Dict:
        """Search across all sources.

        Args:
            query: Search query
            depth: 'shallow' (quick) or 'deep' (comprehensive)

        Returns:
            Combined results from all sources
        """
        results = {
            'query': query,
            'sources': {},
            'total_content': ''
        }

        # Wikipedia search
        wiki_results = await self.wikipedia.search(query, limit=3)
        if wiki_results:
            # Get first article
            article = await self.wikipedia.get_article(wiki_results[0]['title'])
            if article:
                results['sources']['wikipedia'] = article
                results['total_content'] += f"\n\n=== Wikipedia: {article['title']} ===\n{article['content']}"

        # DuckDuckGo instant answer
        ddg_result = await self.ddg.instant_answer(query)
        if ddg_result:
            results['sources']['duckduckgo'] = ddg_result
            results['total_content'] += f"\n\n=== DuckDuckGo ===\n{ddg_result['answer']}"

        # Deep mode: scrape top result
        if depth == 'deep' and wiki_results:
            for result in wiki_results[:2]:
                full_article = await self.wikipedia.get_full_article(result['title'])
                if full_article:
                    results['sources'][f"wikipedia_full_{result['title']}"] = full_article
                    # Add first 1000 words
                    content = ' '.join(full_article['content'].split()[:1000])
                    results['total_content'] += f"\n\n=== Wikipedia Full: {result['title']} ===\n{content}"

        return results

    def get_stats(self) -> Dict:
        """Get combined stats.

        Returns:
            Statistics from all sources
        """
        return {
            'wikipedia': self.wikipedia.get_stats(),
            'duckduckgo': self.ddg.get_stats(),
            'scraper': self.scraper.get_stats()
        }


def generate_search_queries(keywords: List[str], task_context: str = "") -> List[str]:
    """Generate effective search queries from keywords.

    Args:
        keywords: List of extracted keywords
        task_context: Optional task description for context

    Returns:
        List of search queries
    """
    queries = []

    # Single keyword queries (broad)
    for keyword in keywords[:5]:
        queries.append(keyword)

    # Two-keyword combinations (specific)
    for i in range(min(3, len(keywords))):
        for j in range(i + 1, min(i + 3, len(keywords))):
            queries.append(f"{keywords[i]} {keywords[j]}")

    # Context-enhanced queries
    if task_context:
        # Extract verb from task
        task_lower = task_context.lower()
        if 'how' in task_lower or 'make' in task_lower:
            # Add "how to" prefix
            for keyword in keywords[:3]:
                queries.append(f"how to {keyword}")

        if 'what' in task_lower:
            for keyword in keywords[:3]:
                queries.append(f"what is {keyword}")

        if 'why' in task_lower:
            for keyword in keywords[:3]:
                queries.append(f"why {keyword}")

    # Deduplicate and return top queries
    seen = set()
    unique_queries = []
    for q in queries:
        if q.lower() not in seen:
            unique_queries.append(q)
            seen.add(q.lower())

    return unique_queries[:10]  # Top 10 queries
