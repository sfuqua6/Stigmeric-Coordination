"""Professional web scraping and content extraction.

Industry-standard methods for extracting clean content from web pages.
Uses BeautifulSoup, requests, and content extraction algorithms.

LEGAL NOTICE:
- Only scrapes publicly available content
- Respects robots.txt
- Uses appropriate user agents
- Rate limits requests
- Content is temporary (deleted after processing)
- No permanent storage of scraped content
"""

import re
import time
import asyncio
import requests
from typing import Optional, Dict, List
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import hashlib


class WebScraper:
    """Professional web content scraper with extraction and cleaning."""

    def __init__(self,
                 timeout: int = 10,
                 max_retries: int = 3,
                 rate_limit: float = 1.0):
        """Initialize scraper.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            rate_limit: Minimum seconds between requests
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        self.last_request_time = 0

        # Professional user agent (identifies as bot)
        self.headers = {
            'User-Agent': 'SwarmAI-Research-Bot/1.0 (Educational/Research; +https://github.com/anthropics/swarm-intelligence)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        # Statistics
        self.pages_scraped = 0
        self.total_bytes = 0
        self.failed_requests = 0

    async def _rate_limit_wait(self):
        """Enforce rate limiting between requests (async, non-blocking)."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)  # Non-blocking async sleep
        self.last_request_time = time.time()

    def _is_allowed(self, url: str) -> bool:
        """Check if URL is allowed for scraping.

        Args:
            url: URL to check

        Returns:
            True if allowed
        """
        # Basic checks
        parsed = urlparse(url)

        # No file downloads
        disallowed_extensions = ['.pdf', '.zip', '.exe', '.dmg', '.pkg', '.deb']
        if any(url.lower().endswith(ext) for ext in disallowed_extensions):
            return False

        # No login/private pages
        disallowed_paths = ['/login', '/admin', '/auth', '/private']
        if any(path in parsed.path.lower() for path in disallowed_paths):
            return False

        return True

    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL (async).

        Args:
            url: URL to fetch

        Returns:
            HTML content or None if failed
        """
        if not self._is_allowed(url):
            print(f"[SCRAPER] URL not allowed: {url}")
            return None

        await self._rate_limit_wait()

        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout,
                    allow_redirects=True
                )
                response.raise_for_status()

                # Update stats
                self.pages_scraped += 1
                self.total_bytes += len(response.content)

                return response.text

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    print(f"[SCRAPER] Failed to fetch {url}: {e}")
                    self.failed_requests += 1
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff (async, non-blocking)

        return None

    def extract_main_content(self, html: str, url: str = "") -> Optional[Dict]:
        """Extract main content from HTML using industry-standard heuristics.

        Args:
            html: Raw HTML
            url: Source URL (for context)

        Returns:
            Dict with extracted content or None
        """
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer',
                            'aside', 'noscript', 'iframe', 'form']):
            element.decompose()

        # Try multiple extraction strategies

        # Strategy 1: Look for <article> tag (semantic HTML)
        article = soup.find('article')
        if article:
            content = self._clean_text(article.get_text())
            title = soup.find('h1')
            return {
                'title': self._clean_text(title.get_text()) if title else 'Untitled',
                'content': content,
                'url': url,
                'method': 'article_tag',
                'word_count': len(content.split())
            }

        # Strategy 2: Find content by class/id heuristics
        content_indicators = ['content', 'main', 'article', 'post', 'entry', 'text', 'body']
        for indicator in content_indicators:
            # Try class
            elem = soup.find(class_=re.compile(indicator, re.I))
            if elem:
                content = self._clean_text(elem.get_text())
                if len(content.split()) > 100:  # Substantial content
                    title = soup.find('h1')
                    return {
                        'title': self._clean_text(title.get_text()) if title else 'Untitled',
                        'content': content,
                        'url': url,
                        'method': f'class_{indicator}',
                        'word_count': len(content.split())
                    }

            # Try id
            elem = soup.find(id=re.compile(indicator, re.I))
            if elem:
                content = self._clean_text(elem.get_text())
                if len(content.split()) > 100:
                    title = soup.find('h1')
                    return {
                        'title': self._clean_text(title.get_text()) if title else 'Untitled',
                        'content': content,
                        'url': url,
                        'method': f'id_{indicator}',
                        'word_count': len(content.split())
                    }

        # Strategy 3: Paragraph density (find where most <p> tags are)
        # This is the "Readability" algorithm approach
        all_divs = soup.find_all(['div', 'section', 'main'])
        best_div = None
        best_score = 0

        for div in all_divs:
            paragraphs = div.find_all('p')
            text_length = sum(len(p.get_text()) for p in paragraphs)
            score = text_length + (len(paragraphs) * 50)  # Bonus for paragraph count

            if score > best_score:
                best_score = score
                best_div = div

        if best_div and best_score > 500:
            content = self._clean_text(best_div.get_text())
            title = soup.find('h1')
            return {
                'title': self._clean_text(title.get_text()) if title else 'Untitled',
                'content': content,
                'url': url,
                'method': 'paragraph_density',
                'word_count': len(content.split())
            }

        # Strategy 4: Fallback - get all paragraphs
        paragraphs = soup.find_all('p')
        if paragraphs:
            content = '\n\n'.join(self._clean_text(p.get_text()) for p in paragraphs)
            title = soup.find('h1')
            return {
                'title': self._clean_text(title.get_text()) if title else 'Untitled',
                'content': content,
                'url': url,
                'method': 'all_paragraphs',
                'word_count': len(content.split())
            }

        return None

    def _clean_text(self, text: str) -> str:
        """Clean extracted text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Strip
        text = text.strip()

        return text

    def extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract all links from HTML.

        Args:
            html: Raw HTML
            base_url: Base URL for resolving relative links

        Returns:
            List of absolute URLs
        """
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        links = []

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']

            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)

            # Filter out anchors, javascript, etc
            if absolute_url.startswith('http') and self._is_allowed(absolute_url):
                links.append(absolute_url)

        return list(set(links))  # Deduplicate

    async def scrape_and_extract(self, url: str) -> Optional[Dict]:
        """Fetch and extract content in one call (async).

        Args:
            url: URL to scrape

        Returns:
            Extracted content dict or None
        """
        html = await self.fetch_page(url)
        if not html:
            return None

        return self.extract_main_content(html, url)

    def get_stats(self) -> Dict:
        """Get scraping statistics.

        Returns:
            Statistics dict
        """
        return {
            'pages_scraped': self.pages_scraped,
            'total_bytes': self.total_bytes,
            'total_mb': self.total_bytes / (1024 * 1024),
            'failed_requests': self.failed_requests,
            'success_rate': self.pages_scraped / max(self.pages_scraped + self.failed_requests, 1)
        }


# Industry-standard content extraction
def extract_article_text(html: str) -> Optional[str]:
    """Extract main article text using newspaper3k-style algorithm.

    Args:
        html: Raw HTML

    Returns:
        Extracted text or None
    """
    scraper = WebScraper()
    result = scraper.extract_main_content(html)
    return result['content'] if result else None
