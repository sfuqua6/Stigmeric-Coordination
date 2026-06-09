"""Simple web search for information retrieval.

Provides basic web search capability without requiring API keys.
Uses DuckDuckGo lite HTML interface for simple searching.
"""

import asyncio
from typing import Optional


async def simple_web_search(query: str, max_results: int = 3) -> str:
    """Simple web search using requests + beautifulsoup.

    Args:
        query: Search query
        max_results: Maximum number of results to fetch

    Returns:
        Combined text from search results
    """
    try:
        # Try to import required libraries
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("[WEB_SEARCH] Warning: requests or beautifulsoup4 not installed")
        print("[WEB_SEARCH] Install with: pip install requests beautifulsoup4")
        return ""

    try:
        # Add headers to look like a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

        # Try multiple search endpoints
        endpoints = [
            ("https://html.duckduckgo.com/html/", {'q': query}),  # Try HTML version first
            ("https://lite.duckduckgo.com/lite/", {'q': query}),  # Fallback to Lite
        ]

        response = None
        loop = asyncio.get_event_loop()

        for endpoint_url, params in endpoints:
            try:
                # Run in executor to avoid blocking
                response = await loop.run_in_executor(
                    None,
                    lambda url=endpoint_url, p=params: requests.get(url, params=p, headers=headers, timeout=10)
                )

                # Accept any 2xx status code (200-299)
                if 200 <= response.status_code < 300:
                    print(f"[WEB_SEARCH] HTTP {response.status_code} from {endpoint_url.split('/')[2]} - parsing results")
                    break
                else:
                    print(f"[WEB_SEARCH] HTTP {response.status_code} from {endpoint_url.split('/')[2]}, trying next endpoint...")
                    response = None
            except Exception as e:
                print(f"[WEB_SEARCH] Error with {endpoint_url.split('/')[2]}: {e}")
                response = None
                continue

        if not response or response.status_code < 200 or response.status_code >= 300:
            print(f"[WEB_SEARCH] All endpoints failed for query: {query}")
            return ""

        # Parse results
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract result snippets - try multiple approaches
        results = []

        # Approach 1: Look for DuckDuckGo HTML result structure (class="result__snippet")
        result_snippets = soup.find_all('a', class_='result__snippet')[:max_results]
        for snippet_elem in result_snippets:
            text = snippet_elem.get_text(strip=True)
            if text and len(text) > 20:
                results.append(text)

        # Approach 1b: Try result__body for DuckDuckGo HTML
        if not results:
            result_bodies = soup.find_all('div', class_='result__body')[:max_results]
            for body in result_bodies:
                text = body.get_text(strip=True)
                if text and len(text) > 30:
                    results.append(text)

        # Approach 1c: Look for DuckDuckGo lite result structure
        if not results:
            result_links = soup.find_all('a', class_='result-link')[:max_results]
            for link in result_links:
                snippet_elem = link.find_next('td', class_='result-snippet')
                if snippet_elem:
                    snippet = snippet_elem.get_text(strip=True)
                    if snippet and len(snippet) > 20:
                        results.append(snippet)

        # Approach 2: If no results, try finding any table rows with links
        if not results:
            print(f"[WEB_SEARCH] Trying alternate parsing method...")
            rows = soup.find_all('tr')[:max_results * 2]  # Get more rows to parse
            for row in rows:
                # Look for cells with text content
                cells = row.find_all('td')
                for cell in cells:
                    text = cell.get_text(strip=True)
                    # Skip very short text and links-only cells
                    if text and len(text) > 30 and not text.startswith('http'):
                        results.append(text)
                        if len(results) >= max_results:
                            break
                if len(results) >= max_results:
                    break

        # Approach 3: If still no results, get any substantial text blocks
        if not results:
            print(f"[WEB_SEARCH] Trying text block extraction...")
            paragraphs = soup.find_all(['p', 'div', 'span'])
            for elem in paragraphs[:max_results * 3]:
                text = elem.get_text(strip=True)
                if text and len(text) > 50 and len(text) < 500:
                    # Avoid navigation/header text
                    if not any(nav_word in text.lower() for nav_word in ['cookie', 'privacy', 'settings', 'navigation']):
                        results.append(text)
                        if len(results) >= max_results:
                            break

        if not results:
            print(f"[WEB_SEARCH] No results found for: {query}")
            print(f"[WEB_SEARCH] HTML structure preview: {soup.prettify()[:500]}")
            return ""

        # Combine results
        combined = "\n\n".join(results)
        print(f"[WEB_SEARCH] Found {len(results)} results for '{query}' ({len(combined)} chars)")
        return combined

    except Exception as e:
        print(f"[WEB_SEARCH] Error searching '{query}': {e}")
        return ""


async def mock_web_search(query: str) -> str:
    """Mock web search for testing (returns placeholder text).

    Args:
        query: Search query

    Returns:
        Mock search results
    """
    # Simulate network delay
    await asyncio.sleep(0.5)

    # Return mock data based on query keywords
    if "sustainable" in query.lower() or "cities" in query.lower():
        return """
Green infrastructure reduces urban heat island effect by 2-5°C according to EPA studies.
Copenhagen reduced carbon emissions 42% through cycling infrastructure and district heating.
Singapore's vertical gardens cover 40% of building surfaces, reducing cooling needs by 15%.
LED street lighting reduces municipal energy costs by 40-60% in major cities.
Permeable pavements reduce stormwater runoff by 80% compared to conventional surfaces.
        """.strip()
    elif "climate" in query.lower():
        return """
IPCC reports show global temperatures rising 1.1°C above pre-industrial levels.
Renewable energy capacity grew 9.6% in 2021, led by solar and wind.
Cities account for 75% of global CO2 emissions despite covering only 2% of Earth's surface.
        """.strip()
    else:
        return f"Mock search results for: {query}. Install requests+beautifulsoup4 for real web search."


# Default export - try real search, fall back to mock
async def web_search(query: str) -> str:
    """Web search with automatic fallback to mock.

    Args:
        query: Search query

    Returns:
        Search results text
    """
    # Try real search first
    try:
        import requests
        result = await simple_web_search(query)
        if result:
            return result
    except ImportError:
        pass

    # Fall back to mock
    print(f"[WEB_SEARCH] Using mock search (install requests+beautifulsoup4 for real search)")
    return await mock_web_search(query)
