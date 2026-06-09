"""Web ingestion: Buckshot search and knowledge synthesis.

This script performs broad "buckshot" web searches on a topic, ingests relevant
material from multiple sources, and synthesizes insights using the monolith-
breaking document processing pipeline.

Usage:
    python run_web_ingestion.py "What are the latest advances in quantum computing?"
    python run_web_ingestion.py "How does climate change affect ocean ecosystems?"
"""

import asyncio
import sys
import re
from pathlib import Path
from typing import List, Dict
from swarm.monolith_breaking import run_document_swarm
from swarm.documents.processor import DocumentSection
from swarm.knowledge.mcp_client import MCPClient
from swarm.llm.simple_llm import SimpleLLM


class WebContentFetcher:
    """Fetches and extracts text content from web URLs."""

    def __init__(self):
        """Initialize web content fetcher."""
        self.has_requests = False
        self.has_beautifulsoup = False

        # Try to import optional dependencies
        try:
            import requests
            self.requests = requests
            self.has_requests = True
        except ImportError:
            print("[WEB] requests not available (install: pip install requests)")

        try:
            from bs4 import BeautifulSoup
            self.BeautifulSoup = BeautifulSoup
            self.has_beautifulsoup = True
        except ImportError:
            print("[WEB] beautifulsoup4 not available (install: pip install beautifulsoup4)")

    async def fetch_content(self, url: str, timeout: int = 10) -> Dict:
        """Fetch and extract text content from URL.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            Dict with 'text', 'url', 'title', 'success' keys
        """
        if not self.has_requests or not self.has_beautifulsoup:
            return {
                'text': '',
                'url': url,
                'title': '',
                'success': False,
                'error': 'Missing dependencies (requests, beautifulsoup4)'
            }

        try:
            # Fetch with timeout
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.requests.get(
                    url,
                    timeout=timeout,
                    headers={'User-Agent': 'Mozilla/5.0 (research bot)'}
                )
            )

            if response.status_code != 200:
                return {
                    'text': '',
                    'url': url,
                    'title': '',
                    'success': False,
                    'error': f'HTTP {response.status_code}'
                }

            # Parse HTML and extract text
            soup = self.BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()

            # Get title
            title = soup.title.string if soup.title else url

            # Extract text from main content
            # Try to find main content areas first
            main_content = None
            for tag in ['main', 'article', 'div[role="main"]', '.content', '#content']:
                main_content = soup.select_one(tag)
                if main_content:
                    break

            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)

            # Clean up text
            text = self._clean_text(text)

            return {
                'text': text,
                'url': url,
                'title': title,
                'success': True
            }

        except Exception as e:
            return {
                'text': '',
                'url': url,
                'title': '',
                'success': False,
                'error': str(e)
            }

    def _clean_text(self, text: str) -> str:
        """Clean extracted text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)

        # Remove very short lines (likely navigation/junk)
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) > 20]

        return '\n'.join(lines)


class BuckshotSearcher:
    """Performs diverse 'buckshot' searches on a topic."""

    def __init__(self, mcp_client: MCPClient, llm: SimpleLLM):
        """Initialize buckshot searcher.

        Args:
            mcp_client: MCP client for web search
            llm: Language model for query generation
        """
        self.mcp_client = mcp_client
        self.llm = llm

    async def generate_search_queries(self, topic: str, num_queries: int = 8) -> List[str]:
        """Generate diverse search queries for a topic.

        Args:
            topic: Main topic/question
            num_queries: Number of diverse queries to generate

        Returns:
            List of search queries
        """
        prompt = f"""Generate {num_queries} diverse search queries to comprehensively explore this topic from multiple angles:

Topic: {topic}

Generate queries that cover:
- Different aspects and subtopics
- Various perspectives and viewpoints
- Specific details and general overview
- Recent developments and historical context
- Technical details and practical applications

Format: One query per line, numbered 1-{num_queries}. Each query should be 5-10 words.

Search queries:"""

        try:
            result = await self.llm.generate(prompt, max_tokens=250, temperature=0.8, use_cache=False)

            if result:
                # Parse queries
                lines = result.strip().split('\n')
                queries = []

                for line in lines:
                    line = line.strip()
                    # Extract query after number
                    if line and (line[0].isdigit() or line.startswith('-')):
                        # Remove number prefix
                        query = re.sub(r'^\d+[\.\)]\s*', '', line)
                        query = re.sub(r'^-\s*', '', query)
                        query = query.strip()

                        if len(query) > 10:
                            queries.append(query)

                return queries[:num_queries]

        except Exception as e:
            print(f"[BUCKSHOT] Query generation error: {e}")

        # Fallback: return original topic
        return [topic]

    async def search_all_queries(self, queries: List[str], results_per_query: int = 5) -> List[Dict]:
        """Search all queries and aggregate results.

        Args:
            queries: List of search queries
            results_per_query: Results per query

        Returns:
            List of unique search results
        """
        all_results = []
        seen_urls = set()

        for i, query in enumerate(queries):
            print(f"[BUCKSHOT] Searching ({i+1}/{len(queries)}): {query}")

            try:
                results = await self.mcp_client.search(query, max_results=results_per_query)

                for result in results:
                    url = result.get('url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(result)

            except Exception as e:
                print(f"[BUCKSHOT] Search error for '{query}': {e}")

        return all_results


async def run_web_ingestion(
    topic: str,
    num_queries: int = 8,
    results_per_query: int = 5,
    max_sources: int = 20,
    model_name: str = "microsoft/phi-2",
    num_foragers: int = 30,
    num_gatherers: int = 15,
    num_critics: int = 15,
    num_haters: int = 5,
    max_iterations: int = 300,
    convergence_threshold: float = 0.7
):
    """Run buckshot web ingestion and synthesis.

    Args:
        topic: Topic or question to explore
        num_queries: Number of diverse search queries to generate
        results_per_query: Search results per query
        max_sources: Maximum web sources to fetch and process
        model_name: Language model to use
        num_foragers: Number of forager agents
        num_gatherers: Number of gatherer agents
        num_critics: Number of critic agents
        num_haters: Number of hater agents
        max_iterations: Maximum iterations
        convergence_threshold: Gini coefficient threshold for convergence
    """
    print("=" * 80)
    print("WEB INGESTION MODE: Buckshot Search and Synthesis")
    print("=" * 80)
    print(f"\nTopic: {topic}")
    print(f"\nConfiguration:")
    print(f"  - Search queries: {num_queries}")
    print(f"  - Results per query: {results_per_query}")
    print(f"  - Max sources: {max_sources}")
    print()

    # Initialize components
    print("[INIT] Initializing web search and LLM...")
    mcp_client = MCPClient(enable_web_search=True, enable_wikipedia=True)

    # Initialize LLM for query generation
    device = "cuda" if model_name != "gpt2" else "cpu"
    llm = SimpleLLM(model_name, device, enable_cache=True, cache_size=50)
    await llm.load()

    # PHASE 1: Buckshot search
    print("\n" + "=" * 80)
    print("PHASE 1: BUCKSHOT SEARCH")
    print("=" * 80)

    searcher = BuckshotSearcher(mcp_client, llm)

    print(f"\n[BUCKSHOT] Generating {num_queries} diverse search queries...")
    queries = await searcher.generate_search_queries(topic, num_queries)

    print(f"\n[BUCKSHOT] Generated queries:")
    for i, query in enumerate(queries, 1):
        print(f"  {i}. {query}")

    print(f"\n[BUCKSHOT] Searching web (targeting {num_queries * results_per_query} results)...")
    search_results = await searcher.search_all_queries(queries, results_per_query)

    print(f"\n[BUCKSHOT] Found {len(search_results)} unique sources")

    # PHASE 2: Content fetching
    print("\n" + "=" * 80)
    print("PHASE 2: CONTENT FETCHING")
    print("=" * 80)

    fetcher = WebContentFetcher()
    if not fetcher.has_requests or not fetcher.has_beautifulsoup:
        print("\n[ERROR] Missing required dependencies for web fetching:")
        print("  pip install requests beautifulsoup4")
        return None

    # Limit to max_sources
    sources_to_fetch = search_results[:max_sources]
    print(f"\n[WEB] Fetching content from {len(sources_to_fetch)} sources...")

    web_documents = []
    for i, result in enumerate(sources_to_fetch):
        url = result.get('url', '')
        if not url or result.get('type') == 'simulated':
            continue

        print(f"[WEB] Fetching ({i+1}/{len(sources_to_fetch)}): {url[:60]}...")

        content = await fetcher.fetch_content(url)
        if content['success'] and len(content['text']) > 500:
            web_documents.append(content)
            print(f"  SUCCESS: {len(content['text'])} characters")
        else:
            error = content.get('error', 'Unknown error')
            print(f"  FAILED: {error}")

    print(f"\n[WEB] Successfully fetched {len(web_documents)} sources")

    if not web_documents:
        print("\n[ERROR] No web content fetched. Cannot proceed.")
        return None

    # PHASE 3: Convert to document sections
    print("\n" + "=" * 80)
    print("PHASE 3: DOCUMENT PROCESSING")
    print("=" * 80)

    from swarm.documents.processor import DocumentProcessor

    processor = DocumentProcessor(target_tokens=2000, tolerance=500)
    all_sections = []

    for doc in web_documents:
        sections = processor.split_into_sections(
            text=doc['text'],
            source_name=doc['url'],
            metadata={'title': doc['title'], 'url': doc['url']}
        )
        all_sections.extend(sections)

    print(f"\n[DOCS] Created {len(all_sections)} document sections")
    total_tokens = sum(s.token_count for s in all_sections)
    print(f"[DOCS] Total tokens: {total_tokens:,}")

    # PHASE 4: Run monolith-breaking pipeline
    print("\n" + "=" * 80)
    print("PHASE 4: DISTRIBUTED PROCESSING")
    print("=" * 80)

    # Create synthetic "document paths" for tracking
    temp_dir = Path("web_cache")
    temp_dir.mkdir(exist_ok=True)

    # Save web documents to temp files for provenance
    document_paths = []
    for i, doc in enumerate(web_documents):
        temp_file = temp_dir / f"web_source_{i}.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(f"URL: {doc['url']}\n")
            f.write(f"Title: {doc['title']}\n")
            f.write(f"\n{doc['text']}\n")
        document_paths.append(str(temp_file))

    # Process sections directly (skip re-loading)
    from swarm.monolith_breaking import run_document_swarm_from_sections

    # Check if function exists, otherwise we'll need to modify monolith_breaking.py
    try:
        result = await run_document_swarm_from_sections(
            sections=all_sections,
            model_name=model_name,
            num_foragers=num_foragers,
            num_gatherers=num_gatherers,
            num_critics=num_critics,
            num_haters=num_haters,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            enable_mcp=True
        )
    except NameError:
        # Fallback: use regular pipeline with temp files
        result = await run_document_swarm(
            document_paths=document_paths,
            model_name=model_name,
            num_foragers=num_foragers,
            num_gatherers=num_gatherers,
            num_critics=num_critics,
            num_haters=num_haters,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            enable_mcp=True
        )

    if result:
        print("\n" + "=" * 80)
        print("WEB INGESTION COMPLETE")
        print("=" * 80)

        # Save results
        output_file = Path("web_synthesis.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"WEB INGESTION SYNTHESIS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Topic: {topic}\n\n")
            f.write(f"Sources Analyzed: {len(web_documents)}\n")
            f.write(f"Total Sections: {len(all_sections)}\n")
            f.write(f"Total Tokens: {total_tokens:,}\n\n")
            f.write("=" * 80 + "\n")
            f.write("SYNTHESIS\n")
            f.write("=" * 80 + "\n\n")
            f.write(result['synthesis'])
            f.write(f"\n\n{'=' * 80}\n")
            f.write("TOP INSIGHTS\n")
            f.write(f"{'=' * 80}\n\n")
            for i, insight in enumerate(result['top_insights'][:10], 1):
                f.write(f"\n{i}. [Strength: {insight.strength:.3f}]\n")
                f.write(f"{insight.content}\n")
            f.write(f"\n\n{'=' * 80}\n")
            f.write("SOURCES\n")
            f.write(f"{'=' * 80}\n\n")
            for i, doc in enumerate(web_documents, 1):
                f.write(f"{i}. {doc['title']}\n")
                f.write(f"   {doc['url']}\n\n")

        print(f"\nResults saved to: {output_file}")
        print(f"\nKey Metrics:")
        print(f"  - Sources analyzed: {len(web_documents)}")
        print(f"  - Insights discovered: {len(result['top_insights'])}")
        print(f"  - Compression ratio: {result['compression_ratio']:.1f}:1")
        print(f"  - Convergence: {'YES' if result['converged'] else 'NO'}")

        return result

    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_web_ingestion.py \"<topic or question>\"")
        print("\nExamples:")
        print('  python run_web_ingestion.py "What are the latest advances in quantum computing?"')
        print('  python run_web_ingestion.py "How does climate change affect ocean ecosystems?"')
        print('  python run_web_ingestion.py "What are the key challenges in fusion energy?"')
        sys.exit(1)

    topic = sys.argv[1]
    asyncio.run(run_web_ingestion(topic))
