#!/usr/bin/env python3
"""Quick test of retrieval system APIs."""

import asyncio
import sys
sys.path.insert(0, '/home/user/ai_swarm_mechanics')

from swarm.retrieval.search_engine import WikipediaAPI, DuckDuckGoSearch
from swarm.retrieval.knowledge_processor import KnowledgeProcessor
from swarm.retrieval.web_scraper import WebScraper


def test_wikipedia_api():
    """Test Wikipedia API integration."""
    print("\n=== TESTING WIKIPEDIA API ===")

    wiki = WikipediaAPI()

    # Test search
    print("\n1. Testing Wikipedia search...")
    results = wiki.search("renewable energy", limit=3)

    if results:
        print(f"✓ Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']}")
            print(f"     {result['snippet'][:100]}...")
    else:
        print("✗ No results found")
        return False

    # Test get article
    print("\n2. Testing Wikipedia article fetch...")
    article = wiki.get_article(results[0]['title'])

    if article:
        print(f"✓ Fetched article: {article['title']}")
        print(f"  Content length: {len(article['content'])} chars")
        print(f"  Preview: {article['content'][:200]}...")
    else:
        print("✗ Failed to fetch article")
        return False

    # Test full article
    print("\n3. Testing Wikipedia FULL article fetch...")
    full_article = wiki.get_full_article(results[0]['title'])

    if full_article:
        print(f"✓ Fetched FULL article: {full_article['title']}")
        print(f"  Word count: {full_article['word_count']} words")
        print(f"  Content length: {len(full_article['content'])} chars")
    else:
        print("✗ Failed to fetch full article")
        return False

    # Test stats
    stats = wiki.get_stats()
    print(f"\n✓ Wikipedia API Stats:")
    print(f"  Queries made: {stats['queries_made']}")
    print(f"  Articles fetched: {stats['articles_fetched']}")

    return True


def test_duckduckgo_api():
    """Test DuckDuckGo API integration."""
    print("\n=== TESTING DUCKDUCKGO API ===")

    ddg = DuckDuckGoSearch()

    # Test instant answer
    print("\n1. Testing DuckDuckGo instant answer...")
    result = ddg.instant_answer("solar energy")

    if result and result.get('answer'):
        print(f"✓ Got instant answer")
        print(f"  Answer: {result['answer'][:200]}...")
        print(f"  Source: {result.get('source', 'N/A')}")
        if result.get('related_topics'):
            print(f"  Related topics: {len(result['related_topics'])}")
    else:
        print("✗ No instant answer (this is normal for some queries)")

    # Test stats
    stats = ddg.get_stats()
    print(f"\n✓ DuckDuckGo API Stats:")
    print(f"  Queries made: {stats['queries_made']}")

    return True


def test_knowledge_processor():
    """Test knowledge processing."""
    print("\n=== TESTING KNOWLEDGE PROCESSOR ===")

    processor = KnowledgeProcessor(chunk_size=300, overlap=50)

    # Sample content
    content = """
    Renewable energy is energy from sources that are naturally replenishing but flow-limited.
    Solar power is the conversion of energy from sunlight into electricity.
    Wind power is the use of air flow through wind turbines to provide the mechanical power to turn electric generators.
    Hydroelectric power, or hydroelectricity, is electricity produced from hydropower.

    Solar energy is radiant light and heat from the Sun that is harnessed using a range of technologies.
    Photovoltaic systems, commonly called solar panels, convert light into electric power.
    Solar thermal collectors are used for water heating, space heating, or thermal energy storage.

    Wind turbines convert the kinetic energy in the wind into mechanical power.
    This mechanical power can be used for specific tasks or converted into electricity.
    Modern wind turbines range from around 600 kW to 5 MW rated power.
    The largest turbines have a capacity of up to 20 MW.
    """

    print("\n1. Testing content chunking...")
    chunks = processor.chunk_content(content, source="Test Article", source_url="http://test.com")

    print(f"✓ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: {len(chunk.content.split())} words, importance={chunk.importance_score:.2f}")
        print(f"    Keywords: {', '.join(chunk.keywords)}")

    print("\n2. Testing fact extraction...")
    facts = processor.extract_key_facts(content, max_facts=5)

    print(f"✓ Extracted {len(facts)} facts:")
    for i, fact in enumerate(facts, 1):
        print(f"  {i}. {fact[:100]}...")

    print("\n3. Testing scout knowledge base creation...")
    fragments = processor.create_scout_knowledge_base(chunks)

    print(f"✓ Created {len(fragments)} knowledge fragments")
    for i, fragment in enumerate(fragments[:3], 1):
        print(f"  Fragment {i}: {fragment['content'][:80]}...")
        print(f"    Importance: {fragment['importance']:.2f}")

    return True


def test_web_scraper():
    """Test web scraper."""
    print("\n=== TESTING WEB SCRAPER ===")

    scraper = WebScraper()

    print("\n1. Testing HTML content extraction...")

    # Sample HTML
    html = """
    <html>
    <head><title>Test Article</title></head>
    <body>
        <nav>Navigation</nav>
        <article>
            <h1>Renewable Energy Technology</h1>
            <p>Solar power is becoming increasingly efficient and cost-effective.</p>
            <p>Wind energy has seen tremendous growth in recent years.</p>
            <p>These technologies are crucial for combating climate change.</p>
        </article>
        <footer>Footer content</footer>
    </body>
    </html>
    """

    result = scraper.extract_main_content(html, url="http://test.com")

    if result:
        print(f"✓ Extracted content:")
        print(f"  Title: {result['title']}")
        print(f"  Method: {result['method']}")
        print(f"  Word count: {result['word_count']}")
        print(f"  Content: {result['content'][:150]}...")
    else:
        print("✗ Failed to extract content")
        return False

    stats = scraper.get_stats()
    print(f"\n✓ Web Scraper Stats:")
    print(f"  Pages scraped: {stats['pages_scraped']}")
    print(f"  Success rate: {stats['success_rate']:.0%}")

    return True


async def test_external_sources():
    """Test external sources integration."""
    print("\n=== TESTING EXTERNAL SOURCES INTEGRATION ===")

    from swarm.validation.external_sources import WikipediaSource, WebSearchSource

    print("\n1. Testing WikipediaSource with real API...")
    wiki_source = WikipediaSource()
    result = await wiki_source.verify("Solar energy is renewable energy from the sun")

    print(f"  Verified: {result['verified']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Method: {result['method']}")
    print(f"  Evidence: {result['evidence'][:150]}...")

    print("\n2. Testing WebSearchSource with real API...")
    web_source = WebSearchSource(min_sources=1)
    result = await web_source.verify("Wind power generates electricity from air flow")

    print(f"  Verified: {result['verified']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Method: {result['method']}")
    print(f"  Evidence: {result['evidence'][:150]}...")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("RETRIEVAL SYSTEM INTEGRATION TESTS")
    print("=" * 60)

    success = True

    try:
        # Test APIs
        if not test_wikipedia_api():
            success = False

        if not test_duckduckgo_api():
            success = False

        # Test processing
        if not test_knowledge_processor():
            success = False

        if not test_web_scraper():
            success = False

        # Test integration
        asyncio.run(test_external_sources())

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "=" * 60)
    if success:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
