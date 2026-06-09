"""Information retrieval module for stigmergic swarm.

Comprehensive knowledge retrieval system with:
- Web scraping (BeautifulSoup, industry-standard extraction)
- Search engines (Wikipedia API, DuckDuckGo)
- Knowledge processing (chunking, fact extraction)
- Advanced round-aware retrieval (deep ingestion, niche discovery)
"""

# New comprehensive retrieval system
try:
    from .web_scraper import WebScraper, extract_article_text
    from .search_engine import (
        WikipediaAPI,
        DuckDuckGoSearch,
        MultiSourceSearch,
        generate_search_queries
    )
    from .knowledge_processor import (
        KnowledgeProcessor,
        KnowledgeChunk,
        process_search_results
    )
    from .advanced_retriever import (
        AdvancedRetriever,
        ResearchFragment,
        RoundKnowledge
    )

    __all__ = [
        # Web scraping
        'WebScraper',
        'extract_article_text',
        # Search engines
        'WikipediaAPI',
        'DuckDuckGoSearch',
        'MultiSourceSearch',
        'generate_search_queries',
        # Knowledge processing
        'KnowledgeProcessor',
        'KnowledgeChunk',
        'process_search_results',
        # Advanced retrieval
        'AdvancedRetriever',
        'ResearchFragment',
        'RoundKnowledge',
    ]
except ImportError as e:
    # Fallback if dependencies not installed
    print(f"[WARNING] Advanced retrieval system not available: {e}")
    print("[WARNING] Install dependencies: pip install requests beautifulsoup4")
    __all__ = []

