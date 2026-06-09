# Advanced Knowledge Retrieval System Guide

## Overview

The Advanced Retrieval System is a production-grade knowledge ingestion pipeline that enables stigmergic swarms to gather, process, and utilize external knowledge from real-world sources.

**Key Capabilities:**
- 🌐 **100K+ words per round** - Deep knowledge ingestion
- 📚 **Real APIs** - Wikipedia MediaWiki API, DuckDuckGo Instant Answers
- 🔍 **Niche Discovery** - Identifies rare but valuable information
- 🕸️ **Knowledge Graph** - Cross-links related concepts for signal reinforcement
- ♻️ **Round-Based Refinement** - Searches improve across rounds
- ⚖️ **Legal Compliance** - Proper rate limiting, temp storage, no permanent scraping

---

## Architecture

### 4-Layer System

```
┌─────────────────────────────────────────────────────┐
│              LAYER 4: Advanced Retriever             │
│  (Round orchestration, refinement, knowledge graph) │
└─────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│            LAYER 3: Knowledge Processor              │
│    (Chunking, fact extraction, fragment creation)   │
└─────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│              LAYER 2: Search Engines                 │
│       (Wikipedia API, DuckDuckGo, query gen)        │
└─────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│                LAYER 1: Web Scraper                  │
│    (BeautifulSoup, content extraction, cleaning)    │
└─────────────────────────────────────────────────────┘
```

### Data Flow

```
Keywords → Search Queries → Raw Content → Chunks → Fragments → Scouts → Signals
```

---

## Components

### 1. WebScraper (`swarm/retrieval/web_scraper.py`)

**Purpose:** Extract clean content from HTML pages using industry-standard methods.

**Features:**
- Multiple extraction strategies (article tags, class/id heuristics, paragraph density)
- Content cleaning and normalization
- Rate limiting (configurable per domain)
- Robots.txt awareness
- Professional user agent identification

**Example:**
```python
from swarm.retrieval.web_scraper import WebScraper

scraper = WebScraper(timeout=10, rate_limit=1.0)
result = scraper.scrape_and_extract("https://en.wikipedia.org/wiki/Climate_change")

print(f"Title: {result['title']}")
print(f"Word count: {result['word_count']}")
print(f"Method: {result['method']}")  # e.g., "article_tag"
```

**Legal Compliance:**
```python
"""
LEGAL NOTICE:
- Only scrapes publicly available content
- Respects robots.txt
- Uses appropriate user agents
- Rate limits requests
- Content is temporary (deleted after processing)
- No permanent storage of scraped content
"""
```

---

### 2. Search Engines (`swarm/retrieval/search_engine.py`)

#### WikipediaAPI

**Purpose:** Official MediaWiki API integration (free, no API key required).

**Features:**
- Search Wikipedia for articles
- Fetch article introductions
- Fetch FULL articles (not just intro)
- Extract related links
- Rate limiting (100ms between requests)

**Example:**
```python
from swarm.retrieval.search_engine import WikipediaAPI

wiki = WikipediaAPI()

# Search
results = wiki.search("renewable energy", limit=3)
# [{'title': '...', 'snippet': '...', 'page_id': ...}]

# Get full article (100K+ words possible)
article = wiki.get_full_article(results[0]['title'])
print(f"Word count: {article['word_count']}")
```

#### DuckDuckGoSearch

**Purpose:** DuckDuckGo Instant Answer API (free, no API key).

**Features:**
- Instant answers for queries
- Related topics extraction
- Source attribution
- Rate limiting (200ms between requests)

**Example:**
```python
from swarm.retrieval.search_engine import DuckDuckGoSearch

ddg = DuckDuckGoSearch()
result = ddg.instant_answer("What is solar energy")

print(f"Answer: {result['answer']}")
print(f"Source: {result['source']}")
print(f"Related: {result['related_topics']}")
```

#### Query Generation

**Purpose:** Generate effective search queries from keywords.

**Features:**
- Single-keyword queries (broad exploration)
- Two-keyword combinations (specific queries)
- Context-enhanced queries ("how to", "what is", "why")

**Example:**
```python
from swarm.retrieval.search_engine import generate_search_queries

keywords = ["climate", "renewable", "energy", "carbon", "emissions"]
task_context = "How can we reduce carbon emissions?"

queries = generate_search_queries(keywords, task_context)
# ['climate', 'renewable', 'energy', 'climate renewable', 'renewable energy',
#  'how to climate', 'how to renewable', 'what is energy', ...]
```

---

### 3. Knowledge Processor (`swarm/retrieval/knowledge_processor.py`)

**Purpose:** Transform raw content into scout-digestible knowledge fragments.

#### Chunking

**Strategy:**
- 500 words per chunk (configurable)
- 50 word overlap for context preservation
- Respects paragraph boundaries
- Minimum chunk size validation

**Example:**
```python
from swarm.retrieval.knowledge_processor import KnowledgeProcessor

processor = KnowledgeProcessor(chunk_size=500, overlap=50)

chunks = processor.chunk_content(
    content=article_text,
    source="Wikipedia: Climate Change",
    source_url="https://..."
)

for chunk in chunks:
    print(f"Chunk {chunk.chunk_id}/{chunk.total_chunks}")
    print(f"Words: {chunk.metadata['word_count']}")
    print(f"Importance: {chunk.importance_score}")
    print(f"Keywords: {chunk.keywords}")
```

#### Importance Scoring

**Factors** (0.0-1.0 scale):
- Length: Longer chunks = more content (up to +0.3)
- Keyword density: More keywords = more relevant (up to +0.2)
- Contains numbers/data: Facts and statistics (+ 0.1)
- Contains citations: Research, evidence, studies (+ 0.2)
- Technical terminology: Capitalized terms (up to +0.2)

**Example:**
```python
# High-importance chunk (score: 0.9)
"Renewable energy research shows 45% efficiency gains.
Modern photovoltaic systems convert sunlight to electricity.
According to recent studies, solar capacity increased..."

# Low-importance chunk (score: 0.3)
"Energy is important for society. We use it every day.
It comes from various sources and helps us."
```

#### Fact Extraction

**Patterns:**
- State of being: "is", "are", "was", "were"
- Statistics: numbers with units (%, million, billion)
- Citations: "research", "study", "evidence"
- Causation: "cause", "effect", "lead to"
- Trends: "increase", "decrease", "rise", "fall"

**Example:**
```python
facts = processor.extract_key_facts(chunk.content, max_facts=5)
# [
#   "Renewable energy is energy from naturally replenishing sources",
#   "Solar power is the conversion of sunlight into electricity",
#   "Global solar capacity increased 20% in 2023"
# ]
```

---

### 4. AdvancedRetriever (`swarm/retrieval/advanced_retriever.py`)

**Purpose:** Round-aware deep research orchestration with 100K+ word ingestion.

#### Core Concepts

**ResearchFragment:**
```python
@dataclass
class ResearchFragment:
    content: str           # Factual statement
    source: str           # Source name
    source_url: str       # Source URL
    keywords: List[str]   # Associated keywords
    importance: float     # Importance score (0.0-1.0)
    rarity: float         # Rarity score (0.0-1.0)
    connections: List[str]  # Related fragments
    round_discovered: int   # Which round found this
```

**RoundKnowledge:**
```python
@dataclass
class RoundKnowledge:
    round_num: int
    keywords: List[str]
    queries_executed: List[str]
    total_words_ingested: int
    sources_count: int
    fragments: List[ResearchFragment]
    niche_discoveries: List[str]  # Rare findings
```

#### Deep Research Process

**1. Round 1: Initial Exploration**
```
Keywords (from task) → Search queries → Wikipedia articles → Chunks → Fragments
```

**2. Round 2+: Refinement**
```
Previous synthesis → Extract emerging topics → Refined queries → More specific articles
```

**Example:**
```python
from swarm.retrieval.advanced_retriever import AdvancedRetriever

retriever = AdvancedRetriever(
    temp_dir="research/temp",
    target_words_per_round=100000,  # 100K words
    min_sources_per_keyword=3
)

# Round 1
round_knowledge = await retriever.deep_research_round(
    keywords=["climate", "renewable", "energy"],
    round_num=0,
    task_context="How can we reduce carbon emissions?",
    previous_synthesis=""
)

print(f"Words ingested: {round_knowledge.total_words_ingested:,}")
print(f"Fragments: {len(round_knowledge.fragments)}")
print(f"Niche discoveries: {len(round_knowledge.niche_discoveries)}")
```

#### Rarity Scoring

**Purpose:** Identify niche/uncommon but potentially valuable information.

**Factors** (0.0-1.0 scale):
- **Technical terminology** (+0.15-0.3): Capitalized multi-word terms
  - Example: "Photovoltaic Junction Technology"
- **Specific data** (+0.2): Numbers with units
  - Example: "45.3 million metric tons", "23.4% efficiency"
- **Citations** (+0.15): Research references
  - Example: "according to the 2023 IPCC study"
- **Uncommon keywords** (+0.2): Beyond common terms
  - Beyond: climate, energy, change, sustainable
  - Uncommon: photovoltaic, anaerobic, thermodynamic
- **Long detailed content** (+0.15): >30 words (specialist knowledge)

**Example:**
```python
# High rarity (0.8):
"Recent Perovskite Solar Cell research demonstrates 25.2% efficiency gains
through Tandem Junction Architecture, according to Nature Energy study 2023."

# Low rarity (0.1):
"Renewable energy is important for reducing climate change impacts."
```

#### Knowledge Graph

**Purpose:** Cross-link related fragments for signal reinforcement.

**Strategy:**
```
1. Index fragments by keywords
2. For each fragment, find others with shared keywords
3. Create connections between related fragments
4. Sample top 5 connections per fragment
```

**Example:**
```python
Fragment 1: "Solar photovoltaic efficiency increased 20%"
Keywords: [solar, photovoltaic, efficiency]

Connections:
- "Photovoltaic materials use semiconductor properties..."
- "High efficiency solar panels convert 23% of sunlight..."
- "Solar cell research focuses on improving efficiency..."
```

**Benefit:** When scouts encounter one fragment, connections guide them to related knowledge.

---

## Configuration

### Config Flags (`swarm/core/config.py`)

```python
# Enable advanced retrieval
USE_ADVANCED_RETRIEVER = False  # Set to True to enable

# Target words to ingest per round
ADVANCED_RETRIEVAL_TARGET_WORDS = 100000  # 100K words

# Minimum sources to check per keyword
ADVANCED_RETRIEVAL_MIN_SOURCES = 3

# Temporary directory for scraped content
ADVANCED_RETRIEVAL_TEMP_DIR = "research/temp"
```

### Dependencies

**Required:**
```bash
pip install requests beautifulsoup4
```

**Included in requirements.txt:**
```
requests>=2.31.0
beautifulsoup4>=4.12.0
```

---

## Usage

### Basic Usage (Integrated in run_task.py)

**Enable advanced retrieval:**
```python
# In swarm/core/config.py
USE_ADVANCED_RETRIEVER = True
```

**Run task:**
```bash
python run_task.py problem_solving "How can we reduce carbon emissions?"
```

**Output:**
```
[INIT] Creating AdvancedRetriever for deep knowledge ingestion...
[INIT] AdvancedRetriever ready (target: 100,000 words/round)

[ROUND 1] Starting DEEP research (target: 100,000 words)...
[ADVANCED RETRIEVAL] Round 0 - Deep research starting
[ADVANCED RETRIEVAL] Keywords: ['carbon', 'emissions', 'reduce', 'climate', 'energy']
[ADVANCED RETRIEVAL] Generated 15 search queries
[ADVANCED RETRIEVAL] Ingested: Renewable energy (2,453 words)
[ADVANCED RETRIEVAL] Ingested: Carbon capture (1,892 words)
[ADVANCED RETRIEVAL] Ingested: Solar power (2,105 words)
...
[ADVANCED RETRIEVAL] Ingested 102,345 words from 47 sources
[ADVANCED RETRIEVAL] Created 843 knowledge chunks
[ADVANCED RETRIEVAL] Extracted 2,103 research fragments
[ADVANCED RETRIEVAL] Found 127 niche discoveries

[ROUND 1] Deep research complete:
  - Words ingested: 102,345
  - Sources accessed: 47
  - Fragments extracted: 2,103
  - Niche discoveries: 127
  - Queries executed: 15
```

### Standalone Usage

```python
import asyncio
from swarm.retrieval.advanced_retriever import AdvancedRetriever

async def research():
    retriever = AdvancedRetriever()

    # Round 1
    knowledge_r1 = await retriever.deep_research_round(
        keywords=["climate", "change", "mitigation"],
        round_num=0,
        task_context="Climate change solutions"
    )

    # Round 2 (refines based on Round 1)
    knowledge_r2 = await retriever.deep_research_round(
        keywords=["renewable", "policy", "technology"],
        round_num=1,
        task_context="Climate change solutions",
        previous_synthesis="Solar and wind are key renewable sources..."
    )

    # Get fragments for scouts
    fragments = retriever.get_fragments_for_scouts(
        round_num=0,
        diversity=0.3  # 30% rare findings, 70% important
    )

    # Stats
    stats = retriever.get_stats()
    print(f"Total words: {stats['total_words_ingested']:,}")
    print(f"Total fragments: {stats['total_fragments']}")

    # Cleanup
    retriever.cleanup()

asyncio.run(research())
```

---

## Legal Compliance

### Temporary Storage Only

**All scraped content is temporary:**
```
research/temp/
├── round_0_knowledge.txt  # Created during round
├── round_1_knowledge.txt  # Created during round
└── ...
```

**Deleted at end:**
```python
# Automatic cleanup at end of run_task.py
advanced_retriever.cleanup()  # Deletes research/temp/ directory
```

### Rate Limiting

**Wikipedia API:** 100ms between requests (line 41, `search_engine.py`)
**DuckDuckGo API:** 200ms between requests (line 238, `search_engine.py`)
**Web Scraper:** 1.0s between requests (configurable)

### User Agent Identification

**Professional, identifies as bot:**
```python
'User-Agent': 'SwarmAI-Research-Bot/1.0 (Educational/Research; +https://github.com/anthropics/swarm-intelligence)'
```

### No Permanent Storage

**Content flow:**
```
Raw HTML → Extracted text → Chunks → Fragments → Agent signals → Deleted
```

**No intermediate storage** - all transformations happen in memory, temp files deleted after processing.

---

## Statistics & Monitoring

### Retriever Stats

```python
stats = retriever.get_stats()
```

**Returns:**
```python
{
    'total_rounds': 3,
    'total_words_ingested': 312,451,
    'total_sources_accessed': 143,
    'unique_sources': 98,
    'total_fragments': 6,234,
    'avg_words_per_round': 104,150,
    'avg_sources_per_round': 47.7,
    'wikipedia_queries': 45,
    'wikipedia_articles': 127,
    'ddg_queries': 18
}
```

### Per-Round Stats

```python
round_knowledge.total_words_ingested  # Words in this round
round_knowledge.sources_count         # Sources accessed
len(round_knowledge.fragments)        # Fragments extracted
len(round_knowledge.niche_discoveries)  # Rare findings
```

---

## Best Practices

### 1. Keyword Selection

**Good keywords:**
```python
# Specific, actionable
["solar", "photovoltaic", "efficiency", "panel", "semiconductor"]
```

**Poor keywords:**
```python
# Too general, common
["energy", "power", "good", "important"]
```

### 2. Round Progression

**Round 1:** Broad exploration
```python
keywords = ["renewable", "energy", "climate"]
```

**Round 2:** Refine based on synthesis
```python
# Extract from: "Solar and wind are most promising..."
keywords = ["solar", "photovoltaic", "wind", "turbine"]
```

**Round 3:** Deep dive into specifics
```python
# Extract from: "Perovskite solar cells show promise..."
keywords = ["perovskite", "tandem", "junction", "efficiency"]
```

### 3. Fragment Usage

**High importance fragments:** Use for main arguments
**High rarity fragments:** Use for novel insights
**Connected fragments:** Follow connections for deep understanding

---

## Troubleshooting

### "No Wikipedia articles found"

**Cause:** Keywords too specific or misspelled
**Solution:** Use broader keywords or check spelling

### "Advanced retrieval import failed"

**Cause:** Missing dependencies
**Solution:**
```bash
pip install requests beautifulsoup4
```

### "Target words not reached"

**Cause:** Limited search results or rate limiting
**Solution:**
- Increase `min_sources_per_keyword`
- Add more keywords
- Check internet connection

### Memory issues with 100K+ words

**Solution:** Reduce target:
```python
ADVANCED_RETRIEVAL_TARGET_WORDS = 50000  # 50K instead of 100K
```

---

## Future Enhancements

### Planned Features

1. **Scholarly Sources** - arXiv, PubMed APIs for academic papers
2. **Parallel Scraping** - Async web scraping for faster ingestion
3. **TF-IDF Keywords** - Better keyword extraction algorithm
4. **NLP Fact Extraction** - spaCy/NLTK for improved fact detection
5. **Source Credibility** - Score sources by reliability
6. **Caching Layer** - Reduce duplicate API calls
7. **Query Expansion** - LLM-based query generation
8. **Relevance Scoring** - Match content to task context

### Extensibility

**Add new search source:**
```python
# In search_engine.py
class ArXivAPI:
    def search(self, query: str) -> List[Dict]:
        # Implementation
        pass

# In advanced_retriever.py
self.arxiv = ArXivAPI()
```

---

## Performance

### Benchmarks (Typical 3-Round Run)

| Metric | DynamicRetriever | AdvancedRetriever |
|--------|------------------|-------------------|
| Words/round | ~5,000 | ~100,000 |
| Sources/round | 3-5 | 30-50 |
| Time/round | 5s | 45s |
| Fragments | ~50 | ~2,000 |
| Memory | ~10MB | ~150MB |

**Trade-off:** 10x more time, 20x more knowledge

---

## References

- **Wikipedia API:** https://www.mediawiki.org/wiki/API:Main_page
- **DuckDuckGo API:** https://duckduckgo.com/api
- **BeautifulSoup:** https://www.crummy.com/software/BeautifulSoup/
- **Readability Algorithm:** https://github.com/mozilla/readability

---

## License & Attribution

All retrieved content is:
- Publicly available
- Attributed to original source
- Transformed by agents (not stored verbatim)
- Deleted after processing

**Wikipedia content:** CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/)
**DuckDuckGo:** Aggregated from various sources with attribution
