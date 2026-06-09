# Knowledge Retrieval for Swarm Agents

Provides agents with access to external knowledge sources to support deeper, evidence-based signal generation.

## Features

- **Wikipedia Integration**: Search and retrieve Wikipedia summaries
- **Caching**: LRU cache for repeated queries
- **Async Support**: Non-blocking knowledge retrieval

## Usage

```python
from swarm.knowledge.retrieval import KnowledgeRetriever

# Create retriever
retriever = KnowledgeRetriever(cache_size=100)

# Search Wikipedia
summary = await retriever.search_wikipedia("climate change", sentences=3)
print(summary)

# Get broader context
context = await retriever.get_context("renewable energy")
print(context["wikipedia"])

# Cleanup
await retriever.close()
```

## Integration with Agents

**IMPORTANT ARCHITECTURE RULE**: Only **Scouts** should use external knowledge retrieval!

- **Scouts**: Exploratory agents that bring external information into the swarm
- **Foragers/Critics/Haters**: Higher-order agents that ONLY process signals from other agents

This maintains proper swarm intelligence hierarchy:
1. Scouts retrieve raw information from external sources
2. Scouts deposit initial signals in the environment
3. Higher-order agents sample, refine, and critique those signals
4. No direct external access for synthesis/critique agents

To enable knowledge-augmented **Scouts**:

1. Set `ENABLE_KNOWLEDGE_RETRIEVAL = True` in `swarm/core/config.py`
2. Pass retriever instance ONLY to Scout agents during initialization
3. Scouts query retriever before depositing initial signals

Example:

```python
# In run_task.py or run_swarm.py
from swarm.knowledge.retrieval import KnowledgeRetriever

if ENABLE_KNOWLEDGE_RETRIEVAL:
    retriever = KnowledgeRetriever(cache_size=KNOWLEDGE_CACHE_SIZE)

    # Pass ONLY to scouts (not foragers, critics, or haters!)
    for i in range(NUM_SCOUTS):
        scout = Scout(f"Scout_{i}", signal_type, thesis, knowledge_retriever=retriever)
        scouts.append(scout)

    # Foragers, Critics, Haters get NO retriever
    for i in range(NUM_FORAGERS):
        forager = Forager(f"Forager_{i}", input_type, output_type, thesis)  # No retriever!
        foragers.append(forager)

    # Cleanup after swarm completes
    await retriever.close()
```

## Future Enhancements

- **Web Search**: Integrate DuckDuckGo or Google search APIs
- **Academic Papers**: Query arXiv, PubMed, etc.
- **Fact-Checking APIs**: Cross-reference claims with fact-checking databases
- **Custom Knowledge Bases**: Load domain-specific documents
- **Semantic Search**: Use embeddings for better retrieval

## API

### `KnowledgeRetriever`

#### Methods

- `search_wikipedia(query, sentences=3)` - Search Wikipedia and return summary
- `get_context(topic)` - Get multi-source context about a topic
- `close()` - Cleanup resources
- `get_stats()` - Get cache statistics

## Notes

- Requires `aiohttp` package for async HTTP requests
- Wikipedia API rate limits apply (typically generous for personal use)
- Cache persists for the lifetime of the retriever instance
- Knowledge retrieval is **disabled by default** to keep swarm lightweight
