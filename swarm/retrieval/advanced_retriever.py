"""Advanced round-aware knowledge retrieval for emergent discovery.

Supports:
- Deep multi-source ingestion (100K+ words per round)
- Round-based refinement (searches get more specific)
- Cross-linking for signal reinforcement
- Niche fact extraction (finds uncommon but important information)
- Knowledge graph construction across sources
"""

import asyncio
import time
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field

from .search_engine import WikipediaAPI, DuckDuckGoSearch, MultiSourceSearch, generate_search_queries
from .knowledge_processor import KnowledgeProcessor, KnowledgeChunk
from .web_scraper import WebScraper


@dataclass
class ResearchFragment:
    """A research fragment for scouts to discover and synthesize."""
    content: str
    source: str
    source_url: str
    keywords: List[str]
    importance: float
    rarity: float  # How unique/niche this information is (0.0-1.0)
    connections: List[str] = field(default_factory=list)  # Related fragments
    round_discovered: int = 0


@dataclass
class RoundKnowledge:
    """Knowledge gathered in a specific round."""
    round_num: int
    keywords: List[str]
    queries_executed: List[str]
    total_words_ingested: int
    sources_count: int
    fragments: List[ResearchFragment]
    niche_discoveries: List[str]  # Rare/unique findings


class AdvancedRetriever:
    """Advanced knowledge retrieval with round-awareness and deep ingestion.

    Now supports task-adaptive intake via IntakeProfile:
    - Creative tasks: 50K words, high diversity
    - Technical tasks: 300K words, deep research
    - Debate tasks: 250K words, balanced sources
    - Problem solving: 150K words, solution-focused
    """

    def __init__(self,
                 temp_dir: str = "research/temp",
                 target_words_per_round: int = None,  # Now optional (read from intake_profile)
                 min_sources_per_keyword: int = None,  # Now optional (read from intake_profile)
                 intake_profile=None):  # NEW: Task-adaptive configuration
        """Initialize advanced retriever.

        Args:
            temp_dir: Temporary directory for content storage
            target_words_per_round: Target words to ingest per round (overrides intake_profile)
            min_sources_per_keyword: Minimum sources to check per keyword (overrides intake_profile)
            intake_profile: IntakeProfile with task-adaptive parameters (preferred method)
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # TASK-ADAPTIVE PARAMETERS: Read from intake_profile if provided
        self.intake_profile = intake_profile

        if intake_profile:
            # Use profile parameters
            self.target_words_per_round = target_words_per_round or intake_profile.target_words_per_round
            self.min_sources_per_keyword = min_sources_per_keyword or intake_profile.max_sources_per_keyword
            chunk_size = intake_profile.chunk_size
            chunk_overlap = intake_profile.chunk_overlap
        else:
            # Fallback to defaults
            self.target_words_per_round = target_words_per_round or 100000
            self.min_sources_per_keyword = min_sources_per_keyword or 3
            chunk_size = 300
            chunk_overlap = 50

        # Backends
        self.wikipedia = WikipediaAPI()
        self.ddg = DuckDuckGoSearch()
        self.scraper = WebScraper()
        self.processor = KnowledgeProcessor(chunk_size=chunk_size, overlap=chunk_overlap)

        # Round history
        self.round_history: List[RoundKnowledge] = []
        self.all_fragments: List[ResearchFragment] = []

        # Knowledge graph (for cross-linking)
        self.keyword_to_fragments: Dict[str, List[int]] = {}  # keyword -> fragment indices

        # Statistics
        self.total_words_ingested = 0
        self.total_sources_accessed = 0
        self.unique_sources: Set[str] = set()

    async def deep_research_round(self,
                                   keywords: List[str],
                                   round_num: int,
                                   task_context: str = "",
                                   previous_synthesis: str = "") -> RoundKnowledge:
        """Perform deep research for a round.

        Args:
            keywords: Keywords to research
            round_num: Current round number
            task_context: Original task for context
            previous_synthesis: Synthesis from previous round (for refinement)

        Returns:
            RoundKnowledge with all discovered information
        """
        print(f"\n[ADVANCED RETRIEVAL] Round {round_num} - Deep research starting")
        print(f"[ADVANCED RETRIEVAL] Keywords: {keywords[:5]}...")

        # Generate search queries (more comprehensive)
        queries = generate_search_queries(keywords, task_context)

        # If we have previous synthesis, add refinement queries
        if previous_synthesis:
            refined_keywords = self._extract_emerging_topics(previous_synthesis)
            refined_queries = generate_search_queries(refined_keywords, task_context)
            queries.extend(refined_queries)

        print(f"[ADVANCED RETRIEVAL] Generated {len(queries)} search queries")

        # Execute searches across sources
        all_content: Dict[str, Dict] = {}
        words_ingested = 0
        queries_executed = []

        for query in queries:
            if words_ingested >= self.target_words_per_round:
                print(f"[ADVANCED RETRIEVAL] Target {self.target_words_per_round} words reached")
                break

            # Search Wikipedia (FULL articles, not just intro)
            wiki_results = await self.wikipedia.search(query, limit=5)
            for result in wiki_results[:3]:  # Top 3 per query
                if words_ingested >= self.target_words_per_round:
                    break

                # Get FULL article
                article = await self.wikipedia.get_full_article(result['title'])
                if article:
                    key = f"wikipedia_{result['title']}"
                    all_content[key] = article
                    words_ingested += article['word_count']
                    self.unique_sources.add(article['url'])
                    print(f"[ADVANCED RETRIEVAL] Ingested: {result['title']} ({article['word_count']} words)")

            # DuckDuckGo instant answer
            ddg_result = await self.ddg.instant_answer(query)
            if ddg_result and ddg_result['answer']:
                key = f"ddg_{query}"
                all_content[key] = {
                    'title': query,
                    'content': ddg_result['answer'],
                    'source': 'duckduckgo',
                    'url': ddg_result.get('url', '')
                }
                words_ingested += len(ddg_result['answer'].split())

            queries_executed.append(query)

            # Rate limiting
            await asyncio.sleep(0.2)

        print(f"[ADVANCED RETRIEVAL] Ingested {words_ingested} words from {len(all_content)} sources")

        # Process all content into knowledge chunks
        all_chunks: List[KnowledgeChunk] = []
        for source_key, source_data in all_content.items():
            chunks = self.processor.chunk_content(
                content=source_data.get('content', ''),
                source=source_data.get('title', source_key),
                source_url=source_data.get('url', '')
            )
            all_chunks.extend(chunks)

        print(f"[ADVANCED RETRIEVAL] Created {len(all_chunks)} knowledge chunks")

        # Extract research fragments with cross-linking
        fragments = self._extract_research_fragments(all_chunks, round_num)

        print(f"[ADVANCED RETRIEVAL] Extracted {len(fragments)} research fragments")

        # Identify niche discoveries (rare but important)
        niche_discoveries = self._find_niche_discoveries(fragments)

        print(f"[ADVANCED RETRIEVAL] Found {len(niche_discoveries)} niche discoveries")

        # Build knowledge graph (cross-link fragments)
        self._build_knowledge_graph(fragments)

        # Create round knowledge
        round_knowledge = RoundKnowledge(
            round_num=round_num,
            keywords=keywords,
            queries_executed=queries_executed,
            total_words_ingested=words_ingested,
            sources_count=len(all_content),
            fragments=fragments,
            niche_discoveries=niche_discoveries
        )

        # Update history
        self.round_history.append(round_knowledge)
        self.all_fragments.extend(fragments)
        self.total_words_ingested += words_ingested
        self.total_sources_accessed += len(all_content)

        return round_knowledge

    def _extract_research_fragments(self,
                                     chunks: List[KnowledgeChunk],
                                     round_num: int) -> List[ResearchFragment]:
        """Extract research fragments from chunks.

        Args:
            chunks: Knowledge chunks
            round_num: Current round

        Returns:
            List of research fragments
        """
        fragments = []

        for chunk in chunks:
            # Extract multiple facts per chunk (more comprehensive)
            facts = self.processor.extract_key_facts(chunk.content, max_facts=10)

            for fact in facts:
                # Calculate rarity (how unique is this information?)
                rarity = self._calculate_rarity(fact, chunk.keywords)

                fragment = ResearchFragment(
                    content=fact,
                    source=chunk.source,
                    source_url=chunk.source_url,
                    keywords=chunk.keywords,
                    importance=chunk.importance_score,
                    rarity=rarity,
                    connections=[],  # Will be filled by knowledge graph
                    round_discovered=round_num
                )

                fragments.append(fragment)

        return fragments

    def _calculate_rarity(self, content: str, keywords: List[str]) -> float:
        """Calculate how rare/niche this information is.

        Rare information is valuable for discovering novel approaches.

        Args:
            content: Content to analyze
            keywords: Associated keywords

        Returns:
            Rarity score (0.0-1.0, higher = more rare/niche)
        """
        rarity = 0.0

        # Technical terminology (capitalized multi-word terms)
        import re
        technical_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', content)
        if len(technical_terms) >= 3:
            rarity += 0.3
        elif len(technical_terms) >= 1:
            rarity += 0.15

        # Specific numbers/data (indicates detailed research)
        specific_numbers = re.findall(r'\d+\.?\d*\s*(?:percent|%|million|billion|thousand)', content)
        if len(specific_numbers) >= 2:
            rarity += 0.2

        # Citations/sources (indicates specialized knowledge)
        if any(keyword in content.lower() for keyword in ['study', 'research', 'according to', 'evidence']):
            rarity += 0.15

        # Uncommon keywords (not in common keywords list)
        common_keywords = {'energy', 'climate', 'change', 'sustainable', 'environmental'}
        uncommon = [kw for kw in keywords if kw not in common_keywords]
        if len(uncommon) >= 2:
            rarity += 0.2

        # Long, detailed content (more specialized)
        if len(content.split()) > 30:
            rarity += 0.15

        return min(1.0, rarity)

    def _find_niche_discoveries(self, fragments: List[ResearchFragment]) -> List[str]:
        """Find niche discoveries - rare but potentially valuable information.

        Args:
            fragments: Research fragments

        Returns:
            List of niche discovery summaries
        """
        # Sort by rarity
        niche_fragments = sorted(fragments, key=lambda f: f.rarity, reverse=True)

        # Take top 20% most rare
        niche_count = max(5, len(niche_fragments) // 5)
        top_niche = niche_fragments[:niche_count]

        discoveries = []
        for fragment in top_niche:
            if fragment.rarity >= 0.5:  # Only high-rarity items
                summary = f"{fragment.content[:100]}... [Rarity: {fragment.rarity:.2f}, Source: {fragment.source}]"
                discoveries.append(summary)

        return discoveries

    def _build_knowledge_graph(self, fragments: List[ResearchFragment]):
        """Build knowledge graph by cross-linking related fragments.

        This enables signal reinforcement - related concepts strengthen each other.

        Args:
            fragments: Research fragments to link
        """
        # Index fragments by keywords
        for idx, fragment in enumerate(fragments):
            for keyword in fragment.keywords:
                if keyword not in self.keyword_to_fragments:
                    self.keyword_to_fragments[keyword] = []
                self.keyword_to_fragments[keyword].append(len(self.all_fragments) + idx)

        # Create connections between fragments with shared keywords
        for fragment in fragments:
            connections = set()

            for keyword in fragment.keywords:
                if keyword in self.keyword_to_fragments:
                    related_indices = self.keyword_to_fragments[keyword]
                    # Sample up to 5 related fragments
                    for related_idx in related_indices[:5]:
                        if related_idx < len(self.all_fragments):
                            related = self.all_fragments[related_idx]
                            connections.add(f"{related.source}:{related.content[:50]}")

            fragment.connections = list(connections)[:5]  # Keep top 5 connections

    def _extract_emerging_topics(self, synthesis: str) -> List[str]:
        """Extract emerging topics from previous synthesis.

        Args:
            synthesis: Synthesis text from previous round

        Returns:
            List of refined keywords for next round
        """
        # Use knowledge processor to extract keywords
        import re
        words = re.findall(r'\b[a-zA-Z]{5,}\b', synthesis)

        # Count frequency
        from collections import Counter
        word_freq = Counter([w.lower() for w in words])

        # Get high-frequency terms (emerging topics)
        return [word for word, count in word_freq.most_common(10) if count >= 2]

    def get_fragments_for_scouts(self,
                                  round_num: int,
                                  diversity: float = 0.3) -> List[ResearchFragment]:
        """Get research fragments for scouts to ingest.

        Args:
            round_num: Current round number
            diversity: How much to include rare/niche findings (0.0-1.0)

        Returns:
            List of fragments (mix of important and niche)
        """
        if round_num >= len(self.round_history):
            return []

        round_knowledge = self.round_history[round_num]
        fragments = round_knowledge.fragments

        # Sort by importance
        important_fragments = sorted(fragments, key=lambda f: f.importance, reverse=True)

        # Sort by rarity (for niche discoveries)
        niche_fragments = sorted(fragments, key=lambda f: f.rarity, reverse=True)

        # Mix: (1-diversity) important + diversity niche
        important_count = int(len(fragments) * (1.0 - diversity))
        niche_count = int(len(fragments) * diversity)

        selected = important_fragments[:important_count] + niche_fragments[:niche_count]

        return selected

    def save_round_knowledge(self, round_num: int):
        """Save round knowledge to temporary file.

        Args:
            round_num: Round number
        """
        if round_num >= len(self.round_history):
            return

        round_knowledge = self.round_history[round_num]

        # Save fragments to temp file
        filepath = self.temp_dir / f"round_{round_num}_knowledge.txt"

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== ROUND {round_num} KNOWLEDGE ===\n\n")
            f.write(f"Keywords: {', '.join(round_knowledge.keywords)}\n")
            f.write(f"Queries: {len(round_knowledge.queries_executed)}\n")
            f.write(f"Sources: {round_knowledge.sources_count}\n")
            f.write(f"Words ingested: {round_knowledge.total_words_ingested}\n")
            f.write(f"Fragments: {len(round_knowledge.fragments)}\n\n")

            f.write("=== NICHE DISCOVERIES ===\n\n")
            for discovery in round_knowledge.niche_discoveries:
                f.write(f"- {discovery}\n")

            f.write("\n=== ALL FRAGMENTS ===\n\n")
            for i, fragment in enumerate(round_knowledge.fragments, 1):
                f.write(f"{i}. {fragment.content}\n")
                f.write(f"   Source: {fragment.source}\n")
                f.write(f"   Keywords: {', '.join(fragment.keywords)}\n")
                f.write(f"   Importance: {fragment.importance:.2f}, Rarity: {fragment.rarity:.2f}\n")
                if fragment.connections:
                    f.write(f"   Connections: {len(fragment.connections)}\n")
                f.write("\n")

        print(f"[ADVANCED RETRIEVAL] Saved round {round_num} knowledge to {filepath}")

    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"[ADVANCED RETRIEVAL] Cleaned up {self.temp_dir}")

    def get_stats(self) -> Dict:
        """Get retrieval statistics.

        Returns:
            Statistics dict
        """
        return {
            'total_rounds': len(self.round_history),
            'total_words_ingested': self.total_words_ingested,
            'total_sources_accessed': self.total_sources_accessed,
            'unique_sources': len(self.unique_sources),
            'total_fragments': len(self.all_fragments),
            'avg_words_per_round': self.total_words_ingested / max(len(self.round_history), 1),
            'avg_sources_per_round': self.total_sources_accessed / max(len(self.round_history), 1),
            'wikipedia_queries': self.wikipedia.get_stats()['queries_made'],
            'wikipedia_articles': self.wikipedia.get_stats()['articles_fetched'],
            'ddg_queries': self.ddg.get_stats()['queries_made']
        }
