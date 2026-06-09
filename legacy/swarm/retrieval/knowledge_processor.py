"""Knowledge processing and chunking for agent consumption.

Breaks down large documents into digestible chunks that scouts can
ingest, process, and synthesize into signals.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KnowledgeChunk:
    """A chunk of knowledge extracted from a source."""
    content: str
    source: str
    source_url: str
    chunk_id: int
    total_chunks: int
    metadata: Dict
    keywords: List[str]
    importance_score: float


class KnowledgeProcessor:
    """Processes and chunks knowledge for agent consumption."""

    def __init__(self,
                 chunk_size: int = 500,  # words per chunk
                 overlap: int = 50,      # word overlap between chunks
                 min_chunk_size: int = 100):
        """Initialize processor.

        Args:
            chunk_size: Target words per chunk
            overlap: Overlap between chunks for context
            min_chunk_size: Minimum words for a valid chunk
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk_content(self, content: str, source: str = "", source_url: str = "") -> List[KnowledgeChunk]:
        """Break content into chunks for agent processing.

        Args:
            content: Full content to chunk
            source: Source name (e.g., "Wikipedia: Climate Change")
            source_url: Source URL

        Returns:
            List of knowledge chunks
        """
        if not content or len(content.strip()) == 0:
            return []

        # Split into paragraphs first (preserve natural boundaries)
        paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 20]

        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_id = 0

        for paragraph in paragraphs:
            words = paragraph.split()
            para_word_count = len(words)

            # If adding this paragraph exceeds chunk size, create chunk
            if current_word_count + para_word_count > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(chunk_text)
                chunk_id += 1

                # Start new chunk with overlap
                # Keep last paragraph for context
                if current_chunk:
                    overlap_text = current_chunk[-1]
                    overlap_words = len(overlap_text.split())
                    if overlap_words <= self.overlap:
                        current_chunk = [overlap_text]
                        current_word_count = overlap_words
                    else:
                        # Take last N words
                        overlap_words_list = overlap_text.split()[-self.overlap:]
                        current_chunk = [' '.join(overlap_words_list)]
                        current_word_count = len(overlap_words_list)
                else:
                    current_chunk = []
                    current_word_count = 0

            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_word_count += para_word_count

        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text.split()) >= self.min_chunk_size:
                chunks.append(chunk_text)
                chunk_id += 1

        # Convert to KnowledgeChunk objects
        total_chunks = len(chunks)
        knowledge_chunks = []

        for i, chunk_text in enumerate(chunks):
            # Extract keywords from chunk
            keywords = self._extract_keywords(chunk_text)

            # Score importance
            importance = self._score_importance(chunk_text, keywords)

            knowledge_chunks.append(KnowledgeChunk(
                content=chunk_text,
                source=source,
                source_url=source_url,
                chunk_id=i,
                total_chunks=total_chunks,
                metadata={
                    'word_count': len(chunk_text.split()),
                    'char_count': len(chunk_text)
                },
                keywords=keywords,
                importance_score=importance
            ))

        return knowledge_chunks

    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract key terms from text chunk.

        Args:
            text: Text to analyze
            top_k: Number of keywords to extract

        Returns:
            List of keywords
        """
        # Simple frequency-based extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        # Stop words
        stop_words = {
            'this', 'that', 'with', 'from', 'have', 'been', 'were', 'would',
            'their', 'which', 'these', 'those', 'there', 'where', 'when',
            'what', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'also', 'more', 'most', 'some', 'such', 'only',
            'than', 'very', 'other', 'should', 'could', 'will', 'just'
        }

        # Count frequency
        from collections import Counter
        word_freq = Counter([w for w in words if w not in stop_words])

        # Get top keywords
        return [word for word, _ in word_freq.most_common(top_k)]

    def _score_importance(self, text: str, keywords: List[str]) -> float:
        """Score chunk importance.

        Args:
            text: Chunk text
            keywords: Extracted keywords

        Returns:
            Importance score (0.0-1.0)
        """
        score = 0.0

        # Length score (longer = more content)
        word_count = len(text.split())
        if word_count > 300:
            score += 0.3
        elif word_count > 150:
            score += 0.2
        else:
            score += 0.1

        # Keyword density
        if len(keywords) >= 5:
            score += 0.2
        elif len(keywords) >= 3:
            score += 0.1

        # Contains numbers/data (facts)
        if re.search(r'\d+', text):
            score += 0.1

        # Contains citations/sources
        citation_patterns = [
            r'\[\d+\]',  # [1], [2]
            r'\([A-Z][a-z]+ \d{4}\)',  # (Smith 2020)
            r'according to',
            r'research shows',
            r'study found',
            r'evidence suggests'
        ]
        if any(re.search(pattern, text, re.I) for pattern in citation_patterns):
            score += 0.2

        # Contains specific terminology (capitalized terms)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if len(capitalized) >= 5:
            score += 0.2
        elif len(capitalized) >= 2:
            score += 0.1

        return min(1.0, score)

    def extract_key_facts(self, text: str, max_facts: int = 5) -> List[str]:
        """Extract key factual statements from text.

        Args:
            text: Text to analyze
            max_facts: Maximum facts to extract

        Returns:
            List of factual statements
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        facts = []
        for sentence in sentences:
            sentence = sentence.strip()

            # Skip very short sentences
            if len(sentence.split()) < 5:
                continue

            # Check for factual indicators
            factual_patterns = [
                r'\b(is|are|was|were|has|have)\b',  # State of being
                r'\b\d+\s*(percent|%|million|billion|thousand)\b',  # Statistics
                r'\b(research|study|evidence|data|results)\b',  # Citations
                r'\b(cause|effect|lead|result in|due to)\b',  # Causation
                r'\b(increase|decrease|rise|fall|grow)\b'  # Trends
            ]

            if any(re.search(pattern, sentence, re.I) for pattern in factual_patterns):
                facts.append(sentence)

            if len(facts) >= max_facts:
                break

        return facts

    def synthesize_summary(self, chunks: List[KnowledgeChunk], max_words: int = 200) -> str:
        """Create a concise summary from chunks.

        Args:
            chunks: Knowledge chunks to summarize
            max_words: Maximum words in summary

        Returns:
            Summary text
        """
        if not chunks:
            return ""

        # Sort chunks by importance
        sorted_chunks = sorted(chunks, key=lambda c: c.importance_score, reverse=True)

        # Take most important chunks
        summary_parts = []
        current_words = 0

        for chunk in sorted_chunks:
            # Extract key facts from chunk
            facts = self.extract_key_facts(chunk.content, max_facts=2)

            for fact in facts:
                fact_words = len(fact.split())
                if current_words + fact_words <= max_words:
                    summary_parts.append(fact)
                    current_words += fact_words
                else:
                    break

            if current_words >= max_words:
                break

        return ' '.join(summary_parts)

    def create_scout_knowledge_base(self, chunks: List[KnowledgeChunk]) -> List[Dict]:
        """Create knowledge fragments for scouts to use.

        Each fragment is a piece of knowledge a scout can deposit as a signal.

        Args:
            chunks: Knowledge chunks

        Returns:
            List of knowledge fragments
        """
        fragments = []

        for chunk in chunks:
            # Extract key facts
            facts = self.extract_key_facts(chunk.content, max_facts=3)

            for fact in facts:
                fragments.append({
                    'content': fact,
                    'source': chunk.source,
                    'source_url': chunk.source_url,
                    'keywords': chunk.keywords,
                    'importance': chunk.importance_score,
                    'chunk_id': chunk.chunk_id
                })

        # Sort by importance
        fragments.sort(key=lambda f: f['importance'], reverse=True)

        return fragments


def process_search_results(search_results: Dict, task_context: str = "") -> Dict:
    """Process search results into knowledge ready for agents.

    Args:
        search_results: Results from MultiSourceSearch
        task_context: Task description for relevance filtering

    Returns:
        Processed knowledge with chunks and fragments
    """
    processor = KnowledgeProcessor()

    all_chunks = []
    source_metadata = {}

    # Process each source
    for source_name, source_data in search_results.get('sources', {}).items():
        content = source_data.get('content', '')
        url = source_data.get('url', '')

        if content:
            # Create chunks
            chunks = processor.chunk_content(
                content=content,
                source=f"{source_name}: {source_data.get('title', 'Untitled')}",
                source_url=url
            )

            all_chunks.extend(chunks)

            source_metadata[source_name] = {
                'chunks': len(chunks),
                'total_words': sum(c.metadata['word_count'] for c in chunks),
                'url': url
            }

    # Create knowledge fragments for scouts
    fragments = processor.create_scout_knowledge_base(all_chunks)

    # Create summary
    summary = processor.synthesize_summary(all_chunks, max_words=300)

    return {
        'chunks': all_chunks,
        'fragments': fragments,
        'summary': summary,
        'source_metadata': source_metadata,
        'total_chunks': len(all_chunks),
        'total_fragments': len(fragments)
    }
