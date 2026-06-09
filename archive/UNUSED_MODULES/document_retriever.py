"""Simple document retrieval for stigmergic swarm agents.

This module provides keyword-based information retrieval from text documents.
Agents can search for relevant information using keywords extracted from their tasks.
"""

from typing import List, Optional
import re
from pathlib import Path


class DocumentRetriever:
    """Simple keyword-based document retrieval system.

    Provides agents with ability to retrieve information from text documents
    using keyword matching and related word expansion.
    """

    def __init__(self, document_paths: List[str] = None):
        """Initialize retriever with optional document corpus.

        Args:
            document_paths: List of paths to text documents to index
        """
        self.documents = []
        self.document_index = {}

        if document_paths:
            for path in document_paths:
                self.add_document(path)

    def add_document(self, path: str):
        """Add a document to the retrieval corpus.

        Args:
            path: Path to text document
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.documents.append({
                    'path': path,
                    'content': content,
                    'keywords': self._extract_keywords(content)
                })
                print(f"[RETRIEVER] Indexed document: {path} ({len(content)} chars)")
        except Exception as e:
            print(f"[RETRIEVER] Error loading {path}: {e}")

    def add_text(self, text: str, source: str = "inline"):
        """Add inline text to the retrieval corpus.

        Args:
            text: Text content
            source: Source identifier
        """
        self.documents.append({
            'path': source,
            'content': text,
            'keywords': self._extract_keywords(text)
        })

    def search(self, keywords: List[str], max_results: int = 3) -> List[str]:
        """Search for documents matching keywords.

        Args:
            keywords: List of keywords to search for
            max_results: Maximum number of results to return

        Returns:
            List of relevant text snippets
        """
        if not keywords:
            return []

        # Score each document by keyword matches
        scored_docs = []
        for doc in self.documents:
            score = 0
            matches = []

            # Check for keyword matches
            content_lower = doc['content'].lower()
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in content_lower:
                    # Count occurrences
                    count = content_lower.count(keyword_lower)
                    score += count * 10
                    matches.append(keyword)

            # Check for related words in document keywords
            for doc_keyword in doc['keywords']:
                for search_keyword in keywords:
                    if self._are_related(doc_keyword, search_keyword):
                        score += 5

            if score > 0:
                scored_docs.append((score, doc, matches))

        # Sort by score and return top results
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc, matches in scored_docs[:max_results]:
            # Extract relevant snippet around keyword matches
            snippet = self._extract_snippet(doc['content'], matches)
            results.append(snippet)

        return results

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text.

        Args:
            text: Input text

        Returns:
            List of keywords
        """
        # Remove punctuation and split
        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()

        # Filter stopwords and short words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                    'will', 'would', 'should', 'could', 'may', 'might', 'can',
                    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'from', 'by', 'as',
                    'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their'}

        keywords = [w for w in words if w not in stopwords and len(w) > 3]

        # Return unique keywords
        return list(set(keywords))[:50]  # Limit to 50 unique keywords

    def _are_related(self, word1: str, word2: str) -> bool:
        """Check if two words are related (simple heuristic).

        Args:
            word1: First word
            word2: Second word

        Returns:
            True if words are related
        """
        # Simple relatedness: share first 4+ characters (stem matching)
        if len(word1) >= 4 and len(word2) >= 4:
            return word1[:4] == word2[:4]
        return False

    def _extract_snippet(self, text: str, keywords: List[str], context_size: int = 200) -> str:
        """Extract relevant snippet around keywords.

        Args:
            text: Full text
            keywords: Keywords to extract around
            context_size: Size of context window (chars)

        Returns:
            Text snippet
        """
        if not keywords:
            return text[:context_size] + "..."

        # Find first keyword occurrence
        text_lower = text.lower()
        first_pos = -1
        for keyword in keywords:
            pos = text_lower.find(keyword.lower())
            if pos != -1 and (first_pos == -1 or pos < first_pos):
                first_pos = pos

        if first_pos == -1:
            return text[:context_size] + "..."

        # Extract context around keyword
        start = max(0, first_pos - context_size // 2)
        end = min(len(text), first_pos + context_size // 2)

        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet

    def get_stats(self) -> dict:
        """Get retriever statistics.

        Returns:
            Dictionary with stats
        """
        return {
            'num_documents': len(self.documents),
            'total_chars': sum(len(d['content']) for d in self.documents),
            'total_keywords': sum(len(d['keywords']) for d in self.documents)
        }
