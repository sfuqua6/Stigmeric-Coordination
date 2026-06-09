"""Document processing pipeline for loading and splitting documents."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Optional tiktoken import with fallback
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    tiktoken = None


@dataclass
class DocumentSection:
    """A section of a document for agent processing."""
    section_id: int
    content: str
    source_document: str
    token_count: int
    metadata: dict  # Can include page numbers, section headings, etc.


class DocumentProcessor:
    """Processes documents and splits them into sections for distributed reading."""

    def __init__(self, target_tokens: int = 2000, tolerance: int = 500):
        """Initialize document processor.

        Args:
            target_tokens: Target tokens per section
            tolerance: Acceptable deviation from target (±tolerance)
        """
        self.target_tokens = target_tokens
        self.tolerance = tolerance
        self.min_tokens = target_tokens - tolerance
        self.max_tokens = target_tokens + tolerance

        # Initialize tokenizer (using cl100k_base for GPT-4 style tokenization)
        if HAS_TIKTOKEN:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                print(f"[DOC PROCESSOR] Warning: tiktoken initialization failed, using word count approximation: {e}")
                self.tokenizer = None
        else:
            print(f"[DOC PROCESSOR] Warning: tiktoken not available, using word count approximation")
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate as 4 chars per token
            return len(text) // 4

    def load_text_file(self, file_path: Path) -> tuple[str, dict]:
        """Load plain text file.

        Args:
            file_path: Path to text file

        Returns:
            Tuple of (content, metadata)
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        metadata = {
            'format': 'text',
            'filename': file_path.name,
            'size_bytes': file_path.stat().st_size
        }

        return content, metadata

    def load_pdf_file(self, file_path: Path) -> tuple[str, dict]:
        """Load PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Tuple of (content, metadata)
        """
        try:
            import PyPDF2
        except ImportError:
            raise ImportError(
                "PyPDF2 not installed. Install with: pip install PyPDF2"
            )

        content_parts = []
        page_count = 0

        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            page_count = len(pdf_reader.pages)

            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    # Add page marker for provenance tracking
                    content_parts.append(f"\n[Page {page_num + 1}]\n{text}")

        content = "\n".join(content_parts)

        metadata = {
            'format': 'pdf',
            'filename': file_path.name,
            'page_count': page_count,
            'size_bytes': file_path.stat().st_size
        }

        return content, metadata

    def load_document(self, file_path: str) -> tuple[str, dict]:
        """Load document from file.

        Args:
            file_path: Path to document file

        Returns:
            Tuple of (content, metadata)

        Raises:
            ValueError: If file format not supported
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix in ['.txt', '.md', '.markdown']:
            return self.load_text_file(path)
        elif suffix == '.pdf':
            return self.load_pdf_file(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def split_on_paragraphs(self, text: str) -> List[str]:
        """Split text on paragraph boundaries.

        Args:
            text: Text to split

        Returns:
            List of paragraphs
        """
        # Split on double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)

        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return paragraphs

    def group_paragraphs_into_sections(self, paragraphs: List[str]) -> List[str]:
        """Group paragraphs into sections that meet token targets.

        Args:
            paragraphs: List of paragraphs

        Returns:
            List of section texts
        """
        sections = []
        current_section = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If this paragraph alone exceeds max, split it if possible
            if para_tokens > self.max_tokens:
                # Save current section if any
                if current_section:
                    sections.append('\n\n'.join(current_section))
                    current_section = []
                    current_tokens = 0

                # Split very long paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_section = []
                temp_tokens = 0

                for sentence in sentences:
                    sent_tokens = self.count_tokens(sentence)
                    if temp_tokens + sent_tokens > self.max_tokens and temp_section:
                        sections.append(' '.join(temp_section))
                        temp_section = [sentence]
                        temp_tokens = sent_tokens
                    else:
                        temp_section.append(sentence)
                        temp_tokens += sent_tokens

                if temp_section:
                    sections.append(' '.join(temp_section))

            # If adding this paragraph would exceed max, save current section
            elif current_tokens + para_tokens > self.max_tokens and current_section:
                sections.append('\n\n'.join(current_section))
                current_section = [para]
                current_tokens = para_tokens

            # Otherwise add to current section
            else:
                current_section.append(para)
                current_tokens += para_tokens

                # If we've reached a good size, save section
                if current_tokens >= self.min_tokens:
                    sections.append('\n\n'.join(current_section))
                    current_section = []
                    current_tokens = 0

        # Save any remaining content
        if current_section:
            sections.append('\n\n'.join(current_section))

        return sections

    def split_into_sections(self, text: str, source_name: str,
                           metadata: dict) -> List[DocumentSection]:
        """Split text into DocumentSection objects.

        This method is used by web ingestion to convert fetched web content
        into document sections for processing.

        Args:
            text: Text content to split
            source_name: Source identifier (e.g., URL or filename)
            metadata: Metadata dictionary (e.g., title, url, etc.)

        Returns:
            List of DocumentSection objects
        """
        # Split into paragraphs
        paragraphs = self.split_on_paragraphs(text)

        # Group into sections
        section_texts = self.group_paragraphs_into_sections(paragraphs)

        # Convert to DocumentSection objects
        sections = []
        for i, section_text in enumerate(section_texts):
            section_tokens = self.count_tokens(section_text)

            section = DocumentSection(
                section_id=i,
                content=section_text,
                source_document=source_name,
                token_count=section_tokens,
                metadata=metadata.copy() if metadata else {}
            )

            sections.append(section)

        return sections

    def process_documents(self, file_paths: List[str]) -> List[DocumentSection]:
        """Process multiple documents into sections.

        Args:
            file_paths: List of document file paths

        Returns:
            List of DocumentSection objects
        """
        all_sections = []
        section_id = 0

        print(f"[DOC PROCESSOR] Processing {len(file_paths)} documents...")

        for file_path in file_paths:
            print(f"[DOC PROCESSOR] Loading {file_path}...")

            try:
                content, metadata = self.load_document(file_path)
                doc_tokens = self.count_tokens(content)

                print(f"[DOC PROCESSOR]   Loaded: {doc_tokens} tokens, {metadata.get('page_count', 'N/A')} pages")

                # Split into paragraphs
                paragraphs = self.split_on_paragraphs(content)
                print(f"[DOC PROCESSOR]   Split into {len(paragraphs)} paragraphs")

                # Group into sections
                sections = self.group_paragraphs_into_sections(paragraphs)
                print(f"[DOC PROCESSOR]   Created {len(sections)} sections")

                # Create DocumentSection objects
                for section_text in sections:
                    section_tokens = self.count_tokens(section_text)

                    # Extract page numbers from content if available
                    page_markers = re.findall(r'\[Page (\d+)\]', section_text)
                    section_metadata = {
                        'source_file': file_path,
                        'source_format': metadata['format'],
                        'pages': page_markers if page_markers else None
                    }

                    # Remove page markers from content for cleaner text
                    clean_text = re.sub(r'\[Page \d+\]\n', '', section_text)

                    section = DocumentSection(
                        section_id=section_id,
                        content=clean_text,
                        source_document=Path(file_path).name,
                        token_count=section_tokens,
                        metadata=section_metadata
                    )

                    all_sections.append(section)
                    section_id += 1

            except Exception as e:
                print(f"[DOC PROCESSOR] ERROR: Failed to process {file_path}: {e}")
                continue

        print(f"[DOC PROCESSOR] Processing complete: {len(all_sections)} total sections")

        # Print statistics
        if all_sections:
            token_counts = [s.token_count for s in all_sections]
            print(f"[DOC PROCESSOR] Token distribution:")
            print(f"  Min: {min(token_counts)}, Max: {max(token_counts)}")
            print(f"  Mean: {sum(token_counts) // len(token_counts)}")
            print(f"  Total: {sum(token_counts)} tokens")

        return all_sections


def test_processor():
    """Test the document processor."""
    # Create test text file
    test_file = Path("test_document.txt")
    test_content = """
This is the first paragraph of the test document. It contains some information
about testing the document processing pipeline.

This is the second paragraph. It's separated by a blank line from the first
paragraph. This helps the processor identify paragraph boundaries.

Here's a third paragraph with some more content. The processor should group
these paragraphs together into sections based on the target token count.

And a fourth paragraph to ensure we have enough content to test the sectioning
logic. The processor needs to balance between creating sections that are too
small and sections that are too large.

Finally, a fifth paragraph to wrap things up. This should give us enough
content to create at least one or two sections, depending on the token counts.
"""

    test_file.write_text(test_content)

    try:
        processor = DocumentProcessor(target_tokens=100, tolerance=50)
        sections = processor.process_documents([str(test_file)])

        print(f"\nTest Results:")
        print(f"Created {len(sections)} sections:")
        for section in sections:
            print(f"\nSection {section.section_id}:")
            print(f"  Source: {section.source_document}")
            print(f"  Tokens: {section.token_count}")
            print(f"  Preview: {section.content[:100]}...")

    finally:
        # Clean up
        test_file.unlink()


if __name__ == "__main__":
    test_processor()
