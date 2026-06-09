"""Convenience wrapper to run monolith-breaking on repository documents.

This script auto-discovers documentation and can process the repository's
own material without manual file specification.
"""

import asyncio
from pathlib import Path
from swarm.monolith_breaking import run_document_swarm


def discover_documents(base_path: Path, patterns: list = None) -> list:
    """Discover documents in repository.

    Args:
        base_path: Base directory to search
        patterns: List of glob patterns (default: markdown and PDFs)

    Returns:
        List of document paths
    """
    if patterns is None:
        patterns = [
            "*.md",
            "**/*.md",
            "*.pdf",
            "**/*.pdf",
            "*.txt",
            "**/*.txt"
        ]

    documents = []
    exclude_patterns = [
        "node_modules",
        ".git",
        "__pycache__",
        "venv",
        "env",
        ".venv"
    ]

    print(f"[DISCOVERY] Searching for documents in {base_path}")

    for pattern in patterns:
        for file_path in base_path.glob(pattern):
            # Skip excluded directories
            if any(exclude in str(file_path) for exclude in exclude_patterns):
                continue

            # Skip very small files (likely not substantial content)
            if file_path.stat().st_size < 500:  # Less than 500 bytes
                continue

            documents.append(str(file_path))

    # Remove duplicates and sort
    documents = sorted(set(documents))

    return documents


async def main():
    """Run monolith-breaking on discovered documents."""
    print("=" * 80)
    print("AUTO-DISCOVERY MODE: Processing Repository Documentation")
    print("=" * 80)

    # Discover documents in current repository
    repo_path = Path(__file__).parent
    documents = discover_documents(repo_path)

    if not documents:
        print("\nNo documents found in repository!")
        print("Looking for: *.md, *.pdf, *.txt files")
        return

    print(f"\n[DISCOVERY] Found {len(documents)} documents:")
    for i, doc in enumerate(documents, 1):
        file_path = Path(doc)
        size_kb = file_path.stat().st_size / 1024
        print(f"  {i}. {file_path.name} ({size_kb:.1f} KB)")

    print(f"\n[DISCOVERY] Total documents: {len(documents)}")

    # Ask for confirmation
    print("\nProcess all discovered documents? [y/N]: ", end="")
    response = input().strip().lower()

    if response not in ['y', 'yes']:
        print("Cancelled.")
        return

    print("\n" + "=" * 80)
    print("STARTING DISTRIBUTED PROCESSING")
    print("=" * 80)

    # Run processing
    result = await run_document_swarm(
        document_paths=documents,
        model_name="microsoft/phi-2",
        num_foragers=30,  # Smaller for auto-mode
        num_gatherers=10,
        num_critics=10,
        num_haters=5,
        max_iterations=300,  # Shorter for auto-mode
        convergence_threshold=0.7,
        enable_mcp=True  # Enable external validation
    )

    if result:
        print("\n\n" + "=" * 80)
        print("PROCESSING COMPLETE - SUMMARY")
        print("=" * 80)
        print(f"\nDocuments processed: {len(documents)}")
        print(f"Compression ratio: {result['compression_ratio']:.1f}:1")
        print(f"Convergence achieved: {result['converged']}")
        print(f"Processing time: {result['processing_time']:.1f}s")
        print(f"\nTop insights generated: {len(result['top_insights'])}")

        # Save results
        output_file = repo_path / "repository_analysis.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPOSITORY DOCUMENTATION ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Documents Analyzed ({len(documents)}):\n")
            for doc in documents:
                f.write(f"  - {Path(doc).name}\n")
            f.write(f"\n{'=' * 80}\n")
            f.write("SYNTHESIS\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(result['synthesis'])
            f.write(f"\n\n{'=' * 80}\n")
            f.write("TOP INSIGHTS\n")
            f.write(f"{'=' * 80}\n\n")
            for i, insight in enumerate(result['top_insights'][:10], 1):
                f.write(f"{i}. [Strength: {insight.strength:.3f}]\n")
                f.write(f"{insight.content}\n\n")

        print(f"\nResults saved to: {output_file}")

    else:
        print("\nProcessing failed - check logs for errors")


if __name__ == "__main__":
    asyncio.run(main())
