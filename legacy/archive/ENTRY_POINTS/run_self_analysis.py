"""Quick self-analysis: Process repository's own documentation.

This is a simplified version that automatically processes key documentation
files without prompting.
"""

import asyncio
from pathlib import Path
from swarm.monolith_breaking import run_document_swarm


async def main():
    """Process key repository documentation."""
    print("=" * 80)
    print("SELF-ANALYSIS: Processing Repository Documentation")
    print("=" * 80)

    repo_path = Path(__file__).parent

    # Key documentation files to analyze
    key_docs = [
        "A_NOTE_FOR_CLAUDE.md",
        "TRANSFORMATION_STATUS.md",
        "README.md",
        "STIGMERGIC_SWARM_DESIGN.md",
        "PROJECT_STRUCTURE.md"
    ]

    # Find existing files
    documents = []
    for doc_name in key_docs:
        doc_path = repo_path / doc_name
        if doc_path.exists():
            documents.append(str(doc_path))
            print(f"  ✓ Found: {doc_name}")
        else:
            print(f"  ✗ Missing: {doc_name}")

    if not documents:
        print("\nNo documentation found!")
        return

    print(f"\n[ANALYSIS] Processing {len(documents)} documentation files")
    print("\nThis will:")
    print("  1. Read each document in parallel (distributed)")
    print("  2. Extract key observations")
    print("  3. Find patterns across documents")
    print("  4. Validate insights with external sources")
    print("  5. Generate comprehensive synthesis")

    # Run with lighter configuration for documentation
    result = await run_document_swarm(
        document_paths=documents,
        model_name="microsoft/phi-2",
        num_foragers=20,
        num_gatherers=10,
        num_critics=10,
        num_haters=5,
        max_iterations=200,
        convergence_threshold=0.7,
        enable_mcp=True
    )

    if result:
        print("\n\n" + "=" * 80)
        print("REPOSITORY SELF-ANALYSIS COMPLETE")
        print("=" * 80)

        # Save results
        output_file = repo_path / "self_analysis.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("STIGMERGIC SWARM REPOSITORY - SELF-ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write("SYNTHESIS OF REPOSITORY DOCUMENTATION\n")
            f.write("=" * 80 + "\n\n")
            f.write(result['synthesis'])
            f.write(f"\n\n{'=' * 80}\n")
            f.write("KEY INSIGHTS DISCOVERED\n")
            f.write(f"{'=' * 80}\n\n")
            for i, insight in enumerate(result['top_insights'][:10], 1):
                f.write(f"\n{i}. Insight (Strength: {insight.strength:.3f})\n")
                f.write("-" * 80 + "\n")
                f.write(f"{insight.content}\n")

        print(f"\nAnalysis saved to: {output_file}")
        print(f"\nKey Metrics:")
        print(f"  - Documents analyzed: {len(documents)}")
        print(f"  - Insights discovered: {len(result['top_insights'])}")
        print(f"  - Compression ratio: {result['compression_ratio']:.1f}:1")
        print(f"  - Convergence: {'YES' if result['converged'] else 'NO'}")


if __name__ == "__main__":
    asyncio.run(main())
