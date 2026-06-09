"""Prepare research information for swarm agents.

This script helps you gather research information into a format that scouts can use.
Simply dump all relevant text into research/info.txt and scouts will search it.
"""

import sys
from pathlib import Path


def prepare_research_folder():
    """Create research folder if it doesn't exist."""
    research_dir = Path("research")
    research_dir.mkdir(exist_ok=True)

    info_file = research_dir / "info.txt"

    if not info_file.exists():
        # Create template
        template = """# Research Information for Swarm Agents

This file contains information that scouts will search when generating signals.
Add any relevant research, facts, data, or context below.

## Example: Sustainable Cities

Green infrastructure reduces urban heat island effect by 2-5°C.
Studies show bike lanes increase cycling rates by 30-50% in first year.
Copenhagen aims for carbon neutrality by 2025 through district heating and cycling.
Singapore's vertical gardens cover 40% of building surfaces, reducing cooling needs by 15%.
LED street lighting reduces municipal energy costs by 40-60%.
Permeable pavements reduce stormwater runoff by 80% compared to conventional surfaces.

## Your Research

[Add your research here...]

"""
        info_file.write_text(template, encoding='utf-8')
        print(f"[PREP] Created template: {info_file}")
        print(f"[PREP] Edit this file to add your research")
    else:
        content = info_file.read_text(encoding='utf-8')
        word_count = len(content.split())
        print(f"[PREP] Research file exists: {info_file} ({word_count} words)")

    return info_file


if __name__ == "__main__":
    print("=" * 70)
    print("RESEARCH PREPARATION FOR STIGMERGIC SWARM")
    print("=" * 70)

    info_file = prepare_research_folder()

    print(f"\n[NEXT STEPS]")
    print(f"1. Edit {info_file}")
    print(f"2. Add relevant research, facts, statistics")
    print(f"3. Run: python run_task.py problem_solving \"Your question\"")
    print(f"\nScouts will search this file using keywords from your question.")
    print("=" * 70)
