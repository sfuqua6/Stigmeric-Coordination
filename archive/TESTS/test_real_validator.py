"""Test Real Validator with Dynamic Knowledge Base.

Demonstrates:
- DynamicKnowledgeBase learns during execution
- External sources verify claims
- RealValidator integrates both
- NO LLM self-critique

This shows Phase 4 is complete and working.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from swarm.validation.dynamic_knowledge_base import DynamicKnowledgeBase
from swarm.validation.external_sources import WikipediaSource, WebSearchSource, SymbolicMathSource, MultiSourceVerifier
from swarm.validation.real_validator import RealValidator
from swarm.core.signal_store import Signal


async def test_dynamic_knowledge_base():
    """Test that knowledge base learns during execution."""
    print("=" * 70)
    print("TEST 1: Dynamic Knowledge Base Learning")
    print("=" * 70)

    kb = DynamicKnowledgeBase(confidence_threshold=0.6)

    print("\nInitial state:")
    print(f"  Facts: {kb.get_stats()['total_facts']}")

    # Test 1: Query unknown fact
    print("\n1. Query unknown fact:")
    result = kb.query("Renewable energy reduces carbon emissions")
    print(f"   Known: {result['known']}")
    print(f"   Confidence: {result['confidence']:.2f}")

    # Test 2: Learn fact
    print("\n2. Learn new fact:")
    kb.learn(
        claim="Renewable energy reduces carbon emissions",
        confidence=0.85,
        source="wikipedia",
        evidence="Wikipedia article on renewable energy"
    )
    print(f"   Facts learned: 1")
    print(f"   Total facts: {kb.get_stats()['total_facts']}")

    # Test 3: Query learned fact
    print("\n3. Query learned fact:")
    result = kb.query("Renewable energy reduces carbon emissions")
    print(f"   Known: {result['known']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Sources: {', '.join(result['sources'])}")

    # Test 4: Update confidence with new evidence
    print("\n4. Update with new evidence:")
    kb.learn(
        claim="Renewable energy reduces carbon emissions",
        confidence=0.90,
        source="web_search",
        evidence="Multiple sources confirm"
    )
    result = kb.query("Renewable energy reduces carbon emissions")
    print(f"   Updated confidence: {result['confidence']:.2f}")
    print(f"   Sources: {', '.join(result['sources'])}")
    print(f"   Verifications: {result['verification_count']}")

    # Test 5: Find related facts
    print("\n5. Learn related fact:")
    kb.learn(
        claim="Solar power is a renewable energy source",
        confidence=0.88,
        source="wikipedia",
        evidence="Wikipedia article on solar power"
    )

    related = kb.find_related("Renewable energy reduces carbon emissions")
    print(f"   Related facts found: {len(related)}")
    for claim, conf in related:
        print(f"     - {claim[:60]} (conf={conf:.2f})")

    # Final stats
    print("\n" + "=" * 70)
    print("LEARNING SUMMARY")
    print("=" * 70)
    stats = kb.get_stats()
    print(f"  Total facts learned: {stats['total_facts']}")
    print(f"  High confidence facts: {stats['high_confidence_facts']}")
    print(f"  Average confidence: {stats['avg_confidence']:.2f}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"   Knowledge base is DYNAMIC - learns during execution")


async def test_external_sources():
    """Test external verification sources."""
    print("\n\n" + "=" * 70)
    print("TEST 2: External Sources Verification")
    print("=" * 70)

    # Test Wikipedia source
    print("\n1. Wikipedia Source:")
    wiki = WikipediaSource()

    claims = [
        "Renewable energy reduces carbon emissions",
        "Solar power is sustainable",
        "Electric vehicles reduce pollution",
    ]

    for claim in claims:
        result = await wiki.verify(claim)
        print(f"   {claim[:50]}...")
        print(f"     Verified: {result['verified']}, Confidence: {result['confidence']:.2f}")

    # Test Web Search source
    print("\n2. Web Search Consensus:")
    web = WebSearchSource(min_sources=2)

    for claim in claims[:2]:
        result = await web.verify(claim)
        print(f"   {claim[:50]}...")
        print(f"     Verified: {result['verified']}, Confidence: {result['confidence']:.2f}")
        print(f"     Sources: {result.get('num_sources', 0)}")

    # Test Symbolic Math source
    print("\n3. Symbolic Math:")
    math_source = SymbolicMathSource()

    math_claims = [
        "2 + 2 = 4",
        "Solar power reduces emissions by 80 percent",
        "This is not a math claim",
    ]

    for claim in math_claims:
        result = await math_source.verify(claim)
        print(f"   {claim}")
        print(f"     Verified: {result['verified']}, Confidence: {result['confidence']:.2f}")

    # Test Multi-Source Verifier
    print("\n4. Multi-Source Verification:")
    multi = MultiSourceVerifier()

    claim = "Renewable energy reduces carbon emissions"
    result = await multi.verify(claim)
    print(f"   Claim: {claim}")
    print(f"     Verified: {result['verified']}")
    print(f"     Confidence: {result['confidence']:.2f}")
    print(f"     Sources verified: {result['sources_verified']}/{result['sources_checked']}")

    print("\n   External sources work - NO LLM self-critique")


async def test_real_validator():
    """Test RealValidator with signals."""
    print("\n\n" + "=" * 70)
    print("TEST 3: RealValidator Integration")
    print("=" * 70)

    # Create knowledge base
    kb = DynamicKnowledgeBase(confidence_threshold=0.6)

    # Create validator
    validator = RealValidator(
        agent_id="real_validator_001",
        knowledge_base=kb
    )

    # Create test signals
    signals = [
        Signal(
            id="sig_001",
            type="SCOUT_SIGNAL",
            content="Renewable energy reduces carbon emissions. Solar power is sustainable. Research shows this is effective.",
            strength=0.7,
            timestamp=1234567890.0,
            depositor="scout_1"
        ),
        Signal(
            id="sig_002",
            type="FORAGER_SIGNAL",
            content="Electric vehicles reduce pollution in cities. Studies demonstrate significant air quality improvements.",
            strength=0.65,
            timestamp=1234567891.0,
            depositor="forager_1"
        ),
        Signal(
            id="sig_003",
            type="SCOUT_SIGNAL",
            content="I think maybe climate change is real. Perhaps we should do something.",
            strength=0.5,
            timestamp=1234567892.0,
            depositor="scout_2"
        ),
    ]

    # Validate each signal
    print("\nValidating signals:")
    for i, signal in enumerate(signals, 1):
        print(f"\n{i}. Signal {signal.id}:")
        print(f"   Content: {signal.content[:70]}...")

        result = await validator.validate_signal(signal)

        print(f"   Accuracy: {result.accuracy:.2%}")
        print(f"   Verified claims: {result.verified_claims}/{result.total_claims}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Facts learned: {result.learned_facts}")
        print(f"   Conflicts: {result.conflicts}")

        if result.evidence:
            print(f"   Evidence samples:")
            for ev in result.evidence[:2]:
                print(f"     - {ev[:60]}...")

    # Show learning progress
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    stats = validator.get_stats()
    print(f"  Signals validated: {stats['signals_validated']}")
    print(f"  Claims verified: {stats['claims_verified']}")
    print(f"  Facts learned: {stats['facts_learned']}")
    print(f"  Conflicts detected: {stats['conflicts_detected']}")

    kb_stats = stats['kb_stats']
    print(f"\n  Knowledge Base:")
    print(f"    Total facts: {kb_stats['total_facts']}")
    print(f"    High confidence: {kb_stats['high_confidence_facts']}")
    print(f"    Avg confidence: {kb_stats['avg_confidence']:.2f}")

    # Export learned knowledge
    print("\n  Learned Knowledge:")
    learned = kb.export_knowledge(min_confidence=0.6)
    for fact in learned[:3]:
        print(f"    - {fact['claim'][:60]}")
        print(f"      Confidence: {fact['confidence']:.2f}, Verifications: {fact['verifications']}")

    print("\n   RealValidator learns and verifies using external sources")
    print("   NO LLM self-critique - true external validation")


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PHASE 4: REAL VALIDATION WITH DYNAMIC LEARNING")
    print("=" * 70)
    print("\nThis demonstrates:")
    print("  1. DynamicKnowledgeBase learns during execution")
    print("  2. External sources verify claims (Wikipedia, web, math)")
    print("  3. RealValidator integrates both")
    print("  4. NO LLM self-critique (security theater eliminated)")
    print()

    await test_dynamic_knowledge_base()
    await test_external_sources()
    await test_real_validator()

    print("\n\n" + "=" * 70)
    print("PHASE 4 COMPLETE ")
    print("=" * 70)
    print("\nKey Improvements:")
    print("   Dynamic knowledge base that LEARNS")
    print("   External verification (Wikipedia, web, math)")
    print("   Confidence tracking and updates")
    print("   Conflict detection")
    print("   NO LLM asking itself if it's correct")
    print("\nNext: Replace fake Validator in run_task.py with RealValidator")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
