#!/usr/bin/env python3
"""Test symbolic math verification implementation."""

import asyncio
from swarm.validation.external_sources import SymbolicMathSource


async def test_symbolic_math():
    """Test various types of mathematical claims."""

    math_source = SymbolicMathSource()

    test_cases = [
        # Arithmetic
        ("2 + 2 = 4", True, 1.0, "Basic addition"),
        ("5 * 3 = 15", True, 1.0, "Basic multiplication"),
        ("10 - 3 = 7", True, 1.0, "Basic subtraction"),
        ("2 + 2 = 5", False, 0.0, "Incorrect arithmetic"),

        # Percentages
        ("50% of 100 is 50", True, 0.95, "Percentage calculation"),
        ("25 percent of 80 equals 20", True, 0.95, "Percentage with words"),

        # Algebra (if sympy available)
        ("x^2 - 4 = (x-2)(x+2)", True, 0.9, "Algebraic identity"),
        ("a^2 - b^2 = (a+b)(a-b)", True, 0.9, "Difference of squares"),

        # Calculus (if sympy available)
        ("derivative of x^2 is 2*x", True, 0.9, "Simple derivative"),
        ("integral of 2*x is x^2", True, 0.85, "Simple integral"),

        # Non-mathematical
        ("The sky is blue", False, 0.0, "Not a math claim"),
    ]

    print("Testing Symbolic Math Verification")
    print("=" * 60)

    passed = 0
    failed = 0

    for claim, expected_verified, min_confidence, description in test_cases:
        result = await math_source.verify(claim)

        # Check if verification matches expected
        verified_match = result['verified'] == expected_verified

        # For verified claims, check confidence is high enough
        confidence_ok = True
        if expected_verified:
            confidence_ok = result['confidence'] >= min_confidence * 0.8  # Allow 20% tolerance

        status = "✓ PASS" if (verified_match and confidence_ok) else "✗ FAIL"

        if verified_match and confidence_ok:
            passed += 1
        else:
            failed += 1

        print(f"\n{status} | {description}")
        print(f"  Claim: {claim}")
        print(f"  Expected: verified={expected_verified}, confidence>={min_confidence}")
        print(f"  Got: verified={result['verified']}, confidence={result['confidence']:.2f}")
        print(f"  Evidence: {result['evidence']}")
        print(f"  Method: {result['method']}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Success rate: {passed / (passed + failed) * 100:.1f}%")

    return passed, failed


if __name__ == "__main__":
    passed, failed = asyncio.run(test_symbolic_math())
    exit(0 if failed == 0 else 1)
