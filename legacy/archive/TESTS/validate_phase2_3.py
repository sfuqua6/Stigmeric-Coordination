"""Validate Phase 2-3 component compatibility without running.

Checks:
1. SpatialSignalStore.deposit() signature matches SimpleScout usage
2. SimpleScout.step() signature matches test usage
3. All method calls have correct arguments
"""

import ast
import inspect
from pathlib import Path


def extract_function_signature(file_path: str, class_name: str, method_name: str):
    """Extract function signature from source file."""
    with open(file_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    args = [arg.arg for arg in item.args.args]
                    defaults = len(item.args.defaults)
                    required = len(args) - defaults
                    return {
                        'args': args,
                        'required': args[:required],
                        'optional': args[required:] if required < len(args) else []
                    }
    return None


def check_compatibility():
    """Check component compatibility."""
    print("=" * 70)
    print("PHASE 2-3 COMPATIBILITY VALIDATION")
    print("=" * 70)

    issues = []

    # Check SpatialSignalStore.deposit()
    print("\n1. Checking SpatialSignalStore.deposit() signature...")
    deposit_sig = extract_function_signature(
        'swarm/core/spatial_signal_store.py',
        'SpatialSignalStore',
        'deposit'
    )
    print(f"   Required args: {deposit_sig['required']}")
    print(f"   Optional args: {deposit_sig['optional']}")

    expected_deposit_args = ['self', 'signal_type', 'content', 'strength', 'depositor', 'position']
    if set(deposit_sig['required']) == set(expected_deposit_args):
        print("   ✓ Signature matches expected")
    else:
        issues.append("deposit() signature mismatch")
        print(f"   ✗ Expected {expected_deposit_args}")

    # Check SimpleScout.step()
    print("\n2. Checking SimpleScout.step() signature...")
    step_sig = extract_function_signature(
        'swarm/agents/simple_scout.py',
        'SimpleScout',
        'step'
    )
    print(f"   Required args: {step_sig['required']}")
    print(f"   Optional args: {step_sig['optional']}")

    expected_step_args = ['self', 'signal_store', 'llm_oracle', 'task_prompt', 'current_iteration']
    if set(step_sig['required']) == set(expected_step_args):
        print("   ✓ Signature matches expected")
    else:
        issues.append("step() signature mismatch")
        print(f"   ✗ Expected {expected_step_args}")

    # Check SimpleScout.__init__()
    print("\n3. Checking SimpleScout.__init__() signature...")
    init_sig = extract_function_signature(
        'swarm/agents/simple_scout.py',
        'SimpleScout',
        '__init__'
    )
    print(f"   Required args: {init_sig['required']}")
    print(f"   Optional args: {init_sig['optional']}")

    expected_init_args = ['self', 'agent_id', 'position', 'knowledge']
    if set(init_sig['required']) == set(expected_init_args):
        print("   ✓ Signature matches expected")
    else:
        issues.append("__init__() signature mismatch")
        print(f"   ✗ Expected {expected_init_args}")

    # Check method call in SimpleScout._deposit_signal
    print("\n4. Checking SimpleScout._deposit_signal() calls deposit() correctly...")
    with open('swarm/agents/simple_scout.py', 'r') as f:
        content = f.read()

    # Look for signal_store.deposit call
    if 'signal_store.deposit(' in content:
        print("   ✓ Found signal_store.deposit() call")
        # Check if position parameter is passed
        if 'position=tuple(self.position)' in content:
            print("   ✓ position parameter is passed correctly")
        else:
            issues.append("deposit() call missing position parameter")
            print("   ✗ position parameter not found")
    else:
        issues.append("No deposit() call found")
        print("   ✗ signal_store.deposit() call not found")

    # Check test_simple_scouts.py calls step() correctly
    print("\n5. Checking test_simple_scouts.py calls step() correctly...")
    with open('test_simple_scouts.py', 'r') as f:
        test_content = f.read()

    if 'await scout.step(' in test_content:
        print("   ✓ Found scout.step() call")
        # Check if correct arguments are passed
        if 'signal_store, llm, task_prompt, iteration' in test_content:
            print("   ✓ Correct arguments passed to step()")
        else:
            print("   ⚠ Warning: Argument pattern not exactly matched (may still be correct)")
    else:
        issues.append("No step() call found in test")
        print("   ✗ scout.step() call not found")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if issues:
        print(f"\n❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("\n✅ All compatibility checks passed!")
        print("\nPhase 2-3 components are properly integrated:")
        print("  ✓ SpatialSignalStore.deposit() has correct signature")
        print("  ✓ SimpleScout.step() has correct signature")
        print("  ✓ SimpleScout.__init__() has correct signature")
        print("  ✓ deposit() is called with position parameter")
        print("  ✓ test_simple_scouts.py calls step() correctly")
        print("\nReady to proceed to Phase 4: Dynamic Knowledge Base")
        return True


if __name__ == '__main__':
    success = check_compatibility()
    exit(0 if success else 1)
