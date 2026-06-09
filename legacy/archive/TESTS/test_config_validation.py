#!/usr/bin/env python3
"""Test configuration validation."""

import sys
import os

# Test 1: Valid config (both False)
print("Test 1: Both USE_SIMPLE_SCOUTS and USE_SPATIAL_STORE = False")
os.environ['TEST_CONFIG'] = '1'
try:
    import swarm.core.config as config
    print("  ✓ Passed (no error)")
except ValueError as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Valid config (both True)
print("\nTest 2: Both USE_SIMPLE_SCOUTS and USE_SPATIAL_STORE = True")
# Modify config temporarily
config.USE_SIMPLE_SCOUTS = True
config.USE_SPATIAL_STORE = True
try:
    config.validate_config()
    print("  ✓ Passed (no error)")
except ValueError as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Invalid config (USE_SIMPLE_SCOUTS=True, USE_SPATIAL_STORE=False)
print("\nTest 3: USE_SIMPLE_SCOUTS=True but USE_SPATIAL_STORE=False (should fail)")
config.USE_SIMPLE_SCOUTS = True
config.USE_SPATIAL_STORE = False
try:
    config.validate_config()
    print("  ✗ Failed (should have raised ValueError)")
    sys.exit(1)
except ValueError as e:
    print("  ✓ Passed (correctly caught invalid config)")
    print(f"  Error message: {str(e)[:100]}...")

# Test 4: Valid config (USE_SPATIAL_STORE=True, USE_SIMPLE_SCOUTS=False)
print("\nTest 4: USE_SPATIAL_STORE=True but USE_SIMPLE_SCOUTS=False (valid)")
config.USE_SIMPLE_SCOUTS = False
config.USE_SPATIAL_STORE = True
try:
    config.validate_config()
    print("  ✓ Passed (this combination is valid)")
except ValueError as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("\n✓ All configuration validation tests passed!")
