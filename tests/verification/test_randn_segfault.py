#!/usr/bin/env python3
"""Test to verify the randn-specific segfault."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from nabla import add, mul, randn
from nabla.core.execution_context import global_execution_context


def test_randn_only_add():
    """Test only add operations with randn."""
    print("Testing randn with only add operations...")

    global_execution_context.clear()

    try:
        x = randn((2, 2), seed=42)
        y = randn((2, 2), seed=43)

        z = add(x, y)
        z.realize()

        print("✓ randn + add works")
        return True

    except Exception as e:
        print(f"✗ randn + add failed: {e}")
        return False


def test_randn_only_mul():
    """Test only mul operations with randn."""
    print("Testing randn with only mul operations...")

    global_execution_context.clear()

    try:
        x = randn((2, 2), seed=42)
        y = randn((2, 2), seed=43)

        z = mul(x, y)
        z.realize()

        print("✓ randn + mul works")
        return True

    except Exception as e:
        print(f"✗ randn + mul failed: {e}")
        return False


def test_randn_mixed_operations():
    """Test mixed operations with randn - this should segfault."""
    print("Testing randn with mixed operations (add then mul)...")

    global_execution_context.clear()

    try:
        x = randn((2, 2), seed=42)
        y = randn((2, 2), seed=43)

        # First add
        z1 = add(x, y)
        z1.realize()
        print("  Add operation completed")

        # Then mul - this should trigger segfault
        z2 = mul(x, y)
        z2.realize()
        print("  Mul operation completed")

        print("✓ randn + mixed operations works")
        return True

    except Exception as e:
        print(f"✗ randn + mixed operations failed: {e}")
        return False


if __name__ == "__main__":
    print("=== RANDN SEGFAULT ISOLATION TEST ===")

    # Test 1: randn with only add
    test_randn_only_add()
    print()

    # Test 2: randn with only mul
    test_randn_only_mul()
    print()

    # Test 3: randn with mixed operations (this should crash)
    test_randn_mixed_operations()
    print()

    print("=== Test Complete ===")
