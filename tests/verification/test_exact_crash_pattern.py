#!/usr/bin/env python3
"""Reproduce the exact pattern that causes the crash."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_exact_crash_pattern():
    """Reproduce the exact crash pattern from the failing test."""
    print("Testing exact crash pattern...")

    import nabla

    print("   Creating arrays...")
    x = nabla.arange((2, 2))
    y = nabla.randn((2, 2), seed=42)  # This is the key difference

    print("   Testing function calls...")
    # Test function calls - exact same pattern
    z1 = nabla.add(x, y)
    z2 = nabla.mul(x, y)

    print("   Testing operator overloading...")
    # Test operator overloading - exact same pattern
    z3 = x + y
    z4 = x * y

    print("   Realizing results...")
    # Realize results - this is where it should crash
    print("     z1.realize()...")
    z1.realize()
    print("     z2.realize()...")
    z2.realize()
    print("     z3.realize()...")
    z3.realize()
    print("     z4.realize()...")
    z4.realize()

    print(f"   Add result shape: {z1.shape}")
    print(f"   Mul result shape: {z2.shape}")
    print(f"   Operator + works: {z3.shape}")
    print(f"   Operator * works: {z4.shape}")

    print("✅ All operations completed successfully!")


def test_cache_pollution():
    """Test if the issue is related to cache pollution."""
    print("\nTesting cache pollution theory...")

    import nabla
    from nabla.core.execution_context import global_execution_context

    # Clear cache between operations
    global_execution_context.clear()
    print(f"   Cache cleared. Size: {global_execution_context.size()}")

    print("   Creating arrays...")
    x = nabla.arange((2, 2))
    y = nabla.randn((2, 2), seed=42)

    # Test each operation individually with cache clearing
    print("   Testing z1 = add(x, y)...")
    z1 = nabla.add(x, y)
    z1.realize()
    print(f"   Cache size after z1: {global_execution_context.size()}")

    print("   Testing z2 = mul(x, y)...")
    z2 = nabla.mul(x, y)
    z2.realize()
    print(f"   Cache size after z2: {global_execution_context.size()}")

    print("   Testing z3 = x + y...")
    z3 = x + y
    z3.realize()
    print(f"   Cache size after z3: {global_execution_context.size()}")

    print("   Testing z4 = x * y...")
    z4 = x * y
    z4.realize()
    print(f"   Cache size after z4: {global_execution_context.size()}")

    print("✅ Cache pollution test completed!")


if __name__ == "__main__":
    print("=== EXACT CRASH PATTERN REPRODUCTION ===")

    # Test 1: Cache pollution theory
    try:
        test_cache_pollution()
    except Exception as e:
        print(f"❌ Cache pollution test failed: {e}")
        import traceback

        traceback.print_exc()

    # Test 2: Exact crash pattern
    try:
        test_exact_crash_pattern()
    except Exception as e:
        print(f"❌ Exact crash pattern failed: {e}")
        import traceback

        traceback.print_exc()
