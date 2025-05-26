#!/usr/bin/env python3
"""Incremental sum test to find the segfault cause."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_incremental():
    """Test sum operations incrementally."""
    from nabla import graph_improved as nabla
    import numpy as np

    print("Testing sum operations incrementally...")

    # Test 1: 3D array creation
    print("\n1. Creating 3D array...")
    try:
        x = nabla.randn((2, 3, 4), seed=42)
        print(f"   Created array with shape: {x.shape}")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 2: Sum over axis 0
    print("\n2. Testing sum over axis 0...")
    try:
        result = nabla.sum(x, axes=0)
        result.realize()
        print(f"   Result shape: {result.shape}")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 3: Sum over axis 1
    print("\n3. Testing sum over axis 1...")
    try:
        result = nabla.sum(x, axes=1)
        result.realize()
        print(f"   Result shape: {result.shape}")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 4: Sum over axis 2
    print("\n4. Testing sum over axis 2...")
    try:
        result = nabla.sum(x, axes=2)
        result.realize()
        print(f"   Result shape: {result.shape}")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = test_incremental()
    if success:
        print("\n🎉 All incremental tests passed!")
    else:
        print("\n❌ Some tests failed!")
