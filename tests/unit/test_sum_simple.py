#!/usr/bin/env python3
"""Simple sum test to isolate the issue."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_simple_sum():
    """Test simple sum operation scenarios."""
    from nabla import graph_improved as nabla
    import numpy as np

    print("Testing simple sum operations...")

    # Test 1: Simple 2D array, sum over one axis
    print("\n1. Testing simple 2D sum over axis 1...")
    try:
        x = nabla.arange((2, 3))  # [[0,1,2], [3,4,5]]
        print(f"   Input: {x.get_numpy()}")
        result = nabla.sum(x, axes=1)
        result.realize()
        print(f"   Result: {result.get_numpy()}")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test 2: Simple 2D array, sum over axis 0
    print("\n2. Testing simple 2D sum over axis 0...")
    try:
        result = nabla.sum(x, axes=0)
        result.realize()
        print(f"   Result: {result.get_numpy()}")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = test_simple_sum()
    if success:
        print("\n🎉 All simple tests passed!")
    else:
        print("\n❌ Some tests failed!")
