#!/usr/bin/env python3
"""Standalone test for sum operation to verify the fix."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_sum_operations():
    """Test various sum operation scenarios."""
    from nabla import graph_improved as nabla
    import numpy as np

    print("Testing sum operations...")

    # Test 1: Sum all axes (axes=None)
    print("\n1. Testing sum with axes=None...")
    x = nabla.randn((2, 3, 4), seed=42)
    result = nabla.sum(x, axes=None)
    result.realize()
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {result.shape}")
    print(f"   Output value: {result.get_numpy()}")

    # Test 2: Sum specific axis
    print("\n2. Testing sum with specific axis...")
    result_axis1 = nabla.sum(x, axes=1)
    result_axis1.realize()
    print(f"   Sum over axis 1 shape: {result_axis1.shape}")

    # Test 3: Sum multiple axes
    print("\n3. Testing sum with multiple axes...")
    result_multi = nabla.sum(x, axes=[0, 2])
    result_multi.realize()
    print(f"   Sum over axes [0,2] shape: {result_multi.shape}")

    # Test 4: Sum with keepdims
    print("\n4. Testing sum with keepdims=True...")
    result_keepdims = nabla.sum(x, axes=None, keep_dims=True)
    result_keepdims.realize()
    print(f"   Sum with keepdims shape: {result_keepdims.shape}")

    # Test 5: Compare with numpy
    print("\n5. Comparing with numpy...")
    x_np = x.get_numpy()
    np_sum_all = np.sum(x_np)
    nabla_sum_all = result.get_numpy()
    print(f"   NumPy sum all: {np_sum_all}")
    print(f"   Nabla sum all: {nabla_sum_all}")
    print(f"   Close match: {np.allclose(np_sum_all, nabla_sum_all)}")

    print("\n✅ All sum operations completed successfully!")


if __name__ == "__main__":
    test_sum_operations()
