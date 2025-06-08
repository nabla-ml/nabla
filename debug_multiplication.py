#!/usr/bin/env python3

import sys

sys.path.append("/Users/tillife/Documents/CodingProjects/nabla")

import nabla as nb
from nabla.core.trafos import vmap


def debug_multiplication():
    """Debug the multiplication issue."""

    x = nb.array([1.0, 2.0, 3.0])  # shape (3,)

    def test_mul_scalar(a):
        print(f"   Input a shape: {a.shape}")
        result = a * 2
        print(f"   a * 2 shape: {result.shape}")
        print(f"   a * 2 value: {result.to_numpy()}")
        return result

    def test_mul_array(a):
        print(f"   Input a shape: {a.shape}")
        two = nb.array(2.0)
        print(f"   two shape: {two.shape}")
        result = a * two
        print(f"   a * two shape: {result.shape}")
        print(f"   a * two value: {result.to_numpy()}")
        return result

    print("=== Debugging multiplication ===")

    print("\n1. Direct scalar multiplication:")
    direct1 = test_mul_scalar(x[0])  # Single element

    print("\n2. Direct array multiplication:")
    direct2 = test_mul_array(x[0])  # Single element

    print("\n3. Vmap scalar multiplication:")
    vmapped1 = vmap(test_mul_scalar, in_axes=0)
    result1 = vmapped1(x)
    print(f"Final result shape: {result1.shape}")

    print("\n4. Vmap array multiplication:")
    vmapped2 = vmap(test_mul_array, in_axes=0)
    result2 = vmapped2(x)
    print(f"Final result shape: {result2.shape}")


if __name__ == "__main__":
    debug_multiplication()
