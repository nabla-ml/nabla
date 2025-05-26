"""
Test script to verify the refactored graph_improved.py works correctly.
"""

import sys

sys.path.append("/Users/tillife/Documents/CodingProjects/nabla/src")


def test_basic_operations():
    """Test basic operations work in the refactored version."""
    try:
        from nabla.graph_improved import Array, add, mul, sin, cos, randn, matmul
        from nabla.utils.types import DType, CPU

        print("✓ Imports successful")

        # Test array creation
        x = randn((3, 4), mean=0.0, std=1.0)
        print(f"✓ Created random array with shape {x.shape}")

        # Test unary operations
        y = sin(x)
        z = cos(x)
        print(f"✓ Unary ops: sin and cos computed")

        # Test binary operations
        w = add(y, z)
        print(f"✓ Binary op: addition computed")

        # Test matrix multiplication
        a = randn((3, 4))
        b = randn((4, 5))
        c = matmul(a, b)
        print(f"✓ Matrix multiplication: {a.shape} @ {b.shape} = {c.shape}")

        # Test realization
        c.realize()
        print(f"✓ Graph execution successful")

        print("\n🎉 All tests passed! Refactored code works correctly.")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_basic_operations()
