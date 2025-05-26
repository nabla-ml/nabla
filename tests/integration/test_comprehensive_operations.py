#!/usr/bin/env python3
"""Comprehensive test suite for the refactored nabla code."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    try:
        from nabla import graph_improved as nabla

        print("✓ Main import successful")

        # Test that all expected functions are available
        expected_functions = [
            "randn",
            "arange",
            "sin",
            "cos",
            "transpose",
            "reshape",
            "sum",
            "matmul",
            "realize_",
        ]

        for func_name in expected_functions:
            assert hasattr(nabla, func_name), f"Missing function: {func_name}"
            print(f"✓ Function {func_name} available")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_array_creation():
    """Test array creation operations."""
    print("\nTesting array creation...")
    try:
        from nabla import graph_improved as nabla

        # Test arange
        a = nabla.arange((2, 3))
        assert a.shape == (2, 3), f"Expected shape (2, 3), got {a.shape}"
        print("✓ arange works")

        # Test randn with different parameters
        b = nabla.randn((3, 2), mean=1.0, std=2.0, seed=42)
        assert b.shape == (3, 2), f"Expected shape (3, 2), got {b.shape}"
        print("✓ randn with custom parameters works")

        # Test randn with default parameters
        c = nabla.randn((2, 2))
        assert c.shape == (2, 2), f"Expected shape (2, 2), got {c.shape}"
        print("✓ randn with default parameters works")

        return True
    except Exception as e:
        print(f"❌ Array creation test failed: {e}")
        return False


def test_unary_operations():
    """Test unary operations."""
    print("\nTesting unary operations...")
    try:
        from nabla import graph_improved as nabla

        x = nabla.randn((2, 3), seed=123)

        # Test trigonometric functions
        sin_x = nabla.sin(x)
        cos_x = nabla.cos(x)
        assert (
            sin_x.shape == x.shape
        ), f"sin shape mismatch: expected {x.shape}, got {sin_x.shape}"
        assert (
            cos_x.shape == x.shape
        ), f"cos shape mismatch: expected {x.shape}, got {cos_x.shape}"
        print("✓ Trigonometric functions work")

        # Test negation
        neg_x = -x  # This should use the __neg__ method which calls negate
        assert (
            neg_x.shape == x.shape
        ), f"negation shape mismatch: expected {x.shape}, got {neg_x.shape}"
        print("✓ Negation works")

        return True
    except Exception as e:
        print(f"❌ Unary operations test failed: {e}")
        return False


def test_binary_operations():
    """Test binary operations."""
    print("\nTesting binary operations...")
    try:
        from nabla import graph_improved as nabla

        a = nabla.randn((2, 3), seed=456)
        b = nabla.randn((2, 3), seed=789)

        # Test addition
        c = a + b
        assert (
            c.shape == a.shape
        ), f"addition shape mismatch: expected {a.shape}, got {c.shape}"
        print("✓ Addition works")

        # Test multiplication
        d = a * b
        assert (
            d.shape == a.shape
        ), f"multiplication shape mismatch: expected {a.shape}, got {d.shape}"
        print("✓ Multiplication works")

        return True
    except Exception as e:
        print(f"❌ Binary operations test failed: {e}")
        return False


def test_view_operations():
    """Test view/shape operations."""
    print("\nTesting view operations...")
    try:
        from nabla import graph_improved as nabla

        x = nabla.randn((2, 3, 4), seed=111)

        # Test transpose (axis swapping)
        x_t = nabla.transpose(x, 0, 2)  # Swap axis 0 and 2
        assert x_t.shape == (
            4,
            3,
            2,
        ), f"transpose shape mismatch: expected (4, 3, 2), got {x_t.shape}"
        print("✓ Transpose works")

        # Test reshape
        x_reshaped = nabla.reshape(x, (6, 4))
        assert x_reshaped.shape == (
            6,
            4,
        ), f"reshape shape mismatch: expected (6, 4), got {x_reshaped.shape}"
        print("✓ Reshape works")

        return True
    except Exception as e:
        print(f"❌ View operations test failed: {e}")
        return False


def test_reduction_operations():
    """Test reduction operations."""
    print("\nTesting reduction operations...")
    try:
        from nabla import graph_improved as nabla

        x = nabla.randn((3, 4), seed=222)

        # Test sum without axis
        total = nabla.sum(x)
        assert (
            total.shape == ()
        ), f"sum without axis should be scalar, got shape {total.shape}"
        print("✓ Sum without axis works")

        # Test sum with axis
        sum_axis0 = nabla.sum(x, axis=0)
        assert sum_axis0.shape == (
            4,
        ), f"sum along axis 0 shape mismatch: expected (4,), got {sum_axis0.shape}"
        print("✓ Sum with axis works")

        return True
    except Exception as e:
        print(f"❌ Reduction operations test failed: {e}")
        return False


def test_linear_algebra():
    """Test linear algebra operations."""
    print("\nTesting linear algebra operations...")
    try:
        from nabla import graph_improved as nabla

        a = nabla.randn((3, 4), seed=333)
        b = nabla.randn((4, 5), seed=444)

        # Test matrix multiplication
        c = nabla.matmul(a, b)
        assert c.shape == (
            3,
            5,
        ), f"matmul shape mismatch: expected (3, 5), got {c.shape}"
        print("✓ Matrix multiplication works")

        # Test batched matrix multiplication
        batch_a = nabla.randn((2, 3, 4), seed=555)
        batch_b = nabla.randn((2, 4, 5), seed=666)
        batch_c = nabla.matmul(batch_a, batch_b)
        assert batch_c.shape == (
            2,
            3,
            5,
        ), f"batched matmul shape mismatch: expected (2, 3, 5), got {batch_c.shape}"
        print("✓ Batched matrix multiplication works")

        return True
    except Exception as e:
        print(f"❌ Linear algebra test failed: {e}")
        return False


def test_graph_execution():
    """Test graph tracing and execution."""
    print("\nTesting graph execution...")
    try:
        from nabla import graph_improved as nabla

        # Create a computation graph
        x = nabla.randn((2, 3), seed=777)
        y = nabla.randn((2, 3), seed=888)

        # Build a computation: (x + y) * sin(x)
        z = (x + y) * nabla.sin(x)

        # Realize the computation
        z.realize()
        print("✓ Complex graph execution works")

        # Test multiple outputs
        a = nabla.randn((3, 3), seed=999)
        b = nabla.transpose(a, 1, 0)
        c = nabla.sum(a, axis=0)

        # Realize multiple arrays
        nabla.realize_([b, c])
        print("✓ Multiple output realization works")

        return True
    except Exception as e:
        print(f"❌ Graph execution test failed: {e}")
        return False


def test_backward_compatibility():
    """Test that the refactored code maintains backward compatibility."""
    print("\nTesting backward compatibility...")
    try:
        from nabla import graph_improved as nabla

        # Test that we can still access all the original functions
        # in the same way as before
        x = nabla.randn((2, 2))
        y = nabla.arange((2, 2))

        # Test operations
        z = x + y
        w = nabla.sin(z)
        result = nabla.sum(w)

        # Realize
        result.realize()
        print("✓ Backward compatibility maintained")

        return True
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Running comprehensive test suite for refactored nabla...\n")

    tests = [
        test_imports,
        test_array_creation,
        test_unary_operations,
        test_binary_operations,
        test_view_operations,
        test_reduction_operations,
        test_linear_algebra,
        test_graph_execution,
        test_backward_compatibility,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1

    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed / (passed + failed) * 100:.1f}%")

    if failed == 0:
        print("\n🎉 All tests passed! The refactoring is successful.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
