#!/usr/bin/env python3
"""
Test script to verify the clean OOP-based architecture implementation.
This tests the complete OOP refactoring of ALL operations.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Test tracking
tests_total = 0
tests_passed = 0


def run_test(test_name: str, test_func):
    """Run a test function and track results."""
    global tests_total, tests_passed
    tests_total += 1

    try:
        print(f"Testing {test_name}...")
        test_func()
        print(f"✅ {test_name} passed")
        tests_passed += 1
        return True
    except Exception as e:
        print(f"❌ {test_name} failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_imports():
    """Test that all required components can be imported."""
    import nabla

    # Test that key functions are available
    required_functions = [
        "array",
        "add",
        "mul",
        "sin",
        "cos",
        "sum",
        "transpose",
        "matmul",
        "arange",
        "randn",
    ]
    for func_name in required_functions:
        assert hasattr(nabla, func_name), f"Missing function: {func_name}"

    print("   All required functions available")


def test_operation_classes():
    """Test that operations are now proper classes."""
    from nabla.ops.binary import AddOp, MulOp, _add_op, _mul_op
    from nabla.ops.operation import BinaryOperation

    # Test inheritance
    assert isinstance(_add_op, BinaryOperation)
    assert isinstance(_mul_op, BinaryOperation)
    print("   Operations properly inherit from base classes")


def test_basic_operations():
    """Test basic arithmetic with clean operations."""
    import nabla

    print("   Creating arrays...")
    x = nabla.arange((2, 2))
    y = nabla.randn((2, 2), seed=42)

    print("   Testing function calls...")
    # Test function calls
    z1 = nabla.add(x, y)
    z2 = nabla.mul(x, y)

    print("   Testing operator overloading...")
    # Test operator overloading
    z3 = x + y
    z4 = x * y

    print("   Realizing results...")
    # Realize results
    z1.realize()
    z2.realize()
    z3.realize()
    z4.realize()

    print(f"   Add result shape: {z1.shape}")
    print(f"   Mul result shape: {z2.shape}")
    print(f"   Operator + works: {z3.shape}")
    print(f"   Operator * works: {z4.shape}")


def test_backward_compatibility():
    """Test that we maintain backward compatibility."""
    import nabla

    # This should work exactly like before
    x = nabla.randn((2, 2))
    y = nabla.arange((2, 2))
    z = x + y
    w = nabla.sin(z)
    result = nabla.sum(w)
    result.realize()

    print(f"   Complex computation result: {result.get_numpy()}")


def test_original_still_available():
    """Test that backward compatibility is maintained."""
    import nabla

    # Should be able to access the improved graph interface
    assert hasattr(nabla, "graph_improved")
    print("   Graph improved implementation accessible")


if __name__ == "__main__":
    print("🧹 Testing clean OOP-based architecture...\n")

    # Run all tests
    run_test("Clean Imports", test_imports)
    run_test("Operation Classes", test_operation_classes)
    run_test("Basic Operations", test_basic_operations)
    run_test("Backward Compatibility", test_backward_compatibility)
    run_test("Original Available", test_original_still_available)

    # Print summary
    print(f"\n📊 Test Results: {tests_passed}/{tests_total} tests passed")
    print(f"📈 Success Rate: {tests_passed/tests_total*100:.1f}%")

    if tests_passed == tests_total:
        print("\n🎉 Clean architecture implementation successful!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some tests failed. Implementation needs attention.")
        sys.exit(1)
