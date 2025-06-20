#!/usr/bin/env python3
"""
Comprehensive test suite for Array indexing operations with transformations.
Tests edge cases and missing scenarios.
"""

import numpy as np

from nabla import Array, jit, vjp, vmap


def test_functional_array_updates():
    """Test functional array updates using .set() method."""
    print("=== Testing .set() Method (Functional Array Updates) ===")

    # Test basic .set() functionality
    arr = Array.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    print(f"Original array:\n{arr.to_numpy()}")

    # Test single element assignment
    try:
        new_arr = arr.set((1, 2), 99.0)
        print("✓ arr.set((1,2), 99.0) works:")
        print(f"  Original: {arr.to_numpy()}")
        print(f"  New:      {new_arr.to_numpy()}")
    except Exception as e:
        print(f"✗ Single element set failed: {e}")

    # Test slice assignment
    try:
        new_arr2 = arr.set((0, slice(None)), 100.0)
        print("✓ arr.set((0, :), 100.0) works:")
        print(f"  Result: {new_arr2.to_numpy()}")
    except Exception as e:
        print(f"✗ Slice set failed: {e}")

    # Test with different shapes/broadcasting
    try:
        arr2 = Array.from_numpy(
            np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
        )
        new_arr3 = arr2.set(
            (slice(1, 2), slice(1, 3)),
            Array.from_numpy(np.array([[777, 888]], dtype=np.float32)),
        )
        print("✓ Slice assignment with Array works:")
        print(f"  Result: {new_arr3.to_numpy()}")
    except Exception as e:
        print(f"✗ Array slice assignment failed: {e}")

    # Test negative indexing in set
    try:
        arr3 = Array.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float32))
        new_arr4 = arr3.set(-1, 999.0)
        new_arr5 = new_arr4.set(slice(-2, None), 555.0)
        print(f"✓ Negative indexing set works: {new_arr5.to_numpy()}")
    except Exception as e:
        print(f"✗ Negative indexing set failed: {e}")

    # Test ellipsis in set
    try:
        arr4 = Array.from_numpy(np.random.rand(2, 3, 4).astype(np.float32))
        new_arr6 = arr4.set((1, ...), 123.0)  # This should work: arr4[1, :, :] = 123.0
        print(f"✓ Ellipsis set works: new_arr.shape = {new_arr6.shape}")
        print(f"  First slice of result: {new_arr6[1, 0, :].to_numpy()}")
    except Exception as e:
        print(f"✗ Ellipsis set failed: {e}")


def test_boolean_indexing():
    """Test boolean/mask indexing if supported."""
    print("\n=== Testing Boolean Indexing ===")

    arr = Array.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float32))

    try:
        # Create boolean mask
        mask_np = np.array([True, False, True, False, True])
        mask = Array.from_numpy(mask_np)

        # Try boolean indexing
        result = arr[mask]
        print(f"✓ Boolean indexing works: {result.to_numpy()}")
    except Exception as e:
        print(f"✗ Boolean indexing not supported: {e}")


def test_integer_array_indexing():
    """Test fancy indexing with integer arrays."""
    print("\n=== Testing Integer Array Indexing ===")

    arr = Array.from_numpy(np.array([10, 20, 30, 40, 50], dtype=np.float32))

    try:
        # Create index array
        indices = Array.from_numpy(np.array([0, 2, 4], dtype=np.int32))

        # Try fancy indexing
        result = arr[indices]
        print(f"✓ Integer array indexing works: {result.to_numpy()}")
    except Exception as e:
        print(f"✗ Integer array indexing not supported: {e}")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")

    arr = Array.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))

    # Test out of bounds access
    try:
        result = arr[10, 10]
        print(f"✗ Out of bounds access should fail but got: {result}")
    except Exception as e:
        print(f"✓ Out of bounds properly caught: {type(e).__name__}")

    # Test empty slices
    try:
        result = arr[0:0, :]
        print(f"✓ Empty slice works: shape {result.shape}")
    except Exception as e:
        print(f"✗ Empty slice failed: {e}")

    # Test step of 0 (should fail)
    try:
        result = arr[::0]
        print(f"✗ Step=0 should fail but got: {result}")
    except Exception as e:
        print(f"✓ Step=0 properly caught: {type(e).__name__}")


def test_set_with_transformations():
    """Test .set() method behavior with transformations."""
    print("\n=== Testing .set() Method with Transformations ===")

    # Note: .set() is a functional operation that returns a new array,
    # making it perfect for use with transformations.

    def modify_and_process(x):
        """Function that uses .set() and returns processed result."""
        # Use .set() to create a modified version
        modified = x.set(2, 99.0)
        return modified.sum()

    # Test that .set() works with transformations
    try:
        arr = Array.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float32))

        # Test the function directly
        result = modify_and_process(arr)
        print(f"✓ Function with .set(): {result.to_numpy()}")

        # Test VJP with .set() inside function
        value, pullback = vjp(modify_and_process, arr)
        cotangent = Array.from_numpy(np.array(1.0, dtype=np.float32))
        grad = pullback(cotangent)
        print(f"✓ VJP with .set(): grad = {grad.to_numpy()}")

        # Test JIT with .set() inside function
        jit_func = jit(modify_and_process)
        jit_result = jit_func(arr)
        print(f"✓ JIT with .set(): {jit_result.to_numpy()}")

    except Exception as e:
        print(f"✗ .set() with transformations failed: {e}")

    # Test that .set() returns a new array (functional behavior)
    try:
        arr = Array.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        new_arr = arr.set(1, 99.0)

        print("✓ .set() is functional:")
        print(f"  Original: {arr.to_numpy()}")
        print(f"  New:      {new_arr.to_numpy()}")
        print(f"  Arrays are different objects: {arr is not new_arr}")

    except Exception as e:
        print(f"✗ .set() functional test failed: {e}")


def test_mixed_transformations():
    """Test complex combinations of transformations."""
    print("\n=== Testing Mixed Transformations ===")

    def complex_indexing_func(x):
        # Complex indexing operations
        a = x[1:3, :]  # Slice
        b = a[:, ::2]  # Step slice
        c = b[-1, :]  # Negative index
        return c.sum()  # Reduce to scalar

    x = Array.from_numpy(np.random.rand(5, 8).astype(np.float32))

    # Test basic function
    try:
        result = complex_indexing_func(x)
        print(f"✓ Complex indexing function works: {result.to_numpy()}")
    except Exception as e:
        print(f"✗ Complex indexing function failed: {e}")
        return

    # Test vjp
    try:
        value, pullback = vjp(complex_indexing_func, x)
        cotangent = Array.from_numpy(np.array(1.0, dtype=np.float32))
        grad = pullback(cotangent)
        print(f"✓ VJP of complex indexing works: grad shape {grad.shape}")
    except Exception as e:
        print(f"✗ VJP of complex indexing failed: {e}")

    # Test jit
    try:
        jit_func = jit(complex_indexing_func)
        jit_result = jit_func(x)
        print(f"✓ JIT of complex indexing works: {jit_result.to_numpy()}")
    except Exception as e:
        print(f"✗ JIT of complex indexing failed: {e}")

    # Test jit(vjp(...))
    try:
        jit_vjp_func = jit(lambda x: vjp(complex_indexing_func, x)[0])
        jit_vjp_result = jit_vjp_func(x)
        print(f"✓ JIT(VJP) of complex indexing works: {jit_vjp_result.to_numpy()}")
    except Exception as e:
        print(f"✗ JIT(VJP) of complex indexing failed: {e}")

    # Test vmap
    try:
        batched_x = Array.from_numpy(np.random.rand(3, 5, 8).astype(np.float32))
        vmap_func = vmap(complex_indexing_func)
        vmap_result = vmap_func(batched_x)
        print(f"✓ VMAP of complex indexing works: shape {vmap_result.shape}")
    except Exception as e:
        print(f"✗ VMAP of complex indexing failed: {e}")


def test_dtype_edge_cases():
    """Test indexing with different dtypes."""
    print("\n=== Testing Different Dtypes ===")

    dtypes_to_test = [
        (np.float32, "float32"),
        (np.float64, "float64"),
        (np.int32, "int32"),
    ]

    for dtype, name in dtypes_to_test:
        try:
            arr = Array.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype))

            # Test basic indexing
            result = arr[1, 2]
            print(f"✓ {name} indexing works: {result.to_numpy()}")

            # Test setitem
            new_arr = arr.set((0, 0), dtype(99))
            print(f"✓ {name} set works: {new_arr[0, 0].to_numpy()}")

        except Exception as e:
            print(f"✗ {name} failed: {e}")


def test_newaxis_indexing():
    """Test indexing with newaxis/None."""
    print("\n=== Testing Newaxis Indexing ===")

    arr = Array.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float32))

    try:
        # Test adding dimensions with None/newaxis
        result = arr[:, None]
        print(f"✓ arr[:, None] works: shape {result.shape}")

        result2 = arr[None, :]
        print(f"✓ arr[None, :] works: shape {result2.shape}")

    except Exception as e:
        print(f"✗ Newaxis indexing not supported: {e}")


def run_comprehensive_tests():
    """Run all comprehensive indexing tests."""
    print("🧪 Starting Comprehensive Array Indexing Tests...")

    test_functional_array_updates()
    test_set_with_transformations()
    test_boolean_indexing()
    test_integer_array_indexing()
    test_edge_cases()
    test_mixed_transformations()
    test_dtype_edge_cases()
    test_newaxis_indexing()

    print("\n✅ Comprehensive indexing tests completed!")


if __name__ == "__main__":
    run_comprehensive_tests()
