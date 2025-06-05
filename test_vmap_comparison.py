#!/usr/bin/env python3
"""
Test file comparing Nabla and JAX vmap behavior on stack and slice operations.

This test focuses on verifying that Nabla's vmap works correctly with:
- Stack operations along different axes (0, 1, -1)
- Slice operations with various patterns
- Numerical consistency with JAX using .to_numpy() for comparison
"""

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
    # Enable 64-bit precision for better comparison
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

import nabla as nb


def requires_jax(test_func):
    """Decorator to skip tests when JAX is not available."""
    return pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")(test_func)


@requires_jax
class TestVmapStackOperations:
    """Test vmap behavior with stack operations."""
    
    def test_vmap_stack_axis_0(self):
        """Test vmap with stack operation along axis 0."""
        def stack_fn(pair):
            # pair is a list [array1, array2], we need to unpack it
            return nb.stack(pair, axis=0)
        
        def jax_stack_fn(pair):
            # pair is a list [array1, array2], we need to unpack it
            return jnp.stack(pair, axis=0)
        
        # Create test data: batch of array pairs to stack
        batch_size = 5
        array_shape = (3, 4)
        
        # Each element in the batch is a pair of arrays to stack
        batch_arrays1 = [nb.randn(array_shape) for _ in range(batch_size)]
        batch_arrays2 = [nb.randn(array_shape) for _ in range(batch_size)]
        
        # Convert to JAX format
        jax_arrays1 = [jnp.array(arr.to_numpy()) for arr in batch_arrays1]
        jax_arrays2 = [jnp.array(arr.to_numpy()) for arr in batch_arrays2]
        
        # Apply vmap
        nabla_vmap = nb.vmap(stack_fn)
        jax_vmap = jax.vmap(jax_stack_fn)
        
        # Test with pairs of arrays
        nabla_pairs = [[batch_arrays1[i], batch_arrays2[i]] for i in range(batch_size)]
        jax_pairs = [[jax_arrays1[i], jax_arrays2[i]] for i in range(batch_size)]
        
        # Execute
        nabla_result = nabla_vmap(nabla_pairs)
        jax_result = jax_vmap(jax_pairs)
        
        # Compare results
        np.testing.assert_allclose(
            nabla_result.to_numpy(), 
            np.array(jax_result), 
            rtol=1e-6, atol=1e-7
        )
        
        # Check shapes
        expected_shape = (batch_size, 2) + array_shape  # (5, 2, 3, 4)
        assert nabla_result.shape == expected_shape
    
    def test_vmap_stack_axis_1(self):
        """Test vmap with stack operation along axis 1."""
        def stack_axis1(triplet):
            # triplet is a list [array1, array2, array3], we need to unpack it
            return nb.stack(triplet, axis=1)
        
        def jax_stack_axis1(triplet):
            # triplet is a list [array1, array2, array3], we need to unpack it
            return jnp.stack(triplet, axis=1)
        
        # Test data
        batch_size = 4
        array_shape = (3, 5)
        
        batch_arrays1 = [nb.randn(array_shape) for _ in range(batch_size)]
        batch_arrays2 = [nb.randn(array_shape) for _ in range(batch_size)]
        batch_arrays3 = [nb.randn(array_shape) for _ in range(batch_size)]
        
        jax_arrays1 = [jnp.array(arr.to_numpy()) for arr in batch_arrays1]
        jax_arrays2 = [jnp.array(arr.to_numpy()) for arr in batch_arrays2]
        jax_arrays3 = [jnp.array(arr.to_numpy()) for arr in batch_arrays3]
        
        # Apply vmap
        nabla_vmap = nb.vmap(stack_axis1)
        jax_vmap = jax.vmap(jax_stack_axis1)
        
        # Test with triplets
        nabla_triplets = [[batch_arrays1[i], batch_arrays2[i], batch_arrays3[i]] for i in range(batch_size)]
        jax_triplets = [[jax_arrays1[i], jax_arrays2[i], jax_arrays3[i]] for i in range(batch_size)]
        
        nabla_result = nabla_vmap(nabla_triplets)
        jax_result = jax_vmap(jax_triplets)
        
        # Compare
        np.testing.assert_allclose(
            nabla_result.to_numpy(), 
            np.array(jax_result), 
            rtol=1e-6, atol=1e-7
        )
        
        # Check shapes: (batch, shape[0], 3, shape[1])
        expected_shape = (batch_size, array_shape[0], 3, array_shape[1])  # (4, 3, 3, 5)
        assert nabla_result.shape == expected_shape
    
    def test_vmap_stack_negative_axis(self):
        """Test vmap with stack operation along negative axis."""
        def stack_neg_axis(pair):
            # pair is a list [array1, array2], we need to unpack it
            return nb.stack(pair, axis=-1)
        
        def jax_stack_neg_axis(pair):
            # pair is a list [array1, array2], we need to unpack it
            return jnp.stack(pair, axis=-1)
        
        # Test data
        batch_size = 3
        array_shape = (2, 4)
        
        batch_arrays1 = [nb.randn(array_shape) for _ in range(batch_size)]
        batch_arrays2 = [nb.randn(array_shape) for _ in range(batch_size)]
        
        jax_arrays1 = [jnp.array(arr.to_numpy()) for arr in batch_arrays1]
        jax_arrays2 = [jnp.array(arr.to_numpy()) for arr in batch_arrays2]
        
        # Apply vmap
        nabla_vmap = nb.vmap(stack_neg_axis)
        jax_vmap = jax.vmap(jax_stack_neg_axis)
        
        nabla_pairs = [[batch_arrays1[i], batch_arrays2[i]] for i in range(batch_size)]
        jax_pairs = [[jax_arrays1[i], jax_arrays2[i]] for i in range(batch_size)]
        
        nabla_result = nabla_vmap(nabla_pairs)
        jax_result = jax_vmap(jax_pairs)
        
        # Compare
        np.testing.assert_allclose(
            nabla_result.to_numpy(), 
            np.array(jax_result), 
            rtol=1e-6, atol=1e-7
        )
        
        # Check shapes: axis=-1 means last axis
        expected_shape = (batch_size,) + array_shape + (2,)  # (3, 2, 4, 2)
        assert nabla_result.shape == expected_shape


@requires_jax
class TestVmapSliceOperations:
    """Test vmap behavior with slice operations."""
    
    def test_vmap_basic_slice(self):
        """Test vmap with basic array slicing."""
        def slice_fn(arr):
            return arr[1:3, 2:5]
        
        def jax_slice_fn(arr):
            return arr[1:3, 2:5]
        
        # Test data: batch of arrays to slice
        batch_size = 4
        array_shape = (5, 7)
        
        batch_arrays = [nb.randn(array_shape) for _ in range(batch_size)]
        jax_arrays = [jnp.array(arr.to_numpy()) for arr in batch_arrays]
        
        # Apply vmap
        nabla_vmap = nb.vmap(slice_fn)
        jax_vmap = jax.vmap(jax_slice_fn)
        
        # Convert to batch format expected by vmap
        nabla_batch = nb.stack(batch_arrays, axis=0)  # Shape: (4, 5, 7)
        jax_batch = jnp.stack(jax_arrays, axis=0)
        
        nabla_result = nabla_vmap(nabla_batch)
        jax_result = jax_vmap(jax_batch)
        
        # Compare
        np.testing.assert_allclose(
            nabla_result.to_numpy(), 
            np.array(jax_result), 
            rtol=1e-6, atol=1e-7
        )
        
        # Check shapes: (batch_size, 2, 3) from [1:3, 2:5]
        expected_shape = (batch_size, 2, 3)
        assert nabla_result.shape == expected_shape
    
    def test_vmap_negative_slice(self):
        """Test vmap with negative indices in slicing."""
        def neg_slice_fn(arr):
            return arr[-3:-1, :-2]
        
        def jax_neg_slice_fn(arr):
            return arr[-3:-1, :-2]
        
        # Test data
        batch_size = 3
        array_shape = (6, 8)
        
        batch_arrays = [nb.randn(array_shape) for _ in range(batch_size)]
        jax_arrays = [jnp.array(arr.to_numpy()) for arr in batch_arrays]
        
        # Apply vmap
        nabla_vmap = nb.vmap(neg_slice_fn)
        jax_vmap = jax.vmap(jax_neg_slice_fn)
        
        nabla_batch = nb.stack(batch_arrays, axis=0)
        jax_batch = jnp.stack(jax_arrays, axis=0)
        
        nabla_result = nabla_vmap(nabla_batch)
        jax_result = jax_vmap(jax_batch)
        
        # Compare
        np.testing.assert_allclose(
            nabla_result.to_numpy(), 
            np.array(jax_result), 
            rtol=1e-6, atol=1e-7
        )
        
        # Check shapes: [-3:-1, :-2] on (6, 8) gives (2, 6)
        expected_shape = (batch_size, 2, 6)
        assert nabla_result.shape == expected_shape
    
    def test_vmap_mixed_slice_operations(self):
        """Test vmap with more complex slicing patterns."""
        def complex_slice_fn(arr):
            # Multiple slice operations
            slice1 = arr[::2, 1:]  # Every other row, skip first column
            slice2 = slice1[:, :3]  # Take first 3 columns
            return slice2
        
        def jax_complex_slice_fn(arr):
            slice1 = arr[::2, 1:]
            slice2 = slice1[:, :3]
            return slice2
        
        # Test data
        batch_size = 3
        array_shape = (8, 6)
        
        batch_arrays = [nb.randn(array_shape) for _ in range(batch_size)]
        jax_arrays = [jnp.array(arr.to_numpy()) for arr in batch_arrays]
        
        # Apply vmap
        nabla_vmap = nb.vmap(complex_slice_fn)
        jax_vmap = jax.vmap(jax_complex_slice_fn)
        
        nabla_batch = nb.stack(batch_arrays, axis=0)
        jax_batch = jnp.stack(jax_arrays, axis=0)
        
        nabla_result = nabla_vmap(nabla_batch)
        jax_result = jax_vmap(jax_batch)
        
        # Compare
        np.testing.assert_allclose(
            nabla_result.to_numpy(), 
            np.array(jax_result), 
            rtol=1e-6, atol=1e-7
        )
        
        # Check shapes: [::2, 1:] on (8, 6) gives (4, 5), then [:, :3] gives (4, 3)
        expected_shape = (batch_size, 4, 3)
        assert nabla_result.shape == expected_shape


@requires_jax
class TestVmapCombinedOperations:
    """Test vmap with combined stack and slice operations."""
    
    def test_vmap_stack_then_slice(self):
        """Test vmap with stack followed by slice operations."""
        def stack_slice_fn(pair):
            # pair is a list [array1, array2], we need to unpack it
            stacked = nb.stack(pair, axis=0)  # Shape: (2, 3, 4)
            sliced = stacked[1:, :2, 1:3]       # Shape: (1, 2, 2)
            return sliced
        
        def jax_stack_slice_fn(pair):
            # pair is a list [array1, array2], we need to unpack it
            stacked = jnp.stack(pair, axis=0)
            sliced = stacked[1:, :2, 1:3]
            return sliced
        
        # Test data
        batch_size = 3
        array_shape = (3, 4)
        
        batch_arrays1 = [nb.randn(array_shape) for _ in range(batch_size)]
        batch_arrays2 = [nb.randn(array_shape) for _ in range(batch_size)]
        
        jax_arrays1 = [jnp.array(arr.to_numpy()) for arr in batch_arrays1]
        jax_arrays2 = [jnp.array(arr.to_numpy()) for arr in batch_arrays2]
        
        # Apply vmap
        nabla_vmap = nb.vmap(stack_slice_fn)
        jax_vmap = jax.vmap(jax_stack_slice_fn)
        
        nabla_pairs = [[batch_arrays1[i], batch_arrays2[i]] for i in range(batch_size)]
        jax_pairs = [[jax_arrays1[i], jax_arrays2[i]] for i in range(batch_size)]
        
        nabla_result = nabla_vmap(nabla_pairs)
        jax_result = jax_vmap(jax_pairs)
        
        # Compare
        np.testing.assert_allclose(
            nabla_result.to_numpy(), 
            np.array(jax_result), 
            rtol=1e-6, atol=1e-7
        )
        
        # Check shapes: (batch_size, 1, 2, 2)
        expected_shape = (batch_size, 1, 2, 2)
        assert nabla_result.shape == expected_shape
    
    def test_vmap_slice_then_stack(self):
        """Test vmap with slice followed by stack operations."""
        def slice_stack_fn(pair):
            # pair is a list [array1, array2], we need to unpack it
            arr1, arr2 = pair
            slice1 = arr1[1:3, :2]  # Shape: (2, 2)
            slice2 = arr2[1:3, :2]  # Shape: (2, 2)
            stacked = nb.stack([slice1, slice2], axis=-1)  # Shape: (2, 2, 2)
            return stacked
        
        def jax_slice_stack_fn(pair):
            # pair is a list [array1, array2], we need to unpack it
            arr1, arr2 = pair
            slice1 = arr1[1:3, :2]
            slice2 = arr2[1:3, :2]
            stacked = jnp.stack([slice1, slice2], axis=-1)
            return stacked
        
        # Test data
        batch_size = 4
        array_shape = (5, 6)
        
        batch_arrays1 = [nb.randn(array_shape) for _ in range(batch_size)]
        batch_arrays2 = [nb.randn(array_shape) for _ in range(batch_size)]
        
        jax_arrays1 = [jnp.array(arr.to_numpy()) for arr in batch_arrays1]
        jax_arrays2 = [jnp.array(arr.to_numpy()) for arr in batch_arrays2]
        
        # Apply vmap
        nabla_vmap = nb.vmap(slice_stack_fn)
        jax_vmap = jax.vmap(jax_slice_stack_fn)
        
        nabla_pairs = [[batch_arrays1[i], batch_arrays2[i]] for i in range(batch_size)]
        jax_pairs = [[jax_arrays1[i], jax_arrays2[i]] for i in range(batch_size)]
        
        nabla_result = nabla_vmap(nabla_pairs)
        jax_result = jax_vmap(jax_pairs)
        
        # Compare
        np.testing.assert_allclose(
            nabla_result.to_numpy(), 
            np.array(jax_result), 
            rtol=1e-6, atol=1e-7
        )
        
        # Check shapes: (batch_size, 2, 2, 2)
        expected_shape = (batch_size, 2, 2, 2)
        assert nabla_result.shape == expected_shape


if __name__ == "__main__":
    # Simple test runner if executed directly
    print("Running Nabla vs JAX vmap comparison tests...")
    
    if not JAX_AVAILABLE:
        print("JAX not available - skipping tests")
        exit(0)
    
    # Run a simple test
    test_class = TestVmapStackOperations()
    # try:
    test_class.test_vmap_stack_axis_0()
    print("✓ Stack axis 0 test passed")
    
    test_class.test_vmap_stack_axis_1()
    print("✓ Stack axis 1 test passed")
    
    test_class.test_vmap_stack_negative_axis()
    print("✓ Stack negative axis test passed")
    # except Exception as e:
    #     print(f"✗ Stack tests failed: {e}")
    
    slice_tests = TestVmapSliceOperations()
    try:
        slice_tests.test_vmap_basic_slice()
        print("✓ Basic slice test passed")
        
        slice_tests.test_vmap_negative_slice()
        print("✓ Negative slice test passed")
        
        slice_tests.test_vmap_mixed_slice_operations()
        print("✓ Mixed slice operations test passed")
    except Exception as e:
        print(f"✗ Slice tests failed: {e}")
    
    combined_tests = TestVmapCombinedOperations()
    try:
        combined_tests.test_vmap_stack_then_slice()
        print("✓ Stack then slice test passed")
        
        combined_tests.test_vmap_slice_then_stack()
        print("✓ Slice then stack test passed")
    except Exception as e:
        print(f"✗ Combined tests failed: {e}")
    
    print("All tests completed!")
