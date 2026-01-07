# ===----------------------------------------------------------------------=== #
# Nabla 2025
# ===----------------------------------------------------------------------=== #
"""Advanced vmap tests - in_axes variations and edge cases.

This file tests advanced vmap functionality:
1. in_axes with non-zero axis batching (batch over middle/last dimension)
2. out_axes specification
3. Nested pytree handling with in_axes
4. Edge cases and error conditions

These tests complement test_vmap_ops.py which focuses on basic vmap usage.
"""

import pytest
import numpy as np

import nabla
from nabla import vmap
from nabla import (
    add, sub, mul, div,
    relu, sigmoid, tanh,
    reduce_sum, mean,
    matmul,
)

from .conftest import (
    make_array, make_positive_array, tensor_from_numpy, to_numpy,
    assert_allclose, assert_shape, assert_batch_dims,
)


# =============================================================================
# in_axes with Non-Zero Batch Axis
# =============================================================================

class TestVmapInAxesNonZero:
    """Test vmap with in_axes specifying non-zero batch axes."""
    
    def test_vmap_in_axes_1(self):
        """vmap with in_axes=1 - batch over second dimension."""
        # Shape: (features=8, batch=4) - batch is axis 1
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Each slice is (8,) - we're batching over axis 1
        result = vmap(relu, in_axes=1)(x)
        expected = np.maximum(np_x, 0).T  # Result should be (4, 8)
        
        assert_shape(result, (4, 8))
        assert_allclose(result, expected)
    
    def test_vmap_in_axes_negative(self):
        """vmap with in_axes=-1 - batch over last dimension."""
        # Shape: (features=8, batch=4) - batch is last axis
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = vmap(relu, in_axes=-1)(x)
        expected = np.maximum(np_x, 0).T  # Same as in_axes=1 for 2D
        
        assert_shape(result, (4, 8))
        assert_allclose(result, expected)
    
    def test_vmap_in_axes_middle_axis(self):
        """vmap with batch over middle axis of 3D tensor."""
        # Shape: (seq=8, batch=4, features=16) - batch is axis 1
        np_x = make_array(8, 4, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Each slice is (8, 16) - we're batching over axis 1
        result = vmap(relu, in_axes=1)(x)
        expected = np.maximum(np_x.transpose(1, 0, 2), 0)  # (4, 8, 16)
        
        assert_shape(result, (4, 8, 16))
        assert_allclose(result, expected)
    
    def test_vmap_binary_different_in_axes(self):
        """vmap(add) with different in_axes for each input."""
        # x: (batch=4, 8), y: (8, batch=4)
        np_x = make_array(4, 8, seed=42)
        np_y = make_array(8, 4, seed=43)
        
        x = tensor_from_numpy(np_x)
        y = tensor_from_numpy(np_y)
        
        # x batches on axis 0, y batches on axis 1
        result = vmap(add, in_axes=(0, 1))(x, y)
        
        # Expected: each batch element is x[i] + y[:, i]
        expected = np_x + np_y.T
        
        assert_shape(result, (4, 8))
        assert_allclose(result, expected)
    
    def test_vmap_in_axes_none_broadcast(self):
        """vmap with in_axes=None for broadcast (no batching on that input)."""
        # x: (batch=4, 8), w: (8,) - w is broadcast
        np_x = make_array(4, 8, seed=42)
        np_w = make_array(8, seed=43)
        
        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)
        
        def fn(a, b):
            return add(a, b)
        
        result = vmap(fn, in_axes=(0, None))(x, w)
        expected = np_x + np_w
        
        assert_shape(result, (4, 8))
        assert_allclose(result, expected)
    
    def test_vmap_matmul_broadcast_weight(self):
        """vmap(matmul) with batched input and broadcast weight matrix."""
        batch = 4
        M, K, N = 8, 16, 32
        
        np_x = make_array(batch, M, K, seed=42)
        np_w = make_array(K, N, seed=43)
        
        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)
        
        # Batch x, broadcast w
        result = vmap(matmul, in_axes=(0, None))(x, w)
        expected = np.matmul(np_x, np_w)
        
        assert_shape(result, (batch, M, N))
        assert_allclose(result, expected, rtol=1e-4)
    
    def test_vmap_matmul_batch_weight(self):
        """vmap(matmul) with both inputs batched on axis 0."""
        batch = 4
        M, K, N = 8, 16, 32
        
        np_x = make_array(batch, M, K, seed=42)
        np_w = make_array(batch, K, N, seed=43)
        
        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)
        
        result = vmap(matmul)(x, w)
        expected = np.matmul(np_x, np_w)
        
        assert_shape(result, (batch, M, N))
        assert_allclose(result, expected, rtol=1e-4)


# =============================================================================
# out_axes Specification
# =============================================================================

class TestVmapOutAxes:
    """Test vmap with out_axes specification."""
    
    def test_vmap_out_axes_1(self):
        """vmap with out_axes=1 - output batch at axis 1."""
        np_x = make_array(4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        # out_axes=1 means batch dimension goes to position 1
        result = vmap(relu, out_axes=1)(x)
        expected = np.maximum(np_x, 0).T  # (8, 4) instead of (4, 8)
        
        assert_shape(result, (8, 4))
        assert_allclose(result, expected)
    
    def test_vmap_out_axes_negative(self):
        """vmap with out_axes=-1 - output batch at last axis."""
        np_x = make_array(4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = vmap(relu, out_axes=-1)(x)
        expected = np.maximum(np_x, 0).T  # Same as out_axes=1 for 2D->1D
        
        assert_shape(result, (8, 4))
        assert_allclose(result, expected)
    
    def test_vmap_in_out_axes_combined(self):
        """vmap with both in_axes and out_axes specified."""
        # Input: (8, batch=4), batch at axis 1
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        # in_axes=1 (batch from axis 1), out_axes=0 (batch to axis 0)
        result = vmap(relu, in_axes=1, out_axes=0)(x)
        expected = np.maximum(np_x.T, 0)  # (4, 8)
        
        assert_shape(result, (4, 8))
        assert_allclose(result, expected)


# =============================================================================
# Nested vmap with in_axes
# =============================================================================

class TestNestedVmapInAxes:
    """Test nested vmap with in_axes."""
    
    def test_nested_vmap_same_axes(self):
        """Nested vmap both batching on axis 0."""
        # Shape: (outer=2, inner=4, features=8)
        np_x = make_array(2, 4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = vmap(vmap(relu))(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (2, 4, 8))
        assert_allclose(result, expected)
    
    def test_nested_vmap_different_axes(self):
        """Nested vmap with different in_axes at each level."""
        # Shape: (inner=4, outer=2, features=8)
        # We want outer batch at axis 1, inner batch at axis 0 (of the sub-tensor)
        np_x = make_array(4, 2, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Outer vmap batches on axis 1, inner on axis 0
        result = vmap(vmap(relu), in_axes=1)(x)
        expected = np.maximum(np_x.transpose(1, 0, 2), 0)  # (2, 4, 8)
        
        assert_shape(result, (2, 4, 8))
        assert_allclose(result, expected)


# =============================================================================
# vmap with Reductions
# =============================================================================

class TestVmapReductions:
    """Test vmap with reduction operations."""
    
    def test_vmap_reduce_sum_axis_0(self):
        """vmap(reduce_sum) reducing axis 0 of each batch element."""
        # Shape: (batch=4, 8, 16), reduce axis 0 of each (8, 16) -> (16,)
        np_x = make_array(4, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        def fn(a):
            return reduce_sum(a, axis=0)
        
        result = vmap(fn)(x)
        expected = np.sum(np_x, axis=1)  # Physical axis 1 = logical axis 0
        
        assert_shape(result, (4, 16))
        assert_allclose(result, expected)
    
    def test_vmap_reduce_sum_axis_negative(self):
        """vmap(reduce_sum) reducing last axis of each batch element."""
        np_x = make_array(4, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        def fn(a):
            return reduce_sum(a, axis=-1)
        
        result = vmap(fn)(x)
        expected = np.sum(np_x, axis=-1)
        
        assert_shape(result, (4, 8))
        assert_allclose(result, expected)
    
    def test_vmap_mean_broadcast_input(self):
        """vmap(mean) with broadcast input."""
        np_x = make_array(4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        def fn(a):
            return mean(a, axis=0)
        
        result = vmap(fn)(x)
        expected = np.mean(np_x, axis=1)
        
        assert_shape(result, (4,))
        assert_allclose(result, expected)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestVmapEdgeCases:
    """Test vmap edge cases."""
    
    def test_vmap_scalar_output(self):
        """vmap where each element produces a scalar (reduces to single value)."""
        np_x = make_array(4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        def fn(a):
            # Sum all elements to get a scalar-like output
            # a has shape (8,), reduce_sum gives shape () or (1,) depending on keepdims
            return reduce_sum(a, axis=0)  # Reduces (8,) -> scalar
        
        result = vmap(fn)(x)
        expected = np.sum(np_x, axis=1)  # (4,) - one value per batch element
        
        # Result should be (4,) - one scalar per batch element
        assert_shape(result, (4,))
        assert_allclose(result, expected)
    
    def test_vmap_composite_with_broadcast(self):
        """vmap on composite function with internal broadcasting."""
        batch = 4
        features = 16
        
        np_x = make_array(batch, features, seed=42)
        np_scale = make_array(features, seed=43)
        np_bias = make_array(features, seed=44)
        
        x = tensor_from_numpy(np_x)
        scale = tensor_from_numpy(np_scale)
        bias = tensor_from_numpy(np_bias)
        
        def normalize_and_scale(inputs, sc, bi):
            # inputs: (features,), sc: (features,), bi: (features,)
            m = mean(inputs, axis=0)
            centered = sub(inputs, m)
            scaled = mul(centered, sc)
            return add(scaled, bi)
        
        # scale and bias are broadcast (not batched)
        result = vmap(normalize_and_scale, in_axes=(0, None, None))(x, scale, bias)
        
        # Expected computation
        m = np.mean(np_x, axis=1, keepdims=True)
        centered = np_x - m
        scaled = centered * np_scale
        expected = scaled + np_bias
        
        assert_shape(result, (batch, features))
        assert_allclose(result, expected)


# =============================================================================
# Multi-input in_axes Variations
# =============================================================================

class TestVmapMultiInputInAxes:
    """Test vmap with various multi-input in_axes combinations."""
    
    def test_vmap_tuple_in_axes(self):
        """vmap with tuple in_axes for multiple inputs."""
        np_a = make_array(4, 8, seed=42)
        np_b = make_array(4, 8, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        result = vmap(add, in_axes=(0, 0))(a, b)
        expected = np_a + np_b
        
        assert_shape(result, (4, 8))
        assert_allclose(result, expected)
    
    def test_vmap_list_in_axes(self):
        """vmap with list in_axes for multiple inputs."""
        np_a = make_array(4, 8, seed=42)
        np_b = make_array(4, 8, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        result = vmap(add, in_axes=[0, 0])(a, b)
        expected = np_a + np_b
        
        assert_shape(result, (4, 8))
        assert_allclose(result, expected)
    
    def test_vmap_mixed_batch_broadcast(self):
        """vmap with mixed batched and broadcast inputs."""
        batch = 4
        M, K, N = 8, 16, 32
        
        np_x = make_array(batch, M, K, seed=42)
        np_w = make_array(K, N, seed=43)
        np_b = make_array(N, seed=44)
        
        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)
        b = tensor_from_numpy(np_b)
        
        def linear(inputs, weight, bias):
            return add(matmul(inputs, weight), bias)
        
        # x is batched, w and b are broadcast
        result = vmap(linear, in_axes=(0, None, None))(x, w, b)
        expected = np_x @ np_w + np_b
        
        assert_shape(result, (batch, M, N))
        assert_allclose(result, expected, rtol=1e-4)


__all__ = [
    "TestVmapInAxesNonZero",
    "TestVmapOutAxes",
    "TestNestedVmapInAxes",
    "TestVmapReductions",
    "TestVmapEdgeCases",
    "TestVmapMultiInputInAxes",
]
