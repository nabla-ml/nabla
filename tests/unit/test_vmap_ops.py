# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Tests for vmap transform - automatic batching.

vmap automatically handles:
1. Incrementing batch_dims on inputs
2. Running the operation on logical shapes
3. Decrementing batch_dims on outputs

These tests use vmap directly (not manual batch_dims manipulation).
We simply verify input shape → output shape relationships.

Test pattern:
- Create input with shape (batch, *op_shape)
- Apply vmap(fn) where fn operates on op_shape
- Verify output has shape (batch, *expected_shape)
- Verify numerical correctness against numpy

Nested vmap tests:
- vmap(vmap(fn)) for doubly-batched operations
"""

import pytest
import numpy as np

import nabla
from nabla import vmap
from nabla import (
    add, sub, mul, div,
    relu, sigmoid, tanh, exp, neg,
    reduce_sum, mean,
    reshape, squeeze, unsqueeze, swap_axes, broadcast_to,
    matmul,
)
from nabla.sharding.spec import DimSpec

from tests.conftest import (
    make_array, make_positive_array, tensor_from_numpy, to_numpy,
    assert_allclose, assert_shape, assert_batch_dims,
    shard_on_axis, replicated,
)


# =============================================================================
# vmap Binary Ops
# =============================================================================

class TestVmapBinaryOps:
    """Test vmap on binary operations."""
    
    @pytest.mark.parametrize("batch_size", [2, 4, 8])
    def test_vmap_add(self, batch_size):
        """vmap(add) over batched inputs."""
        # Input: (batch, 4, 8) + (batch, 4, 8) → (batch, 4, 8)
        np_a = make_array(batch_size, 4, 8, seed=42)
        np_b = make_array(batch_size, 4, 8, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        def fn(x, y):
            return add(x, y)
        
        result = vmap(fn)(a, b)
        expected = np_a + np_b
        
        assert_shape(result, (batch_size, 4, 8))
        assert_allclose(result, expected)
    
    def test_vmap_mul(self):
        """vmap(mul) over batched inputs."""
        np_a = make_array(4, 8, 16, seed=42)
        np_b = make_array(4, 8, 16, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        result = vmap(mul)(a, b)
        expected = np_a * np_b
        
        assert_shape(result, (4, 8, 16))
        assert_allclose(result, expected)
    
    def test_vmap_add_broadcast_within_batch(self):
        """vmap where the function does broadcasting."""
        # Each batch element: (4, 8) + (8,) → (4, 8)
        # Total: (batch=2, 4, 8) + (batch=2, 8) → (batch=2, 4, 8)
        np_a = make_array(2, 4, 8, seed=42)
        np_b = make_array(2, 8, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        def fn(x, y):
            return add(x, y)
        
        result = vmap(fn)(a, b)
        # Expected: broadcast along batch dimension
        expected = np_a + np_b[:, np.newaxis, :]
        
        assert_shape(result, (2, 4, 8))
        assert_allclose(result, expected)


class TestVmapBinaryOpsNested:
    """Test nested vmap (vmap of vmap) on binary ops."""
    
    def test_nested_vmap_add(self):
        """vmap(vmap(add)) - doubly batched."""
        # Input: (batch1=2, batch2=4, 8) + same → same
        np_a = make_array(2, 4, 8, seed=42)
        np_b = make_array(2, 4, 8, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        def fn(x, y):
            return add(x, y)
        
        result = vmap(vmap(fn))(a, b)
        expected = np_a + np_b
        
        assert_shape(result, (2, 4, 8))
        assert_allclose(result, expected)
    
    def test_nested_vmap_mul(self):
        """vmap(vmap(mul)) - doubly batched."""
        np_a = make_array(2, 4, 8, 16, seed=42)
        np_b = make_array(2, 4, 8, 16, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        result = vmap(vmap(mul))(a, b)
        expected = np_a * np_b
        
        assert_shape(result, (2, 4, 8, 16))
        assert_allclose(result, expected)


# =============================================================================
# vmap Unary Ops
# =============================================================================

class TestVmapUnaryOps:
    """Test vmap on unary operations."""
    
    @pytest.mark.parametrize("batch_size", [2, 4, 8])
    def test_vmap_relu(self, batch_size):
        """vmap(relu) over batched input."""
        np_x = make_array(batch_size, 4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = vmap(relu)(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (batch_size, 4, 8))
        assert_allclose(result, expected)
    
    def test_vmap_sigmoid(self):
        """vmap(sigmoid) over batched input."""
        np_x = make_array(4, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = vmap(sigmoid)(x)
        expected = 1 / (1 + np.exp(-np_x))
        
        assert_shape(result, (4, 8, 16))
        assert_allclose(result, expected)
    
    def test_vmap_tanh(self):
        """vmap(tanh) over batched input."""
        np_x = make_array(4, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = vmap(tanh)(x)
        expected = np.tanh(np_x)
        
        assert_shape(result, (4, 8, 16))
        assert_allclose(result, expected)
    
    def test_vmap_neg(self):
        """vmap(neg) over batched input."""
        np_x = make_array(4, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = vmap(neg)(x)
        expected = -np_x
        
        assert_shape(result, (4, 8, 16))
        assert_allclose(result, expected)


class TestVmapUnaryOpsNested:
    """Test nested vmap on unary ops."""
    
    def test_nested_vmap_relu(self):
        """vmap(vmap(relu)) - doubly batched."""
        np_x = make_array(2, 4, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = vmap(vmap(relu))(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (2, 4, 8, 16))
        assert_allclose(result, expected)
    
    def test_triple_vmap_relu(self):
        """vmap(vmap(vmap(relu))) - triply batched."""
        np_x = make_array(2, 3, 4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = vmap(vmap(vmap(relu)))(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (2, 3, 4, 8))
        assert_allclose(result, expected)


# =============================================================================
# vmap Reduction Ops
# =============================================================================

class TestVmapReductionOps:
    """Test vmap on reduction operations."""
    
    def test_vmap_reduce_sum(self):
        """vmap(reduce_sum) - reduce within each batch element."""
        # Input: (batch=4, M=8, N=16) → reduce axis=1 → (batch=4, M=8)
        np_x = make_array(4, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        def fn(t):
            return reduce_sum(t, axis=1)  # Reduce N dimension
        
        result = vmap(fn)(x)
        expected = np.sum(np_x, axis=2)  # Physical axis 2 = logical axis 1
        
        assert_shape(result, (4, 8))
        assert_allclose(result, expected)
    
    def test_vmap_reduce_sum_axis_0(self):
        """vmap(reduce_sum) reducing first logical axis."""
        # Input: (batch=4, M=8, N=16) → reduce axis=0 → (batch=4, N=16)
        np_x = make_array(4, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        def fn(t):
            return reduce_sum(t, axis=0)  # Reduce M dimension
        
        result = vmap(fn)(x)
        expected = np.sum(np_x, axis=1)  # Physical axis 1 = logical axis 0
        
        assert_shape(result, (4, 16))
        assert_allclose(result, expected)
    
    def test_vmap_mean(self):
        """vmap(mean) - mean within each batch element."""
        np_x = make_array(4, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        def fn(t):
            return mean(t, axis=1)
        
        result = vmap(fn)(x)
        expected = np.mean(np_x, axis=2)
        
        assert_shape(result, (4, 8))
        assert_allclose(result, expected)
    
    def test_nested_vmap_reduce_sum(self):
        """vmap(vmap(reduce_sum)) - doubly batched reduction."""
        # (b1=2, b2=4, M=8, N=16) → reduce N → (b1=2, b2=4, M=8)
        np_x = make_array(2, 4, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        def fn(t):
            return reduce_sum(t, axis=1)
        
        result = vmap(vmap(fn))(x)
        expected = np.sum(np_x, axis=3)  # Physical axis 3 = logical axis 1
        
        assert_shape(result, (2, 4, 8))
        assert_allclose(result, expected)


# =============================================================================
# vmap View Ops
# =============================================================================

class TestVmapViewOps:
    """Test vmap on view operations."""
    
    def test_vmap_reshape(self):
        """vmap(reshape) - reshape within each batch element."""
        # Input: (batch=4, 8, 16) → reshape to (128,) → (batch=4, 128)
        np_x = make_array(4, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        def fn(t):
            return reshape(t, (128,))
        
        result = vmap(fn)(x)
        expected = np_x.reshape(4, 128)
        
        assert_shape(result, (4, 128))
        assert_allclose(result, expected)
    
    def test_vmap_squeeze(self):
        """vmap(squeeze) - squeeze within each batch element."""
        # Input: (batch=4, 1, 8) → squeeze axis=0 → (batch=4, 8)
        np_x = make_array(4, 1, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        def fn(t):
            return squeeze(t, axis=0)  # Squeeze logical axis 0
        
        result = vmap(fn)(x)
        expected = np_x.reshape(4, 8)  # Squeeze removes the 1
        
        assert_shape(result, (4, 8))
        assert_allclose(result, expected)
    
    def test_vmap_unsqueeze(self):
        """vmap(unsqueeze) - unsqueeze within each batch element."""
        # Input: (batch=4, 8) → unsqueeze axis=0 → (batch=4, 1, 8)
        np_x = make_array(4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        def fn(t):
            return unsqueeze(t, axis=0)  # Add dim at logical axis 0
        
        result = vmap(fn)(x)
        expected = np_x.reshape(4, 1, 8)
        
        assert_shape(result, (4, 1, 8))
        assert_allclose(result, expected)
    
    def test_vmap_swap_axes(self):
        """vmap(swap_axes) - swap within each batch element."""
        # Input: (batch=4, 8, 16) → swap(0,1) → (batch=4, 16, 8)
        np_x = make_array(4, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        def fn(t):
            return swap_axes(t, 0, 1)
        
        result = vmap(fn)(x)
        expected = np.swapaxes(np_x, 1, 2)  # Physical: swap 1,2
        
        assert_shape(result, (4, 16, 8))
        assert_allclose(result, expected)
    
    def test_nested_vmap_reshape(self):
        """vmap(vmap(reshape)) - doubly batched reshape."""
        # (b1=2, b2=4, 8, 16) → reshape to (128,) → (b1=2, b2=4, 128)
        np_x = make_array(2, 4, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        def fn(t):
            return reshape(t, (128,))
        
        result = vmap(vmap(fn))(x)
        expected = np_x.reshape(2, 4, 128)
        
        assert_shape(result, (2, 4, 128))
        assert_allclose(result, expected)


# =============================================================================
# vmap Matmul
# =============================================================================

class TestVmapMatmul:
    """Test vmap on matrix multiplication."""
    
    def test_vmap_matmul(self):
        """vmap(matmul) - batched matrix multiplication."""
        # Input: (batch=4, M=8, K=16) @ (batch=4, K=16, N=32) → (batch=4, M=8, N=32)
        np_a = make_array(4, 8, 16, seed=42)
        np_b = make_array(4, 16, 32, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        def fn(x, y):
            return matmul(x, y)
        
        result = vmap(fn)(a, b)
        expected = np.matmul(np_a, np_b)
        
        assert_shape(result, (4, 8, 32))
        assert_allclose(result, expected, rtol=1e-4)
    
    def test_nested_vmap_matmul(self):
        """vmap(vmap(matmul)) - doubly batched matmul."""
        # (b1=2, b2=4, 8, 16) @ (b1=2, b2=4, 16, 32) → (b1=2, b2=4, 8, 32)
        np_a = make_array(2, 4, 8, 16, seed=42)
        np_b = make_array(2, 4, 16, 32, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        result = vmap(vmap(matmul))(a, b)
        expected = np.matmul(np_a, np_b)
        
        assert_shape(result, (2, 4, 8, 32))
        assert_allclose(result, expected, rtol=1e-4)


# =============================================================================
# vmap Composite Functions
# =============================================================================

class TestVmapComposite:
    """Test vmap on composite functions (multiple ops)."""
    
    def test_vmap_mlp_layer(self):
        """vmap over a simple MLP layer: relu(x @ W + b)."""
        batch = 4
        in_features = 16
        out_features = 32
        
        np_x = make_array(batch, in_features, seed=42)
        np_w = make_array(in_features, out_features, seed=43)
        np_b = make_array(out_features, seed=44)
        
        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)
        b = tensor_from_numpy(np_b)
        
        def mlp_layer(inputs):
            return relu(add(matmul(inputs, w), b))
        
        result = vmap(mlp_layer)(x)
        expected = np.maximum(np_x @ np_w + np_b, 0)
        
        assert_shape(result, (batch, out_features))
        assert_allclose(result, expected, rtol=1e-4)
    
    def test_vmap_normalize(self):
        """vmap over a normalization function: (x - mean) / std."""
        batch = 4
        features = 8
        
        np_x = make_array(batch, features, seed=42)
        
        x = tensor_from_numpy(np_x)
        
        def normalize(inputs):
            m = mean(inputs, axis=0)
            # Simple normalize: subtract mean
            return sub(inputs, m)
        
        result = vmap(normalize)(x)
        expected = np_x - np.mean(np_x, axis=1, keepdims=True)
        
        assert_shape(result, (batch, features))
        assert_allclose(result, expected)


# =============================================================================
# vmap + Sharding
# =============================================================================

class TestVmapWithSharding:
    """Test vmap combined with sharding."""
    
    def test_vmap_relu_sharded(self, mesh_1d):
        """vmap(relu) on sharded input."""
        # Shard the batch dimension
        np_x = make_array(8, 4, 16, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)  # Shard batch
        
        result = vmap(relu)(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (8, 4, 16))
        assert_allclose(result, expected)
    
    def test_vmap_add_sharded(self, mesh_1d):
        """vmap(add) on sharded inputs."""
        np_a = make_array(8, 4, 16, seed=42)
        np_b = make_array(8, 4, 16, seed=43)
        
        a = shard_on_axis(tensor_from_numpy(np_a), mesh_1d, axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh_1d, axis=0)
        
        result = vmap(add)(a, b)
        expected = np_a + np_b
        
        assert_shape(result, (8, 4, 16))
        assert_allclose(result, expected)
    
    def test_vmap_matmul_sharded_batch(self, mesh_1d):
        """vmap(matmul) with batch dimension sharded."""
        np_a = make_array(8, 4, 16, seed=42)
        np_b = make_array(8, 16, 8, seed=43)
        
        a = shard_on_axis(tensor_from_numpy(np_a), mesh_1d, axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh_1d, axis=0)
        
        result = vmap(matmul)(a, b)
        expected = np.matmul(np_a, np_b)
        
        assert_shape(result, (8, 4, 8))
        assert_allclose(result, expected, rtol=1e-4)
    
    def test_nested_vmap_sharded(self, mesh_1d):
        """Nested vmap on sharded tensor."""
        np_x = make_array(8, 4, 16, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh_1d, axis=0)
        
        result = vmap(vmap(relu))(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (8, 4, 16))
        assert_allclose(result, expected)
