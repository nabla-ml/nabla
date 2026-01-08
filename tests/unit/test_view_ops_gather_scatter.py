# ===----------------------------------------------------------------------=== #
# Nabla 2025
# ===----------------------------------------------------------------------=== #
"""Tests for gather, scatter, concatenate, and stack operations.

These operations are critical for Pipeline Parallelism and need thorough testing:
1. Basic functionality (no sharding)
2. Vmap correctness with various batch sizes  
3. Sharding with multiple mesh configurations
4. Combined vmap + sharding

Each test validates:
- Correct output shape
- Numerical correctness against numpy reference
- Proper sharding propagation
"""

import pytest
import numpy as np

import nabla
from nabla import vmap, DeviceMesh, P
from nabla import gather, scatter, concatenate, stack
from nabla import unsqueeze, squeeze, reshape
from nabla.sharding.spec import DimSpec

from tests.conftest import (
    make_array, make_positive_array, tensor_from_numpy, to_numpy,
    assert_allclose, assert_shape, assert_is_sharded, assert_batch_dims,
    shard_on_axis, shard_on_axes, replicated,
)


# =============================================================================
# Gather Tests
# =============================================================================

class TestGatherBasic:
    """Test gather operation basic functionality."""
    
    def test_gather_1d_indices(self):
        """Gather with 1D indices along axis 0."""
        # Data: (8,) -> gather with indices (3,) -> (3,)
        np_data = make_array(8, seed=42)
        np_indices = np.array([0, 3, 7], dtype=np.int32)
        
        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)
        
        result = gather(data, indices, axis=0)
        expected = np_data[np_indices]
        
        assert_shape(result, (3,))
        assert_allclose(result, expected)
    
    def test_gather_2d_axis0(self):
        """Gather from 2D tensor along axis 0."""
        # Data: (8, 4) -> gather indices (3,) -> (3, 4)
        np_data = make_array(8, 4, seed=42)
        np_indices = np.array([1, 5, 7], dtype=np.int32)
        
        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)
        
        result = gather(data, indices, axis=0)
        expected = np_data[np_indices, :]
        
        assert_shape(result, (3, 4))
        assert_allclose(result, expected)
    
    def test_gather_2d_axis1(self):
        """Gather from 2D tensor along axis 1."""
        # Data: (4, 8) -> gather indices (3,) at axis=1 -> (4, 3)
        np_data = make_array(4, 8, seed=42)
        np_indices = np.array([0, 2, 6], dtype=np.int32)
        
        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)
        
        result = gather(data, indices, axis=1)
        expected = np_data[:, np_indices]
        
        assert_shape(result, (4, 3))
        assert_allclose(result, expected)
    
    def test_gather_3d_axis0(self):
        """Gather from 3D tensor along axis 0."""
        # Data: (8, 4, 6) -> gather indices (2,) -> (2, 4, 6)
        np_data = make_array(8, 4, 6, seed=42)
        np_indices = np.array([1, 5], dtype=np.int32)
        
        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)
        
        result = gather(data, indices, axis=0)
        expected = np_data[np_indices, :, :]
        
        assert_shape(result, (2, 4, 6))
        assert_allclose(result, expected)
    
    def test_gather_3d_axis1(self):
        """Gather from 3D tensor along axis 1 (middle axis)."""
        # Data: (4, 8, 6) -> gather indices (3,) at axis=1 -> (4, 3, 6)
        np_data = make_array(4, 8, 6, seed=42)
        np_indices = np.array([0, 3, 7], dtype=np.int32)
        
        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)
        
        result = gather(data, indices, axis=1)
        expected = np_data[:, np_indices, :]
        
        assert_shape(result, (4, 3, 6))
        assert_allclose(result, expected)
    
    def test_gather_negative_axis(self):
        """Gather with negative axis."""
        np_data = make_array(4, 8, seed=42)
        np_indices = np.array([1, 3, 5], dtype=np.int32)
        
        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)
        
        result = gather(data, indices, axis=-1)
        expected = np_data[:, np_indices]
        
        assert_shape(result, (4, 3))
        assert_allclose(result, expected)


class TestGatherVmap:
    """Test gather with vmap (automatic batching)."""
    
    @pytest.mark.parametrize("batch_size", [2, 4, 8])
    def test_vmap_gather_axis0(self, batch_size):
        """Vmap over gather with batch in data."""
        # Each batch element: (8, 4) -> gather (3,) -> (3, 4)
        # Batched: (batch, 8, 4) + (3,) -> (batch, 3, 4)
        np_data = make_array(batch_size, 8, 4, seed=42)
        np_indices = np.array([0, 3, 7], dtype=np.int32)
        
        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)
        
        def fn(x):
            return gather(x, indices, axis=0)
        
        result = vmap(fn)(data)
        expected = np_data[:, np_indices, :]  # Numpy advanced indexing
        
        assert_shape(result, (batch_size, 3, 4))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_vmap_gather_batched_indices(self, batch_size):
        """Vmap with both data and indices batched."""
        # Each batch: data (8,), indices (3,) -> (3,)
        np_data = make_array(batch_size, 8, seed=42)
        np_indices = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 2, 3]][:batch_size], dtype=np.int32)
        
        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)
        
        def fn(x, idx):
            return gather(x, idx, axis=0)
        
        result = vmap(fn)(data, indices)
        
        # Manual computation
        expected = np.array([np_data[i, np_indices[i]] for i in range(batch_size)])
        
        assert_shape(result, (batch_size, 3))
        assert_allclose(result, expected)


class TestGatherSharding:
    """Test gather with sharding."""
    
    def test_gather_replicated(self, mesh_1d):
        """Gather from replicated tensor produces correct result."""
        np_data = make_array(8, 4, seed=42)
        np_indices = np.array([1, 5], dtype=np.int32)
        
        data = tensor_from_numpy(np_data)
        data_repl = replicated(data, mesh_1d)
        indices = tensor_from_numpy(np_indices)
        
        result = gather(data_repl, indices, axis=0)
        expected = np_data[np_indices, :]
        
        assert_shape(result, (2, 4))
        assert_allclose(result, expected)
    
    def test_gather_sharded_non_gather_axis(self, mesh_1d):
        """Gather from tensor sharded on non-gather axis."""
        # Data sharded on axis 1, gather on axis 0
        np_data = make_array(8, 4, seed=42)
        np_indices = np.array([0, 3, 7], dtype=np.int32)
        
        data = tensor_from_numpy(np_data)
        data_sharded = shard_on_axis(data, mesh_1d, axis=1)
        indices = tensor_from_numpy(np_indices)
        
        result = gather(data_sharded, indices, axis=0)
        expected = np_data[np_indices, :]
        
        assert_shape(result, (3, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


# =============================================================================
# Scatter Tests
# =============================================================================

class TestScatterBasic:
    """Test scatter operation basic functionality."""
    
    def test_scatter_1d(self):
        """Scatter updates into 1D tensor."""
        np_data = make_array(8, seed=42)
        np_indices = np.array([1, 3, 5], dtype=np.int32)
        np_updates = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        
        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)
        updates = tensor_from_numpy(np_updates)
        
        result = scatter(data, indices, updates, axis=0)
        
        expected = np_data.copy()
        expected[np_indices] = np_updates
        
        assert_shape(result, (8,))
        assert_allclose(result, expected)
    
    def test_scatter_2d_axis0(self):
        """Scatter into 2D tensor along axis 0."""
        np_data = make_array(8, 4, seed=42)
        np_indices = np.array([0, 3], dtype=np.int32)
        np_updates = make_array(2, 4, seed=43)  # (num_indices, 4)
        
        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)
        updates = tensor_from_numpy(np_updates)
        
        result = scatter(data, indices, updates, axis=0)
        
        expected = np_data.copy()
        expected[np_indices, :] = np_updates
        
        assert_shape(result, (8, 4))
        assert_allclose(result, expected)
    
    def test_scatter_2d_axis1(self):
        """Scatter into 2D tensor along axis 1."""
        np_data = make_array(4, 8, seed=42)
        np_indices = np.array([1, 5, 7], dtype=np.int32)
        np_updates = make_array(4, 3, seed=43)  # (4, num_indices)
        
        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)
        updates = tensor_from_numpy(np_updates)
        
        result = scatter(data, indices, updates, axis=1)
        
        expected = np_data.copy()
        expected[:, np_indices] = np_updates
        
        assert_shape(result, (4, 8))
        assert_allclose(result, expected)


class TestScatterSharding:
    """Test scatter with sharding."""
    
    def test_scatter_replicated(self, mesh_1d):
        """Scatter into replicated tensor."""
        np_data = make_array(8, 4, seed=42)
        np_indices = np.array([2, 5], dtype=np.int32)
        np_updates = make_array(2, 4, seed=43)
        
        data = tensor_from_numpy(np_data)
        data_repl = replicated(data, mesh_1d)
        indices = tensor_from_numpy(np_indices)
        updates = tensor_from_numpy(np_updates)
        
        result = scatter(data_repl, indices, updates, axis=0)
        
        expected = np_data.copy()
        expected[np_indices, :] = np_updates
        
        assert_shape(result, (8, 4))
        assert_allclose(result, expected)


# =============================================================================
# Concatenate Tests  
# =============================================================================

class TestConcatenateBasic:
    """Test concatenate operation basic functionality."""
    
    def test_concatenate_1d(self):
        """Concatenate 1D tensors."""
        np_a = make_array(4, seed=42)
        np_b = make_array(6, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        result = concatenate([a, b], axis=0)
        expected = np.concatenate([np_a, np_b], axis=0)
        
        assert_shape(result, (10,))
        assert_allclose(result, expected)
    
    def test_concatenate_2d_axis0(self):
        """Concatenate 2D tensors along axis 0."""
        np_a = make_array(2, 4, seed=42)
        np_b = make_array(3, 4, seed=43)
        np_c = make_array(5, 4, seed=44)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        c = tensor_from_numpy(np_c)
        
        result = concatenate([a, b, c], axis=0)
        expected = np.concatenate([np_a, np_b, np_c], axis=0)
        
        assert_shape(result, (10, 4))
        assert_allclose(result, expected)
    
    def test_concatenate_2d_axis1(self):
        """Concatenate 2D tensors along axis 1."""
        np_a = make_array(4, 2, seed=42)
        np_b = make_array(4, 3, seed=43)
        np_c = make_array(4, 5, seed=44)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        c = tensor_from_numpy(np_c)
        
        result = concatenate([a, b, c], axis=1)
        expected = np.concatenate([np_a, np_b, np_c], axis=1)
        
        assert_shape(result, (4, 10))
        assert_allclose(result, expected)
    
    def test_concatenate_3d(self):
        """Concatenate 3D tensors along middle axis."""
        np_a = make_array(2, 3, 4, seed=42)
        np_b = make_array(2, 5, 4, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        result = concatenate([a, b], axis=1)
        expected = np.concatenate([np_a, np_b], axis=1)
        
        assert_shape(result, (2, 8, 4))
        assert_allclose(result, expected)
    
    def test_concatenate_negative_axis(self):
        """Concatenate with negative axis."""
        np_a = make_array(4, 2, seed=42)
        np_b = make_array(4, 3, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        result = concatenate([a, b], axis=-1)
        expected = np.concatenate([np_a, np_b], axis=-1)
        
        assert_shape(result, (4, 5))
        assert_allclose(result, expected)


class TestConcatenateVmap:
    """Test concatenate with vmap."""
    
    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_vmap_concatenate(self, batch_size):
        """Vmap over concatenate."""
        np_a = make_array(batch_size, 4, 2, seed=42)
        np_b = make_array(batch_size, 4, 3, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        def fn(x, y):
            return concatenate([x, y], axis=-1)
        
        result = vmap(fn)(a, b)
        expected = np.concatenate([np_a, np_b], axis=-1)
        
        assert_shape(result, (batch_size, 4, 5))
        assert_allclose(result, expected)


class TestConcatenateSharding:
    """Test concatenate with sharding."""
    
    def test_concatenate_replicated(self, mesh_1d):
        """Concatenate replicated tensors."""
        np_a = make_array(4, 8, seed=42)
        np_b = make_array(4, 8, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        a_repl = replicated(a, mesh_1d)
        b_repl = replicated(b, mesh_1d)
        
        result = concatenate([a_repl, b_repl], axis=0)
        expected = np.concatenate([np_a, np_b], axis=0)
        
        assert_shape(result, (8, 8))
        assert_allclose(result, expected)
    
    def test_concatenate_sharded_non_concat_axis(self, mesh_1d):
        """Concatenate tensors sharded on non-concat axis."""
        # Shard on axis 1, concat on axis 0
        np_a = make_array(4, 8, seed=42)
        np_b = make_array(6, 8, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        a_sharded = shard_on_axis(a, mesh_1d, axis=1)
        b_sharded = shard_on_axis(b, mesh_1d, axis=1)
        
        result = concatenate([a_sharded, b_sharded], axis=0)
        expected = np.concatenate([np_a, np_b], axis=0)
        
        assert_shape(result, (10, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


# =============================================================================
# Stack Tests
# =============================================================================

class TestStackBasic:
    """Test stack operation basic functionality."""
    
    def test_stack_1d(self):
        """Stack 1D tensors."""
        np_a = make_array(4, seed=42)
        np_b = make_array(4, seed=43)
        np_c = make_array(4, seed=44)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        c = tensor_from_numpy(np_c)
        
        result = stack([a, b, c], axis=0)
        expected = np.stack([np_a, np_b, np_c], axis=0)
        
        assert_shape(result, (3, 4))
        assert_allclose(result, expected)
    
    def test_stack_2d_axis0(self):
        """Stack 2D tensors along axis 0."""
        np_a = make_array(4, 8, seed=42)
        np_b = make_array(4, 8, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        result = stack([a, b], axis=0)
        expected = np.stack([np_a, np_b], axis=0)
        
        assert_shape(result, (2, 4, 8))
        assert_allclose(result, expected)
    
    def test_stack_2d_axis1(self):
        """Stack 2D tensors along axis 1."""
        np_a = make_array(4, 8, seed=42)
        np_b = make_array(4, 8, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        result = stack([a, b], axis=1)
        expected = np.stack([np_a, np_b], axis=1)
        
        assert_shape(result, (4, 2, 8))
        assert_allclose(result, expected)
    
    def test_stack_2d_axis_last(self):
        """Stack 2D tensors along last axis."""
        np_a = make_array(4, 8, seed=42)
        np_b = make_array(4, 8, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        result = stack([a, b], axis=-1)
        expected = np.stack([np_a, np_b], axis=-1)
        
        assert_shape(result, (4, 8, 2))
        assert_allclose(result, expected)


class TestStackVmap:
    """Test stack with vmap."""
    
    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_vmap_stack(self, batch_size):
        """Vmap over stack."""
        np_a = make_array(batch_size, 4, seed=42)
        np_b = make_array(batch_size, 4, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        def fn(x, y):
            return stack([x, y], axis=0)
        
        result = vmap(fn)(a, b)
        expected = np.stack([np_a, np_b], axis=1)  # After vmap, stack axis is 1
        
        assert_shape(result, (batch_size, 2, 4))
        assert_allclose(result, expected)


class TestStackSharding:
    """Test stack with sharding."""
    
    def test_stack_replicated(self, mesh_1d):
        """Stack replicated tensors."""
        np_a = make_array(4, 8, seed=42)
        np_b = make_array(4, 8, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        a_repl = replicated(a, mesh_1d)
        b_repl = replicated(b, mesh_1d)
        
        result = stack([a_repl, b_repl], axis=0)
        expected = np.stack([np_a, np_b], axis=0)
        
        assert_shape(result, (2, 4, 8))
        assert_allclose(result, expected)


# =============================================================================
# Integration Tests
# =============================================================================

class TestGatherScatterRoundTrip:
    """Test gather followed by scatter produces expected results."""
    
    def test_gather_scatter_identity(self):
        """Gather then scatter back to same positions."""
        np_data = make_array(8, 4, seed=42)
        np_indices = np.array([1, 3, 5], dtype=np.int32)
        
        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)
        
        # Gather some elements
        gathered = gather(data, indices, axis=0)
        
        # Create zero buffer and scatter back
        buffer = tensor_from_numpy(np.zeros((8, 4), dtype=np.float32))
        result = scatter(buffer, indices, gathered, axis=0)
        
        # Check that scattered positions match original
        expected = np.zeros((8, 4), dtype=np.float32)
        expected[np_indices, :] = np_data[np_indices, :]
        
        assert_shape(result, (8, 4))
        assert_allclose(result, expected)
