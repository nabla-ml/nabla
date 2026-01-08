# ===----------------------------------------------------------------------=== #
# Nabla 2025
# ===----------------------------------------------------------------------=== #
"""Tests for physical operations - the foundation layer.

Physical ops work on full physical shapes (no batch_dims awareness).
These must pass before testing logical ops or vmap transforms.

Physical ops tested:
- reduce_sum_physical: Sum reduction on physical axis
- mean_physical: Mean reduction on physical axis  
- squeeze_physical: Remove dim of size 1 at physical axis
- unsqueeze_physical: Add dim of size 1 at physical axis
- broadcast_to_physical: Broadcast to physical target shape

All physical ops should also work correctly with sharding propagation.
"""

import pytest
import numpy as np

import nabla
from nabla import (
    reduce_sum_physical, mean_physical,
    squeeze_physical, unsqueeze_physical, broadcast_to_physical,
)
from nabla.sharding.spec import DimSpec

from .conftest import (
    make_array, make_positive_array, tensor_from_numpy, to_numpy,
    assert_allclose, assert_shape, assert_physical_shape, assert_is_sharded,
    shard_on_axis, replicated,
)


# =============================================================================
# reduce_sum_physical Tests
# =============================================================================

class TestReduceSumPhysical:
    """Test reduce_sum_physical on various shapes and axes."""
    
    @pytest.mark.parametrize("shape,axis", [
        ((8,), 0),           # 1D, reduce all
        ((4, 8), 0),         # 2D, reduce first
        ((4, 8), 1),         # 2D, reduce last
        ((4, 8), -1),        # 2D, negative index
        ((2, 4, 8), 0),      # 3D, reduce first
        ((2, 4, 8), 1),      # 3D, reduce middle
        ((2, 4, 8), 2),      # 3D, reduce last
        ((2, 4, 8), -2),     # 3D, negative index
    ])
    def test_reduce_axis(self, shape, axis):
        """Test reduction along specific axis."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = reduce_sum_physical(x, axis=axis)
        expected = np.sum(np_x, axis=axis)
        
        assert_shape(result, expected.shape)
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("shape,axis", [
        ((4, 8), 0),
        ((4, 8), 1),
        ((2, 4, 8), 1),
    ])
    def test_keepdims(self, shape, axis):
        """Test reduction with keepdims=True."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = reduce_sum_physical(x, axis=axis, keepdims=True)
        expected = np.sum(np_x, axis=axis, keepdims=True)
        
        assert_shape(result, expected.shape)
        assert_allclose(result, expected)
    
    def test_sharded_reduce_non_sharded_axis(self, mesh_1d):
        """Reduce on axis that is NOT sharded - should work without gather."""
        # Shape (8, 4) sharded on axis 0, reduce axis 1
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)
        
        result = reduce_sum_physical(x, axis=1)
        expected = np.sum(np_x, axis=1)
        
        assert_shape(result, expected.shape)
        assert_is_sharded(result, True)  # Should still be sharded on axis 0
        assert_allclose(result, expected)
    
    def test_sharded_reduce_sharded_axis(self, mesh_1d):
        """Reduce on axis that IS sharded - requires all_reduce."""
        # Shape (8, 4) sharded on axis 0, reduce axis 0
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)
        
        result = reduce_sum_physical(x, axis=0)
        expected = np.sum(np_x, axis=0)
        
        assert_shape(result, expected.shape)
        assert_allclose(result, expected)


# =============================================================================
# mean_physical Tests
# =============================================================================

class TestMeanPhysical:
    """Test mean_physical on various shapes and axes."""
    
    @pytest.mark.parametrize("shape,axis", [
        ((8,), 0),
        ((4, 8), 0),
        ((4, 8), 1),
        ((2, 4, 8), 0),
        ((2, 4, 8), 1),
        ((2, 4, 8), 2),
    ])
    def test_mean_axis(self, shape, axis):
        """Test mean along specific axis."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = mean_physical(x, axis=axis)
        expected = np.mean(np_x, axis=axis)
        
        assert_shape(result, expected.shape)
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("shape,axis", [
        ((4, 8), 0),
        ((4, 8), 1),
    ])
    def test_keepdims(self, shape, axis):
        """Test mean with keepdims=True."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = mean_physical(x, axis=axis, keepdims=True)
        expected = np.mean(np_x, axis=axis, keepdims=True)
        
        assert_shape(result, expected.shape)
        assert_allclose(result, expected)
    
    def test_sharded_mean_non_sharded_axis(self, mesh_1d):
        """Mean on non-sharded axis."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)
        
        result = mean_physical(x, axis=1)
        expected = np.mean(np_x, axis=1)
        
        assert_shape(result, expected.shape)
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


# =============================================================================
# squeeze_physical Tests
# =============================================================================

class TestSqueezePhysical:
    """Test squeeze_physical on various shapes."""
    
    @pytest.mark.parametrize("shape,axis,expected_shape", [
        ((1, 8), 0, (8,)),           # Remove first dim
        ((4, 1), 1, (4,)),           # Remove last dim
        ((4, 1, 8), 1, (4, 8)),      # Remove middle dim
        ((1, 4, 8), 0, (4, 8)),      # Remove first of 3D
        ((4, 8, 1), 2, (4, 8)),      # Remove last of 3D
        ((4, 8, 1), -1, (4, 8)),     # Negative index
    ])
    def test_squeeze_axis(self, shape, axis, expected_shape):
        """Test squeeze at specific axis."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = squeeze_physical(x, axis=axis)
        expected = np.squeeze(np_x, axis=axis)
        
        assert_shape(result, expected_shape)
        assert_allclose(result, expected)
    
    def test_sharded_squeeze_non_sharded_dim(self, mesh_1d):
        """Squeeze a non-sharded dimension."""
        # Shape (4, 1, 8) sharded on axis 0, squeeze axis 1
        np_x = make_array(4, 1, 8, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)
        
        result = squeeze_physical(x, axis=1)
        expected = np.squeeze(np_x, axis=1)
        
        assert_shape(result, (4, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)
    
    def test_sharded_squeeze_after_sharded_dim(self, mesh_1d):
        """Squeeze a dim that comes after the sharded dim."""
        # Shape (4, 8, 1) sharded on axis 0, squeeze axis 2
        np_x = make_array(4, 8, 1, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)
        
        result = squeeze_physical(x, axis=2)
        expected = np.squeeze(np_x, axis=2)
        
        assert_shape(result, (4, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


# =============================================================================
# unsqueeze_physical Tests  
# =============================================================================

class TestUnsqueezePhysical:
    """Test unsqueeze_physical on various shapes."""
    
    @pytest.mark.parametrize("shape,axis,expected_shape", [
        ((8,), 0, (1, 8)),           # Add first dim
        ((8,), 1, (8, 1)),           # Add last dim
        ((4, 8), 0, (1, 4, 8)),      # Add first of 2D
        ((4, 8), 1, (4, 1, 8)),      # Add middle of 2D
        ((4, 8), 2, (4, 8, 1)),      # Add last of 2D
        ((4, 8), -1, (4, 8, 1)),     # Negative index
    ])
    def test_unsqueeze_axis(self, shape, axis, expected_shape):
        """Test unsqueeze at specific axis."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = unsqueeze_physical(x, axis=axis)
        expected = np.expand_dims(np_x, axis=axis)
        
        assert_shape(result, expected_shape)
        assert_allclose(result, expected)
    
    def test_sharded_unsqueeze_before_sharded(self, mesh_1d):
        """Unsqueeze before sharded dimension - sharded dim index shifts."""
        # Shape (4, 8) sharded on axis 0, unsqueeze at 0
        np_x = make_array(4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)
        
        result = unsqueeze_physical(x, axis=0)
        expected = np.expand_dims(np_x, axis=0)
        
        assert_shape(result, (1, 4, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)
    
    def test_sharded_unsqueeze_after_sharded(self, mesh_1d):
        """Unsqueeze after sharded dimension."""
        # Shape (4, 8) sharded on axis 0, unsqueeze at 1
        np_x = make_array(4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)
        
        result = unsqueeze_physical(x, axis=1)
        expected = np.expand_dims(np_x, axis=1)
        
        assert_shape(result, (4, 1, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


# =============================================================================
# broadcast_to_physical Tests
# =============================================================================

class TestBroadcastToPhysical:
    """Test broadcast_to_physical on various shapes."""
    
    @pytest.mark.parametrize("shape,target_shape", [
        ((1,), (4,)),                  # Expand scalar-like
        ((8,), (4, 8)),                # Add leading dim
        ((1, 8), (4, 8)),              # Expand first dim
        ((4, 1), (4, 8)),              # Expand last dim
        ((1, 1, 8), (2, 4, 8)),        # Expand multiple dims
        ((1, 4, 1), (2, 4, 8)),        # Expand first and last
    ])
    def test_broadcast(self, shape, target_shape):
        """Test broadcast to target shape."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = broadcast_to_physical(x, target_shape)
        expected = np.broadcast_to(np_x, target_shape)
        
        # When rank increases, leading dims become batch_dims, so check physical shape
        assert_physical_shape(result, target_shape)
        assert_allclose(result, expected)
    
    def test_broadcast_preserves_sharding(self, mesh_1d):
        """Broadcast should preserve sharding on non-broadcast dims."""
        # Shape (4, 1) sharded on axis 0, broadcast to (4, 8)
        np_x = make_array(4, 1, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)
        
        result = broadcast_to_physical(x, (4, 8))
        expected = np.broadcast_to(np_x, (4, 8))
        
        assert_shape(result, (4, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)
    
    # @pytest.mark.xfail(reason="Known bug: sharded tensor broadcasting by adding leading dim produces incorrect values (all same value)")
    def test_broadcast_add_leading_dim_sharded(self, mesh_1d):
        """Broadcast by adding leading dim to sharded tensor."""
        # Shape (4,) sharded → (2, 4)
        np_x = make_array(4, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)
        
        result = broadcast_to_physical(x, (2, 4))
        expected = np.broadcast_to(np_x, (2, 4))
        
        # When rank increases, leading dims become batch_dims, so check physical shape
        assert_physical_shape(result, (2, 4))
        assert_allclose(result, expected)


# =============================================================================
# Edge Cases
# =============================================================================

class TestPhysicalOpsEdgeCases:
    """Test edge cases for physical ops."""
    
    def test_reduce_sum_single_element(self):
        """Reduce a single-element tensor."""
        np_x = make_array(1, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = reduce_sum_physical(x, axis=0)
        expected = np.sum(np_x, axis=0)
        
        # Result should be scalar-like
        assert_allclose(result, expected)
    
    def test_mean_single_element(self):
        """Mean of single-element tensor."""
        np_x = make_array(1, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = mean_physical(x, axis=0)
        expected = np.mean(np_x, axis=0)
        
        assert_allclose(result, expected)
    
    def test_squeeze_multiple_ones(self):
        """Squeeze when there are multiple dims of size 1."""
        np_x = make_array(1, 4, 1, 8, 1, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Squeeze only one axis at a time
        result = squeeze_physical(x, axis=0)
        expected = np.squeeze(np_x, axis=0)
        
        assert_shape(result, (4, 1, 8, 1))
        assert_allclose(result, expected)
    
    def test_broadcast_no_change(self):
        """Broadcast to same shape (no-op)."""
        np_x = make_array(4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = broadcast_to_physical(x, (4, 8))
        expected = np.broadcast_to(np_x, (4, 8))
        
        assert_shape(result, (4, 8))
        assert_allclose(result, expected)
