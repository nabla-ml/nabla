# ===----------------------------------------------------------------------=== #
# Nabla 2025
# ===----------------------------------------------------------------------=== #
"""Tests for multi-output operations with sharding.

Multi-output ops:
- split: Split tensor into multiple parts
- chunk: Split tensor into chunks
- unbind: Unbind along an axis

All tested with:
- Basic functionality
- Sharding on the split axis
- Sharding on non-split axes
- vmap integration
"""

import pytest
import numpy as np

import nabla
from nabla import DeviceMesh, P, vmap
from nabla.ops.multi_output import split, chunk, unbind

from .conftest import (
    make_array, tensor_from_numpy, to_numpy,
    assert_allclose, assert_shape, assert_is_sharded,
)


# =============================================================================
# Test SplitOp
# =============================================================================

class TestSplitOp:
    """Test split operation - split tensor into multiple parts."""
    
    def test_split_basic(self):
        """Basic split without sharding."""
        np_x = make_array(16, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Split into 2 parts along axis 0
        results = split(x, num_splits=2, axis=0)
        expected = np.split(np_x, 2, axis=0)
        
        assert len(results) == 2
        for r, e in zip(results, expected):
            assert_shape(r, e.shape)
            assert_allclose(r, e)
    
    def test_split_axis1(self):
        """Split along axis 1."""
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Split into 4 parts along axis 1
        results = split(x, num_splits=4, axis=1)
        expected = np.split(np_x, 4, axis=1)
        
        assert len(results) == 4
        for r, e in zip(results, expected):
            assert_shape(r, (8, 4))
            assert_allclose(r, e)
    
    def test_split_sharded_non_split_axis(self, mesh_1d):
        """Split with sharding on non-split axis."""
        np_x = make_array(16, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard on axis 0
        x_sharded = x.shard(mesh_1d, P("dp", None))
        
        # Split along axis 1 (non-sharded axis)
        results = split(x_sharded, num_splits=2, axis=1)
        expected = np.split(np_x, 2, axis=1)
        
        assert len(results) == 2
        for r, e in zip(results, expected):
            assert_shape(r, (16, 4))
            assert_allclose(r, e)
            # Should preserve sharding on axis 0
            assert_is_sharded(r, True)
    
    def test_split_sharded_split_axis(self, mesh_1d):
        """Split with sharding on the split axis itself."""
        np_x = make_array(16, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard on axis 0
        x_sharded = x.shard(mesh_1d, P("dp", None))
        
        # Split along axis 0 (the sharded axis)
        results = split(x_sharded, num_splits=2, axis=0)
        expected = np.split(np_x, 2, axis=0)
        
        assert len(results) == 2
        for r, e in zip(results, expected):
            assert_shape(r, (8, 8))
            assert_allclose(r, e)
    
    def test_split_with_vmap(self):
        """Split inside vmap."""
        batch = 4
        np_x = make_array(batch, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        def f(row):  # row: (16,)
            return split(row, num_splits=2, axis=0)
        
        results = vmap(f)(x)
        
        # Results is a tuple of (batch, 8) tensors
        assert len(results) == 2
        for r in results:
            assert_shape(r, (batch, 8))
        
        expected_parts = np.split(np_x, 2, axis=1)
        for r, e in zip(results, expected_parts):
            assert_allclose(r, e)
    
    def test_split_vmap_with_sharding(self, mesh_2x4):
        """Split inside vmap with spmd_axis_name + logical sharding."""
        batch = 8
        mesh = mesh_2x4
        
        np_x = make_array(batch, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        @vmap(spmd_axis_name="dp")
        def f(row):  # row: (16,)
            row_sharded = row.shard(mesh, P("tp"))
            return split(row_sharded, num_splits=2, axis=0)
        
        results = f(x)
        
        assert len(results) == 2
        for r in results:
            assert_shape(r, (batch, 8))
            # Each should be sharded on both dp and tp
            spec = r._impl.sharding
            if spec:
                assert "dp" in spec.dim_specs[0].axes
                assert "tp" in spec.dim_specs[1].axes


# =============================================================================
# Test ChunkOp
# =============================================================================

class TestChunkOp:
    """Test chunk operation - split into chunks (may have uneven last chunk)."""
    
    def test_chunk_basic(self):
        """Basic chunk without sharding."""
        np_x = make_array(10, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Chunk into 3 parts along axis 0
        results = chunk(x, chunks=3, axis=0)
        expected = np.array_split(np_x, 3, axis=0)
        
        assert len(results) == 3
        for r, e in zip(results, expected):
            assert_shape(r, e.shape)
            assert_allclose(r, e)
    
    def test_chunk_axis1(self):
        """Chunk along axis 1."""
        np_x = make_array(8, 10, seed=42)
        x = tensor_from_numpy(np_x)
        
        results = chunk(x, chunks=4, axis=1)
        expected = np.array_split(np_x, 4, axis=1)
        
        assert len(results) == 4
        for r, e in zip(results, expected):
            assert_allclose(r, e)
    
    def test_chunk_sharded(self, mesh_1d):
        """Chunk with sharding."""
        np_x = make_array(12, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        x_sharded = x.shard(mesh_1d, P("dp", None))
        
        results = chunk(x_sharded, chunks=3, axis=0)
        expected = np.array_split(np_x, 3, axis=0)
        
        assert len(results) == 3
        for r, e in zip(results, expected):
            assert_allclose(r, e)
    
    def test_chunk_with_vmap(self):
        """Chunk inside vmap."""
        batch = 4
        np_x = make_array(batch, 10, seed=42)
        x = tensor_from_numpy(np_x)
        
        def f(row):
            return chunk(row, chunks=3, axis=0)
        
        results = vmap(f)(x)
        
        assert len(results) == 3
        expected_parts = [np.array_split(np_x, 3, axis=1)[i] for i in range(3)]
        for r, e in zip(results, expected_parts):
            assert_allclose(r, e)


# =============================================================================
# Test UnbindOp
# =============================================================================

class TestUnbindOp:
    """Test unbind operation - unbind along an axis."""
    
    def test_unbind_basic(self):
        """Basic unbind without sharding."""
        np_x = make_array(4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Unbind along axis 0
        results = unbind(x, axis=0)
        
        # Should get 4 tensors of shape (8,)
        assert len(results) == 4
        for i, r in enumerate(results):
            assert_shape(r, (8,))
            assert_allclose(r, np_x[i])
    
    def test_unbind_axis1(self):
        """Unbind along axis 1."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        results = unbind(x, axis=1)
        
        assert len(results) == 4
        for i, r in enumerate(results):
            assert_shape(r, (8,))
            assert_allclose(r, np_x[:, i])
    
    def test_unbind_3d(self):
        """Unbind 3D tensor."""
        np_x = make_array(2, 4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        results = unbind(x, axis=1)
        
        assert len(results) == 4
        for i, r in enumerate(results):
            assert_shape(r, (2, 8))
            assert_allclose(r, np_x[:, i, :])
    
    def test_unbind_sharded(self, mesh_1d):
        """Unbind with sharding."""
        np_x = make_array(4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard on axis 1
        x_sharded = x.shard(mesh_1d, P(None, "dp"))
        
        # Unbind along axis 0 (non-sharded)
        results = unbind(x_sharded, axis=0)
        
        assert len(results) == 4
        for i, r in enumerate(results):
            assert_shape(r, (8,))
            assert_allclose(r, np_x[i])
            # Should preserve sharding
            assert_is_sharded(r, True)
    
    def test_unbind_with_vmap(self):
        """Unbind inside vmap."""
        batch = 4
        np_x = make_array(batch, 3, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        def f(batch_x):  # batch_x: (3, 8)
            return unbind(batch_x, axis=0)
        
        results = vmap(f)(x)
        
        # Results is a tuple of 3 tensors, each (batch, 8)
        assert len(results) == 3
        for i, r in enumerate(results):
            assert_shape(r, (batch, 8))
            assert_allclose(r, np_x[:, i, :])


# =============================================================================
# Edge Cases
# =============================================================================

class TestMultiOutputEdgeCases:
    """Test edge cases for multi-output ops."""
    
    def test_split_single_part(self):
        """Split into single part (essentially a no-op)."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        results = split(x, num_splits=1, axis=0)
        
        assert len(results) == 1
        assert_allclose(results[0], np_x)
    
    def test_chunk_more_chunks_than_elements(self):
        """Chunk with more chunks than elements."""
        np_x = make_array(2, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        results = chunk(x, chunks=5, axis=0)
        
        # Should get 2 chunks (can't split into more than we have)
        assert len(results) == 2
        for i, r in enumerate(results):
            assert_allclose(r, np_x[i:i+1])
    
    def test_unbind_single_element(self):
        """Unbind along dimension of size 1."""
        np_x = make_array(1, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        results = unbind(x, axis=0)
        
        assert len(results) == 1
        assert_shape(results[0], (8,))
        assert_allclose(results[0], np_x[0])


__all__ = [
    "TestSplitOp",
    "TestChunkOp",
    "TestUnbindOp",
    "TestMultiOutputEdgeCases",
]
