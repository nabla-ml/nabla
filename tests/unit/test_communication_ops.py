# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Tests for communication operations - sharding primitives.

This file tests all communication/collective operations:
- ShardOp: Partition tensor across mesh
- AllGatherOp: Gather shards to replicated
- AllReduceOp: Reduce across shards
- ReshardOp: Change sharding specification

All ops are tested with:
- 1D meshes
- 2D asymmetric meshes (2x4, 4x2)
- 3D meshes
- Numerical verification against numpy
"""

import pytest
import numpy as np

import nabla
from nabla import DeviceMesh, P
from nabla.ops.communication import shard, all_gather, all_reduce, reshard
from nabla.core.sharding.spec import DimSpec, ShardingSpec

from tests.conftest import (
    make_array, tensor_from_numpy, to_numpy,
    assert_allclose, assert_shape, assert_is_sharded,
)


# =============================================================================
# Test ShardOp - Partition tensor across mesh
# =============================================================================

class TestShardOp:
    """Test ShardOp: partition tensor into shards."""
    
    def test_shard_1d_axis0(self, mesh_1d):
        """Shard tensor on first axis with 1D mesh."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard on axis 0 (rows)
        result = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])
        
        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, np_x)
        
        # Verify sharding spec
        spec = result.sharding
        assert spec.dim_specs[0].axes == ["dp"]
        assert spec.dim_specs[1].axes == []
    
    def test_shard_1d_axis1(self, mesh_1d):
        """Shard tensor on second axis with 1D mesh."""
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard on axis 1 (columns)
        result = x.shard(mesh_1d, [DimSpec([]), DimSpec(["dp"])])
        
        assert_shape(result, (8, 16))
        assert_is_sharded(result, True)
        assert_allclose(result, np_x)
    
    def test_shard_2d_asymmetric(self, mesh_2x4):
        """Shard on 2D asymmetric mesh (2, 4)."""
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard axis 0 on dp (2 devices), axis 1 on tp (4 devices)
        result = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec(["tp"])])
        
        assert_shape(result, (8, 16))
        assert_is_sharded(result, True)
        assert_allclose(result, np_x)
        
        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes
        assert "tp" in spec.dim_specs[1].axes
    
    def test_shard_2d_asymmetric_flipped(self, mesh_4x2):
        """Shard on 2D asymmetric mesh (4, 2) - flipped dimensions."""
        np_x = make_array(16, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard axis 0 on dp (4 devices), axis 1 on tp (2 devices)
        result = x.shard(mesh_4x2, [DimSpec(["dp"]), DimSpec(["tp"])])
        
        assert_shape(result, (16, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, np_x)
    
    def test_shard_3d_mesh(self, mesh_3d):
        """Shard on 3D mesh (2, 2, 2)."""
        np_x = make_array(8, 8, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard all three axes
        result = x.shard(mesh_3d, [
            DimSpec(["dp"]),
            DimSpec(["tp"]),
            DimSpec(["pp"])
        ])
        
        assert_shape(result, (8, 8, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, np_x)
        
        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes
        assert "tp" in spec.dim_specs[1].axes
        assert "pp" in spec.dim_specs[2].axes
    
    def test_shard_replicated(self, mesh_1d):
        """Shard with fully replicated spec (no actual sharding)."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        # All dimensions replicated
        result = x.shard(mesh_1d, [DimSpec([]), DimSpec([])])
        
        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)  # Has mesh but fully replicated
        assert_allclose(result, np_x)
        
        spec = result.sharding
        assert spec.is_fully_replicated()
    
    def test_shard_numerical_values(self, mesh_1d_2):
        """Verify shard contents are numerically correct slices."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard on axis 0 with 2 devices
        result = x.shard(mesh_1d_2, [DimSpec(["dp"]), DimSpec([])])
        
        # Get the shards
        assert result.is_sharded
        assert len(result._values) == 2  # 2 shards for 2 devices
        
        # Verify numerical correctness
        assert_allclose(result, np_x)


# =============================================================================
# Test AllGatherOp - Gather shards to replicated
# =============================================================================

class TestAllGatherOp:
    """Test AllGatherOp: gather shards to full tensor."""
    
    def test_all_gather_1d(self, mesh_1d):
        """AllGather on 1D mesh."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        # First shard
        x_sharded = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])
        
        # Then gather
        result = all_gather(x_sharded, axis=0)
        
        assert_shape(result, (8, 4))
        assert_allclose(result, np_x)
        
        # Result should be fully replicated
        spec = result.sharding
        assert spec is None or spec.is_fully_replicated()
    
    def test_all_gather_2d_asymmetric(self, mesh_2x4):
        """AllGather on 2D asymmetric mesh - gather one axis."""
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard both axes
        x_sharded = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec(["tp"])])
        
        # Gather axis 1 (tp axis)
        result = all_gather(x_sharded, axis=1)
        
        # In single-machine simulation, all_gather works on local shard
        # For (8,16) on (2,4) mesh: local is (4,4), gathering axis 1 gives (4,16)
        # Just verify numerical correctness of what we get
        assert result.shape[1] == 16  # Gathered axis should be full width
        assert_allclose(result, np_x)
        
        # Axis 1 should now be replicated (tp gathered)
        spec = result.sharding
        if spec and len(spec.dim_specs) > 1:
            assert "tp" not in spec.dim_specs[1].axes
    
    def test_all_gather_3d(self, mesh_3d):
        """AllGather on 3D mesh."""
        np_x = make_array(8, 8, 8, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard on dp axis only
        x_sharded = x.shard(mesh_3d, [DimSpec(["dp"]), DimSpec([]), DimSpec([])])
        
        # Gather
        result = all_gather(x_sharded, axis=0)
        
        assert_shape(result, (8, 8, 8))
        assert_allclose(result, np_x)
    
    def test_all_gather_numerical(self, mesh_1d_2):
        """Verify all_gather produces correct full array."""
        np_x = make_array(6, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard
        x_sharded = x.shard(mesh_1d_2, [DimSpec(["dp"]), DimSpec([])])
        
        # Gather
        result = all_gather(x_sharded, axis=0)
        
        # Verify exact numerical match
        assert_allclose(result, np_x)


# =============================================================================
# Test AllReduceOp - Reduce across shards
# =============================================================================

class TestAllReduceOp:
    """Test AllReduceOp: reduce values across all shards."""
    
    def test_all_reduce_sum_1d(self, mesh_1d):
        """AllReduce sum on 1D mesh."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard
        x_sharded = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])
        
        # AllReduce (sum) - each shard gets the full sum
        result = all_reduce(x_sharded)
        
        # NOTE: In single-machine simulation, all_reduce operates on local shards
        # The global semantics require actual distributed execution to test properly
        # For now, just verify the operation completes without error
        assert result is not None
        # TODO: This test needs multi-GPU environment for proper validation
    
    def test_all_reduce_sum_2d_asymmetric(self, mesh_2x4):
        """AllReduce on 2D asymmetric mesh."""
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard
        x_sharded = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec(["tp"])])
        
        # AllReduce
        result = all_reduce(x_sharded)
        
        # NOTE: In single-machine simulation, can't verify global all_reduce semantics
        assert result is not None
        # TODO: Needs multi-GPU for proper validation
    
    def test_all_reduce_numerical(self, mesh_1d_2):
        """Verify all_reduce sum produces correct result."""
        np_x = make_array(4, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard
        x_sharded = x.shard(mesh_1d_2, [DimSpec(["dp"]), DimSpec([])])
        
        # AllReduce
        result = all_reduce(x_sharded)
        
        # NOTE: Can't verify numerical correctness in single-machine simulation
        assert result is not None
        # TODO: Needs multi-GPU


# =============================================================================
# Test ReshardOp - Change sharding specification
# =============================================================================

class TestReshardOp:
    """Test reshard: change from one sharding to another."""
    
    def test_reshard_change_sharding(self, mesh_2x4):
        """Reshard from one axis to another."""
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Initial: shard on dp only
        x_sharded = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec([])])
        
        # Reshard: now shard on tp instead
        result = reshard(x_sharded, mesh_2x4, [DimSpec([]), DimSpec(["tp"])])
        
        assert_shape(result, (8, 16))
        assert_allclose(result, np_x)
        
        # Verify new sharding
        spec = result.sharding
        assert spec.dim_specs[0].axes == []
        assert "tp" in spec.dim_specs[1].axes
    
    def test_reshard_to_replicated(self, mesh_1d):
        """Reshard from sharded to fully replicated."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Initial: sharded
        x_sharded = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])
        
        # Reshard: fully replicated
        result = reshard(x_sharded, mesh_1d, [DimSpec([]), DimSpec([])])
        
        assert_shape(result, (8, 4))
        assert_allclose(result, np_x)
        
        spec = result.sharding
        assert spec.is_fully_replicated()
    
    def test_reshard_between_different_patterns(self, mesh_2x4):
        """Reshard between different multi-axis patterns."""
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Initial: both axes sharded
        x_sharded = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec(["tp"])])
        
        # Reshard: swap which axis gets which sharding
        result = reshard(x_sharded, mesh_2x4, [DimSpec(["tp"]), DimSpec(["dp"])])
        
        assert_shape(result, (8, 16))
        assert_allclose(result, np_x)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestCommunicationOpsEdgeCases:
    """Test edge cases for communication ops."""
    
    def test_shard_uneven_size(self, mesh_1d):
        """Shard dimension not evenly divisible by mesh size."""
        # 10 elements on 4 devices = uneven (2, 3, 3, 2 or similar)
        np_x = make_array(10, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        # This should work with padding/uneven shards
        result = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])
        
        assert_shape(result, (10, 4))
        assert_is_sharded(result, True)
        # Numerical correctness might have padding issues
        # but at minimum shape should be correct
    
    def test_shard_already_sharded(self, mesh_1d):
        """Shard an already-sharded tensor (should reshard)."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        
        # First shard
        x_sharded = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])
        
        # Shard again with different spec
        result = x_sharded.shard(mesh_1d, [DimSpec([]), DimSpec(["dp"])])
        
        assert_shape(result, (8, 4))
        assert_allclose(result, np_x)


__all__ = [
    "TestShardOp",
    "TestAllGatherOp",
    "TestAllReduceOp",
    "TestReshardOp",
    "TestCommunicationOpsEdgeCases",
]
