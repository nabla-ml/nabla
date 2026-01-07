"""Numerical verification tests for sharding operations.

These tests verify that sharding operations produce correct structure results
by checking shapes, shard counts, and metadata.
"""
import pytest
import numpy as np
from nabla.core.tensor import Tensor
from nabla.sharding.spec import DeviceMesh, DimSpec, ShardingSpec
from nabla.ops.communication import shard_op, reshard, all_gather


@pytest.fixture
def mesh():
    """4-device mesh for testing."""
    return DeviceMesh("test_mesh", (4,), ("dp",))


@pytest.fixture
def mesh_2d():
    """2x2 device mesh for 2D sharding tests."""
    return DeviceMesh("test_mesh_2d", (2, 2), ("dp", "mp"))


class TestShardNumericalCorrectness:
    """Test that ShardOp produces correct shard structure."""
    
    def test_shard_dim0_values(self, mesh):
        """Verify sharding dim 0 produces correct number of shards."""
        # Create tensor with shape (4, 4)
        x = Tensor.normal((4, 4))
        
        # Shard on dim 0 (4 shards of shape (1, 4))
        y = shard_op(x, mesh, [DimSpec(["dp"]), DimSpec([])])
        
        # Verify we have 4 shard values
        values = y._impl._values
        assert len(values) == 4, f"Expected 4 shards, got {len(values)}"
        
        # Each shard should have shape (1, 4)
        for i, val in enumerate(values):
            shard_shape = tuple(val.type.shape)
            assert shard_shape == (1, 4), f"Shard {i} has shape {shard_shape}, expected (1, 4)"
    
    def test_shard_then_gather_roundtrip(self, mesh):
        """Shard then gather should produce replicated tensor."""
        x = Tensor.normal((4, 4))
        
        # Shard on dim 0
        y = shard_op(x, mesh, [DimSpec(["dp"]), DimSpec([])])
        
        # Gather back
        z = all_gather(y, axis=0)
        
        # Verify sharding is now replicated on dim 0
        assert z._impl.sharding is not None
        assert z._impl.sharding.dim_specs[0].is_replicated(), "Dim 0 should be replicated after gather"
        
        # Shape should match original
        assert tuple(z.shape) == (4, 4), f"Shape mismatch: {z.shape} vs (4, 4)"


class TestReshardNumericalCorrectness:
    """Test that ReshardOp produces correct values across sharding changes."""
    
    def test_reshard_replicated_to_sharded(self, mesh):
        """Reshard from replicated to sharded should produce correct shards."""
        x = Tensor.normal((4, 2))
        
        # Reshard to shard dim 0
        y = reshard(x, mesh, [DimSpec(["dp"]), DimSpec([])])
        
        # Verify sharding spec
        assert y._impl.sharding is not None
        assert y._impl.sharding.dim_specs[0].axes == ["dp"]
        
        # Verify we have 4 shard values  
        assert len(y._impl._values) == 4
    
    def test_reshard_axis_change(self, mesh):
        """Changing sharding axis should work correctly."""
        x = Tensor.normal((4, 4))
        
        # Shard on dim 0
        y = shard_op(x, mesh, [DimSpec(["dp"]), DimSpec([])])
        
        # Reshard to dim 1 (should gather dim 0, shard dim 1)
        z = reshard(y, mesh, [DimSpec([]), DimSpec(["dp"])])
        
        # Verify new sharding
        assert z._impl.sharding is not None
        assert z._impl.sharding.dim_specs[0].is_replicated(), "Dim 0 should be replicated"
        assert z._impl.sharding.dim_specs[1].axes == ["dp"], "Dim 1 should be sharded on dp"
    
    def test_reshard_noop(self, mesh):
        """Resharding to same spec should return same tensor."""
        x = Tensor.normal((4, 2))
        
        # Shard dim 0
        y = shard_op(x, mesh, [DimSpec(["dp"]), DimSpec([])])
        
        # Reshard to same spec
        z = reshard(y, mesh, [DimSpec(["dp"]), DimSpec([])])
        
        # Should be identical object (no-op optimization)
        assert z is y, "No-op reshard should return same tensor"


class TestBatchDimsNumericalCorrectness:
    """Test sharding with batch dimensions."""
    
    def test_batch_dims_sharding(self, mesh):
        """Sharding a tensor with batch_dims should work correctly."""
        x = Tensor.normal((4, 8))
        
        # Simulate vmap by setting batch_dims
        x._impl.batch_dims = 1
        
        # Shard logical dim 0 (physical dim 1) - user provides logical spec
        y = reshard(x, mesh, [DimSpec(["dp"])])
        
        # Verify physical spec has 2 dims
        assert len(y._impl.sharding.dim_specs) == 2, "Should have 2 dim specs (batch + logical)"
        assert y._impl.sharding.dim_specs[0].is_replicated(), "Batch dim should be replicated"
        assert y._impl.sharding.dim_specs[1].axes == ["dp"], "Logical dim should be sharded"

