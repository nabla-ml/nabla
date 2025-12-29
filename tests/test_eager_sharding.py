#!/usr/bin/env python3
"""Tests for eager sharding propagation.

These tests validate the new eager sharding architecture where:
1. shard() immediately creates per-shard TensorValues
2. Operations propagate sharding at call-time (in __call__)
3. Collectives are inserted when shardings conflict

Run with: python -m pytest tests/test_eager_sharding.py -v
"""

import pytest
from nabla import (
    Tensor,
    DeviceMesh,
    DimSpec,
    ShardingSpec,
    add,
    matmul,
)
from nabla.ops.communication import shard, all_gather, all_reduce


class TestShardOperation:
    """Test that shard() creates correct multi-value tensors."""
    
    def test_shard_creates_multiple_values(self):
        """shard() should create N TensorValues for N devices."""
        mesh = DeviceMesh("m", (2,), ("x",))
        x = Tensor.ones((4, 8))
        sharded = shard(x, mesh, [DimSpec(["x"]), DimSpec([])])
        
        # Should have 2 TensorValues (one per device)
        assert len(sharded._impl._values) == 2
        assert sharded._impl.is_sharded
        assert sharded._impl.sharding is not None
    
    def test_shard_spec_attached(self):
        """ShardingSpec should be attached to sharded tensor."""
        mesh = DeviceMesh("m", (2,), ("x",))
        x = Tensor.ones((4, 8))
        sharded = shard(x, mesh, [DimSpec(["x"]), DimSpec([])])
        
        spec = sharded._impl.sharding
        assert isinstance(spec, ShardingSpec)
        assert spec.mesh.name == "m"
        assert spec.dim_specs[0].axes == ["x"]
        assert spec.dim_specs[1].axes == []


class TestBinaryOperationsSharded:
    """Test binary operations with sharded inputs."""
    
    def test_sharded_plus_sharded(self):
        """Sharded + Sharded → Sharded output."""
        mesh = DeviceMesh("m", (2,), ("x",))
        
        a = shard(Tensor.ones((4, 8)), mesh, [DimSpec(["x"]), DimSpec([])])
        b = shard(Tensor.ones((4, 8)), mesh, [DimSpec(["x"]), DimSpec([])])
        
        c = a + b
        
        assert c._impl.is_sharded
        assert len(c._impl._values) == 2
        assert c._impl.sharding is not None
    
    def test_sharded_plus_replicated(self):
        """Sharded + Replicated → Sharded output."""
        mesh = DeviceMesh("m", (2,), ("x",))
        
        # a is sharded
        a = shard(Tensor.ones((4, 8)), mesh, [DimSpec(["x"]), DimSpec([])])
        # b is NOT sharded, but gets replicated sharding assigned
        b = Tensor.ones((4, 8))
        
        c = a + b
        
        assert c._impl.is_sharded
        assert len(c._impl._values) == 2


class TestShardingPropagation:
    """Test that sharding propagates correctly through operations."""
    
    def test_elementwise_propagation(self):
        """Sharding should propagate through elementwise ops."""
        mesh = DeviceMesh("m", (2,), ("x",))
        
        a = shard(Tensor.ones((4, 8)), mesh, [DimSpec(["x"]), DimSpec([])])
        b = a + a
        c = b * Tensor.constant(2.0)
        
        # All should be sharded on first dim
        assert b._impl.is_sharded
        assert c._impl.is_sharded
        assert c._impl.sharding.dim_specs[0].axes == ["x"]


class TestTensorShardMethod:
    """Test the Tensor.shard() method for inline sharding annotation."""
    
    def test_shard_method_returns_self(self):
        """Tensor.shard() should return self for chaining."""
        mesh = DeviceMesh("test", (2,), ("x",))
        t = Tensor.ones((4, 8))
        result = t.shard(mesh, [DimSpec(["x"]), DimSpec([])])
        assert result is t
    
    def test_shard_method_sets_sharding(self):
        """Tensor.shard() should set the sharding spec."""
        mesh = DeviceMesh("test", (2,), ("x",))
        t = Tensor.ones((4, 8))
        t.shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        assert t._impl.sharding is not None
        assert t._impl.sharding.mesh.name == "test"
        assert t._impl.sharding.dim_specs[0].axes == ["x"]


class TestCollectivesInterface:
    """Test that collective ops have correct interface."""
    
    def test_collective_op_names(self):
        """Collective ops should have correct names."""
        from nabla.ops.communication import ShardOp, AllGatherOp, AllReduceOp, ReduceScatterOp
        
        assert AllGatherOp().name == "all_gather"
        assert AllReduceOp().name == "all_reduce"
        assert ReduceScatterOp().name == "reduce_scatter"


class TestAllGather:
    """Test all_gather collective operation."""
    
    def test_all_gather_returns_replicated(self):
        """all_gather should convert sharded to replicated."""
        mesh = DeviceMesh("m", (2,), ("x",))
        
        # Create sharded tensor
        a = shard(Tensor.ones((4, 8)), mesh, [DimSpec(["x"]), DimSpec([])])
        assert len(a._impl._values) == 2
        
        # Gather to replicated
        gathered = all_gather(a, axis=0)
        
        # Result should have same number of values (one per device)
        assert len(gathered._impl._values) == 2
        # Sharding should be replicated (all axes empty)
        assert gathered._impl.sharding is not None
        assert gathered._impl.sharding.is_fully_replicated()


class TestInputAlignment:
    """Test that inputs with conflicting shardings are aligned."""
    
    def test_replicated_aligned_to_sharded(self):
        """Replicated input should be sliced to match sharded input."""
        mesh = DeviceMesh("m", (2,), ("x",))
        
        # a is sharded on first dim
        a = shard(Tensor.ones((4, 8)), mesh, [DimSpec(["x"]), DimSpec([])])
        # b is replicated - different sharding
        b = Tensor.ones((4, 8))
        
        # This should succeed - b gets sliced to match a's sharding
        c = a + b
        
        assert c._impl.is_sharded
        assert c._impl.sharding.dim_specs[0].axes == ["x"]
    
    def test_sharded_chain(self):
        """Chain of sharded operations should maintain sharding."""
        mesh = DeviceMesh("m", (2,), ("x",))
        
        a = shard(Tensor.ones((4, 8)), mesh, [DimSpec(["x"]), DimSpec([])])
        b = a + a
        c = b + a
        d = c + Tensor.ones((4, 8))  # Mix with replicated
        
        # All intermediate results should be sharded
        assert b._impl.is_sharded
        assert c._impl.is_sharded
        assert d._impl.is_sharded


class TestResharding:
    """Test resharding between different sharding specifications."""
    
    def test_needs_reshard_same_spec(self):
        """Same sharding spec should not need resharding."""
        from nabla.ops.binary import AddOp
        
        mesh = DeviceMesh("m", (2,), ("x",))
        spec1 = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
        spec2 = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
        
        op = AddOp()
        assert not op._needs_reshard(spec1, spec2)
    
    def test_needs_reshard_different_axes(self):
        """Different sharding axes should need resharding."""
        from nabla.ops.binary import AddOp
        
        mesh = DeviceMesh("m", (2,), ("x",))
        spec1 = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
        spec2 = ShardingSpec(mesh, [DimSpec([]), DimSpec(["x"])])
        
        op = AddOp()
        assert op._needs_reshard(spec1, spec2)
    
    def test_needs_reshard_none_vs_spec(self):
        """None vs spec should need resharding."""
        from nabla.ops.binary import AddOp
        
        mesh = DeviceMesh("m", (2,), ("x",))
        spec = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
        
        op = AddOp()
        assert op._needs_reshard(None, spec)
        assert op._needs_reshard(spec, None)


class TestNumericalCorrectness:
    """Test that sharded operations produce numerically identical results."""
    
    def test_add_sharded_matches_unsharded(self):
        """Sharded addition should match unsharded result after gather."""
        import numpy as np
        
        mesh = DeviceMesh("m", (2,), ("x",))
        
        # Create input arrays
        a_data = np.arange(8, dtype=np.float32).reshape(4, 2)
        b_data = np.ones((4, 2), dtype=np.float32) * 10
        
        # Unsharded computation
        a_unsharded = Tensor.from_dlpack(a_data.copy())
        b_unsharded = Tensor.from_dlpack(b_data.copy())
        c_unsharded = a_unsharded + b_unsharded
        c_unsharded._sync_realize()
        expected = c_unsharded._impl._storages[0].to_numpy()
        
        # Sharded computation
        a_sharded = shard(Tensor.from_dlpack(a_data.copy()), mesh, [DimSpec(["x"]), DimSpec([])])
        b_sharded = shard(Tensor.from_dlpack(b_data.copy()), mesh, [DimSpec(["x"]), DimSpec([])])
        c_sharded = a_sharded + b_sharded
        
        # Gather and verify
        c_gathered = all_gather(c_sharded, axis=0)
        c_gathered._sync_realize()
        result = c_gathered._impl._storages[0].to_numpy()
        
        assert np.allclose(result, expected), f"Sharded result {result} != expected {expected}"
    
    def test_mul_sharded_matches_unsharded(self):
        """Sharded multiplication should match unsharded result."""
        import numpy as np
        
        mesh = DeviceMesh("m", (2,), ("x",))
        
        # Create input
        data = np.arange(8, dtype=np.float32).reshape(4, 2)
        
        # Unsharded: x * 2
        x_un = Tensor.from_dlpack(data.copy())
        y_un = x_un * Tensor.constant(2.0)
        y_un._sync_realize()
        expected = y_un._impl._storages[0].to_numpy()
        
        # Sharded: shard(x) * 2
        x_sh = shard(Tensor.from_dlpack(data.copy()), mesh, [DimSpec(["x"]), DimSpec([])])
        y_sh = x_sh * Tensor.constant(2.0)
        y_gathered = all_gather(y_sh, axis=0)
        y_gathered._sync_realize()
        result = y_gathered._impl._storages[0].to_numpy()
        
        assert np.allclose(result, expected), f"Sharded result {result} != expected {expected}"


class TestMatmulSharding:
    """Test matmul with sharded inputs.
    
    These tests require per-input factor-based slicing: B's dims map to factors
    k,n while A maps to m,k. When A is row-sharded (factor m), B shouldn't be
    sliced because it doesn't have factor m on any dim.
    """
    

    def test_matmul_row_sharded_a(self):
        """Matmul with A sharded on rows -> output sharded on rows."""
        mesh = DeviceMesh("m", (2,), ("x",))
        
        # A: (4, 2) sharded on dim 0 (rows)
        # B: (2, 3) replicated
        # C = A @ B: (4, 3) should be sharded on dim 0
        a = shard(Tensor.ones((4, 2)), mesh, [DimSpec(["x"]), DimSpec([])])
        b = Tensor.ones((2, 3))
        
        c = matmul(a, b)
        
        assert c._impl.is_sharded
        assert len(c._impl._values) == 2
        # Output sharded on rows (dim 0)
        assert c._impl.sharding.dim_specs[0].axes == ["x"]
        assert c._impl.sharding.dim_specs[1].axes == []
    

    def test_matmul_col_sharded_b(self):
        """Matmul with B sharded on cols -> output sharded on cols."""
        mesh = DeviceMesh("m", (2,), ("x",))
        
        # A: (4, 2) replicated
        # B: (2, 6) sharded on dim 1 (cols)
        # C = A @ B: (4, 6) should be sharded on dim 1
        a = Tensor.ones((4, 2))
        b = shard(Tensor.ones((2, 6)), mesh, [DimSpec([]), DimSpec(["x"])])
        
        c = matmul(a, b)
        
        assert c._impl.is_sharded
        assert len(c._impl._values) == 2
        # Output sharded on cols (dim 1)
        assert c._impl.sharding.dim_specs[1].axes == ["x"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

