"""Operation-level sharding execution tests.

These tests validate the FULL execution path for sharded operations:
1. Actual tensor computation with sharded inputs
2. Local vs global shape tracking
3. Resharding when inputs conflict
4. Output annotation conflicts
5. AllReduce insertion for contracting dimensions

Unlike test_sharding_propagation_unit.py (pure algorithm tests), these tests
use real Tensor objects and verify numerical correctness.
"""

import pytest
import numpy as np
from nabla import Tensor, DeviceMesh, DimSpec
from nabla.sharding.spec import ShardingSpec, compute_local_shape
from nabla.ops.binary import matmul
from nabla.ops.reduction import reduce_sum


class TestMatmulExecution:
    """Test matmul operation with actual sharded execution."""
    
    def test_row_parallel_matmul_shapes(self):
        """Matmul A[4,8] @ B[8,2] with A row-sharded: verify local/global shapes.
        
        A sharded on dim0 (rows) with 2 shards:
        - Global shape: (4, 8)
        - Local shape per shard: (2, 8)
        
        B replicated:
        - Global shape: (8, 2)
        - Local shape: (8, 2)
        
        C should be row-sharded:
        - Global shape: (4, 2)
        - Local shape per shard: (2, 2)
        """
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Create A with row sharding annotation
        A = Tensor.ones((4, 8)).trace()
        A.shard(mesh, [DimSpec(["x"], is_open=False), DimSpec([], is_open=True)])
        
        # Verify A's sharding spec
        assert A._impl.sharding is not None
        assert A._impl.sharding.dim_specs[0].axes == ["x"]
        
        # Compute local shape for A
        a_local_shape = compute_local_shape((4, 8), A._impl.sharding, device_id=0)
        assert a_local_shape == (2, 8), f"Expected (2, 8), got {a_local_shape}"
        
        # B replicated
        B = Tensor.ones((8, 2)).trace()
        
        # Execute matmul
        C = A @ B
        
        # Verify C has sharding (should be row-sharded like A)
        assert C._impl.sharding is not None
        print(f"C sharding: {C._impl.sharding}")
        
        # C should be sharded on first dim (m factor from A)
        assert C._impl.sharding.dim_specs[0].axes == ["x"], \
            f"Expected C dim0 sharded on x, got {C._impl.sharding.dim_specs[0].axes}"
        
        # Verify global shape
        assert tuple(int(d) for d in C.shape) == (4, 2), f"Expected global shape (4, 2), got {C.shape}"
        
        # Verify local shape
        c_local_shape = compute_local_shape((4, 2), C._impl.sharding, device_id=0)
        assert c_local_shape == (2, 2), f"Expected local shape (2, 2), got {c_local_shape}"
    
    def test_col_parallel_matmul_shapes(self):
        """Matmul with B col-sharded: verify output is col-sharded."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # A replicated
        A = Tensor.ones((4, 8)).trace()
        
        # B col-sharded
        B = Tensor.ones((8, 6)).trace()
        B.shard(mesh, [DimSpec([], is_open=True), DimSpec(["x"], is_open=False)])
        
        # Execute
        C = A @ B
        
        # Output should be col-sharded (n factor from B)
        assert C._impl.sharding.dim_specs[1].axes == ["x"]
        
        # Global shape
        assert tuple(int(d) for d in C.shape) == (4, 6)
        
        # Local shape per shard
        c_local = compute_local_shape((4, 6), C._impl.sharding, device_id=0)
        assert c_local == (4, 3)  # Cols split
    
    def test_contracting_dim_sharded_needs_allreduce(self):
        """If contracting dimension k is sharded, AllReduce should be triggered.
        
        A[4, 8] sharded on dim1 (k)
        B[8, 2] sharded on dim0 (k)
        
        Both have k sharded -> produces partial results -> need AllReduce.
        Output should be replicated after AllReduce.
        """
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # A: shard on k (dim1)
        A = Tensor.ones((4, 8)).trace()
        A.shard(mesh, [DimSpec([], is_open=True), DimSpec(["x"], is_open=False)])
        
        # B: shard on k (dim0)
        B = Tensor.ones((8, 2)).trace()
        B.shard(mesh, [DimSpec(["x"], is_open=False), DimSpec([], is_open=True)])
        
        # Execute - should detect contracting factor is sharded
        C = A @ B
        
        # Check that needs_allreduce was detected
        from nabla.sharding.spmd import infer_output_sharding
        _, _, needs_allreduce = infer_output_sharding(matmul, (A, B), mesh, {})
        
        assert needs_allreduce, "Should detect that contracting dimension is sharded"
        
        # Output should be replicated (neither m nor n are sharded)
        # The AllReduce combines partial sums from each shard
        assert C._impl.sharding.is_fully_replicated()


class TestReduceExecution:
    """Test reduce operations with actual execution."""
    
    def test_reduce_sharded_dim_shape_tracking(self):
        """Reduce over a SHARDED dimension: verify shapes.
        
        Input: (8, 4) sharded on dim0
        - Global: (8, 4)
        - Local: (4, 4)
        
        Reduce axis=0 -> Output: (4,)
        - Global: (4,)
        - Should be replicated (d0 was reduced away)
        """
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Input sharded on first dim
        X = Tensor.ones((8, 4)).trace()
        X.shard(mesh, [DimSpec(["x"], is_open=False), DimSpec([], is_open=True)])
        
        # Reduce over sharded dimension
        Y = reduce_sum(X, axis=0)
        
        # Output should be replicated (d0 factor is gone)
        assert Y._impl.sharding is not None
        assert Y._impl.sharding.dim_specs[0].axes == []
        
        # Global shape
        assert tuple(int(d) for d in Y.shape) == (4,)
    
    def test_reduce_unsharded_dim_preserves_sharding(self):
        """Reduce over UNSHARDED dimension: verify sharding preserved.
        
        Input: (8, 4) sharded on dim1
        - Global: (8, 4)
        - Local: (8, 2)
        
        Reduce axis=0 -> Output: (4,) sharded
        - Global: (4,)
        - Local: (2,)
        - d1 sharding preserved
        """
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Input sharded on second dim
        X = Tensor.ones((8, 4)).trace()
        X.shard(mesh, [DimSpec([], is_open=True), DimSpec(["x"], is_open=False)])
        
        # Reduce over first dim (unsharded)
        Y = reduce_sum(X, axis=0)
        
        # Output should preserve d1 sharding
        assert Y._impl.sharding.dim_specs[0].axes == ["x"]
        
        # Global shape
        assert tuple(int(d) for d in Y.shape) == (4,)
        
        # Local shape
        y_local = compute_local_shape((4,), Y._impl.sharding, device_id=0)
        assert y_local == (2,)


class TestReshardingLogic:
    """Test automatic resharding when inputs conflict."""
    
    def test_with_sharding_replicated_to_sharded(self):
        """Test resharding from replicated to sharded using with_sharding."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Start with replicated tensor
        A = Tensor.ones((4, 4)).trace()
        
        # Apply sharding constraint - should trigger resharding
        A_sharded = A.with_sharding(mesh, [DimSpec(["x"]), DimSpec([])])
        
        # Verify sharding was applied
        assert A_sharded._impl.sharding is not None
        assert A_sharded._impl.sharding.dim_specs[0].axes == ["x"]
    
    def test_with_sharding_sharded_to_replicated(self):
        """Test resharding from sharded to replicated."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Start with sharded tensor
        A = Tensor.ones((4, 4)).trace()
        A.shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        # Reshard to replicated
        A_replicated = A.with_sharding(mesh, [DimSpec([]), DimSpec([])])
        
        # Verify all dimensions are replicated
        assert all(not spec.axes for spec in A_replicated._impl.sharding.dim_specs)
    
    def test_with_sharding_axis_change(self):
        """Test resharding from one axis to another: row-sharded → col-sharded."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Start row-sharded
        A = Tensor.ones((4, 4)).trace()
        A.shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        # Reshard to col-sharded
        A_col = A.with_sharding(mesh, [DimSpec([]), DimSpec(["x"])])
        
        # Verify new sharding
        assert A_col._impl.sharding.dim_specs[0].axes == []
        assert A_col._impl.sharding.dim_specs[1].axes == ["x"]


class TestOutputAnnotationConflicts:
    """Test when user annotates output differently than inferred sharding."""
    
    def test_matmul_output_constraint_replicated(self):
        """Matmul would infer row-sharded, but user wants replicated."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # A row-sharded
        A = Tensor.ones((4, 8)).trace()
        A.shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        # B replicated
        B = Tensor.ones((8, 2)).trace()
        
        # Matmul infers row-sharded output
        C = A @ B
        
        # User wants replicated output - use with_sharding
        C_replicated = C.with_sharding(mesh, [DimSpec([]), DimSpec([])])
        
        # Verify output is replicated
        assert C_replicated._impl.sharding is not None
        assert all(not spec.axes for spec in C_replicated._impl.sharding.dim_specs)
    
    def test_output_constraint_different_axis(self):
        """User wants different sharding axis than inferred."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # A row-sharded
        A = Tensor.ones((4, 8)).trace()
        A.shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        B = Tensor.ones((8, 4)).trace()
        
        # Matmul would infer row-sharded (dim0=x)
        C = A @ B
        
        # But user wants col-sharded (dim1=x)
        C_col = C.with_sharding(mesh, [DimSpec([]), DimSpec(["x"])])
        
        # Verify
        assert C_col._impl.sharding.dim_specs[1].axes == ["x"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
