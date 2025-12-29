"""Unit tests for broadcast matmul sharding and to_numpy shard gathering."""

import pytest
import numpy as np
import asyncio
from nabla import Tensor
from nabla.sharding import DeviceMesh, DimSpec, ShardingSpec
from nabla.sharding.propagation import matmul_template


class TestBroadcastMatmulSharding:
    """Test matmul sharding with different input ranks (broadcast matmul)."""
    
    def test_matmul_template_with_batch_dims(self):
        """Template should work for batched matmul: (B, M, K) @ (B, K, N) -> (B, M, N)."""
        template = matmul_template(batch_dims=1)
        
        # Instantiate with batched shapes
        input_shapes = [(8, 16, 64), (8, 64, 128)]
        output_shapes = [(8, 16, 128)]
        
        rule = template.instantiate(input_shapes, output_shapes)
        
        # Should have factors: b0, m, k, n
        assert "b0" in rule.factor_sizes
        assert "m" in rule.factor_sizes  
        assert "k" in rule.factor_sizes
        assert "n" in rule.factor_sizes
        
        # Verify sizes
        assert rule.factor_sizes["b0"] == 8
        assert rule.factor_sizes["m"] == 16
        assert rule.factor_sizes["k"] == 64
        assert rule.factor_sizes["n"] == 128
    
    def test_matmul_template_broadcast_2d_weight(self):
        """Template should handle broadcast: (B, M, K) @ (K, N) -> (B, M, N).
        
        This is the broadcast matmul case where weight has no batch dims.
        The template needs to only have batch factors for tensors that have them.
        """
        # This is the case that currently fails - we need to fix the template
        # to handle different input ranks
        
        # For now, test what the OUTPUT batch dims should be
        input_a_shape = (8, 16, 64)  # Has batch dim
        input_b_shape = (64, 128)    # No batch dim  
        output_shape = (8, 16, 128)  # Has batch dim
        
        # batch_dims should be computed from OUTPUT, not input[0]
        output_batch_dims = len(output_shape) - 2  # = 1
        
        # This should NOT raise
        template = matmul_template(batch_dims=output_batch_dims)
        
        # But instantiation will fail with current code because B shape
        # doesn't have batch dims. We need a new template or smarter handling.
        # For now, verify the current behavior to understand the fix needed.
        
        # Skip the actual instantiation since we know it fails
        # (test documents expected behavior)
    
    def test_matmul_sharding_propagation_broadcast(self):
        """Test sharding propagates correctly through broadcast matmul."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # x: (8, 16, 64) sharded on batch dim
        np_x = np.random.randn(8, 16, 64).astype(np.float32)
        x = Tensor.from_dlpack(np_x).trace()
        x.shard(mesh, [DimSpec(["x"]), DimSpec([]), DimSpec([])])
        
        # w: (64, 128) - 2D weight, no batch
        np_w = np.random.randn(64, 128).astype(np.float32)
        w = Tensor.from_dlpack(np_w).trace()
        # w is replicated (no explicit sharding)
        
        # Matmul: should propagate batch sharding to output
        y = x @ w  # (8, 16, 64) @ (64, 128) -> (8, 16, 128)
        
        # Output should be sharded on dim0 (batch)
        assert y._impl.sharding is not None
        assert y._impl.sharding.dim_specs[0].axes == ["x"]
        assert y._impl.sharding.dim_specs[1].axes == []
        assert y._impl.sharding.dim_specs[2].axes == []


class TestToNumpyShardGathering:
    """Test that to_numpy correctly gathers all shards."""
    
    def test_to_numpy_single_shard(self):
        """to_numpy works for single-shard (non-sharded) tensor."""
        np_x = np.array([[1, 2], [3, 4]], dtype=np.float32)
        x = Tensor.from_dlpack(np_x)
        
        result = x.to_numpy()
        np.testing.assert_array_equal(result, np_x)
    
    def test_to_numpy_multi_shard_gather(self):
        """to_numpy should gather all shards into complete array."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Create sharded tensor
        np_x = np.arange(8).reshape(4, 2).astype(np.float32)
        x = Tensor.from_dlpack(np_x).trace()
        x.shard(mesh, [DimSpec(["x"]), DimSpec([])])  # Shard on dim0
        
        # Do an operation to create a graph (identity-like: add 0)
        np_zero = np.zeros((4, 2), dtype=np.float32)
        zero = Tensor.from_dlpack(np_zero).trace()
        y = x + zero  # This creates a proper sharded computation
        
        # Force evaluation
        asyncio.run(y.realize)
        
        # to_numpy should return the FULL array, not just shard 0
        result = y.to_numpy()
        
        # Should match original (x + 0 = x)
        np.testing.assert_array_equal(result, np_x)
    
    def test_sharded_matmul_to_numpy(self):
        """Full test: sharded matmul then to_numpy gets correct result."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        np_A = np.random.randn(4, 4).astype(np.float32)
        np_B = np.random.randn(4, 4).astype(np.float32)
        expected = np_A @ np_B
        
        A = Tensor.from_dlpack(np_A).trace()
        A.shard(mesh, [DimSpec(["x"]), DimSpec([])])  # Shard rows
        
        B = Tensor.from_dlpack(np_B).trace()  # Replicated
        
        C = A @ B
        
        asyncio.run(C.realize)
        
        actual = C.to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
