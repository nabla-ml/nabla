"""Phase 1: Unit tests for 2D mesh shard slicing and execution.

These tests isolate the slicing/gathering logic for 2D meshes to find where
numerical errors occur in the hybrid parallelism demo.
"""

import pytest
import numpy as np
import asyncio
from nabla import Tensor, ops
from nabla.sharding import DeviceMesh, DimSpec, ShardingSpec
from nabla.sharding import spmd


class TestPhase1_SliceForShard:
    """Test slice_for_shard with 2D mesh."""
    
    def test_1d_mesh_single_axis_sharding(self):
        """Baseline: 1D mesh slicing works correctly."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Global tensor (4, 4), sharded on dim0
        global_shape = (4, 4)
        sharding = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
        
        # Create a mock tensor value for slicing
        np_val = np.arange(16).reshape(4, 4).astype(np.float32)
        
        from nabla.core.compute_graph import GRAPH
        from max import graph as g
        
        with GRAPH.graph:
            tensor_val = g.constant(np_val)
            
            # Shard 0 should get rows 0-1
            shard0 = spmd.slice_for_shard(tensor_val, global_shape, sharding, 0)
            # Shard 1 should get rows 2-3
            shard1 = spmd.slice_for_shard(tensor_val, global_shape, sharding, 1)
        
        # Verify shapes
        assert tuple(shard0.type.shape) == (2, 4), f"Got {shard0.type.shape}"
        assert tuple(shard1.type.shape) == (2, 4), f"Got {shard1.type.shape}"
    
    def test_2d_mesh_single_axis_sharding_data(self):
        """2D mesh with only 'data' axis used for sharding."""
        mesh = DeviceMesh("hybrid", (2, 2), ("data", "model"))
        
        # Global tensor (8, 64), sharded on dim0 with "data" only
        global_shape = (8, 64)
        sharding = ShardingSpec(mesh, [DimSpec(["data"]), DimSpec([])])
        
        np_val = np.arange(512).reshape(8, 64).astype(np.float32)
        
        from nabla.core.compute_graph import GRAPH
        from max import graph as g
        
        with GRAPH.graph:
            tensor_val = g.constant(np_val)
            
            # With 2x2 mesh and "data" axis (size 2):
            # Shards 0,1 should get rows 0-3 (data=0)
            # Shards 2,3 should get rows 4-7 (data=1)
            shard0 = spmd.slice_for_shard(tensor_val, global_shape, sharding, 0)
            shard1 = spmd.slice_for_shard(tensor_val, global_shape, sharding, 1)
            shard2 = spmd.slice_for_shard(tensor_val, global_shape, sharding, 2)
            shard3 = spmd.slice_for_shard(tensor_val, global_shape, sharding, 3)
        
        # All shards should be (4, 64) - sliced along data axis
        for i, shard in enumerate([shard0, shard1, shard2, shard3]):
            assert tuple(shard.type.shape) == (4, 64), f"Shard {i}: got {shard.type.shape}"
    
    def test_2d_mesh_both_axes_sharding(self):
        """2D mesh with both axes used for sharding different dims."""
        mesh = DeviceMesh("hybrid", (2, 2), ("data", "model"))
        
        # Global tensor (8, 64), sharded on dim0 with "data" AND dim1 with "model"
        global_shape = (8, 64)
        sharding = ShardingSpec(mesh, [DimSpec(["data"]), DimSpec(["model"])])
        
        np_val = np.arange(512).reshape(8, 64).astype(np.float32)
        
        from nabla.core.compute_graph import GRAPH
        from max import graph as g
        
        with GRAPH.graph:
            tensor_val = g.constant(np_val)
            
            # Each shard should be (4, 32)
            shard0 = spmd.slice_for_shard(tensor_val, global_shape, sharding, 0)
            shard1 = spmd.slice_for_shard(tensor_val, global_shape, sharding, 1)
            shard2 = spmd.slice_for_shard(tensor_val, global_shape, sharding, 2)
            shard3 = spmd.slice_for_shard(tensor_val, global_shape, sharding, 3)
        
        # Each shard: (8/2, 64/2) = (4, 32)
        for i, shard in enumerate([shard0, shard1, shard2, shard3]):
            assert tuple(shard.type.shape) == (4, 32), f"Shard {i}: got {shard.type.shape}"


class TestPhase1_2DShardExecution:
    """Test actual execution with 2D mesh sharding."""
    
    def test_elementwise_add_2d_mesh_single_axis(self):
        """Elementwise add with 2D mesh, single-axis sharding."""
        mesh = DeviceMesh("hybrid", (2, 2), ("data", "model"))
        
        # A: (8, 64) sharded on dim0 with "data"
        np_A = np.random.randn(8, 64).astype(np.float32)
        A = Tensor.from_dlpack(np_A).trace()
        A.shard(mesh, [DimSpec(["data"]), DimSpec([])])
        
        # B: (8, 64) replicated
        np_B = np.random.randn(8, 64).astype(np.float32)
        B = Tensor.from_dlpack(np_B).trace()
        
        # C = A + B
        C = A + B
        
        # Verify shape
        assert tuple(int(d) for d in C.shape) == (8, 64)
        
        # Evaluate and verify numerical correctness
        asyncio.run(C.realize)
        actual = C.to_numpy()
        expected = np_A + np_B
        
        np.testing.assert_allclose(actual, expected, rtol=1e-5)
    
    def test_matmul_2d_mesh_column_parallel(self):
        """Matmul with column-parallel weights (2D mesh, model axis only)."""
        mesh = DeviceMesh("hybrid", (2, 2), ("data", "model"))
        
        # x: (4, 32) replicated
        np_x = np.random.randn(4, 32).astype(np.float32)
        x = Tensor.from_dlpack(np_x).trace()
        
        # w: (32, 64) sharded on dim1 with "model"
        np_w = np.random.randn(32, 64).astype(np.float32)
        w = Tensor.from_dlpack(np_w).trace()
        w.shard(mesh, [DimSpec([]), DimSpec(["model"])])
        
        # y = x @ w -> (4, 64) sharded on dim1
        y = x @ w
        
        # Verify shape
        assert tuple(int(d) for d in y.shape) == (4, 64)
        
        # Verify sharding: dim1 should have "model"
        assert y._impl.sharding is not None
        assert y._impl.sharding.dim_specs[1].axes == ["model"]
        
        # Evaluate and verify numerical correctness
        asyncio.run(y.realize)
        actual = y.to_numpy()
        expected = np_x @ np_w
        
        np.testing.assert_allclose(actual, expected, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
