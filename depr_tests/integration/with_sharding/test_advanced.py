"""
Advanced Sharding Tests
=======================

Tests for edge cases and advanced patterns beyond the basic functional tests.
"""

import numpy as np
import pytest
from nabla import Tensor
from nabla.sharding import DeviceMesh, DimSpec
from nabla.ops.unary import relu

from depr_tests.common.sharding_utils import make_array, make_randn



# =============================================================================
# Advanced Matmul Tests
# =============================================================================

class TestMatmulAdvanced:
    """Advanced matmul sharding scenarios."""
    
    def test_matmul_4x1_mesh(self):
        """Matmul with 4-device 1D mesh (high data parallelism)."""
        batch, in_dim, out_dim = 16, 32, 64
        
        np_x = make_randn(batch, in_dim, seed=1) * 0.1
        np_w = make_randn(in_dim, out_dim, seed=2) * 0.1
        np_expected = np_x @ np_w
        
        mesh = DeviceMesh("dp4", (4,), ("batch",))
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["batch"]), DimSpec([])])
        w = Tensor.from_dlpack(np_w)
        
        result = x @ w
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-4)
        print(f"\n  ✓ 4x1 mesh matmul: {actual.shape}")
    
    def test_matmul_1x4_mesh_tensor_parallel(self):
        """Matmul with 1x4 mesh (high tensor parallelism on output dim)."""
        batch, in_dim, out_dim = 8, 32, 64
        
        np_x = make_randn(batch, in_dim, seed=3) * 0.1
        np_w = make_randn(in_dim, out_dim, seed=4) * 0.1
        np_expected = np_x @ np_w
        
        mesh = DeviceMesh("tp4", (4,), ("model",))
        x = Tensor.from_dlpack(np_x)  # Replicated
        # Column parallel: shard output dim
        w = Tensor.from_dlpack(np_w).shard(mesh, [DimSpec([]), DimSpec(["model"])])
        
        result = x @ w
        actual = result.to_numpy()
        
        # Relax tolerance slightly for distributed dot product order differences
        np.testing.assert_allclose(actual, np_expected, rtol=1e-3, atol=1e-5)
        print(f"\n  ✓ 1x4 tensor parallel: {actual.shape}")
    
    def test_matmul_2x4_hybrid_mesh(self):
        """Matmul with 2x4 = 8 device mesh."""
        batch, in_dim, out_dim = 8, 32, 64
        
        np_x = make_randn(batch, in_dim, seed=5) * 0.1
        np_w = make_randn(in_dim, out_dim, seed=6) * 0.1
        np_expected = np_x @ np_w
        
        mesh = DeviceMesh("hybrid", (2, 4), ("data", "model"))
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["data"]), DimSpec([])])
        w = Tensor.from_dlpack(np_w).shard(mesh, [DimSpec([]), DimSpec(["model"])])
        
        result = x @ w
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-4)
        print(f"\n  ✓ 2x4 hybrid mesh: {actual.shape}")
    
    def test_matmul_4x2_asymmetric_mesh(self):
        """Matmul with asymmetric 4x2 mesh (more data than model parallel)."""
        batch, in_dim, out_dim = 16, 32, 64
        
        np_x = make_randn(batch, in_dim, seed=7) * 0.1
        np_w = make_randn(in_dim, out_dim, seed=8) * 0.1
        np_expected = np_x @ np_w
        
        mesh = DeviceMesh("asym", (4, 2), ("data", "model"))
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["data"]), DimSpec([])])
        w = Tensor.from_dlpack(np_w).shard(mesh, [DimSpec([]), DimSpec(["model"])])
        
        result = x @ w
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-4)
        print(f"\n  ✓ 4x2 asymmetric mesh: {actual.shape}")
    
    def test_batched_matmul_sharded(self):
        """Batched matmul [B, M, K] @ [K, N] with batch sharding."""
        batch, m, k, n = 8, 16, 32, 64
        
        np_x = make_randn(batch, m, k, seed=9) * 0.1
        np_w = make_randn(k, n, seed=10) * 0.1
        np_expected = np_x @ np_w  # (8, 16, 64)
        
        mesh = DeviceMesh("m", (4,), ("x",))
        x = Tensor.from_dlpack(np_x).shard(
            mesh, [DimSpec(["x"]), DimSpec([]), DimSpec([])]
        )
        w = Tensor.from_dlpack(np_w)
        
        result = x @ w
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-4)
        print(f"\n  ✓ Batched matmul: {actual.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


# =============================================================================
# Sub-Axis Sharding Tests
# =============================================================================

class TestSubAxisSharding:
    """Tests for complex sub-axis patterns like 'x:(2)4'."""
    
    def test_sub_axis_basic(self):
        """Split a mesh axis 'devices' into two logical sub-axes."""
        # 8 devices total
        batch, hidden = 16, 32
        
        np_x = make_array(batch, hidden, scale=0.1)
        np_expected = np.maximum(0, np_x)
        
        # Mesh has 1 axis "devices" size 8
        mesh = DeviceMesh("cluster", (8,), ("devices",))
        
        # We shard batch on the first part of "devices" (size 2)
        # We shard hidden on the second part of "devices" (size 4)
        # Logic: 8 devices -> logical 2x4 grid
        # axis names: "devices:(2)4" means: 
        #   parent="devices", pre_size=1 (implied), size=2 -> "devices:(1)2" usually? 
        #   Wait, parser expects "name:(pre)size".
        #   First sub-axis "rows": size 2. Stride 4. "devices:(1)2" ?
        #   Second sub-axis "cols": size 4. Stride 1. "devices:(2)4" ?
        
        # Let's verify parser logic in DeviceMesh.get_coordinate:
        # parent, pre, size = parsed
        # parent_total = 8
        # post = parent_total // (pre * size)
        # coord = (parent_coord // post) % size
        
        # Want logical 2x4 grid from physical 8.
        # Logical Row (size 2): pre=1? post=4. -> (d // 4) % 2. 
        #   "devices:(1)2" -> pre=1, size=2. Total=1*2=2 covered? No pre*size is divisor.
        #   pre=1, size=2. post = 8 // 2 = 4. correct.
        
        # Logical Col (size 4): pre=2? post=1. -> (d // 1) % 4.
        #   "devices:(2)4" -> pre=2, size=4. post = 8 // 8 = 1. correct.
        
        # So we use axes "devices:(1)2" and "devices:(2)4"
        
        x = Tensor.from_dlpack(np_x).shard(
            mesh, 
            [DimSpec(["devices:(1)2"]), DimSpec(["devices:(2)4"])]
        )
        
        # Check sharding
        # Total shards = 8
        assert len(x._impl._values) == 8
        # Each shard: batch/2=8, hidden/4=8 -> (8, 8)
        for v in x._impl._values:
            assert tuple(v.type.shape) == (8, 8)
            
        result = relu(x)
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Sub-axis 2x4 split: {actual.shape}")

    def test_sub_axis_replicated_mix(self):
        """Use one sub-axis for sharding, other implicit refined."""
        # 4 devices
        batch, hidden = 8, 16
        np_x = make_array(batch, hidden)
        np_expected = np_x * 2
        
        mesh = DeviceMesh("small", (4,), ("gpu",))
        
        # Split "gpu" into 2x2.
        # Shard batch on first half: "gpu:(1)2"
        # Replicate hidden (second half "gpu:(2)2" is unused -> implicit replication)
        
        x = Tensor.from_dlpack(np_x).shard(
            mesh, 
            [DimSpec(["gpu:(1)2"]), DimSpec([])]
        )
        
        # Each shard should have full hidden dim
        # batch is split by 2 -> 4
        # But we have 4 devices. 
        # Devices 0,1 get first half of batch. Devices 2,3 get second half?
        # "gpu:(1)2" -> (d // 2) % 2. 
        # d=0,1 -> coord 0.
        # d=2,3 -> coord 1.
        # So yes, 0/1 have same data? No, they are distinct shards in the list.
        # But they hold the SAME logical slice of the tensor if the other axis is not used.
        # The system treats them as replicas for the unspecified axes.
        
        result = x * 2
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Sub-axis mixed replication: {actual.shape}")


# =============================================================================
# Uneven Sharding Tests (Non-divisible dim sizes)
# =============================================================================

class TestUnevenSharding:
    """Tests for shapes that don't divide evenly by mesh size (padding/uneven shards)."""
    

    def test_uneven_shard_sizes(self):
        """Shard dim 5 across 4 devices -> shards [2, 2, 1, 0]."""
        batch = 5
        hidden = 16
        
        np_x = make_array(batch, hidden)
        np_expected = np_x * 2
        
        mesh = DeviceMesh("uneven", (4,), ("d",))
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["d"]), DimSpec([])])
        
        # Verify shard shapes
        # 5 / 4 -> ceil=2.
        # 0: [0, 2) -> 2
        # 1: [2, 4) -> 2
        # 2: [4, 6) -> min(6, 5)-4 = 1
        # 3: [6, 8) -> min(8, 5)-6 = -1 -> 0
        expected_shapes = [(2, 16), (2, 16), (1, 16), (0, 16)]
        shard_shapes = [tuple(v.type.shape) for v in x._impl._values]
        assert shard_shapes == expected_shapes
        
        # Op execution handles 0-sized tensors?
        result = x * 2
        
        # Result shards should match input layout
        res_shapes = [tuple(v.type.shape) for v in result._impl._values]
        assert res_shapes == expected_shapes
        
        # Gather correctness
        actual = result.to_numpy()
        np.testing.assert_allclose(actual, np_expected)
        print(f"\n  ✓ Uneven sharding (5->4 devs): {actual.shape} -> shards {shard_shapes}")
