"""
Sharding Functional Tests
=========================

Every test follows this workflow:
1. Create NumPy arrays
2. Create Tensors and shard them (functional - returns NEW tensor)
3. Perform operations  
4. Call to_numpy() which gathers shards automatically
5. Compare against pure NumPy reference

Start with ONE comprehensive MLP test to verify basic functionality.
"""

import numpy as np
import pytest
from nabla import Tensor
from nabla.sharding import DeviceMesh, DimSpec
from nabla.ops.unary import relu

from sharding_test_utils import make_array, make_randn


class TestMLPSharded:
    """Test MLP-like computation with sharding."""
    
    def test_simple_mlp_1d_mesh(self):
        """MLP forward pass: X @ W1 -> ReLU -> @ W2 -> + bias
        
        Sharding: X sharded on batch dim, weights replicated.
        This is basic data parallelism.
        """
        # Shapes
        batch, in_dim, hidden, out_dim = 8, 16, 32, 8
        
        # 1. NumPy reference data
        np_x = make_array(batch, in_dim, scale=0.1)
        np_w1 = make_randn(in_dim, hidden, seed=1) * 0.1
        np_w2 = make_randn(hidden, out_dim, seed=2) * 0.1
        np_bias = make_array(out_dim, scale=0.01)
        
        # NumPy reference computation
        np_h1 = np_x @ np_w1
        np_h1_relu = np.maximum(0, np_h1)
        np_h2 = np_h1_relu @ np_w2
        np_out = np_h2 + np_bias
        
        # 2. Create mesh (2 devices for data parallelism)
        mesh = DeviceMesh("dp", (2,), ("batch",))
        
        # 3. Shard X on batch dimension, weights replicated
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["batch"]), DimSpec([])])
        w1 = Tensor.from_dlpack(np_w1)  # Replicated
        w2 = Tensor.from_dlpack(np_w2)  # Replicated
        bias = Tensor.from_dlpack(np_bias)  # Replicated
        
        # Verify sharding happened
        assert len(x._impl._values) == 2, f"Expected 2 shards, got {len(x._impl._values)}"
        print(f"\n  X sharded into {len(x._impl._values)} parts")
        for i, v in enumerate(x._impl._values):
            print(f"    Shard {i}: {tuple(v.type.shape)}")
        
        # 4. Forward pass
        h1 = x @ w1
        h1_relu = relu(h1)
        h2 = h1_relu @ w2
        out = h2 + bias
        
        # 5. Get result (to_numpy gathers shards)
        actual = out.to_numpy()
        
        # 6. Verify
        np.testing.assert_allclose(actual, np_out, rtol=1e-4, atol=1e-5)
        print(f"  ✓ MLP output shape: {actual.shape}")
        print(f"  ✓ Max diff: {np.abs(actual - np_out).max():.2e}")
    
    def test_mlp_2d_mesh_hybrid_parallelism(self):
        """MLP with 2D mesh: data parallelism + tensor parallelism.
        
        Sharding:
        - X: sharded on batch (data axis)
        - W1: sharded on output dim (model axis) - column parallel
        - W2: sharded on input dim (model axis) - row parallel
        
        This is Megatron-LM style hybrid parallelism.
        """
        # Shapes
        batch, in_dim, hidden, out_dim = 8, 16, 32, 16
        
        # 1. NumPy reference data
        np_x = make_array(batch, in_dim, scale=0.1)
        np_w1 = make_randn(in_dim, hidden, seed=10) * 0.1
        np_w2 = make_randn(hidden, out_dim, seed=20) * 0.1
        
        # NumPy reference computation
        np_h1 = np_x @ np_w1
        np_h1_relu = np.maximum(0, np_h1)
        np_out = np_h1_relu @ np_w2
        
        # 2. Create 2x2 mesh (data=2, model=2)
        mesh = DeviceMesh("hybrid", (2, 2), ("data", "model"))
        
        # 3. Shard tensors
        # X: [batch, in_dim] -> shard batch on "data"
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["data"]), DimSpec([])])
        
        # W1: [in_dim, hidden] -> shard hidden on "model" (column parallel)
        w1 = Tensor.from_dlpack(np_w1).shard(mesh, [DimSpec([]), DimSpec(["model"])])
        
        # W2: [hidden, out_dim] -> shard hidden on "model" (row parallel)
        w2 = Tensor.from_dlpack(np_w2).shard(mesh, [DimSpec(["model"]), DimSpec([])])
        
        # Verify 4 shards (2x2 mesh)
        assert len(x._impl._values) == 4, f"X: expected 4 shards, got {len(x._impl._values)}"
        assert len(w1._impl._values) == 4, f"W1: expected 4 shards, got {len(w1._impl._values)}"
        assert len(w2._impl._values) == 4, f"W2: expected 4 shards, got {len(w2._impl._values)}"
        
        print(f"\n  X shards: {[tuple(v.type.shape) for v in x._impl._values]}")
        print(f"  W1 shards: {[tuple(v.type.shape) for v in w1._impl._values]}")
        print(f"  W2 shards: {[tuple(v.type.shape) for v in w2._impl._values]}")
        
        # 4. Forward pass
        h1 = x @ w1
        h1_relu = relu(h1)
        out = h1_relu @ w2
        
        # 5. Get result
        actual = out.to_numpy()
        
        # 6. Verify
        np.testing.assert_allclose(actual, np_out, rtol=1e-4, atol=1e-4)
        print(f"  ✓ Hybrid MLP output shape: {actual.shape}")
        print(f"  ✓ Max diff: {np.abs(actual - np_out).max():.2e}")


class TestDifferentMeshes:
    """Test different mesh configurations."""
    
    def test_mesh_4_devices(self):
        """1D mesh with 4 devices for higher parallelism."""
        batch, features = 16, 32
        
        np_x = make_array(batch, features, scale=0.1)
        np_expected = np.maximum(0, np_x)  # ReLU
        
        mesh = DeviceMesh("dp4", (4,), ("batch",))
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["batch"]), DimSpec([])])
        
        assert len(x._impl._values) == 4
        # Each shard should be batch/4 = 4 rows
        for v in x._impl._values:
            assert tuple(v.type.shape) == (4, 32)
        
        result = relu(x)
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ 4-device mesh: {actual.shape}")
    
    def test_mesh_2x4(self):
        """2D mesh 2x4 = 8 virtual devices."""
        batch, seq, hidden = 8, 16, 32
        
        np_x = make_array(batch, seq, hidden, scale=0.01)
        np_expected = np.exp(np_x)
        
        mesh = DeviceMesh("big", (2, 4), ("data", "model"))
        # Shard batch on "data" axis only
        x = Tensor.from_dlpack(np_x).shard(
            mesh, [DimSpec(["data"]), DimSpec([]), DimSpec([])]
        )
        
        assert len(x._impl._values) == 8  # 2 * 4 = 8
        
        from nabla.ops.unary import exp
        result = exp(x)
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ 2x4 mesh: {actual.shape}")


class TestDifferentShapes:
    """Test operations on various tensor shapes."""
    
    def test_3d_tensor_batch_sharded(self):
        """3D tensor [batch, seq, hidden] with batch sharding."""
        batch, seq, hidden = 8, 16, 64
        
        np_x = make_array(batch, seq, hidden, scale=0.1)
        np_w = make_randn(hidden, hidden, seed=5) * 0.1
        np_expected = np_x @ np_w
        
        mesh = DeviceMesh("m", (2,), ("x",))
        x = Tensor.from_dlpack(np_x).shard(
            mesh, [DimSpec(["x"]), DimSpec([]), DimSpec([])]
        )
        w = Tensor.from_dlpack(np_w)  # replicated
        
        result = x @ w
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-4)
        print(f"\n  ✓ 3D batch-sharded matmul: {actual.shape}")
    
    def test_large_reduction_dim(self):
        """Matmul with large reduction dimension."""
        batch, in_dim, out_dim = 4, 512, 64
        
        np_x = make_randn(batch, in_dim, seed=10) * 0.01
        np_w = make_randn(in_dim, out_dim, seed=20) * 0.01
        np_expected = np_x @ np_w
        
        mesh = DeviceMesh("m", (2,), ("x",))
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["x"]), DimSpec([])])
        w = Tensor.from_dlpack(np_w)
        
        result = x @ w
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-3, atol=1e-5)
        print(f"\n  ✓ Large reduction dim: {actual.shape}")
    
    def test_square_matrices(self):
        """Square matrix multiply with sharding."""
        n = 64
        
        np_a = make_randn(n, n, seed=1) * 0.1
        np_b = make_randn(n, n, seed=2) * 0.1
        np_expected = np_a @ np_b
        
        mesh = DeviceMesh("m", (4,), ("x",))
        a = Tensor.from_dlpack(np_a).shard(mesh, [DimSpec(["x"]), DimSpec([])])
        b = Tensor.from_dlpack(np_b)
        
        result = a @ b
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-4, atol=1e-5)
        print(f"\n  ✓ Square matrices: {actual.shape}")


class TestParallelizationStrategies:
    """Test different parallelization strategies."""
    
    def test_pure_tensor_parallelism(self):
        """Column-parallel: shard weight output dim only, input replicated."""
        batch, in_dim, out_dim = 8, 32, 64
        
        np_x = make_randn(batch, in_dim, seed=1)
        np_w = make_randn(in_dim, out_dim, seed=2) * 0.1
        np_expected = np.maximum(0, np_x @ np_w)  # matmul + relu
        
        mesh = DeviceMesh("tp", (2,), ("model",))
        # X is replicated (no sharding)
        x = Tensor.from_dlpack(np_x)  
        # W sharded on output dim (columns)
        w = Tensor.from_dlpack(np_w).shard(mesh, [DimSpec([]), DimSpec(["model"])])
        
        result = x @ w
        result = relu(result)
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-4)
        print(f"\n  ✓ Pure tensor parallelism: {actual.shape}")
    
    def test_row_parallel_with_allreduce(self):
        """Row-parallel: shard weight input dim, requires AllReduce."""
        batch, in_dim, out_dim = 4, 32, 16
        
        np_x = make_randn(batch, in_dim, seed=1)
        np_w = make_randn(in_dim, out_dim, seed=2) * 0.1
        np_expected = np_x @ np_w
        
        mesh = DeviceMesh("tp", (2,), ("model",))
        # X sharded on feature dim to match W's row sharding
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec([]), DimSpec(["model"])])
        # W sharded on input dim (rows)
        w = Tensor.from_dlpack(np_w).shard(mesh, [DimSpec(["model"]), DimSpec([])])
        
        result = x @ w  # Should trigger AllReduce internally
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-4)
        print(f"\n  ✓ Row-parallel with AllReduce: {actual.shape}")
    
    def test_sequence_parallel(self):
        """Sequence parallelism: shard sequence dimension."""
        batch, seq, hidden = 4, 32, 64
        
        np_x = make_array(batch, seq, hidden, scale=0.01)
        np_expected = np.maximum(0, np_x)  # ReLU preserves sharding
        
        mesh = DeviceMesh("sp", (4,), ("seq",))
        # Shard on sequence dimension
        x = Tensor.from_dlpack(np_x).shard(
            mesh, [DimSpec([]), DimSpec(["seq"]), DimSpec([])]
        )
        
        # Each shard: (4, 8, 64) since seq=32/4=8
        assert tuple(x._impl._values[0].type.shape) == (4, 8, 64)
        
        result = relu(x)
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Sequence parallelism: {actual.shape}")
    
    def test_multi_axis_sharding_single_tensor(self):
        """Shard one tensor on multiple axes (e.g., both batch and features)."""
        batch, hidden = 16, 32
        
        np_x = make_array(batch, hidden, scale=0.1)
        np_expected = np.maximum(0, np_x)
        
        # 2D mesh, shard dim 0 on "data" and dim 1 on "model"
        mesh = DeviceMesh("2d", (2, 2), ("data", "model"))
        x = Tensor.from_dlpack(np_x).shard(
            mesh, [DimSpec(["data"]), DimSpec(["model"])]
        )
        
        # Should have 4 shards, each (8, 16)
        assert len(x._impl._values) == 4
        for v in x._impl._values:
            assert tuple(v.type.shape) == (8, 16)
        
        result = relu(x)
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Multi-axis sharding: {actual.shape}")


# =============================================================================
# Unit Tests: Operations
# =============================================================================

class TestUnaryOps:
    """Test one representative unary operation."""
    
    def test_exp_sharded(self):
        """Exp operation on sharded tensor."""
        from nabla.ops.unary import exp
        
        np_x = make_array(8, 16, scale=0.1)  # Keep small to avoid overflow
        np_expected = np.exp(np_x)
        
        mesh = DeviceMesh("m", (2,), ("x",))
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        result = exp(x)
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Exp sharded: {actual.shape}")


class TestBinaryOps:
    """Test one representative binary operation."""
    
    def test_mul_sharded_both(self):
        """Multiply two sharded tensors with same sharding."""
        np_a = make_array(8, 16, scale=0.1)
        np_b = make_randn(8, 16, seed=10) * 0.5
        np_expected = np_a * np_b
        
        mesh = DeviceMesh("m", (2,), ("x",))
        a = Tensor.from_dlpack(np_a).shard(mesh, [DimSpec(["x"]), DimSpec([])])
        b = Tensor.from_dlpack(np_b).shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        result = a * b
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Mul both sharded: {actual.shape}")
    
    def test_add_sharded_replicated(self):
        """Add sharded tensor with replicated tensor."""
        np_a = make_array(8, 16, scale=0.1)
        np_b = make_randn(8, 16, seed=20) * 0.5
        np_expected = np_a + np_b
        
        mesh = DeviceMesh("m", (2,), ("x",))
        a = Tensor.from_dlpack(np_a).shard(mesh, [DimSpec(["x"]), DimSpec([])])
        b = Tensor.from_dlpack(np_b)  # Replicated
        
        result = a + b
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Add sharded+replicated: {actual.shape}")


# =============================================================================
# Unit Tests: View Operations (comprehensive)
# =============================================================================

class TestUnsqueezeOp:
    """Test unsqueeze on sharded tensors."""
    
    def test_unsqueeze_axis0_sharded_dim0(self):
        """Unsqueeze at axis 0 when dim 0 is sharded."""
        from nabla.ops.view import unsqueeze
        
        np_x = make_array(8, 16)
        np_expected = np.expand_dims(np_x, axis=0)  # (1, 8, 16)
        
        mesh = DeviceMesh("m", (2,), ("x",))
        # Shard on dim 0, then unsqueeze at dim 0 -> sharding shifts to dim 1
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        result = unsqueeze(x, axis=0)
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Unsqueeze axis=0 sharded: {np_x.shape} -> {actual.shape}")
    
    def test_unsqueeze_axis_last(self):
        """Unsqueeze at last axis."""
        from nabla.ops.view import unsqueeze
        
        np_x = make_array(8, 16)
        np_expected = np.expand_dims(np_x, axis=2)  # (8, 16, 1)
        
        mesh = DeviceMesh("m", (2,), ("x",))
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        result = unsqueeze(x, axis=2)
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Unsqueeze axis=2: {np_x.shape} -> {actual.shape}")


class TestSqueezeOp:
    """Test squeeze on sharded tensors."""
    
    def test_squeeze_axis0(self):
        """Squeeze dimension 0 when it's size 1."""
        from nabla.ops.view import squeeze
        
        np_x = make_array(1, 8, 16)
        np_expected = np.squeeze(np_x, axis=0)  # (8, 16)
        
        mesh = DeviceMesh("m", (2,), ("x",))
        # Shard on dim 1 (which becomes dim 0 after squeeze)
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec([]), DimSpec(["x"]), DimSpec([])])
        
        result = squeeze(x, axis=0)
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Squeeze axis=0: {np_x.shape} -> {actual.shape}")
    
    def test_squeeze_middle_axis(self):
        """Squeeze middle dimension."""
        from nabla.ops.view import squeeze
        
        np_x = make_array(8, 1, 16)
        np_expected = np.squeeze(np_x, axis=1)  # (8, 16)
        
        mesh = DeviceMesh("m", (2,), ("x",))
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["x"]), DimSpec([]), DimSpec([])])
        
        result = squeeze(x, axis=1)
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Squeeze axis=1: {np_x.shape} -> {actual.shape}")


class TestSwapAxesOp:
    """Test swap_axes (transpose) on sharded tensors."""
    
    def test_swap_sharded_axes(self):
        """Swap axes where one is sharded."""
        from nabla.ops.view import swap_axes
        
        np_x = make_array(8, 16)
        np_expected = np.swapaxes(np_x, 0, 1)  # (16, 8)
        
        mesh = DeviceMesh("m", (2,), ("x",))
        # Shard on dim 0, after swap sharding should be on dim 1
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        result = swap_axes(x, 0, 1)
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ SwapAxes(0,1): {np_x.shape} -> {actual.shape}")
    
    def test_swap_3d_tensor(self):
        """Swap axes on 3D sharded tensor."""
        from nabla.ops.view import swap_axes
        
        np_x = make_array(4, 8, 16)
        np_expected = np.swapaxes(np_x, 1, 2)  # (4, 16, 8)
        
        mesh = DeviceMesh("m", (2,), ("x",))
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["x"]), DimSpec([]), DimSpec([])])
        
        result = swap_axes(x, 1, 2)
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ SwapAxes 3D: {np_x.shape} -> {actual.shape}")


class TestBroadcastToOp:
    """Test broadcast_to on sharded tensors."""
    
    def test_broadcast_row_vector(self):
        """Broadcast 1D to 2D."""
        from nabla.ops.view import broadcast_to
        
        np_x = make_array(16)
        np_expected = np.broadcast_to(np_x, (8, 16))
        
        mesh = DeviceMesh("m", (2,), ("x",))
        # Shard on the feature dim (will replicate on new batch dim)
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["x"])])
        
        result = broadcast_to(x, shape=(8, 16))
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Broadcast 1D->2D: {np_x.shape} -> {actual.shape}")
    
    def test_broadcast_add_batch(self):
        """Broadcast 2D to 3D (add batch dim)."""
        from nabla.ops.view import broadcast_to
        
        np_x = make_array(8, 16)
        np_expected = np.broadcast_to(np_x, (4, 8, 16))
        
        mesh = DeviceMesh("m", (2,), ("x",))
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        result = broadcast_to(x, shape=(4, 8, 16))
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Broadcast 2D->3D: {np_x.shape} -> {actual.shape}")


class TestReshapeOp:
    """Test reshape on sharded tensors."""
    
    def test_reshape_flatten(self):
        """Reshape 2D to 1D (flatten)."""
        from nabla.ops.view import reshape
        
        np_x = make_array(8, 16)
        np_expected = np_x.reshape(-1)  # (128,)
        
        mesh = DeviceMesh("m", (2,), ("x",))
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        result = reshape(x, shape=(128,))
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Reshape flatten: {np_x.shape} -> {actual.shape}")
    
    def test_reshape_split_dim(self):
        """Reshape to split a dimension."""
        from nabla.ops.view import reshape
        
        np_x = make_array(8, 16)
        np_expected = np_x.reshape(2, 4, 16)  # Split first dim
        
        mesh = DeviceMesh("m", (2,), ("x",))
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        result = reshape(x, shape=(2, 4, 16))
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Reshape split dim: {np_x.shape} -> {actual.shape}")
    
    def test_reshape_merge_dims(self):
        """Reshape to merge dimensions."""
        from nabla.ops.view import reshape
        
        np_x = make_array(4, 8, 16)
        np_expected = np_x.reshape(4, 128)  # Merge last two dims
        
        mesh = DeviceMesh("m", (2,), ("x",))
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["x"]), DimSpec([]), DimSpec([])])
        
        result = reshape(x, shape=(4, 128))
        actual = result.to_numpy()
        
        np.testing.assert_allclose(actual, np_expected, rtol=1e-5)
        print(f"\n  ✓ Reshape merge dims: {np_x.shape} -> {actual.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

