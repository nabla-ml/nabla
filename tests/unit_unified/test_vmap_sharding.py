# ===----------------------------------------------------------------------=== #
# Unified Test: VMap + Sharding Combined
# ===----------------------------------------------------------------------=== #

"""Test vmap with sharding - realistic patterns from production ML workloads."""

import pytest
import nabla as nb
import jax
import jax.numpy as jnp
import numpy as np

from .common import MESH_CONFIGS
from nabla.core.sharding.spec import DeviceMesh, P
from nabla import Tensor

# ============================================================================
# Unary Operations with Sharding
# ============================================================================

class TestVmapShardingUnary:
    """Test vmap(unary_op) with sharding inside function."""
    
    @pytest.mark.parametrize("mesh_name,mesh_shape,mesh_axes", MESH_CONFIGS[:3])
    def test_relu_sharded_features(self, mesh_name, mesh_shape, mesh_axes):
        """vmap(relu) with feature dimension sharded (tensor parallelism)."""
        batch, features = 4, 16
        mesh = DeviceMesh(f"mesh_relu_{mesh_name}", mesh_shape, mesh_axes)
        
        def f(x):  # x: (features,) logical shape
            x_sharded = x.shard(mesh, P(mesh_axes[-1]))
            return nb.relu(x_sharded)
        
        np_x = np.random.randn(batch, features).astype(np.float32)
        x = nb.Tensor.from_dlpack(np_x)
        
        result = nb.vmap(f)(x)
        expected = np.maximum(np_x, 0)
        
        assert tuple(int(d) for d in result.shape) == (batch, features)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

# ============================================================================
# Binary Operations - Batched Data + Sharded Weights
# ============================================================================

class TestVmapShardingBinary:
    """Test vmap(binary_op) with realistic batched+broadcast patterns."""
    
    @pytest.mark.parametrize("mesh_name,mesh_shape,mesh_axes", MESH_CONFIGS[:3])
    def test_add_batched_data_sharded_bias(self, mesh_name, mesh_shape, mesh_axes):
        """vmap(add) with batched input + broadcast sharded bias."""
        batch, hidden = 4, 16
        mesh = DeviceMesh(f"mesh_add_{mesh_name}", mesh_shape, mesh_axes)
        
        def f(x, bias):  # x: (hidden,), bias: (hidden,)
            bias_sharded = bias.shard(mesh, P(mesh_axes[-1]))
            return nb.add(x, bias_sharded)
        
        np_x = np.random.randn(batch, hidden).astype(np.float32)
        np_bias = np.random.randn(hidden).astype(np.float32)
        
        x = nb.Tensor.from_dlpack(np_x)
        bias = nb.Tensor.from_dlpack(np_bias)
        
        result = nb.vmap(f, in_axes=(0, None))(x, bias)
        expected = np_x + np_bias
        
        assert tuple(int(d) for d in result.shape) == (batch, hidden)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

# ============================================================================
# Matmul - Megatron-Style Tensor Parallelism
# ============================================================================

class TestVmapShardingMatmul:
    """Test vmap(matmul) with Megatron-style column/row parallel patterns."""
    
    def test_column_parallel(self):
        """Column-parallel matmul: shard output dimension."""
        batch, in_feat, out_feat = 4, 16, 32
        mesh = DeviceMesh("mesh_matmul_col", (2,), ("tp",))
        
        def f(x, w):  # x: (in_feat,), w: (in_feat, out_feat)
            w_sharded = w.shard(mesh, P(None, "tp"))  # Column parallel
            return nb.matmul(x, w_sharded)
        
        np_x = np.random.randn(batch, in_feat).astype(np.float32)
        np_w = np.random.randn(in_feat, out_feat).astype(np.float32)
        
        x = nb.Tensor.from_dlpack(np_x)
        w = nb.Tensor.from_dlpack(np_w)
        
        result = nb.vmap(f, in_axes=(0, None))(x, w)
        expected = np_x @ np_w
        
        assert tuple(int(d) for d in result.shape) == (batch, out_feat)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4)
    
    def test_row_parallel(self):
        """Row-parallel matmul: shard contracting dimension (needs AllReduce)."""
        batch, in_feat, out_feat = 4, 32, 16
        mesh = DeviceMesh("mesh_matmul_row", (2,), ("tp",))
        
        def f(x, w):  # x: (in_feat,), w: (in_feat, out_feat)
            x_sharded = x.shard(mesh, P("tp"))
            w_sharded = w.shard(mesh, P("tp", None))  # Row parallel
            return nb.matmul(x_sharded, w_sharded)  # Auto AllReduce
        
        np_x = np.random.randn(batch, in_feat).astype(np.float32)
        np_w = np.random.randn(in_feat, out_feat).astype(np.float32)
        
        x = nb.Tensor.from_dlpack(np_x)
        w = nb.Tensor.from_dlpack(np_w)
        
        result = nb.vmap(f, in_axes=(0, None))(x, w)
        expected = np_x @ np_w
        
        assert tuple(int(d) for d in result.shape) == (batch, out_feat)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4)

# ============================================================================
# Reductions with Sharding
# ============================================================================

class TestVmapShardingReduction:
    """Test vmap(reduction) with sharded inputs."""
    
    @pytest.mark.parametrize("mesh_name,mesh_shape,mesh_axes", MESH_CONFIGS[:2])
    def test_reduce_sum_sharded_axis(self, mesh_name, mesh_shape, mesh_axes):
        """vmap(reduce_sum) reducing over sharded dimension (needs AllReduce)."""
        pytest.skip("Pending investigation: suspected double-counting issue in vmap+sharding simulation")
        batch = 4
        mesh = DeviceMesh(f"mesh_reduce_{mesh_name}", mesh_shape, mesh_axes)
        
        def f(x):  # x: (16,)
            x_sharded = x.shard(mesh, P(mesh_axes[-1]))
            return nb.reduce_sum(x_sharded, axis=0)  # Reduce sharded axis
        
        np_x = np.random.randn(batch, 16).astype(np.float32)
        x = nb.Tensor.from_dlpack(np_x)
        
        result = nb.vmap(f)(x)
        expected = np.sum(np_x, axis=1)
        
        assert tuple(int(d) for d in result.shape) == (batch,)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

# ============================================================================
# Composite - Realistic MLP Layer
# ============================================================================

class TestVmapShardingComposite:
    """Test realistic composite functions with vmap+sharding."""
    
    def test_mlp_layer_megatron(self):
        """Full MLP: column parallel → ReLU → row parallel."""
        batch, hidden, ffn = 4, 16, 32
        mesh = DeviceMesh("mesh_mlp", (2,), ("tp",))
        
        def mlp(x, w1, w2, b1):
            # Layer 1: Column parallel
            w1_sharded = w1.shard(mesh, P(None, "tp"))
            h = nb.matmul(x, w1_sharded)
            
            # Bias
            b1_sharded = b1.shard(mesh, P("tp"))
            h = nb.add(h, b1_sharded)
            
            # Activation
            h = nb.relu(h)
            
            # Layer 2: Row parallel (AllReduce)
            w2_sharded = w2.shard(mesh, P("tp", None))
            out = nb.matmul(h, w2_sharded)
            
            return out
        
        np_x = np.random.randn(batch, hidden).astype(np.float32) * 0.1
        np_w1 = np.random.randn(hidden, ffn).astype(np.float32) * 0.1
        np_w2 = np.random.randn(ffn, hidden).astype(np.float32) * 0.1
        np_b1 = np.random.randn(ffn).astype(np.float32) * 0.1
        
        x = nb.Tensor.from_dlpack(np_x)
        w1 = nb.Tensor.from_dlpack(np_w1)
        w2 = nb.Tensor.from_dlpack(np_w2)
        b1 = nb.Tensor.from_dlpack(np_b1)
        
        result = nb.vmap(mlp, in_axes=(0, None, None, None))(x, w1, w2, b1)
        
        # Expected
        h1 = np_x @ np_w1 + np_b1
        h1 = np.maximum(h1, 0)
        expected = h1 @ np_w2
        
        assert tuple(int(d) for d in result.shape) == (batch, hidden)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4)
