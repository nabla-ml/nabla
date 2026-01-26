
import numpy as np
import pytest
import nabla as nb
from nabla.core.sharding import DeviceMesh, DimSpec
from .utils import check_vjp, HAS_JAX

np.random.seed(1337)

def test_sharded_tp_1d_basic():
    print("\n=== Testing Sharded TP (1D) Basic Ops ===")
    
    mesh = DeviceMesh("mesh_1d", (2,), ("tp",))
    
    x = np.random.randn(8, 8).astype(np.float32)
    spec = [DimSpec([]), DimSpec(["tp"])]
    
    check_vjp(
        "sharded_relu_1d",
        nb.ops.relu,
        pytest.importorskip("jax.nn").relu if HAS_JAX else None,
        (x,),
        mesh=mesh,
        input_specs=(spec,)
    )
    
    y = np.random.randn(8, 8).astype(np.float32)
    check_vjp(
        "sharded_add_1d",
        nb.add,
        np.add if HAS_JAX else None,
        (x, y),
        mesh=mesh,
        input_specs=(spec, spec)
    )

def test_sharded_matmul_1d_col_row():
    print("\n=== Testing Sharded Matmul 1D (Split K) ===")
    
    mesh = DeviceMesh("mesh_1d_mm", (2,), ("tp",))
    
    x = np.random.randn(8, 4).astype(np.float32)
    y = np.random.randn(4, 8).astype(np.float32)
    
    x_spec = [DimSpec([]), DimSpec(["tp"])]
    y_spec = [DimSpec(["tp"]), DimSpec([])]
    
    check_vjp(
        "sharded_matmul_split_k",
        nb.matmul,
        np.matmul if HAS_JAX else None,
        (x, y),
        mesh=mesh,
        input_specs=(x_spec, y_spec)
    )

def test_sharded_complex_mesh_hybrid():
    print("\n=== Testing Sharded Hybrid Mesh (2, 4) ===")
    
    mesh = DeviceMesh("mesh_2d_hybrid", (2, 4), ("dp", "tp"))
    
    B, D = 16, 16
    x = np.random.randn(B, D).astype(np.float32)
    
    spec = [DimSpec(["dp"]), DimSpec(["tp"])]
    
    check_vjp(
        "hybrid_sigmoid",
        nb.ops.sigmoid,
        pytest.importorskip("jax.nn").sigmoid if HAS_JAX else None,
        (x,),
        mesh=mesh,
        input_specs=(spec,)
    )
    
    def reduce_feat(t):
        return nb.reduce_sum(t, axis=1)
        
    def reduce_feat_jax(t):
        import jax.numpy as jnp
        return jnp.sum(t, axis=1)

    check_vjp(
        "hybrid_reduce_sum_tp",
        reduce_feat,
        reduce_feat_jax if HAS_JAX else None,
        (x,),
        mesh=mesh,
        input_specs=(spec,)
    )

    def reduce_batch(t):
        return nb.reduce_sum(t, axis=0)

    def reduce_batch_jax(t):
        import jax.numpy as jnp
        return jnp.sum(t, axis=0)

    check_vjp(
        "hybrid_reduce_sum_dp",
        reduce_batch,
        reduce_batch_jax if HAS_JAX else None,
        (x,),
        mesh=mesh,
        input_specs=(spec,)
    )

def test_sharded_reduction_all_axes():
    print("\n=== Testing Global Reduction 2D ===")
    mesh = DeviceMesh("mesh_2d_global", (2, 2), ("x", "y"))
    
    x = np.random.randn(4, 4).astype(np.float32)
    spec = [DimSpec(["x"]), DimSpec(["y"])]
    
    def global_sum(t):
        return nb.reduce_sum(t)
        
    check_vjp(
        "global_sum_2d",
        global_sum,
        np.sum if HAS_JAX else None,
        (x,),
        mesh=mesh,
        input_specs=(spec,)
    )

def test_sharded_mlp_grads_explicit():
    print("\n=== Testing Full Sharded MLP Grads ===")
    mesh = DeviceMesh("dgx", (2,), ("tp",))

    W_data = np.random.randn(8, 8).astype(np.float32)
    X_data = np.random.randn(4, 8).astype(np.float32)

    W = nb.ops.shard(
        nb.Tensor.from_dlpack(W_data), mesh, [DimSpec([]), DimSpec(["tp"])]
    )
    X = nb.ops.shard(
        nb.Tensor.from_dlpack(X_data), mesh, [DimSpec([]), DimSpec([])]
    )

    def mlp_step(x, w):
        h = nb.matmul(x, w)
        return nb.reduce_sum(h)

    t = nb.core.graph.tracing.trace(mlp_step, X, W)
    cot = nb.ops.full_like(t.outputs, 1.0)
    grads = nb.core.autograd.backward_on_trace(t, cot)

    grad_W = grads[W]
    grad_X = grads[X]

    from nabla.core.sharding.spec import needs_reshard
    assert not needs_reshard(grad_W.sharding, W.sharding)
    assert not needs_reshard(grad_X.sharding, X.sharding)

    if HAS_JAX:
        import jax.numpy as jnp
        import jax
        def expected_fn(x, w):
            h = jnp.matmul(x, w)
            return jnp.sum(h)
        
        grad_fn = jax.grad(expected_fn, argnums=(0, 1))
        gX_j, gW_j = grad_fn(X_data, W_data)
        
        np.testing.assert_allclose(grad_W.to_numpy(), gW_j, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(grad_X.to_numpy(), gX_j, rtol=1e-5, atol=1e-5)
    
    print("✓ MLP gradients verified against JAX")

if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
