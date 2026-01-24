import numpy as np
import jax
import jax.numpy as jnp
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.transforms import vmap

def assert_grads_match(nabla_fn, jax_fn, inputs_np, mesh, mesh_dim_specs, name, vmapping=None):
    """Generic helper to compare Nabla gradients against JAX."""
    print(f"\nTesting: {name}")
    
    # 1. JAX Reference
    if vmapping:
        jax_vmap_fn = jax.vmap(jax_fn, in_axes=vmapping[0], out_axes=vmapping[1])
        jax_grad_target = lambda *args: jnp.sum(jax_vmap_fn(*args))
    else:
        jax_grad_target = lambda *args: jnp.sum(jax_fn(*args))
        
    grad_fn_jax = jax.grad(jax_grad_target, argnums=tuple(range(len(inputs_np))))
    grads_jax = grad_fn_jax(*inputs_np)
    if not isinstance(grads_jax, (list, tuple)):
        grads_jax = (grads_jax,)

    # 2. Nabla Computation
    inputs_nb = [nb.Tensor.from_dlpack(x.copy()) for x in inputs_np]
    
    # Apply sharding
    sharded_inputs = []
    for inp, specs in zip(inputs_nb, mesh_dim_specs):
        if specs:
            sharded_inputs.append(nb.ops.shard(inp, mesh, [DimSpec(s) for s in specs]))
        else:
            sharded_inputs.append(inp)
            
    if vmapping:
        def outer_fn(*args):
             res = vmap(nabla_fn, in_axes=vmapping[0], out_axes=vmapping[1])(*args)
             return nb.ops.reduce_sum(res, axis=list(range(len(res.shape))))
        nabla_target = outer_fn
    else:
        def nabla_target(*args):
             res = nabla_fn(*args)
             return nb.ops.reduce_sum(res, axis=list(range(len(res.shape))))

    traced = trace(nabla_target, *sharded_inputs)
    cotangent = nb.Tensor.from_dlpack(np.array(1.0, dtype=np.float32))
    grads_nb_map = backward_on_trace(traced, cotangent)
    
    grads_nb = [grads_nb_map[inp] for inp in sharded_inputs]

    # 3. Compare
    def to_np(t):
        from nabla.core.graph.engine import GRAPH
        if not t._impl.is_realized:
            GRAPH.evaluate(t)
        return np.asarray(t.to_numpy())

    for i, (gnb, gjax) in enumerate(zip(grads_nb, grads_jax)):
        gnb_np = to_np(gnb)
        np.testing.assert_allclose(gnb_np, gjax, rtol=1e-4, atol=1e-5, 
                                   err_msg=f"Gradient mismatch for input {i} in '{name}'")
    
    print(f"  ✓ {name} passed!")

def test_complex_shardings():
    # 2D Mesh: DP=2, TP=2
    mesh_2d = DeviceMesh("mesh2d", (2, 2), ("dp", "tp"))
    
    # 1. 2D Mesh Gradient: y = relu(x), x sharded on (DP, TP)
    def n_relu(x): return nb.ops.relu(x)
    def j_relu(x): return jax.nn.relu(x)
    x_np = np.random.randn(8, 4).astype(np.float32)
    # Shard axis 0 on 'dp', axis 1 on 'tp'
    assert_grads_match(n_relu, j_relu, (x_np,), mesh_2d, ([["dp"], ["tp"]],), "relu_2d_mesh")

    # 2. Sharding Conflict: z = x + y, x sharded on DP, y sharded on TP
    # Nabla should reshard inputs to align them
    def n_add(x, y): return x + y
    def j_add(x, y): return x + y
    x_np = np.random.randn(8, 8).astype(np.float32)
    y_np = np.random.randn(8, 8).astype(np.float32)
    assert_grads_match(n_add, j_add, (x_np, y_np), mesh_2d, ([["dp"], []], [[], ["tp"]]), "add_conflict")

    # 3. TP-style Matrix Multiplication Chain
    def n_tp_chain(x, w1, w2):
        h = nb.ops.matmul(x, w1)
        h_act = nb.ops.relu(h)
        out = nb.ops.matmul(h_act, w2)
        return out
        
    def j_tp_chain(x, w1, w2):
        h = jnp.matmul(x, w1)
        h_act = jax.nn.relu(h)
        return jnp.matmul(h_act, w2)

    x_np = np.random.randn(4, 8).astype(np.float32)
    w1_np = np.random.randn(8, 16).astype(np.float32)
    w2_np = np.random.randn(16, 4).astype(np.float32)
    
    mesh_1d = DeviceMesh("mesh1d", (2,), ("tp",))
    
    specs = [
        [], # x
        [[], ["tp"]], # w1
        [["tp"], []], # w2
    ]
    assert_grads_match(n_tp_chain, j_tp_chain, (x_np, w1_np, w2_np), mesh_1d, specs, "tp_chain_grads")

    # 4. VMapped 2D Mesh with conflicts
    def n_vmap_2d(x):
        return nb.ops.exp(x)
    def j_vmap_2d(x):
        return jnp.exp(x)
    
    x_np = np.random.randn(8, 4).astype(np.float32)
    assert_grads_match(n_vmap_2d, j_vmap_2d, (x_np,), mesh_2d, ([["dp"], ["tp"]],), "vmap_2d_mesh", vmapping=(0, 0))

if __name__ == "__main__":
    np.random.seed(42)
    try:
        test_complex_shardings()
        print("\n" + "=" * 70)
        print("✅ COMPLEX SHARDINGS GRADIENT TESTS PASSED!")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ COMPLEX SHARDINGS GRADIENT TESTS FAILED:")
        import traceback
        traceback.print_exc()
        exit(1)
