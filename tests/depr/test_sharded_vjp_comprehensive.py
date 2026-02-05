import numpy as np
import jax
import jax.numpy as jnp
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.transforms import vmap


def assert_grads_match(
    nabla_fn, jax_fn, inputs_np, mesh, mesh_dim_specs, name, vmapping=None
):
    """Generic helper to compare Nabla gradients against JAX."""
    print(f"\nTesting: {name}")

    # 1. JAX Gradients
    if vmapping:
        # vmapping is (in_axes, out_axes)
        jax_vmap_fn = jax.vmap(jax_fn, in_axes=vmapping[0], out_axes=vmapping[1])
        # If we want a scalar output for jax.grad, we sum the result
        jax_grad_target = lambda *args: jnp.sum(jax_vmap_fn(*args))
    else:
        jax_grad_target = lambda *args: jnp.sum(jax_fn(*args))

    grad_fn_jax = jax.grad(jax_grad_target, argnums=tuple(range(len(inputs_np))))
    grads_jax = grad_fn_jax(*inputs_np)
    if not isinstance(grads_jax, (list, tuple)):
        grads_jax = (grads_jax,)

    # 2. Nabla Gradients
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
        nabla_target = lambda *args: nb.ops.reduce_sum(
            nabla_fn(*args), axis=list(range(len(nabla_fn(*args).shape)))
        )

    traced = trace(nabla_target, *sharded_inputs)
    # Trace output is always a scalar in our test setup
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
        np.testing.assert_allclose(
            gnb_np,
            gjax,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Gradient mismatch for input {i} in '{name}'",
        )

    print(f"  ✓ {name} passed!")


def test_all_ops_sharded_vjp():
    mesh = DeviceMesh("mesh", (2,), ("tp",))

    # 1. Unary Operations
    unary_ops = [
        ("relu", nb.ops.relu, jax.nn.relu),
        ("sigmoid", nb.ops.sigmoid, jax.nn.sigmoid),
        ("tanh", nb.ops.tanh, jnp.tanh),
        ("exp", nb.ops.exp, jnp.exp),
        ("neg", nb.ops.neg, lambda x: -x),
        ("abs", nb.ops.abs, jnp.abs),
        ("log", nb.ops.log, jnp.log),
        ("sqrt", nb.ops.sqrt, jnp.sqrt),
    ]

    for name, n_op, j_op in unary_ops:
        x_np = np.random.uniform(0.1, 1.1, (8, 4)).astype(np.float32)
        assert_grads_match(
            n_op, j_op, (x_np,), mesh, ([["tp"], []],), name, vmapping=(0, 0)
        )

    # 2. Binary Operations
    binary_ops = [
        ("add", nb.ops.add, jnp.add),
        ("mul", nb.ops.mul, jnp.multiply),
        ("sub", nb.ops.sub, jnp.subtract),
        ("div", nb.ops.div, jnp.divide),
    ]

    for name, n_op, j_op in binary_ops:
        x_np = np.random.randn(8, 4).astype(np.float32)
        y_np = np.random.uniform(0.5, 1.5, (8, 4)).astype(np.float32)
        assert_grads_match(
            n_op,
            j_op,
            (x_np, y_np),
            mesh,
            ([["tp"], []], [["tp"], []]),
            name,
            vmapping=(0, 0),
        )

    # 3. Matmul
    # (B, M, K) @ (K, N) -> (B, M, N) inside vmap
    # or (B, M, K) @ (B, K, N)
    x_np = np.random.randn(8, 4, 3).astype(np.float32)
    y_np = np.random.randn(8, 3, 5).astype(np.float32)
    assert_grads_match(
        nb.ops.matmul,
        jnp.matmul,
        (x_np, y_np),
        mesh,
        ([["tp"], [], []], [["tp"], [], []]),
        "matmul",
        vmapping=(0, 0),
    )

    # 4. Reduction (inside vmap)
    # vmap(reduce_sum)
    def n_sum(x):
        return nb.ops.reduce_sum(x, axis=0)  # reduce over rows of (4, 4)

    def j_sum(x):
        return jnp.sum(x, axis=0)

    x_np = np.random.randn(8, 4, 4).astype(np.float32)
    assert_grads_match(
        n_sum,
        j_sum,
        (x_np,),
        mesh,
        ([["tp"], [], []],),
        "reduce_sum_vmap",
        vmapping=(0, 0),
    )

    # 5. View Ops
    # unsqueeze
    def n_unsq(x):
        return nb.ops.unsqueeze(x, axis=1)

    def j_unsq(x):
        return jnp.expand_dims(x, axis=1)

    x_np = np.random.randn(10, 5).astype(np.float32)  # vmap(10), inner(5)
    assert_grads_match(
        n_unsq,
        j_unsq,
        (x_np,),
        mesh,
        ([["tp"], []],),
        "unsqueeze_vmap",
        vmapping=(0, 0),
    )

    # squeeze
    def n_sq(x):
        return nb.ops.squeeze(x, axis=0)

    def j_sq(x):
        return jnp.squeeze(x, axis=0)

    x_np = np.random.randn(8, 1, 4).astype(np.float32)
    assert_grads_match(
        n_sq, j_sq, (x_np,), mesh, ([["tp"], [], []],), "squeeze_vmap", vmapping=(0, 0)
    )

    # swap_axes
    def n_swap(x):
        return nb.ops.swap_axes(x, 0, 1)

    def j_swap(x):
        return jnp.swapaxes(x, 0, 1)

    x_np = np.random.randn(8, 4, 5).astype(np.float32)
    assert_grads_match(
        n_swap,
        j_swap,
        (x_np,),
        mesh,
        ([["tp"], [], []],),
        "swap_axes_vmap",
        vmapping=(0, 0),
    )

    # 6. Softmax (Composition)
    from nabla.ops.unary import softmax

    def j_softmax(x):
        return jax.nn.softmax(x, axis=-1)

    x_np = np.random.randn(8, 16).astype(np.float32)
    assert_grads_match(
        softmax,
        j_softmax,
        (x_np,),
        mesh,
        ([["tp"], []],),
        "softmax_vmap",
        vmapping=(0, 0),
    )

    # 7. Communication Ops (explicit)
    # reshard
    def n_reshard(x):
        s2 = [DimSpec([]), DimSpec(["tp"])]
        return nb.ops.reshard(x, mesh, s2)

    def j_reshard(x):
        return x  # identity in JAX

    x_np = np.random.randn(8, 4).astype(np.float32)
    assert_grads_match(
        n_reshard,
        j_reshard,
        (x_np,),
        mesh,
        ([["tp"], []],),
        "reshard_vmap",
        vmapping=(0, 0),
    )


if __name__ == "__main__":
    np.random.seed(42)
    try:
        test_all_ops_sharded_vjp()
        print("\n" + "=" * 70)
        print("✅ COMPREHENSIVE SHARDED VJP TEST PASSED!")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ COMPREHENSIVE SHARDED VJP TEST FAILED:")
        import traceback

        traceback.print_exc()
        exit(1)
