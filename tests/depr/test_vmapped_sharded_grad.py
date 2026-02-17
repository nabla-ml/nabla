import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb
from nabla.core.autograd import backward_on_trace
from nabla.core.graph.tracing import trace
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.transforms import vmap


def test_vmapped_sharded_grad():
    """Test VJP rules on sharded and vmapped inputs."""
    print("\n" + "=" * 70)
    print("Test: VMapped + Sharded Gradients")
    print("=" * 70)

    mesh = DeviceMesh("mesh", (2,), ("tp",))

    # 8 samples, each size (4, 4)
    # Total shape: (8, 4, 4)
    np.random.seed(42)
    x_np = np.random.randn(8, 4, 4).astype(np.float32)
    y_np = np.random.randn(8, 4, 4).astype(np.float32)

    # 1. JAX Reference
    # vmap over the first axis (8)
    def simple_op(x, y):
        return jnp.sum(jax.nn.relu(x) * y)

    def jax_vmap_fn(x, y):
        # vmap returns (8,)
        res = jax.vmap(simple_op)(x, y)
        return jnp.sum(res)

    grad_fn = jax.grad(jax_vmap_fn, argnums=(0, 1))
    grads_jax = grad_fn(x_np, y_np)

    # 2. Nabla
    x_nb = nb.Tensor.from_dlpack(x_np.copy())
    y_nb = nb.Tensor.from_dlpack(y_np.copy())

    # Shard the batch dimension (8) across 2 devices
    x_nb = nb.ops.shard(x_nb, mesh, [DimSpec(["tp"]), DimSpec([]), DimSpec([])])
    y_nb = nb.ops.shard(y_nb, mesh, [DimSpec(["tp"]), DimSpec([]), DimSpec([])])

    def nabla_inner_fn(x, y):
        # Operations inside vmap see logical shape (4, 4)
        r = nb.ops.relu(x)
        m = nb.ops.mul(r, y)
        # sum over (4, 4) -> scalar (inside vmap)
        return nb.ops.reduce_sum(m, axis=0).sum(0)

    def nabla_outer_fn(x, y):
        # vmap over 0, out_axes=0 -> result shape (8,)
        v_res = vmap(nabla_inner_fn, in_axes=0, out_axes=0)(x, y)
        # sum over (8,) -> scalar
        return nb.ops.reduce_sum(v_res, axis=0)

    # Note: In our current Trace/backward_on_trace, we don't handle `vmap` as a meta-op,
    # it is fully unfolded into basic ops (incr_batch_dims, etc.)
    traced = trace(nabla_outer_fn, x_nb, y_nb)
    print("\nCaptured Trace (VMapped + Sharded):")
    print(traced)

    # Backward
    cotangent = nb.Tensor.from_dlpack(np.array(1.0, dtype=np.float32))
    grads_nb = backward_on_trace(traced, cotangent)

    # Verify
    def to_np(t):
        from nabla.core.graph.engine import GRAPH

        if not t._impl.is_realized:
            GRAPH.evaluate(t)
        return np.asarray(t.to_numpy())

    grad_x_nb = to_np(grads_nb[x_nb])
    grad_y_nb = to_np(grads_nb[y_nb])

    print(f"\nGradient x shape: Nabla {grad_x_nb.shape}, JAX {grads_jax[0].shape}")

    np.testing.assert_allclose(grad_x_nb, grads_jax[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad_y_nb, grads_jax[1], rtol=1e-5, atol=1e-6)

    print("\n✓ SUCCESS: VMapped + Sharded gradients match JAX!")


if __name__ == "__main__":
    try:
        test_vmapped_sharded_grad()
    except Exception:
        print("\n✗ Test FAILED:")
        import traceback

        traceback.print_exc()
        exit(1)
