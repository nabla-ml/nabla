import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb
from nabla.core.autograd import backward_on_trace
from nabla.core.graph.tracing import trace
from nabla.core.sharding import DeviceMesh, DimSpec


def test_sharded_relu_mul():
    """Simple test with sharded inputs for ReLU and Multiplication."""
    print("\n" + "=" * 70)
    print("Test: Sharded ReLU and Mul - grad(sum(relu(x) * y))")
    print("=" * 70)

    # 1. Setup Mesh
    mesh = DeviceMesh("mesh", (2,), ("tp",))

    # Random seeding for reproducibility
    np.random.seed(42)
    x_np = np.random.randn(8, 4).astype(np.float32)
    y_np = np.random.randn(8, 4).astype(np.float32)

    # 2. JAX Reference
    def jax_fn(x, y):
        # We use a sum at the end as requested
        out = jax.nn.relu(x) * y
        return jnp.sum(out)

    grad_fn = jax.grad(jax_fn, argnums=(0, 1))
    grads_jax = grad_fn(x_np, y_np)

    # 3. Nabla Computation
    x_nb = nb.Tensor.from_dlpack(x_np.copy())
    y_nb = nb.Tensor.from_dlpack(y_np.copy())

    # Shard inputs along the first axis (8)
    # 8 elements / 2 devices = 4 elements per shard
    x_nb = nb.ops.shard(x_nb, mesh, [DimSpec(["tp"]), DimSpec([])])
    y_nb = nb.ops.shard(y_nb, mesh, [DimSpec(["tp"]), DimSpec([])])

    print(f"Input x sharding: {x_nb.sharding}")
    print(f"Input y sharding: {y_nb.sharding}")

    def nabla_fn(x, y):
        # Unary: relu
        r = nb.ops.relu(x)
        # Binary: mul
        m = nb.ops.mul(r, y)
        # Reduction: sum over all axes (result is scalar)
        s1 = nb.ops.reduce_sum(m, axis=0)
        return nb.ops.reduce_sum(s1, axis=0)

    # Trace the computation
    traced = trace(nabla_fn, x_nb, y_nb)
    print("\nCaptured Trace:")
    print(traced)

    # 4. Backward
    # Since output is a scalar, cotangent is 1.0
    cotangent = nb.Tensor.from_dlpack(np.array(1.0, dtype=np.float32))
    grads_nb = backward_on_trace(traced, cotangent)

    # 5. Verification
    def to_np(t):
        from nabla.core.graph.engine import GRAPH

        if not t._impl.is_realized:
            GRAPH.evaluate(t)
        return np.asarray(t.to_numpy())

    grad_x_nb = to_np(grads_nb[x_nb])
    grad_y_nb = to_np(grads_nb[y_nb])

    print(f"\nGradient x shape: Nabla {grad_x_nb.shape}, JAX {grads_jax[0].shape}")
    print(f"Gradient y shape: Nabla {grad_y_nb.shape}, JAX {grads_jax[1].shape}")

    np.testing.assert_allclose(grad_x_nb, grads_jax[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad_y_nb, grads_jax[1], rtol=1e-5, atol=1e-6)

    print("\n✓ SUCCESS: Sharded ReLU/Mul gradients match JAX!")


if __name__ == "__main__":
    try:
        test_sharded_relu_mul()
    except Exception:
        print("\n✗ Test FAILED:")
        import traceback

        traceback.print_exc()
        exit(1)
