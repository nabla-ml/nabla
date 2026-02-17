import numpy as np

import nabla as nb
from nabla import ops
from nabla.core.sharding import DeviceMesh, DimSpec


def test_dp_simple():
    DP_SIZE = 2
    DIM = 4
    mesh = DeviceMesh("dp", (DP_SIZE,), ("dp",))
    print(f"Running Simple DP Test (DP={DP_SIZE})")

    np.random.seed(42)
    w_np = np.random.randn(DIM, DIM).astype(np.float32)
    x_np = np.random.randn(DP_SIZE, DIM).astype(np.float32)
    y_np = np.random.randn(DP_SIZE, DIM).astype(np.float32)

    # Weights: Replicated on 'dp'
    w_spec = [None, None]
    w_nb = ops.shard(nb.Tensor.from_dlpack(w_np), mesh, w_spec).realize()

    # Data: Sharded on 'dp'
    x_spec = [DimSpec.from_raw("dp"), None]
    x_nb = ops.shard(nb.Tensor.from_dlpack(x_np), mesh, x_spec).realize()
    y_nb = ops.shard(nb.Tensor.from_dlpack(y_np), mesh, x_spec).realize()

    def loss_fn(x, w, y):
        pred = ops.matmul(x, w)
        diff = pred - y
        return ops.mean(diff * diff)

    from nabla.core.autograd import grad

    grad_fn = grad(loss_fn, argnums=1)

    w_grad = grad_fn(x_nb, w_nb, y_nb)
    w_grad_np = w_grad.to_numpy()

    # JAX
    import jax
    import jax.numpy as jnp

    def jax_loss(x, w, y):
        pred = x @ w
        return jnp.mean((pred - y) ** 2)

    jax_grad_fn = jax.grad(jax_loss, argnums=1)
    w_ref = jax_grad_fn(x_np, w_np, y_np)

    print("Nabla Grad Sample:", w_grad_np[0, :3])
    print("JAX Grad Sample:  ", w_ref[0, :3])

    diff = np.max(np.abs(w_grad_np - w_ref))
    print(f"Max Diff: {diff:.6f}")
    assert diff < 1e-5


if __name__ == "__main__":
    test_dp_simple()
