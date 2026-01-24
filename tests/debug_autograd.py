import unittest
import numpy as np
import jax
import jax.numpy as jnp
import nabla as nb
from nabla import ops
from nabla.core import trace
from nabla.core.autograd import backward_on_trace
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.transforms.vmap import vmap

class TestAutogradDebug(unittest.TestCase):
    def test_01_simple_matmul_relu(self):
        print("\n[Test 01] Simple Matmul + ReLU (Unsharded)")
        def fn(x, w):
            return ops.relu(x @ w)

        x_np = np.random.randn(4, 8).astype(np.float32)
        w_np = np.random.randn(8, 16).astype(np.float32)

        x_nb = nb.Tensor.from_dlpack(x_np.copy())
        w_nb = nb.Tensor.from_dlpack(w_np.copy())

        traced = trace(fn, x_nb, w_nb)
        print(f"  Nodes: {len(traced.nodes)}")
        
        # Loss = sum(output)
        def loss_fn(x, w):
            return jnp.sum(jax.nn.relu(x @ w))
        
        gw_jax = jax.grad(loss_fn, argnums=1)(x_np, w_np)
        gx_jax = jax.grad(loss_fn, argnums=0)(x_np, w_np)

        out_nb = fn(x_nb, w_nb)
        cotangent = ops.ones_like(out_nb)
        grads = backward_on_trace(traced, cotangent)

        np.testing.assert_allclose(grads[w_nb].to_numpy(), gw_jax, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(grads[x_nb].to_numpy(), gx_jax, rtol=1e-5, atol=1e-5)
        print("  ✓ Matmul + ReLU gradients matched!")

    def test_02_simple_vmap_unsharded(self):
        print("\n[Test 02] Simple VMap (Unsharded)")
        def layer(x, w):
            return ops.relu(x @ w)

        vmapped_layer = vmap(layer, in_axes=(0, None), out_axes=0)

        x_np = np.random.randn(2, 4, 8).astype(np.float32)
        w_np = np.random.randn(8, 16).astype(np.float32)

        x_nb = nb.Tensor.from_dlpack(x_np.copy())
        w_nb = nb.Tensor.from_dlpack(w_np.copy())

        def full_fn(x, w):
            y = vmapped_layer(x, w)
            return ops.reduce_sum(y, axis=list(range(len(y.shape))))

        traced = trace(full_fn, x_nb, w_nb)
        print(f"  Nodes: {len(traced.nodes)}")
        
        def ref_fn(x, w):
            y = jax.vmap(lambda xi: jax.nn.relu(xi @ w))(x)
            return jnp.sum(y)

        gx_jax, gw_jax = jax.grad(ref_fn, argnums=(0, 1))(x_np, w_np)

        cotangent = nb.Tensor.from_dlpack(np.array(1.0, dtype=np.float32))
        grads = backward_on_trace(traced, cotangent)

        np.testing.assert_allclose(grads[x_nb].to_numpy(), gx_jax, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(grads[w_nb].to_numpy(), gw_jax, rtol=1e-5, atol=1e-5)
        print("  ✓ VMap (unsharded) gradients matched!")

    def test_03_data_parallel_vmap(self):
        print("\n[Test 03] Data Parallel VMap")
        np.random.seed(42)
        mesh = DeviceMesh("dp", (2,), ("data",))
        
        def layer(x, w):
            return ops.relu(x @ w)

        vmapped_layer = vmap(layer, in_axes=(0, None), out_axes=0, spmd_axis_name="data", mesh=mesh)

        x_np = np.random.randn(4, 8, 16).astype(np.float32)
        w_np = np.random.randn(16, 32).astype(np.float32)

        x_nb = nb.Tensor.from_dlpack(x_np.copy())
        w_nb = nb.Tensor.from_dlpack(w_np.copy())
        
        # Shard x along batch dim
        x_nb = ops.shard(x_nb, mesh, [DimSpec(["data"]), DimSpec([]), DimSpec([])])
        
        # Explicitly replicate w to ensure correct gradient accumulation (AllReduce)
        w_nb = ops.shard(w_nb, mesh, [DimSpec([]), DimSpec([])])

        def full_fn(x, w):
            y = vmapped_layer(x, w)
            return ops.reduce_sum(y, axis=list(range(len(y.shape))))

        traced = trace(full_fn, x_nb, w_nb)
        with open("trace.txt", "w") as f:
            f.write(str(traced))

        def ref_fn(x, w):
            y = jax.vmap(lambda xi: jax.nn.relu(xi @ w))(x)
            return jnp.sum(y)

        gx_jax, gw_jax = jax.grad(ref_fn, argnums=(0, 1))(x_np, w_np)

        cotangent = nb.Tensor.from_dlpack(np.array(1.0, dtype=np.float32))
        grads = backward_on_trace(traced, cotangent)

        print("\n  Checking Inputs gradients (x)...")
        np.testing.assert_allclose(grads[x_nb].to_numpy(), gx_jax, rtol=1e-5, atol=1e-5)
        print("  ✓ x gradients matched!")

        print("  Checking Weights gradients (w)...")
        np.testing.assert_allclose(grads[w_nb].to_numpy(), gw_jax, rtol=1e-5, atol=1e-5)
        print("  ✓ w gradients matched!")
        print("  ✓ DP VMap gradients matched!")

if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestAutogradDebug("test_03_data_parallel_vmap"))
    unittest.TextTestRunner().run(suite)
