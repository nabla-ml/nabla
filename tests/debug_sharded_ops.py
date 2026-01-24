
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import nabla as nb
from nabla import ops
from nabla.core import trace
from nabla.core.autograd import backward_on_trace
from nabla.core.sharding import DeviceMesh, DimSpec

class TestShardedOps(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create a 2-device mesh
        self.mesh = DeviceMesh("dp", (2,), ("data",))

    def test_01_sharded_relu(self):
        print("\n[Test 01] Sharded ReLU")
        
        def fn(x):
            return ops.relu(x)

        # Input: (4, 4), sharded on axis 0 ("data" -> split into 2 chunks of (2,4))
        x_np = np.random.randn(4, 4).astype(np.float32)
        x_nb = nb.Tensor.from_dlpack(x_np.copy())
        x_nb = ops.shard(x_nb, self.mesh, [DimSpec(["data"]), DimSpec([])])

        traced = trace(fn, x_nb)
        
        # JAX reference
        cot_np = np.ones_like(x_np)
        gx_jax = jax.grad(lambda x: jnp.sum(jax.nn.relu(x)))(x_np)

        cotangent = nb.Tensor.from_dlpack(cot_np)
        # Cotangent needs to be sharded matching the output (which inherits input sharding)
        cotangent = ops.shard(cotangent, self.mesh, [DimSpec(["data"]), DimSpec([])])
        
        grads = backward_on_trace(traced, cotangent)
        
        np.testing.assert_allclose(grads[x_nb].to_numpy(), gx_jax, rtol=1e-5, atol=1e-5)
        print("  ✓ Sharded ReLU matched")

    def test_02_sharded_reduce_sum(self):
        print("\n[Test 02] Sharded Reduce Sum (AllReduce)")
        
        def fn(x):
            # Sum over everyone -> Output is scalar (replicated)
            return ops.reduce_sum(x, axis=None)

        # Input: (4, 4), sharded on axis 0
        x_np = np.random.randn(4, 4).astype(np.float32)
        x_nb = nb.Tensor.from_dlpack(x_np.copy())
        x_nb = ops.shard(x_nb, self.mesh, [DimSpec(["data"]), DimSpec([])])
        
        traced = trace(fn, x_nb)
        
        # VJP of Sum is Broadcast. 
        # Input sharded -> Output replicated.
        # Backward: Cotangent (replicated) -> Broadcast -> Output (Sharded?)
        
        # JAX reference
        def ref_fn(x):
            return jnp.sum(x)
        gx_jax = jax.grad(ref_fn)(x_np)

        cotangent = nb.Tensor.from_dlpack(np.array(1.0, dtype=np.float32)) # Replicated scalar
        grads = backward_on_trace(traced, cotangent)

        np.testing.assert_allclose(grads[x_nb].to_numpy(), gx_jax, rtol=1e-5, atol=1e-5)
        print("  ✓ Sharded Reduce Sum matched")

    def test_03_sharded_matmul_rhs_replicated(self):
        print("\n[Test 03] Sharded Matmul (X_sharded @ W_replicated -> Y_sharded)")
        
        def fn(x, w):
            return x @ w

        # X: (4, 8) sharded on 0
        # W: (8, 4) replicated
        x_np = np.random.randn(4, 8).astype(np.float32)
        w_np = np.random.randn(8, 4).astype(np.float32)

        x_nb = nb.Tensor.from_dlpack(x_np.copy())
        w_nb = nb.Tensor.from_dlpack(w_np.copy())
        
        x_nb = ops.shard(x_nb, self.mesh, [DimSpec(["data"]), DimSpec([])])
        w_nb = ops.shard(w_nb, self.mesh, [DimSpec([]), DimSpec([])])

        traced = trace(fn, x_nb, w_nb)
        
        # JAX reference
        def ref_fn(x, w):
            return jnp.sum(x @ w)
        
        gx_jax, gw_jax = jax.grad(ref_fn, argnums=(0,1))(x_np, w_np)
        
        # Output is (4, 4) sharded on 0
        cot_np = np.ones((4, 4), dtype=np.float32)
        cotangent = nb.Tensor.from_dlpack(cot_np)
        cotangent = ops.shard(cotangent, self.mesh, [DimSpec(["data"]), DimSpec([])])
        
        grads = backward_on_trace(traced, cotangent)
        
        print("  Checking dX...")
        np.testing.assert_allclose(grads[x_nb].to_numpy(), gx_jax, rtol=1e-5, atol=1e-5)
        print("  ✓ dX matched")

        print("  Checking dW...")
        np.testing.assert_allclose(grads[w_nb].to_numpy(), gw_jax, rtol=1e-5, atol=1e-5)
        print("  ✓ dW matched")

if __name__ == "__main__":
    unittest.main()
