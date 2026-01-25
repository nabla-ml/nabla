import unittest
import numpy as np
import jax
import jax.numpy as jnp
import nabla as nb
from nabla import ops
from nabla.core import trace
from nabla.core.autograd import backward_on_trace
from nabla.core import pytree
from nabla.core.sharding import DeviceMesh, DimSpec


class TestMatmulShardingVariations(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.mesh = DeviceMesh("dp", (2,), ("data",))
        # Shapes
        self.B, self.M, self.K, self.N = 4, 8, 16, 8

    def _run_matmul_test(self, x_sharding_spec, w_sharding_spec, case_name):
        print(f"\n[{case_name}] X:{x_sharding_spec} @ W:{w_sharding_spec}")

        def fn(x, w):
            return x @ w

        x_np = np.random.randn(self.B, self.M, self.K).astype(np.float32)
        w_np = np.random.randn(self.B, self.K, self.N).astype(np.float32)

        # JAX Ref
        def ref_fn(x, w):
            return jnp.sum(x @ w)

        gx_jax, gw_jax = jax.grad(ref_fn, argnums=(0, 1))(x_np, w_np)

        # Nabla
        x_nb = nb.Tensor.from_dlpack(x_np.copy())
        w_nb = nb.Tensor.from_dlpack(w_np.copy())

        if x_sharding_spec:
            x_nb = ops.shard(x_nb, self.mesh, x_sharding_spec)
        if w_sharding_spec:
            w_nb = ops.shard(w_nb, self.mesh, w_sharding_spec)

        traced = trace(fn, x_nb, w_nb)

        cot_np = np.ones((self.B, self.M, self.N), dtype=np.float32)
        cotangent = nb.Tensor.from_dlpack(cot_np)

        # Determine cotangent sharding
        # For Case 3, Output is Replicated because contraction axis (K) is sharded.
        is_case_3 = case_name == "Case 3: Contracting Axis Sharded"
        if not is_case_3:
            # Assume sharding on axis 0
            cotangent = ops.shard(
                cotangent, self.mesh, [DimSpec(["data"]), DimSpec([]), DimSpec([])]
            )

        try:
            grads = backward_on_trace(traced, cotangent)

            print("  Checking dX...")
            np.testing.assert_allclose(
                grads[x_nb].to_numpy(), gx_jax, rtol=1e-5, atol=1e-5
            )
            print("  ✓ dX matched")

            print("  Checking dW...")
            np.testing.assert_allclose(
                grads[w_nb].to_numpy(), gw_jax, rtol=1e-5, atol=1e-5
            )
            print("  ✓ dW matched")
        except AssertionError as e:
            print(f"  MISMATCH: {e}")
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise

    # def test_case_01(self):
    #     # Case 1: X (data, ., .), W (., ., .) -> dW needs Reduce
    #     self._run_matmul_test(
    #         [DimSpec(["data"]), DimSpec([]), DimSpec([])],
    #         [DimSpec([]), DimSpec([]), DimSpec([])],
    #         "Case 1: X sharded(0), W repl"
    #     )

    # def test_case_02(self):
    #     # Case 2: X (., ., .), W (data, ., .) -> dX needs Reduce
    #     self._run_matmul_test(
    #         [DimSpec([]), DimSpec([]), DimSpec([])],
    #         [DimSpec(["data"]), DimSpec([]), DimSpec([])],
    #         "Case 2: X repl, W sharded(0)"
    #     )

    def test_case_03(self):
        # Case 3: X (., ., data), W (., data, .) -> Dot product sharding!
        # Contraction happens on a sharded axis.
        self._run_matmul_test(
            [DimSpec([]), DimSpec([]), DimSpec(["data"])],
            [DimSpec([]), DimSpec(["data"]), DimSpec([])],
            "Case 3: Contracting Axis Sharded",
        )


if __name__ == "__main__":
    unittest.main()
