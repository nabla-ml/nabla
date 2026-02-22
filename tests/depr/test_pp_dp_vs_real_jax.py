import unittest

import jax
import jax.numpy as jnp
import numpy as np

from nabla import ops
from nabla.transforms import grad
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.ops import communication
from nabla.transforms.vmap import vmap


class TestPP_DP_vs_Real_JAX(unittest.TestCase):
    def test_pp_dp_comparison(self):
        print("\n=== Comparing Nabla PP+DP Gradients vs JAX ===")

        # Shapes
        STAGES = 2
        DP = 2
        B_LOC = 2
        B_GLOBAL = B_LOC * DP
        D = 8

        # Use fixed seed for both
        key = jax.random.PRNGKey(1337)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Initialize Data (JAX)
        w1_jax = jax.random.normal(k1, (STAGES, D, D))
        w2_jax = jax.random.normal(k2, (STAGES, D, D))
        x_jax = jax.random.normal(k3, (STAGES, B_GLOBAL, D))
        target_jax = jax.random.normal(k4, (STAGES, B_GLOBAL, D))

        # Convert to Numpy for Nabla
        w1_np = np.array(w1_jax)
        w2_np = np.array(w2_jax)
        x_np = np.array(x_jax)
        target_np = np.array(target_jax)

        # --- JAX Implementation ---
        # We simulate the pipeline logic:
        # 1. vmap over stages
        # 2. explicit permutation
        # 3. loss

        def stage_fn_jax(x, w1, w2):
            # x: (B, D), w: (D, D)
            h = jax.nn.relu(x @ w1)
            out = h @ w2
            return out

        # Map over stages (axis 0)
        # x: (STAGES, B, D) -> out: (STAGES, B, D)
        stage_mapped_jax = jax.vmap(stage_fn_jax, in_axes=(0, 0, 0), out_axes=0)

        def loss_fn_jax(w1, w2, x, target):
            out = stage_mapped_jax(x, w1, w2)

            # Permutation: 0->1, 1->0 (simple swap for 2 stages)
            # Nabla ppermute was: [(0, 2), (1, 3), (2, 0), (3, 1)] for flat mesh
            # Logically this is just rolling the stage dimension by +1?
            # Stage 0 output goes to Stage 1. Stage 1 output goes to Stage 0.
            # out shape: (STAGES, B, D)
            # out[0] moves to index 1. out[1] moves to index 0.
            out_permuted = jnp.roll(out, shift=1, axis=0)

            diff = out_permuted - target
            return jnp.mean(diff * diff)

        jax_grad_fn = jax.grad(loss_fn_jax, argnums=(0, 1))
        gw1_jax, gw2_jax = jax_grad_fn(w1_jax, w2_jax, x_jax, target_jax)

        print(f"JAX grad_w1 mean: {jnp.mean(gw1_jax)}")

        # --- Nabla Implementation ---
        mesh = DeviceMesh("pp_dp_mesh", (2, 2), ("stage", "dp"))

        # Nabla Sharding
        # w: <stage, *, *>
        # x: <stage, dp, *>
        w1_nb = ops.shard(
            ops.constant(w1_np), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
        )
        w2_nb = ops.shard(
            ops.constant(w2_np), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
        )
        x_nb = ops.shard(
            ops.constant(x_np), mesh, [DimSpec(["stage"]), DimSpec(["dp"]), DimSpec([])]
        )
        target_nb = ops.shard(
            ops.constant(target_np),
            mesh,
            [DimSpec(["stage"]), DimSpec(["dp"]), DimSpec([])],
        )

        params = {"w1": w1_nb, "w2": w2_nb}

        def stage_fn_nb(x, p):
            h = ops.relu(x @ p["w1"])
            out = h @ p["w2"]
            return out

        stage_mapped_nb = vmap(
            stage_fn_nb, in_axes=(0, 0), out_axes=0, spmd_axis_name="stage", mesh=mesh
        )

        def pipeline_loss_nb(params, x, target):
            out = stage_mapped_nb(x, params)

            # Permutation Logic matching JAX roll(shift=1)
            # JAX: (S, B, D). Roll S: S0->S1, S1->S0.
            # Nabla Mesh (2, 2):
            # S0: devs 0,1. S1: devs 2,3.
            # S0 (dev 0) -> S1 (dev 2)
            # S0 (dev 1) -> S1 (dev 3)
            # S1 (dev 2) -> S0 (dev 0)
            # S1 (dev 3) -> S0 (dev 1)
            perm = [(0, 2), (1, 3), (2, 0), (3, 1)]

            passed_out = communication.ppermute(out, perm)

            diff = passed_out - target
            return ops.mean(diff * diff)

        nb_grad_fn = grad(pipeline_loss_nb)
        grads_nb = nb_grad_fn(params, x_nb, target_nb)

        gw1_nb = grads_nb["w1"].to_numpy()
        gw2_nb = grads_nb["w2"].to_numpy()

        print(f"Nabla grad_w1 mean: {gw1_nb.mean()}")

        # --- Comparison ---
        # Tolerance: 1e-5
        np.testing.assert_allclose(
            gw1_nb, gw1_jax, rtol=1e-5, atol=1e-5, err_msg="W1 Gradients mismatch"
        )
        np.testing.assert_allclose(
            gw2_nb, gw2_jax, rtol=1e-5, atol=1e-5, err_msg="W2 Gradients mismatch"
        )

        print("✅ Success: Nabla matched JAX gradients perfectly.")


if __name__ == "__main__":
    unittest.main()
