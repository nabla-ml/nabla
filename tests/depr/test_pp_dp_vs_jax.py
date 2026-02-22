import unittest

import numpy as np

from nabla import ops
from nabla.transforms import grad
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.ops import communication
from nabla.transforms.vmap import vmap


class TestPP_DP_vs_JAX(unittest.TestCase):
    def test_pp_dp_mlp_grad_correctness(self):
        print("\n=== Testing PP + DP Gradients vs NumPy Ground Truth ===")

        # Mesh: 2 Stages, 2 DP Replicas per stage
        mesh = DeviceMesh("pp_dp_mesh", (2, 2), ("stage", "dp"))
        STAGES = 2
        DP = 2
        B_LOC = 2
        B_GLOBAL = B_LOC * DP
        D = 8

        np.random.seed(1337)

        # Weights: (STAGES, D, D)
        w1_data = np.random.randn(STAGES, D, D).astype(np.float32)
        w2_data = np.random.randn(STAGES, D, D).astype(np.float32)

        # Inputs: (STAGES, B_GLOBAL, D) - feeding independent inputs to each stage for symmetry test
        x_data = np.random.randn(STAGES, B_GLOBAL, D).astype(np.float32)
        target_data = np.random.randn(STAGES, B_GLOBAL, D).astype(np.float32)

        # --- NumPy Ground Truth ---
        # Simulate the pipeline + DP logic sequentially
        # Loss = mean((ppermute(f(x)) - target)^2)
        # f(x) = relu(x @ w1) @ w2
        # ppermute shifts stage outputs: 0->1, 1->0

        # 1. Forward Pass
        out_stage = np.zeros_like(target_data)
        for s in range(STAGES):
            w1_s = w1_data[s]
            w2_s = w2_data[s]
            x_s = x_data[s]  # (B, D)

            # Local computation
            h = np.maximum(x_s @ w1_s, 0)
            out = h @ w2_s
            out_stage[s] = out

        # 2. Pipeline Communication (0->1, 1->0)
        out_permuted = np.zeros_like(out_stage)
        out_permuted[0] = out_stage[1]
        out_permuted[1] = out_stage[0]

        # 3. Loss
        diff = out_permuted - target_data
        loss = np.mean(diff**2)

        # 4. Backward Pass (Manual)
        # dL/d_out_permuted = 2 * diff / N_elements
        N_elements = diff.size
        d_out_permuted = 2 * diff / N_elements

        # Permute Back gradients: 0<-1, 1<-0
        d_out_stage = np.zeros_like(d_out_permuted)
        d_out_stage[0] = d_out_permuted[1]
        d_out_stage[1] = d_out_permuted[0]

        grad_w1 = np.zeros_like(w1_data)
        grad_w2 = np.zeros_like(w2_data)

        for s in range(STAGES):
            x_s = x_data[s]
            w1_s = w1_data[s]
            w2_s = w2_data[s]
            d_out = d_out_stage[s]

            # d_out = dH @ W2.T
            # dW2 = H.T @ d_out
            h = np.maximum(x_s @ w1_s, 0)

            grad_w2[s] = h.T @ d_out

            d_h = d_out @ w2_s.T
            # ReLU grad
            d_h[h <= 0] = 0

            # dW1 = X.T @ d_h
            grad_w1[s] = x_s.T @ d_h

        print(f"NumPy Loss: {loss:.6f}")
        print(f"NumPy grad_w1 mean: {grad_w1.mean():.6f}")

        # --- Nabla PP+DP Execution ---

        # Weights: Sharded over 'stage', Replicated over 'dp'
        w1_sharded = ops.shard(
            ops.constant(w1_data), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
        )
        w2_sharded = ops.shard(
            ops.constant(w2_data), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
        )

        # Inputs: Sharded over 'stage' and 'dp' (axis 1 of x_data is batch)
        # x_data: (STAGES, B_GLOBAL, D).
        # DimSpec(["stage"]), DimSpec(["dp"]), DimSpec([])
        x_sharded = ops.shard(
            ops.constant(x_data),
            mesh,
            [DimSpec(["stage"]), DimSpec(["dp"]), DimSpec([])],
        )
        target_sharded = ops.shard(
            ops.constant(target_data),
            mesh,
            [DimSpec(["stage"]), DimSpec(["dp"]), DimSpec([])],
        )

        params = {"w1": w1_sharded, "w2": w2_sharded}

        def stage_fn(x, p):
            h = ops.relu(x @ p["w1"])
            out = h @ p["w2"]
            return out

        stage_mapped = vmap(
            stage_fn, in_axes=(0, 0), out_axes=0, spmd_axis_name="stage", mesh=mesh
        )

        def pipeline_loss(params, x, target):
            out = stage_mapped(x, params)

            # Permutation logic for DP mesh (2, 2)
            # Row 0 (S0): 0, 1. Row 1 (S1): 2, 3.
            # DP pairs: (0, 2), (1, 3).
            # Perm: 0->2, 1->3, 2->0, 3->1.
            perm = [(0, 2), (2, 0), (1, 3), (3, 1)]

            passed_out = communication.ppermute(out, perm)

            diff = passed_out - target
            return ops.mean(diff * diff)

        # Compute Gradients
        grads = grad(pipeline_loss)(params, x_sharded, target_sharded)

        # Verify
        gw1_nabla = grads["w1"].to_numpy()
        gw2_nabla = grads["w2"].to_numpy()

        print(f"Nabla grad_w1 mean: {gw1_nabla.mean():.6f}")

        diff_w1 = np.abs(gw1_nabla - grad_w1).max()
        diff_w2 = np.abs(gw2_nabla - grad_w2).max()

        print(f"Diff W1: {diff_w1:.6f}")
        print(f"Diff W2: {diff_w2:.6f}")

        np.testing.assert_allclose(gw1_nabla, grad_w1, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(gw2_nabla, grad_w2, rtol=1e-4, atol=1e-4)
        print("✅ Success: PP+DP gradients match NumPy ground truth!")


if __name__ == "__main__":
    unittest.main()
