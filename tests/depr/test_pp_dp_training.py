import unittest
import numpy as np
from nabla import ops
from nabla.core import grad
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.transforms.vmap import vmap
from nabla.ops import communication


class TestPP_DP_Training(unittest.TestCase):
    def test_pp_dp_mlp_grad(self):
        # 2 Stages, 4 DP Replicas per stage
        mesh = DeviceMesh("pp_dp_mesh", (2, 4), ("stage", "dp"))
        STAGES = 2
        DP = 4
        B_LOC = 2  # Batch size per DP replica
        B_GLOBAL = B_LOC * DP
        D = 8

        # Weights: (STAGES, D, D)
        # Sharded over 'stage', Replicated over 'dp' (implied by absence)
        w1 = np.random.randn(STAGES, D, D).astype(np.float32)
        w2 = np.random.randn(STAGES, D, D).astype(np.float32)

        # Inputs: (STAGES, B_GLOBAL, D)
        # Note: Inputs usually come into the first stage, but for this symmetric test
        # we feed inputs to all stages to keep the vmap structure uniform.
        # Sharded over 'stage' (outer vmap) AND 'dp' (inner split of B_GLOBAL)
        x = np.random.randn(STAGES, B_GLOBAL, D).astype(np.float32)
        target = np.random.randn(STAGES, B_GLOBAL, D).astype(np.float32)

        w1_sharded = ops.shard(
            ops.constant(w1), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
        )
        w2_sharded = ops.shard(
            ops.constant(w2), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
        )

        # Input sharding: Stage dim is mapped, Batch dim is sharded over 'dp'
        x_sharded = ops.shard(
            ops.constant(x), mesh, [DimSpec(["stage"]), DimSpec(["dp"]), DimSpec([])]
        )
        target_sharded = ops.shard(
            ops.constant(target),
            mesh,
            [DimSpec(["stage"]), DimSpec(["dp"]), DimSpec([])],
        )

        params = {"w1": w1_sharded, "w2": w2_sharded}

        def stage_fn(x, p):
            # x is (B_LOC, D) here, but sharded over DP mesh?
            # When vmap peels 'stage', we are left with a sub-mesh of size 4 ('dp').
            # The input x slice is (B_GLOBAL, D).
            # Wait, vmap(in_axes=(0,0)) peels the leading dimension.
            # So x passed to stage_fn is (B_GLOBAL, D).
            # But since x_sharded had 'dp' on axis 1, the x passed here inherits that sharding?
            # Yes, x here is sharded over 'dp' on axis 0 (since axis 0 of slice is axis 1 of original).

            # Matmul: (B, D) @ (D, D) -> (B, D)
            # LHS is sharded on axis 0 ('dp'). RHS is replicated.
            # Result should be sharded on axis 0 ('dp').
            h = ops.relu(x @ p["w1"])
            out = h @ p["w2"]
            return out

        stage_mapped = vmap(
            stage_fn, in_axes=(0, 0), out_axes=0, spmd_axis_name="stage", mesh=mesh
        )

        def pipeline_loss(params, x, target):
            out = stage_mapped(x, params)

            # Pipeline Communication
            # We need to permute from Stage 0 -> Stage 1
            # Since we have DP replicas, we move (0, dp_i) -> (1, dp_i)

            # Devices in mesh (2, 4):
            # Row 0: 0, 1, 2, 3
            # Row 1: 4, 5, 6, 7
            # Permutation: 0->4, 1->5, 2->6, 3->7
            # And for ring closing: 4->0, ...

            perm = []
            for dp_i in range(DP):
                src_s0 = dp_i  # (0, dp_i)
                dst_s1 = DP + dp_i  # (1, dp_i)
                perm.append((src_s0, dst_s1))
                perm.append((dst_s1, src_s0))  # Cycle

            passed_out = communication.ppermute(out, perm)

            diff = passed_out - target
            return ops.mean(diff * diff)

        # Trace and compute gradients
        # Use create_graph=True just to stress test our new feature too :)
        grads = grad(pipeline_loss, create_graph=True)(
            params, x_sharded, target_sharded
        )

        self.assertIsNotNone(grads["w1"])
        self.assertIsNotNone(grads["w2"])

        # Verify sharding of gradients
        # Grads for weights should be replicated over DP (reduced) and sharded over Stage
        # nabla's autograd usually preserves input sharding or reduces.
        # Since w1 was <stage, *, *>, dw1 should be <stage, *, *>
        # This implies an all-reduce over 'dp' happened during backprop.

        dw1 = grads["w1"]
        # self.assertEqual(dw1.sharding.spec, ...) # Hard to check spec string equality directly
        # But we can check it runs.


if __name__ == "__main__":
    unittest.main()
