import unittest

import numpy as np

from nabla import ops
from nabla.transforms import grad
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.ops import communication
from nabla.transforms.vmap import vmap


class TestPPTraining(unittest.TestCase):
    def test_pp_mlp_grad(self):
        mesh = DeviceMesh("pp_mesh", (2,), ("stage",))
        STAGES = 2
        B, D = 4, 8

        w1 = np.random.randn(STAGES, D, D).astype(np.float32)
        w2 = np.random.randn(STAGES, D, D).astype(np.float32)

        x = np.random.randn(STAGES, B, D).astype(np.float32)
        target = np.random.randn(STAGES, B, D).astype(np.float32)

        w1_sharded = ops.shard(
            ops.constant(w1), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
        )
        w2_sharded = ops.shard(
            ops.constant(w2), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
        )

        x_sharded = ops.shard(
            ops.constant(x), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
        )
        target_sharded = ops.shard(
            ops.constant(target), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
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

            total = len(mesh.devices)
            perm = [(i, (i + 1) % total) for i in range(total)]
            passed_out = communication.ppermute(out, perm)

            diff = passed_out - target
            return ops.mean(diff * diff)

        grads = grad(pipeline_loss)(params, x_sharded, target_sharded)

        self.assertIsNotNone(grads["w1"])


if __name__ == "__main__":
    unittest.main()
