import unittest

import numpy as np

from nabla import ops
from nabla.core.autograd import grad
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.transforms.vmap import vmap


class TestDPOnlyMatmul(unittest.TestCase):
    def test_dp_mlp_grad(self):
        mesh = DeviceMesh("dp_mesh", (2,), ("data",))
        B, D_in, D_hidden, D_out = 4, 8, 16, 8

        x = np.random.randn(B, D_in).astype(np.float32)
        target = np.random.randn(B, D_out).astype(np.float32)

        w1 = np.random.randn(D_in, D_hidden).astype(np.float32)
        b1 = np.random.randn(D_hidden).astype(np.float32)
        w2 = np.random.randn(D_hidden, D_out).astype(np.float32)
        b2 = np.random.randn(D_out).astype(np.float32)

        params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

        x_sharded = ops.shard(ops.constant(x), mesh, [DimSpec(["data"]), DimSpec([])])
        target_sharded = ops.shard(
            ops.constant(target), mesh, [DimSpec(["data"]), DimSpec([])]
        )

        params_sharded = {
            k: ops.shard(ops.constant(v), mesh, [DimSpec([])] * v.ndim)
            for k, v in params.items()
        }

        def mlp(params, x):
            h = ops.relu(x @ params["w1"] + params["b1"])
            out = h @ params["w2"] + params["b2"]
            return out

        def forward(params, x):
            return mlp(params, x)

        forward_dp = vmap(
            forward, in_axes=(None, 0), out_axes=0, spmd_axis_name="data", mesh=mesh
        )

        def final_loss(params, x, target):
            preds = forward_dp(params, x)
            diff = preds - target
            return ops.mean(diff * diff)

        grads = grad(final_loss)(params_sharded, x_sharded, target_sharded)

        for k, v in grads.items():
            # Convert Shape (list of Dims) to tuple of ints
            v_shape = tuple(int(d) for d in v.shape)
            self.assertEqual(v_shape, params[k].shape)


if __name__ == "__main__":
    unittest.main()
