# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import unittest

import numpy as np

from nabla.core import Tensor, trace
from nabla.core.sharding.spec import DeviceMesh, DimSpec, ShardingSpec
from nabla.transforms.shard_map import shard_map


class TestShardMapRigorous(unittest.TestCase):
    def setUp(self):
        self.mesh = DeviceMesh("test_mesh", (2, 2), ("dp", "tp"), devices=[0, 1, 2, 3])

    def test_input_sharding_trace(self):
        def func(x):
            return x * 2

        sharded_fn = shard_map(
            func,
            self.mesh,
            in_specs={0: ShardingSpec(self.mesh, [DimSpec(["dp"]), DimSpec([])])},
        )

        data = np.random.rand(4, 4).astype(np.float32)
        x = Tensor.from_dlpack(data)
        t = trace(sharded_fn, x)

        self.assertIn("shard", str(t))
        self.assertIn("dp", str(t))

        result = sharded_fn(x)
        np.testing.assert_allclose(result.to_numpy(), data * 2)

    def test_io_constraints_sharding(self):
        in_spec = ShardingSpec(self.mesh, [DimSpec(["dp"]), DimSpec([])])
        out_spec = ShardingSpec(self.mesh, [DimSpec([]), DimSpec(["tp"])])

        def simple_add(a, b):
            return a + b

        sharded_fn = shard_map(
            simple_add,
            self.mesh,
            in_specs={0: in_spec, 1: in_spec},
            out_specs={0: out_spec},
        )

        d1, d2 = (
            np.random.rand(4, 4).astype(np.float32),
            np.random.rand(4, 4).astype(np.float32),
        )
        t1, t2 = Tensor.from_dlpack(d1), Tensor.from_dlpack(d2)

        t = trace(sharded_fn, t1, t2)
        result = sharded_fn(t1, t2)

        self.assertEqual(result.sharding.dim_specs[1].axes, ["tp"])
        np.testing.assert_allclose(result.to_numpy(), d1 + d2, atol=1e-5)

    def test_complex_graph_trace(self):
        def complex_func(x, w1, w2):
            h = (x @ w1) * 2.0
            return h @ w2

        sharded_fn = shard_map(
            complex_func,
            self.mesh,
            in_specs={
                0: None,
                1: ShardingSpec(self.mesh, [DimSpec([]), DimSpec(["tp"])]),
                2: ShardingSpec(self.mesh, [DimSpec(["tp"]), DimSpec([])]),
            },
            out_specs={0: None},
        )

        B, H = 4, 4
        d_x, d_w1, d_w2 = (
            np.random.rand(B, H).astype(np.float32),
            np.random.rand(H, 4 * H).astype(np.float32),
            np.random.rand(4 * H, H).astype(np.float32),
        )
        t_x, t_w1, t_w2 = (
            Tensor.from_dlpack(d_x),
            Tensor.from_dlpack(d_w1),
            Tensor.from_dlpack(d_w2),
        )

        t = trace(sharded_fn, t_x, t_w1, t_w2)
        self.assertIn("shard", str(t))
        self.assertIn("tp", str(t))

        result = sharded_fn(t_x, t_w1, t_w2)
        expected = (d_x @ d_w1 * 2.0) @ d_w2
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
