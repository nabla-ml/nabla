# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import unittest
import numpy as np
from nabla.core import Tensor
from nabla.core import trace
from nabla.transforms.shard_map import shard_map
from nabla.core.sharding.spec import DeviceMesh, ShardingSpec, DimSpec
from nabla.ops import binary, view, reduction

class TestShardMapComplex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = DeviceMesh("test_mesh", (2, 2), ("x", "y"), devices=[0, 1, 2, 3])

    def test_reduction_on_sharded_axis(self):
        in_specs = {0: ShardingSpec(self.mesh, [DimSpec(["x"]), DimSpec([])])}
        out_specs = {0: ShardingSpec(self.mesh, [DimSpec([])])}

        def my_reduce(x):
            return reduction.reduce_sum(x, axis=0)

        x_np = np.ones((4, 4), dtype=np.float32)
        x = Tensor.from_dlpack(x_np)
        sharded_fn = shard_map(my_reduce, self.mesh, in_specs, out_specs)
        
        res = sharded_fn(x)
        np.testing.assert_allclose(res.to_numpy(), np.sum(x_np, axis=0), rtol=1e-5)
    
    def test_reshape_resharding(self):
        in_specs = {0: ShardingSpec(self.mesh, [DimSpec(["x"]), DimSpec([])])}
        out_specs = {0: ShardingSpec(self.mesh, [DimSpec(["x"])])}

        def my_reshape(x):
            return view.reshape(x, (16,))

        x_np = np.arange(16, dtype=np.float32).reshape(4, 4)
        x = Tensor.from_dlpack(x_np)
        sharded_fn = shard_map(my_reshape, self.mesh, in_specs, out_specs)
        
        res = sharded_fn(x)
        np.testing.assert_allclose(res.to_numpy(), x_np.reshape(16), rtol=1e-5)

    def test_matmul_contraction(self):
        M, K, N = 4, 8, 4
        in_specs = {
            0: ShardingSpec(self.mesh, [DimSpec([]), DimSpec(["x"])]),
            1: ShardingSpec(self.mesh, [DimSpec(["x"]), DimSpec([])])
        }
        out_specs = {0: ShardingSpec(self.mesh, [DimSpec([]), DimSpec([])])}

        a_np, b_np = np.random.randn(M, K).astype(np.float32), np.random.randn(K, N).astype(np.float32)
        a, b = Tensor.from_dlpack(a_np), Tensor.from_dlpack(b_np)
        sharded_fn = shard_map(binary.matmul, self.mesh, in_specs, out_specs)
        
        res = sharded_fn(a, b)
        np.testing.assert_allclose(res.to_numpy(), a_np @ b_np, rtol=1e-4)

    def test_complex_composition(self):
        in_specs = {0: ShardingSpec(self.mesh, [DimSpec(["x"]), DimSpec([])])}
        out_specs = {0: ShardingSpec(self.mesh, [DimSpec([])])}
        W_np = np.eye(4, dtype=np.float32) * 2.0
        
        def my_model(x):
            h1 = binary.add(x, 1.0)
            W = Tensor.from_dlpack(W_np)
            return reduction.reduce_sum(binary.matmul(h1, W), axis=1)

        x_np = np.ones((4, 4), dtype=np.float32)
        x = Tensor.from_dlpack(x_np)
        sharded_fn = shard_map(my_model, self.mesh, in_specs, out_specs)
        
        res = sharded_fn(x)
        np.testing.assert_allclose(res.to_numpy(), np.sum((x_np + 1.0) @ W_np, axis=1), rtol=1e-5)

    def test_output_spec_enforcement(self):
        in_specs = {0: ShardingSpec(self.mesh, [DimSpec([]), DimSpec([])])}
        out_specs = {0: ShardingSpec(self.mesh, [DimSpec(["x"]), DimSpec([])])}

        x_np = np.ones((4, 4), dtype=np.float32)
        x = Tensor.from_dlpack(x_np)
        sharded_fn = shard_map(lambda x: x, self.mesh, in_specs, out_specs)
        
        res = sharded_fn(x)
        self.assertEqual(res._impl.sharding.dim_specs[0].axes, ["x"])
        np.testing.assert_allclose(res.to_numpy(), x_np)

if __name__ == "__main__":
    unittest.main()

