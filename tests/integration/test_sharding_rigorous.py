import unittest
import numpy as np
import nabla
from nabla import ops
from nabla.core.trace import trace
from nabla.sharding.spec import DeviceMesh, DimSpec, ShardingSpec

class TestShardingRigorous(unittest.TestCase):
    def setUp(self):
        self.mesh = DeviceMesh(name="mesh", shape=(2, 2), axis_names=("x", "y"))
        
    def test_concat_sharding_trace(self):
        def func(a, b):
            a = ops.shard(a, self.mesh, [DimSpec(["x"]), DimSpec([])])
            b = ops.shard(b, self.mesh, [DimSpec(["x"]), DimSpec([])])
            return ops.concatenate([a, b], axis=0)

        a, b = nabla.Tensor.zeros((4, 4)), nabla.Tensor.zeros((4, 4))
        t = str(trace(func, a, b))
        
        self.assertIn("(<x, *>)", t)
        self.assertRegex(t, r"shard.*%a1")
        self.assertRegex(t, r"shard.*%a2")
        
    def test_reduce_sharding_trace(self):
        def func_keep(x):
            x = ops.shard(x, self.mesh, [DimSpec(["x"]), DimSpec(["y"])])
            return ops.reduce_sum(x, axis=1, keepdims=True)
            
        def func_drop(x):
            x = ops.shard(x, self.mesh, [DimSpec(["x"]), DimSpec(["y"])])
            return ops.reduce_sum(x, axis=1, keepdims=False)
            
        x = nabla.Tensor.zeros((4, 4))
        self.assertIn("(<x, *>)", str(trace(func_keep, x)))
        self.assertIn("(<x>)", str(trace(func_drop, x)))
        
    def test_mlp_sharding_trace(self):
        def mlp(x, w1, w2):
            x = ops.shard(x, self.mesh, [DimSpec(["x"]), DimSpec([])])
            w1 = ops.shard(w1, self.mesh, [DimSpec([]), DimSpec(["y"])])
            w2 = ops.shard(w2, self.mesh, [DimSpec(["y"]), DimSpec([])])
            h1 = ops.relu(x @ w1)
            return h1 @ w2

        x, w1, w2 = nabla.Tensor.zeros((16, 32)), nabla.Tensor.zeros((32, 64)), nabla.Tensor.zeros((64, 32))
        t = str(trace(mlp, x, w1, w2))
        
        self.assertIn("matmul", t)
        self.assertIn("all_reduce", t)
        self.assertIn("(<x, y>)", t)
        self.assertIn("(<x, *>)", t)

if __name__ == "__main__":
    unittest.main()

