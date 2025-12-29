
import unittest
from nabla.sharding.spec import DeviceMesh, ShardingSpec, DimSpec
from nabla.sharding.propagation import (
    propagate_sharding, OpShardingRule, FactorShardingState, reshape_template
)

class TestReshardingPropagation(unittest.TestCase):
    def setUp(self):
        # 16 devices
        self.mesh = DeviceMesh("mesh", (4, 4), ("x", "y"))
        
    def test_reshape_template_merge(self):
        """Test merge using high-level template (existing implementation)."""
        # (4, 4) -> (16)
        # Should infer factors automatically
        template = reshape_template((4, 4), (16,))
        
        # Instantiate with shapes to infer factor sizes: M=4, K=4
        rule = template.instantiate([(4, 4)], [(16,)])
        
        in_spec = ShardingSpec(self.mesh, [DimSpec(["x"]), DimSpec(["y"])])
        out_spec = ShardingSpec(self.mesh, [DimSpec([], is_open=True)])
        
        changed = propagate_sharding(rule, [in_spec], [out_spec])
        
        self.assertTrue(changed)
        self.assertEqual(out_spec.dim_specs[0].axes, ["x", "y"])

    def test_merge_dimensions(self):
        """Test merging dims (i, j) -> k where k maps to [i, j]."""
        # Rule: Input 0 [M, K] -> Output 0 [M*K]
        # M maps to 'm', K maps to 'k'. Output maps to ['m', 'k'].
        
        # M=4, K=4. Output=16.
        # Factories sizes: m=4, k=4.
        # Mesh axes x=4, y=4.
        
        rule = OpShardingRule(
            input_mappings=[{0: ["m"], 1: ["k"]}],
            output_mappings=[{0: ["m", "k"]}],
            factor_sizes={"m": 4, "k": 4}
        )
        
        # Case 1: Input M on "x", K on "y". Output should start with "x", "y".
        in_spec = ShardingSpec(self.mesh, [DimSpec(["x"]), DimSpec(["y"])])
        out_spec = ShardingSpec(self.mesh, [DimSpec([], is_open=True)])
        
        changed = propagate_sharding(rule, [in_spec], [out_spec])
        
        self.assertTrue(changed)
        self.assertEqual(out_spec.dim_specs[0].axes, ["x", "y"])

    def test_split_dimension(self):
        """Test splitting dim k -> (i, j)."""
        # Rule: Input 0 [M*K] -> Output 0 [M, K]
        # Input maps to ['m', 'k']. Output maps to ['m'], ['k'].
        
        rule = OpShardingRule(
            input_mappings=[{0: ["m", "k"]}],
            output_mappings=[{0: ["m"], 1: ["k"]}],
            factor_sizes={"m": 4, "k": 4}
        )
        
        # Case 1: Input on "x", "y". Output 0 should get "x", Output 1 should get "y".
        # This relies on m=4, k=4 matching x=4, y=4 sizes.
        
        in_spec = ShardingSpec(self.mesh, [DimSpec(["x", "y"])])
        out_spec = ShardingSpec(self.mesh, [DimSpec([], is_open=True), DimSpec([], is_open=True)])
        
        changed = propagate_sharding(rule, [in_spec], [out_spec])
        
        self.assertTrue(changed)
        self.assertEqual(out_spec.dim_specs[0].axes, ["x"])
        self.assertEqual(out_spec.dim_specs[1].axes, ["y"])
        
    def test_split_dimension_subaxes(self):
        """Test splitting dim k -> (i, j) where k is sharded on SINGLE axis 'x' which must split."""
        # Rule: Input [M*K] -> [M, K]
        # M=2, K=2. Mesh X=4.
        
        rule = OpShardingRule(
            input_mappings=[{0: ["m", "k"]}],
            output_mappings=[{0: ["m"], 1: ["k"]}],
            factor_sizes={"m": 2, "k": 2}
        )
        
        # Input on "x". X has size 4. M*K = 4.
        # Should split X into x:(1)2 for M and x:(2)2 for K.
        
        in_spec = ShardingSpec(self.mesh, [DimSpec(["x"])])
        out_spec = ShardingSpec(self.mesh, [DimSpec([], is_open=True), DimSpec([], is_open=True)])
        
        changed = propagate_sharding(rule, [in_spec], [out_spec])
        
        expected_m = "x:(1)2"
        expected_k = "x:(2)2"
        
        self.assertTrue(changed)
        self.assertEqual(out_spec.dim_specs[0].axes, [expected_m])
        self.assertEqual(out_spec.dim_specs[1].axes, [expected_k])

if __name__ == "__main__":
    unittest.main()
