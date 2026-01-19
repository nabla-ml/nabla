# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
from nabla.core.sharding.spec import DeviceMesh, ShardingSpec, DimSpec
from nabla.core.sharding.propagation import (
    OpShardingRuleTemplate, 
    propagate_sharding, 
    FactorShardingState, 
    OpPriority
)

class TestRuleTemplate:
    def test_parse_einsum(self):
        tmpl = OpShardingRuleTemplate.parse("m k, k n -> m n")
        assert len(tmpl.input_mappings) == 2
        assert len(tmpl.output_mappings) == 1
        assert tmpl.input_mappings[0][0] == ["m"]
        assert tmpl.input_mappings[0][1] == ["k"]
        
    def test_instantiate(self):
        tmpl = OpShardingRuleTemplate.parse("m k, k n -> m n")
        input_shapes = [(128, 64), (64, 32)]
        rule = tmpl.instantiate(input_shapes)
        assert rule.factor_sizes["m"] == 128
        assert rule.factor_sizes["k"] == 64
        assert rule.factor_sizes["n"] == 32

    def test_instantiate_broadcast(self):
        tmpl = OpShardingRuleTemplate.parse("m, n -> m n")
        rule = tmpl.instantiate([(128,), (32,)])
        assert rule.factor_sizes["m"] == 128
        assert rule.factor_sizes["n"] == 32

class TestPropagation:
    @pytest.fixture
    def mesh(self):
        return DeviceMesh("mesh", (4, 4), ("x", "y"))

    def test_propagate_elementwise(self, mesh):
        tmpl = OpShardingRuleTemplate.parse("a, a -> a")
        rule = tmpl.instantiate([(32,), (32,)])
        in_spec_0 = ShardingSpec(mesh, [DimSpec(["x"], is_open=False)]) 
        in_spec_1 = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        out_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        changed = propagate_sharding(rule, [in_spec_0, in_spec_1], [out_spec])
        assert changed
        assert in_spec_1.dim_specs[0].axes == ["x"]
        assert out_spec.dim_specs[0].axes == ["x"]

    def test_propagate_matmul(self, mesh):
        tmpl = OpShardingRuleTemplate.parse("m k, k n -> m n")
        rule = tmpl.instantiate([(128, 128), (128, 128)])
        in_a = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
        in_b = ShardingSpec(mesh, [DimSpec([]), DimSpec(["y"])])
        out = ShardingSpec(mesh, [DimSpec([], is_open=True), DimSpec([], is_open=True)])
        propagate_sharding(rule, [in_a, in_b], [out])
        assert out.dim_specs[0].axes == ["x"]
        assert out.dim_specs[1].axes == ["y"]
        
    def test_conflict_resolution(self, mesh):
        tmpl = OpShardingRuleTemplate.parse("m, m -> m")
        rule = tmpl.instantiate([(32,), (32,)])
        in_a = ShardingSpec(mesh, [DimSpec(["x"], priority=1)])
        in_b = ShardingSpec(mesh, [DimSpec(["y"], priority=0)])
        out = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        propagate_sharding(rule, [in_a, in_b], [out])
        assert out.dim_specs[0].axes == ["y"]
        assert in_a.dim_specs[0].axes == ["y"]
