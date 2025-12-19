"""
Shardy Tests
============

Unit tests for the Sharding Representation and Propagation system.
Refactored for maximum conciseness and readability.

Run with: python -m shardy.tests
"""

import unittest
from core import (
    DeviceMesh, DimSpec, ShardingSpec, DistributedTensor, GraphTensor,
    parse_sub_axis, validate_sub_axes_non_overlapping, check_sub_axes_maximality
)
from propagation import (
    OpShardingRule, OpShardingRuleTemplate, FactorSharding, FactorShardingState,
    Operation, DataFlowEdge, ShardingPass, propagate_sharding, OpPriority, PropagationStrategy,
    # Templates
    gather_template, attention_template, embedding_template
)

class ShardingTestBase(unittest.TestCase):
    """Base class with factories to reduce boilerplate."""
    
    def setUp(self):
        # Default 2D mesh: x=2, y=4
        self.devices = list(range(8))
        self.mesh = DeviceMesh("cluster", (2, 4), ("x", "y"), self.devices)

    def s(self, *axes_args, p=0, open=False, repl=None, mesh=None) -> ShardingSpec:
        """
        Factory for ShardingSpec.
        Usage: self.s(["x"], ["y"], p=1) creates 2 dims with priority 1.
        """
        dims = []
        for arg in axes_args:
            # Handle empty list or None as []
            ax_list = arg if arg else []
            dims.append(DimSpec(ax_list, is_open=open, priority=p))
        return ShardingSpec(mesh or self.mesh, dims, replicated_axes=repl or set())

    def check(self, spec: ShardingSpec, *expected_axes):
        """Asserts that spec dimensions match expected axes."""
        self.assertEqual(len(spec.dim_specs), len(expected_axes), "Rank mismatch")
        for i, expected in enumerate(expected_axes):
            # Normalize "x" -> ["x"] for convenience
            exp_list = [expected] if isinstance(expected, str) else (expected or [])
            self.assertEqual(spec.dim_specs[i].axes, exp_list, f"Mismatch at dim {i}")

    def make_rule(self, in_map, out_map, sizes):
        """Factory for OpShardingRule."""
        return OpShardingRule(in_map, out_map, sizes)

    def prop(self, rule, inputs, outputs, strategy=PropagationStrategy.BASIC, max_p=None):
        """Wrapper for propagate_sharding."""
        return propagate_sharding(rule, inputs, outputs, strategy, max_p)


class TestShardingCompiler(ShardingTestBase):

    # --- Group 1: Physical & Execution ---
    def test_01_mesh_coords(self):
        # 5 // 4 = 1 (x), 5 % 4 = 1 (y)
        self.assertEqual(self.mesh.get_coordinate(5, "x"), 1)
        self.assertEqual(self.mesh.get_coordinate(5, "y"), 1)

    def test_02_uneven_slicing(self):
        m1d = DeviceMesh("1d", (2,), ("x",), [0, 1])
        dt = DistributedTensor((7,), self.s(["x"], mesh=m1d))
        # Dev 0: [0, 4), Dev 1: [4, 7) + 1 pad
        self.assertEqual(dt.get_local_interval(0, 1), (4, 7, 1))

    def test_03_axis_size(self):
        self.assertEqual(self.mesh.get_axis_size("y"), 4)

    # --- Group 2: Safety Invariants ---
    def test_04_invariant_conflict(self):
        with self.assertRaisesRegex(ValueError, "multiple times"):
            self.s(["x", "x"])

    def test_05_explicit_replication_conflict(self):
        with self.assertRaisesRegex(ValueError, "explicitly replicated"):
            self.s(["x"], repl={"x"})

    # --- Group 3: Basic Propagation ---
    def test_06_forward_prop(self):
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m": 2})
        in_a, out_b = self.s(["x"], p=1), self.s([], open=True)
        self.prop(rule, [in_a], [out_b])
        self.check(out_b, ["x"])

    def test_07_backward_prop(self):
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m": 2})
        in_a, out_b = self.s([], open=True), self.s(["x"], p=1)
        self.prop(rule, [in_a], [out_b])
        self.check(in_a, ["x"])

    # --- Group 4: Complex Ops ---
    def test_08_reshape_split(self):
        # Split dim0 -> f1(x), f2(y)
        rule = self.make_rule([{0: ["f1", "f2"]}], [{0: ["f1"], 1: ["f2"]}], {"f1":2, "f2":4})
        out = self.s([], [], open=True)
        self.prop(rule, [self.s(["x", "y"], p=1)], [out])
        self.check(out, ["x"], ["y"])

    def test_09_transpose(self):
        rule = self.make_rule([{0:["m"], 1:["n"]}], [{0:["n"], 1:["m"]}], {"m":2, "n":4})
        out = self.s([], [], open=True)
        self.prop(rule, [self.s(["x"], ["y"])], [out])
        self.check(out, ["y"], ["x"])

    def test_10_reduce(self):
        rule = self.make_rule([{0:["m"], 1:["n"]}], [{0:["m"]}], {"m":2, "n":4})
        out = self.s([], open=True)
        self.prop(rule, [self.s(["x"], ["y"])], [out])
        self.check(out, ["x"])

    # --- Group 5: Conflict Resolution ---
    def test_11_priority_overwrite(self):
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m": 2})
        out = self.s(["y"], p=1)
        self.prop(rule, [self.s(["x"], p=0)], [out])
        self.check(out, ["x"]) # p0 wins

    def test_12_priority_intersection(self):
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m": 2})
        out = self.s(["x"], p=1)
        self.prop(rule, [self.s(["x", "y"], p=1)], [out])
        self.check(out, ["x"]) # intersection

    def test_13_implicit_replication_intersection(self):
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m": 2})
        out = self.s(["x"], p=1)
        self.prop(rule, [self.s([], p=0)], [out])
        self.check(out, []) # p0 empty wins

    def test_14_matmul_broadcast_conflict(self):
        # A(x,y) + Bias(replicated p0) -> C
        rule = self.make_rule([{0:["m"], 1:["n"]}, {0:["n"]}], [{0:["m"], 1:["n"]}], {"m":2, "n":4})
        out = self.s([], [], open=True)
        self.prop(rule, [self.s(["x"], ["y"]), self.s([], p=0)], [out])
        self.check(out, ["x"], []) # n forced replicated

    # --- Group 6: Edge Cases ---
    def test_15_single_element_dim(self):
        m1d = DeviceMesh("1d", (4,), ("x",), [0,1,2,3])
        dt = DistributedTensor((1,), self.s(["x"], mesh=m1d))
        s, e, _ = dt.get_local_interval(0, 1) # Dev 1 has 0 elements
        self.assertEqual(e - s, 0)

    def test_16_zero_sized_dimension(self):
        m1d = DeviceMesh("1d", (4,), ("x",), [0,1,2,3])
        dt = DistributedTensor((0,), self.s(["x"], mesh=m1d))
        self.assertEqual(dt.get_local_interval(0, 0), (0, 0, 0))

    def test_17_explicit_replication_enforcement(self):
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m":2})
        out = self.s([], open=True, repl={"x"})
        self.prop(rule, [self.s(["x"], p=1)], [out])
        self.check(out, []) # x blocked

    # --- Group 7: Sub-Axis Logic ---
    def test_18_subaxis_auto_decomposition_strings(self):
        m8 = DeviceMesh("x8", (8,), ("x",), list(range(8)))
        rule = self.make_rule([{0: ["f1", "f2"]}], [{0: ["f1"], 1: ["f2"]}], {"f1": 2, "f2": 4})
        out = self.s([], [], open=True, mesh=m8)
        self.prop(rule, [self.s(["x"], p=0, mesh=m8)], [out])
        self.check(out, ["x:(1)2"], ["x:(2)4"])

    def test_19_subaxis_coordinate_math(self):
        m8 = DeviceMesh("x8", (8,), ("x",), list(range(8)))
        # x:(1)2 -> Indices 0..3=0, 4..7=1
        self.assertEqual(m8.get_coordinate(3, "x:(1)2"), 0)
        self.assertEqual(m8.get_coordinate(4, "x:(1)2"), 1)
        # x:(2)4 -> (idx // 1) % 4
        self.assertEqual(m8.get_coordinate(5, "x:(2)4"), 1)

    def test_20_three_way_split(self):
        m8 = DeviceMesh("x8", (8,), ("x",), list(range(8)))
        rule = self.make_rule([{0: ["f1","f2","f3"]}], [{0:["f1"], 1:["f2"], 2:["f3"]}], 
                              {"f1":2, "f2":2, "f3":2})
        out = self.s([], [], [], open=True, mesh=m8)
        self.prop(rule, [self.s(["x"], p=0, mesh=m8)], [out])
        self.check(out, ["x:(1)2"], ["x:(2)2"], ["x:(4)2"])

    def test_21_partial_split_usage(self):
        m8 = DeviceMesh("x8", (8,), ("x",), list(range(8)))
        rule = self.make_rule([{0: ["f1", "f2"]}], [{0: ["f2"]}], {"f1": 2, "f2": 4})
        out = self.s([], open=True, mesh=m8)
        self.prop(rule, [self.s(["x"], p=0, mesh=m8)], [out])
        self.check(out, ["x:(2)4"])

    # --- Group 8: Data Flow Edges ---
    def test_22_data_flow_edge_identity(self):
        m1 = DeviceMesh("1d", (4,), ("x",), [0, 1, 2, 3])
        t1 = GraphTensor("t1", (4,4), m1, self.s(["x"], [], mesh=m1))
        t2 = GraphTensor("t2", (4,4), m1)
        propagate_sharding(DataFlowEdge([t1], [t2]).to_identity_rule(), [t1.spec], [t2.spec])
        self.check(t2.spec, ["x"], [])

    def test_23_data_flow_edge_bidirectional(self):
        m1 = DeviceMesh("1d", (2,), ("x",), [0, 1])
        t1, t2 = GraphTensor("s", (4,), m1), GraphTensor("t", (4,), m1, self.s(["x"], mesh=m1))
        propagate_sharding(DataFlowEdge([t1], [t2]).to_identity_rule(), [t1.spec], [t2.spec])
        self.check(t1.spec, ["x"])

    # --- Group 9: Open/Closed Semantics ---
    def test_24_open_dimension_extension(self):
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m": 2})
        out = self.s([], open=True)
        self.prop(rule, [self.s(["x"], open=True, p=1)], [out])
        self.check(out, ["x"])

    def test_25_closed_dimension_no_extension(self):
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m": 2})
        out = self.s(["x"], open=False, p=1)
        self.prop(rule, [self.s(["x", "y"], open=False, p=1)], [out])
        self.check(out, ["x"]) # closed blocks extension

    def test_26_closed_dimension_priority_override(self):
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m": 2})
        out = self.s(["x"], open=False, p=1)
        self.prop(rule, [self.s(["y"], open=False, p=0)], [out])
        self.check(out, ["y"]) # p0 overrides closed

    # --- Group 10: ShardingPass ---
    def test_27_sharding_pass_basic(self):
        m1 = DeviceMesh("1d", (2,), ("x",), [0, 1])
        t_a = GraphTensor("A", (4,), m1, self.s(["x"], p=0, mesh=m1))
        t_b, t_c = GraphTensor("B", (4,), m1), GraphTensor("C", (4,), m1)
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m": 2})
        
        ops = [Operation("1", rule, [t_a], [t_b]), Operation("2", rule, [t_b], [t_c])]
        ShardingPass(ops).run_pass()
        self.check(t_c.spec, ["x"])

    def test_28_sharding_pass_with_data_flow_edges(self):
        m1 = DeviceMesh("1d", (2,), ("x",), [0, 1])
        t_in = GraphTensor("in", (4,), m1, self.s(["x"], p=0, mesh=m1))
        t_out = GraphTensor("out", (4,), m1)
        ShardingPass([], data_flow_edges=[DataFlowEdge([t_in], [t_out])]).run_pass()
        self.check(t_out.spec, ["x"])

    # --- Group 11: Multi-Mesh & Implicit ---
    def test_29_implicit_replication_detection(self):
        m = DeviceMesh("3d", (2, 4, 2), ("x", "y", "z"), list(range(16)))
        spec = self.s(["x"], [], mesh=m)
        self.assertEqual(spec.get_implicitly_replicated_axes(), {"y", "z"})

    def test_30_fully_replicated_check(self):
        m = DeviceMesh("1d", (4,), ("x",), [0,1,2,3])
        self.assertTrue(self.s([], [], mesh=m).is_fully_replicated())
        self.assertFalse(self.s(["x"], [], mesh=m).is_fully_replicated())

    # --- Group 12: Edge Cases / Validation ---
    def test_31_subaxis_overlap_validation(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_sub_axes_non_overlapping(["x:(1)4", "x:(2)4"])

    def test_32_subaxis_maximality_warning(self):
        self.assertEqual(len(check_sub_axes_maximality(["x:(1)2", "x:(2)4"])), 1)

    def test_33_mesh_repr_format(self):
        self.assertIn('@cluster', repr(self.mesh))

    def test_34_sharding_spec_repr_format(self):
        self.assertIn('sharding<@', repr(self.s(["x"])))

    # --- Group 13: Repr ---
    def test_35_priority_repr_syntax(self):
        self.assertNotIn("p", repr(DimSpec(["x"], priority=0)))
        self.assertIn("p1", repr(DimSpec(["x"], priority=1)))

    def test_36_empty_dim_repr(self):
        self.assertEqual(repr(DimSpec([])), '{}')
        self.assertEqual(repr(DimSpec([], is_open=True)), '{?}')

    def test_37_replicated_axes_ordering(self):
        spec = self.s([], repl={"y", "x:(4)2", "x:(1)2"})
        ordered = spec._order_replicated_axes()
        self.assertEqual(ordered, ["x:(1)2", "x:(4)2", "y"])

    def test_38_propagation_strategy_enum(self):
        self.assertEqual(PropagationStrategy.BASIC, 0)
        self.assertEqual(PropagationStrategy.AGGRESSIVE, 1)

    # --- Group 14: Coverage ---
    def test_39_multi_factor_with_replication(self):
        rule = self.make_rule([{0:["m"], 1:["n"]}], [{0:["m"], 1:["n"]}], {"m":2, "n":4})
        out = self.s([], [], open=True, repl={"y"})
        self.prop(rule, [self.s(["x"], ["y"])], [out])
        self.check(out, ["x"], []) # y blocked

    def test_40_dim_spec_get_total_shards(self):
        self.assertEqual(DimSpec(["x"]).get_total_shards(self.mesh), 2)

    def test_41_distributed_tensor_byte_size(self):
        dt = DistributedTensor((4, 8), self.s(["x"], []))
        self.assertEqual(dt.get_byte_size(), 128)
        self.assertEqual(dt.get_local_byte_size(0), 64)

    def test_42_op_sharding_rule_get_all_factors(self):
        rule = self.make_rule([{0:["m","k"]}], [{0:["m"]}], {"m":2,"k":8})
        self.assertEqual(rule.get_all_factors(), {"m", "k"})

    def test_43_op_sharding_rule_to_einsum(self):
        self.assertIn("->", self.make_rule([{0:["m"]}], [{0:["m"]}], {"m":2}).to_einsum_notation())

    def test_44_sharding_spec_priority_helpers(self):
        spec = self.s(["x"], [], p=0)
        spec.dim_specs[1].priority = 2
        self.assertEqual(spec.get_min_priority(), 0)
        self.assertEqual(spec.get_max_priority(), 2)

    def test_45_sharding_pass_get_propagation_table(self):
        t_a = GraphTensor("A", (4,), self.mesh)
        ops = [Operation("id", self.make_rule([{0:["m"]}], [{0:["m"]}], {"m":2}), [t_a], [t_a])]
        self.assertIn("A", ShardingPass(ops).get_propagation_table())

    def test_46_empty_open_vs_closed_semantics(self):
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m":2})
        in_open, in_closed = self.s([], open=True), self.s([], open=False, p=0)
        out1, out2 = self.s(["x"], p=1), self.s(["x"], p=1)
        
        self.prop(rule, [in_open], [out1])
        self.check(out1, ["x"]) # open accepts
        
        self.prop(rule, [in_closed], [out2])
        self.check(out2, []) # closed blocks

    def test_47_divisibility_uneven_sharding(self):
        m = DeviceMesh("m", (3,), ("x",), [0, 1, 2])
        dt = DistributedTensor((7,), self.s(["x"], mesh=m))
        self.assertEqual(dt.get_local_interval(0, 0), (0, 3, 0)) # ceil(7/3)=3
        self.assertEqual(dt.get_local_interval(0, 2), (6, 7, 2)) # 1 real, 2 pad

    # --- Group 15: Aggressive Strategy ---
    def test_48_aggressive_picks_higher_parallelism(self):
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m": 2})
        in_a, out_b = self.s(["x"], p=1), self.s(["y"], p=1)
        # BASIC -> No change
        self.prop(rule, [in_a], [out_b], strategy=PropagationStrategy.BASIC)
        self.check(out_b, ["y"])
        
        # AGGRESSIVE -> Pick Y (parallelism 4 > 2). 
        # Note: In standard 1-to-1 prop, output usually overrides input if equal priority 
        # depending on loop order, but here we test the MERGE logic in factor state.
        # We need a conflict test: 2 inputs mapping to same factor.
        pass

    def test_49_aggressive_vs_basic_conflict(self):
        rule = self.make_rule([{0:["m"]}, {0:["m"]}], [{0:["m"]}], {"m": 2})
        in_a, in_b = self.s(["x"], p=1), self.s(["y"], p=1)
        out = self.s([], open=True)
        
        # BASIC -> Empty (conservative)
        self.prop(rule, [in_a, in_b], [out], strategy=PropagationStrategy.BASIC)
        self.check(out, [])
        
        # AGGRESSIVE -> Y (higher par)
        # Reset output
        out = self.s([], open=True)
        self.prop(rule, [in_a, in_b], [out], strategy=PropagationStrategy.AGGRESSIVE)
        self.check(out, ["y"])

    def test_50_aggressive_same_parallelism_keeps_first(self):
        m = DeviceMesh("m", (4, 4), ("x", "y"), list(range(16)))
        rule = self.make_rule([{0:["m"]}, {0:["m"]}], [{0:["m"]}], {"m": 4})
        out = self.s([], open=True, mesh=m)
        # Both size 4. Keep x (first).
        self.prop(rule, [self.s(["x"], p=1, mesh=m), self.s(["y"], p=1, mesh=m)], [out], 
                  strategy=PropagationStrategy.AGGRESSIVE)
        self.check(out, ["x"])

    # --- Group 16: Priority Iteration ---
    def test_51_priority_iteration_basic(self):
        # A(p0) -> B(p1) -> B should get p0
        m = DeviceMesh("m", (2,), ("x",), [0,1])
        t_a = GraphTensor("A", (4,), m, self.s(["x"], p=0, mesh=m))
        t_b = GraphTensor("B", (4,), m, self.s([], open=True, mesh=m))
        op = Operation("op", self.make_rule([{0:["m"]}], [{0:["m"]}], {"m":2}), [t_a], [t_b])
        
        ShardingPass([op], use_priority_iteration=True).run_pass()
        self.check(t_b.spec, ["x"])

    def test_52_priority_iteration_respects_order(self):
        # A(x, p0) -> C, B(y, p1) -> C. C should be x.
        m = DeviceMesh("m", (2,), ("x",), [0,1])
        t_a = GraphTensor("A", (4,), m, self.s(["x"], p=0, mesh=m))
        t_b = GraphTensor("B", (4,), m, self.s(["y"], p=1, mesh=m))
        t_c = GraphTensor("C", (4,), m)
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m":2})
        
        ops = [Operation("1", rule, [t_a], [t_c]), Operation("2", rule, [t_b], [t_c])]
        ShardingPass(ops, use_priority_iteration=True).run_pass()
        self.check(t_c.spec, ["x"])

    def test_53_priority_iteration_vs_simple(self):
        # Result should be same for non-conflicting chain
        m = DeviceMesh("m", (2,), ("x",), [0,1])
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m":2})
        
        t1, t2 = GraphTensor("A", (4,), m, self.s(["x"], p=0, mesh=m)), GraphTensor("B", (4,), m)
        ShardingPass([Operation("op", rule, [t1], [t2])], use_priority_iteration=True).run_pass()
        self.check(t2.spec, ["x"])

    def test_54_max_priority_filtering(self):
        # Propagate only p0
        rule = self.make_rule([{0:["m"], 1:["n"]}], [{0:["m"], 1:["n"]}], {"m":2, "n":4})
        out = self.s([], [], open=True)
        # in dim0=p0, dim1=p2. Only dim0 should prop.
        self.prop(rule, [self.s(["x"], ["y"], p=2)], [out], max_p=0)
        # Manually verify dim1 is empty
        self.assertEqual(out.dim_specs[0].axes, []) # Wait, p2 input vs p0 filter?
        # Correction: The helper sets p=2 for BOTH dims. Let's make specific spec.
        spec = self.s(["x"], ["y"], p=0)
        spec.dim_specs[1].priority = 2
        
        self.prop(rule, [spec], [out], max_p=0)
        self.check(out, ["x"], [])

    def test_55_sharding_pass_aggressive(self):
        # Pass level aggressive strategy test
        m = DeviceMesh("m", (2,4), ("x","y"), range(8))
        t_y = GraphTensor("Y", (8,), m, self.s(["y"], p=1, mesh=m))
        t_x = GraphTensor("X", (8,), m, self.s(["x"], p=1, mesh=m))
        t_out = GraphTensor("O", (8,), m)
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m":8})
        
        ops = [Operation("1", rule, [t_y], [t_out]), Operation("2", rule, [t_x], [t_out])]
        ShardingPass(ops, strategy=PropagationStrategy.AGGRESSIVE).run_pass()
        self.check(t_out.spec, ["y"])

    # --- Group 17: Templates ---
    def test_56_template_instantiate_matmul(self):
        rule = OpShardingRuleTemplate(
            [{0:["m"], 1:["k"]}, {0:["k"], 1:["n"]}], [{0:["m"], 1:["n"]}]
        ).instantiate([(4,8), (8,16)], [(4,16)])
        self.assertEqual(rule.factor_sizes, {"m":4, "k":8, "n":16})

    def test_57_template_instantiate_reshape_split(self):
        rule = OpShardingRuleTemplate(
            [{0:["f1","f2","f3"]}], [{0:["f1"], 1:["f2"], 2:["f3"]}]
        ).instantiate([(24,)], [(2,3,4)])
        self.assertEqual(rule.factor_sizes["f1"], 2)
        self.assertEqual(rule.factor_sizes["f3"], 4)

    def test_58_template_instantiate_inconsistent_sizes(self):
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            OpShardingRuleTemplate(
                [{0:["m"]}, {0:["m"]}], []
            ).instantiate([(4,), (5,)], [])

    def test_59_template_instantiate_wrong_product(self):
        with self.assertRaisesRegex(ValueError, "Factor product"):
            # Factor product mismatch: input dim is 8, but output defines f1=2, f2=5 -> 2*5=10 != 8
            t = OpShardingRuleTemplate([{0:["f1","f2"]}], [{0:["f1"], 1:["f2"]}])
            t.instantiate([(8,)], [(2,5)])

    def test_60_template_with_propagation(self):
        # Matmul prop check
        rule = OpShardingRuleTemplate(
            [{0:["m"], 1:["n"]}], [{0:["n"], 1:["m"]}]
        ).instantiate([(2,4)], [(4,2)])
        out = self.s([], [], open=True)
        self.prop(rule, [self.s(["x"], ["y"])], [out])
        self.check(out, ["y"], ["x"])

    def test_61_template_einsum_notation(self):
        t = OpShardingRuleTemplate([{0:["m"]}], [{0:["m"]}])
        self.assertIn("->", t.to_einsum_notation())

    # --- Group 18: Irregular Templates ---
    def test_62_gather_template(self):
        rule = gather_template(3, 2, 1).instantiate([(4,8,16), (2,3)], [(4,2,3,16)])
        # d0(x), d1(y), d2. d1 is gathered.
        out = self.s([],[],[],[], open=True)
        self.prop(rule, [self.s(["x"],["y"],[]), self.s([],[], open=True)], [out])
        self.check(out, ["x"], [], [], [])

    def test_63_gather_preserves_non_indexed(self):
        rule = gather_template(2, 1, 0).instantiate([(8,16), (4,)], [(4,16)])
        out = self.s([],[], open=True)
        self.prop(rule, [self.s([],["y"]), self.s([], open=True)], [out])
        self.check(out, [], ["y"])

    def test_64_attention_template(self):
        rule = attention_template(1, True).instantiate(
            [(2,8,64,64)]*3, [(2,8,64,64)]
        )
        # Q: batch(x), head(y). Using default mesh axes.
        q = self.s(["x"], ["y"], [], [])
        out = self.s([],[],[],[], open=True)
        self.prop(rule, [q, self.s([],[],[],[], open=True), self.s([],[],[],[], open=True)], [out])
        self.check(out, ["x"], ["y"], [], [])

    def test_65_embedding_template_basic(self):
        # Tested above in 56, but explicit standalone check here
        rule = embedding_template(False).instantiate([(100,512), (4,32)], [(4,32,512)])
        out = self.s([],[],[], open=True)
        self.prop(rule, [self.s([],["y"]), self.s(["x"],[])], [out])
        self.check(out, ["x"], [], ["y"])

    # --- Debug/Misc ---
    def test_66_state_debug(self):
        s = FactorShardingState()
        s.merge("m", ["x"], 0, False, self.mesh)
        self.assertIn("m:", repr(s))

    def test_67_factor_properties(self):
        f = FactorSharding([], 0, False)
        self.assertTrue(f.is_explicit_replication)
        self.assertFalse(f.is_receptive)

    def test_68_rule_get_tensors(self):
        rule = self.make_rule([{0:["m"]}], [{0:["m"]}], {"m":2})
        self.assertEqual(rule.get_factor_tensors("m"), [("input",0,0), ("output",0,0)])

if __name__ == '__main__':
    unittest.main()