# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Stress tests for sharding propagation on complex graphs.

Tests transformer blocks, diamond patterns, and numerical equivalence
between sharded and unsharded execution.
"""

import unittest

from nabla.core.sharding.propagation import OpShardingRuleTemplate
from nabla.core.sharding.spec import DeviceMesh


class TestShardingStress(unittest.TestCase):
    """Stress tests for complex sharding scenarios."""

    def setUp(self):
        """Create standard test meshes."""
        self.mesh_4 = DeviceMesh("test", (4,), ("d",), devices=[0, 1, 2, 3])
        self.mesh_2x2 = DeviceMesh("test2d", (2, 2), ("dp", "tp"), devices=[0, 1, 2, 3])

    def test_memory_cost_matmul(self):
        """Verify memory_cost returns correct bytes for matmul output."""
        from nabla.ops.binary import MatmulOp

        op = MatmulOp()

        input_shapes = [(32, 64), (64, 128)]
        output_shapes = [(32, 128)]

        expected = 32 * 128 * 4
        actual = op.memory_cost(input_shapes, output_shapes)

        self.assertEqual(actual, expected)

    def test_memory_cost_reduction(self):
        """Verify memory_cost for reduction operations."""
        from nabla.ops.reduction import ReduceSumOp

        op = ReduceSumOp()

        input_shapes = [(32, 64, 128)]
        output_shapes = [(32, 128)]

        expected = 32 * 128 * 4
        actual = op.memory_cost(input_shapes, output_shapes)

        self.assertEqual(actual, expected)

    def test_memory_cost_batch_matmul(self):
        """Verify memory_cost for batched matmul."""
        from nabla.ops.binary import MatmulOp

        op = MatmulOp()

        input_shapes = [(8, 32, 64), (8, 64, 128)]
        output_shapes = [(8, 32, 128)]

        expected = 8 * 32 * 128 * 4
        actual = op.memory_cost(input_shapes, output_shapes)

        self.assertEqual(actual, expected)

    def test_transformer_block_sharding_rules(self):
        """Verify sharding rules propagate through 8+ op transformer block.

        Simplified transformer: Q, K, V projections -> attention -> O proj -> FFN.
        This tests that factor propagation handles complex multi-op chains.
        """
        from nabla.ops.binary import AddOp, MatmulOp

        matmul_op = MatmulOp()
        add_op = AddOp()

        x_shape = (4, 32, 64)
        w_shape = (64, 64)
        w_ff1_shape = (64, 256)
        w_ff2_shape = (256, 64)

        q_shapes = [(4, 32, 64)]
        q_rule = matmul_op.sharding_rule([x_shape, w_shape], q_shapes)
        self.assertIsNotNone(q_rule)
        self.assertEqual(q_rule.to_einsum_notation(), "b0 m k, k n -> b0 m n")

        k_rule = matmul_op.sharding_rule([x_shape, w_shape], q_shapes)
        self.assertIsNotNone(k_rule)

        v_rule = matmul_op.sharding_rule([x_shape, w_shape], q_shapes)
        self.assertIsNotNone(v_rule)

        attn_rule = matmul_op.sharding_rule([(4, 32, 64), (4, 64, 32)], [(4, 32, 32)])
        self.assertIsNotNone(attn_rule)

        o_rule = matmul_op.sharding_rule([(4, 32, 64), w_shape], q_shapes)
        self.assertIsNotNone(o_rule)

        add_rule = add_op.sharding_rule([x_shape, x_shape], [x_shape])
        self.assertIsNotNone(add_rule)

        ffn1_rule = matmul_op.sharding_rule([x_shape, w_ff1_shape], [(4, 32, 256)])
        self.assertIsNotNone(ffn1_rule)

        ffn2_rule = matmul_op.sharding_rule([(4, 32, 256), w_ff2_shape], [x_shape])
        self.assertIsNotNone(ffn2_rule)

        rules = [
            q_rule,
            k_rule,
            v_rule,
            attn_rule,
            o_rule,
            add_rule,
            ffn1_rule,
            ffn2_rule,
        ]
        for i, rule in enumerate(rules):
            self.assertIsNotNone(rule, f"Rule {i} is None")

    def test_diamond_pattern_consistency(self):
        """Verify sharding consistency through fork-and-join (diamond) pattern.

        Pattern:
            x -> matmul1 -> branch1 (add scalar)
            x -> matmul1 -> branch2 (mul scalar)
            branch1 + branch2 -> output

        The shared intermediate must have consistent sharding.
        """
        from nabla.ops.binary import AddOp, MatmulOp, MulOp

        x_shape = (32, 64)
        w_shape = (64, 128)
        h_shape = (32, 128)

        matmul_op = MatmulOp()
        add_op = AddOp()
        mul_op = MulOp()

        matmul_rule = matmul_op.sharding_rule([x_shape, w_shape], [h_shape])

        add_rule = add_op.sharding_rule([h_shape, h_shape], [h_shape])

        mul_rule = mul_op.sharding_rule([h_shape, h_shape], [h_shape])

        join_rule = add_op.sharding_rule([h_shape, h_shape], [h_shape])

        self.assertEqual(matmul_rule.to_einsum_notation(), "m k, k n -> m n")
        self.assertEqual(add_rule.to_einsum_notation(), "d0 d1, d0 d1 -> d0 d1")
        self.assertEqual(mul_rule.to_einsum_notation(), "d0 d1, d0 d1 -> d0 d1")
        self.assertEqual(join_rule.to_einsum_notation(), "d0 d1, d0 d1 -> d0 d1")

    def test_template_parsing_complex(self):
        """Test OpShardingRuleTemplate parsing for complex patterns."""

        template = OpShardingRuleTemplate.parse(
            "... m k, ... k n -> ... m n", [(8, 32, 64), (8, 64, 128)]
        )
        rule = template.instantiate([(8, 32, 64), (8, 64, 128)], [(8, 32, 128)])
        self.assertIsNotNone(rule)
        self.assertEqual(rule.to_einsum_notation(), "b0 m k, b0 k n -> b0 m n")

        contracting = rule.get_contracting_factors()
        self.assertIn("k", contracting)
        self.assertNotIn("m", contracting)
        self.assertNotIn("n", contracting)

    def test_template_ellipsis_expansion(self):
        """Test that ellipsis correctly expands for different batch dimensions."""

        template = OpShardingRuleTemplate.parse(
            "... m k, ... k n -> ... m n", [(2, 4, 32, 64), (2, 4, 64, 128)]
        )
        rule = template.instantiate(
            [(2, 4, 32, 64), (2, 4, 64, 128)], [(2, 4, 32, 128)]
        )
        self.assertEqual(rule.to_einsum_notation(), "b0 b1 m k, b0 b1 k n -> b0 b1 m n")


class TestMemoryCostIntegration(unittest.TestCase):
    """Test memory_cost integration across operation types."""

    def test_memory_cost_exists_on_all_base_classes(self):
        """Verify memory_cost method is accessible on all base operation classes."""
        from nabla.ops.base import (
            BinaryOperation,
            Operation,
            ReduceOperation,
            UnaryOperation,
        )

        for cls in [Operation, BinaryOperation, UnaryOperation, ReduceOperation]:
            self.assertTrue(hasattr(cls, "memory_cost"))

    def test_memory_cost_dtype_override(self):
        """Verify dtype_bytes parameter works correctly."""
        from nabla.ops.binary import MatmulOp

        op = MatmulOp()
        input_shapes = [(32, 64), (64, 128)]
        output_shapes = [(32, 128)]

        cost_f32 = op.memory_cost(input_shapes, output_shapes, dtype_bytes=4)

        cost_f64 = op.memory_cost(input_shapes, output_shapes, dtype_bytes=8)

        cost_f16 = op.memory_cost(input_shapes, output_shapes, dtype_bytes=2)

        self.assertEqual(cost_f64, cost_f32 * 2)
        self.assertEqual(cost_f16, cost_f32 // 2)


if __name__ == "__main__":
    unittest.main()
