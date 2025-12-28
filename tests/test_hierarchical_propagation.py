"""Tests for hierarchical sharding propagation.

These tests verify that the propagation system correctly enforces
the XLA Shardy hierarchy:
1. User priorities (p0, p1, p2...)
2. Operation priorities (PASSTHROUGH vs CONTRACTION vs REDUCTION)
3. Propagation strategies (AGGRESSIVE vs BASIC)
"""

import pytest
from nabla.sharding.spec import DeviceMesh, DimSpec, ShardingSpec
from nabla.sharding.propagation import (
    OpShardingRule,
    propagate_sharding,
    elementwise_template, 
    matmul_template,
    PropagationStrategy,
    OpPriority,
)


class TestUserPriorityEnforcement:
    """Test that user priorities prevent lower-priority overrides."""
    
    def test_priority_0_beats_priority_1(self):
        """Priority 0 (user/default) should not be overridden by priority 1."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Spec 1: priority 0, sharded on x
        spec1 = ShardingSpec(mesh, [DimSpec(["x"], is_open=False, priority=0)])
        
        # Spec 2: priority 1, sharded on y (hypothetical conflict)
        # Since we only have x axis, let's test with empty (replicated)
        spec2 = ShardingSpec(mesh, [DimSpec([], is_open=True, priority=1)])
        
        # Output initially open
        out_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        # Elementwise: both inputs contribute to same factor
        template = elementwise_template(rank=1)
        rule = template.instantiate([(4,), (4,)], [(4,)])
        
        # Propagate with max_priority=0 (only honor p0)
        propagate_sharding(rule, [spec1, spec2], [out_spec], max_priority=0)
        
        # Output should get x from spec1 (priority 0), not influenced by spec2 (priority 1)
        assert out_spec.dim_specs[0].axes == ["x"]
        assert out_spec.dim_specs[0].priority == 0
    
    def test_max_priority_filtering(self):
        """max_priority parameter should filter out higher priorities."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Only priority 1 spec
        spec1 = ShardingSpec(mesh, [DimSpec(["x"], is_open=False, priority=1)])
        out_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        template = elementwise_template(rank=1)
        rule = template.instantiate([(4,)], [(4,)])
        
        # With max_priority=0, should NOT propagate priority 1
        changed = propagate_sharding(rule, [spec1], [out_spec], max_priority=0)
        
        # Output should remain empty (priority 1 filtered out)
        assert out_spec.dim_specs[0].axes == []
        
        # With max_priority=1, should propagate
        changed = propagate_sharding(rule, [spec1], [out_spec], max_priority=1)
        assert out_spec.dim_specs[0].axes == ["x"]


class TestIterativePropagation:
    """Test that multiple propagation iterations work correctly."""
    
    def test_priority_0_then_1_propagation(self):
        """Simulate iterative propagation: first p0, then p1."""
        mesh = DeviceMesh("test", (2, 2), ("x", "y"))
        
        # Create a chain: A (p0: x) -> B (p1: y) -> C
        # First iteration (p0): A's x should reach C
        # Second iteration (p1): B's y should NOT override x
        
        # A: priority 0, sharded on x
        a_spec = ShardingSpec(mesh, [DimSpec(["x"], is_open=False, priority=0)])
        
        # B: priority 1, sharded on y
        b_spec = ShardingSpec(mesh, [DimSpec(["y"], is_open=True, priority=1)])
        
        # C: initially open
        c_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        template = elementwise_template(rank=1)
        rule = template.instantiate([(4,), (4,)], [(4,)])
        
        # Iteration 1: max_priority=0
        propagate_sharding(rule, [a_spec, b_spec], [c_spec], max_priority=0)
        
        # C should get x (from A with p0)
        assert c_spec.dim_specs[0].axes == ["x"]
        assert c_spec.dim_specs[0].priority == 0
        
        # Iteration 2: max_priority=1 (but x is already closed at p0)
        c_spec_before = c_spec.dim_specs[0].axes.copy()
        propagate_sharding(rule, [a_spec, b_spec], [c_spec], max_priority=1)
        
        # C should STILL have x (p0 protected from p1)
        assert c_spec.dim_specs[0].axes == ["x"]


class TestHierarchicalPass:
    """Test the complete hierarchical propagation pass."""
    
    def test_simple_chain_with_priorities(self):
        """Test propagation through a simple chain with mixed priorities."""
        from nabla.sharding.propagation import run_hierarchical_propagation_pass
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Create specs for chain: A (p0) -> B -> C (p1) -> D
        a_spec = ShardingSpec(mesh, [DimSpec(["x"], is_open=False, priority=0)])
        b_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        c_spec = ShardingSpec(mesh, [DimSpec([], is_open=True, priority=1)])
        d_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        # Create rules for two elementwise operations
        template = elementwise_template(rank=1)
        rule1 = template.instantiate([(4,), (4,)], [(4,)])  # A + B -> C
        rule2 = template.instantiate([(4,), (4,)], [(4,)])  # C + ? -> D
        
        # Mock operations (don't actually need real ops, just need to group specs)
        from nabla.ops.binary import add  # Use as placeholder
        operations_with_rules = [
            (add, rule1, [a_spec, b_spec], [c_spec]),
            (add, rule2, [c_spec, d_spec], [d_spec]),  # Second operation reuses d_spec as both input and output for simplicity
        ]
        
        # Run hierarchical pass
        changes = run_hierarchical_propagation_pass(operations_with_rules, max_user_priority=1)
        
        # Verify propagation
        assert changes > 0, "Should have made changes"
        
        # B should get x from A (p0)
        assert b_spec.dim_specs[0].axes == ["x"]
        
        # C should get x from A via B (p0)
        assert c_spec.dim_specs[0].axes == ["x"]
        
        # D should get x from C
        assert d_spec.dim_specs[0].axes == ["x"]


class TestOperationPriorities:
    """Test operation-based priority ordering."""
    
    def test_passthrough_before_contraction(self):
        """PASSTHROUGH operations should propagate before CONTRACTION."""
        from nabla.sharding.propagation import OpPriority
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Simulate: A (sharded) -> elementwise -> B -> matmul -> C
        # Elementwise (PASSTHROUGH) should propagate first
        # Matmul (CONTRACTION) should propagate second
        
        a_spec = ShardingSpec(mesh, [DimSpec(["x"], is_open=False, priority=0)])
        b_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        c_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        # Elementwise rule (A -> B)
        elem_template = elementwise_template(rank=1)
        elem_rule = elem_template.instantiate([(4,)], [(4,)])
        
        # Matmul rule (B, B -> C)  
        matmul_template_fn = matmul_template()
        matmul_rule = matmul_template_fn.instantiate([(4, 4), (4, 4)], [(4, 4)])
        
        # Mock operations with priorities
        class MockElemOp:
            op_priority = OpPriority.PASSTHROUGH
        class MockMatmulOp:
            op_priority = OpPriority.CONTRACTION
        
        operations = [
            (MockMatmulOp(), matmul_rule, [b_spec, b_spec], [c_spec]),  # Should run second
            (MockElemOp(), elem_rule, [a_spec], [b_spec]),  # Should run first
        ]
        
        # Run hierarchical pass (should order by op priority)
        from nabla.sharding.propagation import run_hierarchical_propagation_pass
        changes = run_hierarchical_propagation_pass(operations, max_user_priority=0)
        
        # B should get x from A (elementwise ran first)
        assert b_spec.dim_specs[0].axes == ["x"]


class TestPropagationStrategies:
    """Test AGGRESSIVE vs BASIC propagation strategies."""
    
    def test_basic_takes_common_prefix(self):
        """BASIC strategy should take longest common prefix when conflict."""
        from nabla.sharding.propagation import (
            propagate_sharding,
            PropagationStrategy,
            FactorShardingState,
        )
        
        mesh = DeviceMesh("test", (2, 2), ("x", "y"))
        
        # Two inputs with different shardings on same factor
        # Input 1: sharded on [x, y]
        # Input 2: sharded on [x]
        # BASIC should pick [x] (common prefix)
        
        spec1 = ShardingSpec(mesh, [DimSpec(["x", "y"], is_open=False, priority=0)])
        spec2 = ShardingSpec(mesh, [DimSpec(["x"], is_open=False, priority=0)])
        out_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        template = elementwise_template(rank=1)
        rule = template.instantiate([(4,), (4,)], [(4,)])
        
        # Use BASIC strategy
        changed = propagate_sharding(rule, [spec1, spec2], [out_spec], 
                                    strategy=PropagationStrategy.BASIC)
        
        # Output should have [x] (common prefix of [x,y] and [x])
        assert out_spec.dim_specs[0].axes == ["x"]
    
    def test_aggressive_picks_max_parallelism(self):
        """AGGRESSIVE strategy should pick sharding with most parallelism."""
        from nabla.sharding.propagation import propagate_sharding, PropagationStrategy
        
        mesh = DeviceMesh("test", (2, 4), ("x", "y"))
        
        # Input 1: sharded on [x] (parallelism = 2)
        # Input 2: sharded on [y] (parallelism = 4)
        # AGGRESSIVE should pick [y] (more parallelism)
        
        spec1 = ShardingSpec(mesh, [DimSpec(["x"], is_open=True, priority=0)])
        spec2 = ShardingSpec(mesh, [DimSpec(["y"], is_open=True, priority=0)])
        out_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        template = elementwise_template(rank=1)
        rule = template.instantiate([(8,), (8,)], [(8,)])
        
        # Use AGGRESSIVE strategy
        changed = propagate_sharding(rule, [spec1, spec2], [out_spec],
                                    strategy=PropagationStrategy.AGGRESSIVE)
        
        # Output should have [y] (more parallelism: 4 > 2)
        assert out_spec.dim_specs[0].axes == ["y"]


class TestComplexPropagationScenarios:
    """Test complex multi-operation scenarios."""
    
    def test_multi_stage_propagation(self):
        """Test propagation through multiple operations."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Chain: A (sharded) -> op1 -> B -> op2 -> C -> op3 -> D
        a_spec = ShardingSpec(mesh, [DimSpec(["x"], is_open=False, priority=0)])
        b_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        c_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        d_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        template = elementwise_template(rank=1)
        rule1 = template.instantiate([(4,)], [(4,)])
        rule2 = template.instantiate([(4,)], [(4,)])
        rule3 = template.instantiate([(4,)], [(4,)])
        
        from nabla.ops.binary import add
        operations = [
            (add, rule1, [a_spec], [b_spec]),
            (add, rule2, [b_spec], [c_spec]),
            (add, rule3, [c_spec], [d_spec]),
        ]
        
        from nabla.sharding.propagation import run_hierarchical_propagation_pass
        changes = run_hierarchical_propagation_pass(operations, max_user_priority=0)
        
        # All should get x from A
        assert b_spec.dim_specs[0].axes == ["x"]
        assert c_spec.dim_specs[0].axes == ["x"]
        assert d_spec.dim_specs[0].axes == ["x"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
