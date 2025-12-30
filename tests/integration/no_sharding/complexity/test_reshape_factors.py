"""Tests for reshape operations with compound factors.

Verifies that reshapes correctly handle sharding propagation using
compound factors, following XLA Shardy's specification.
"""

import pytest
from nabla.sharding.spec import DeviceMesh, DimSpec, ShardingSpec
from nabla.sharding.propagation import OpShardingRuleTemplate


class TestReshapeCompoundFactors:
    """Test reshape with compound factor mappings."""
    
    def test_reshape_merge_factors(self):
        """Test (2, 4) -> (8,) merge: two factors -> one compound factor."""
        # Input: (d0, d1) with d0=2, d1=4
        # Output: (compound) where compound = (d0*d1) = 8
        # Mapping: {0: ["d0"], 1: ["d1"]} -> {0: ["d0", "d1"]}
        
        input_mapping = {0: ["d0"], 1: ["d1"]}
        output_mapping = {0: ["d0", "d1"]}  # Compound factor
        
        template = OpShardingRuleTemplate([input_mapping], [output_mapping])
        
        # Instantiate with shapes
        rule = template.instantiate([(2, 4)], [(8,)])
        
        # Verify factor sizes inferred correctly
        assert rule.factor_sizes["d0"] == 2
        assert rule.factor_sizes["d1"] == 4
        
        # Verify einsum notation  
        assert rule.to_einsum_notation() == "(d0, d1) -> ((d0,d1))"
    
    def test_reshape_split_factors(self):
        """Test (8,) -> (2, 4) split: one compound factor -> two factors."""
        # Input: (compound) where compound = (d0*d1) = 8
        # Output: (d0, d1) with d0=2, d1=4
        # Mapping: {0: ["d0", "d1"]} -> {0: ["d0"], 1: ["d1"]}
        
        input_mapping = {0: ["d0", "d1"]}  # Compound factor
        output_mapping = {0: ["d0"], 1: ["d1"]}
        
        template = OpShardingRuleTemplate([input_mapping], [output_mapping])
        
        # Instantiate
        rule = template.instantiate([(8,)], [(2, 4)])
        
        # Verify factors
        assert rule.factor_sizes["d0"] == 2
        assert rule.factor_sizes["d1"] == 4
    
    def test_reshape_partial_compound(self):
        """Test (8, 32) -> (2, 4, 32): partial merge."""
        # Input: (compound, k) where compound = (d0*d1) = 8, k=32
        # Output: (d0, d1, k) with d0=2, d1=4, k=32
        
        input_mapping = {0: ["d0", "d1"], 1: ["k"]}
        output_mapping = {0: ["d0"], 1: ["d1"], 2: ["k"]}
        
        template = OpShardingRuleTemplate([input_mapping], [output_mapping])
        rule = template.instantiate([(8, 32)], [(2, 4, 32)])
        
        assert rule.factor_sizes["d0"] == 2
        assert rule.factor_sizes["d1"] == 4
        assert rule.factor_sizes["k"] == 32


class TestReshapeShardingPropagation:
    """Test sharding propagation through reshapes."""
    
    def test_merge_preserves_first_factor_sharding(self):
        """(2, 4) -> (8,) with first dim sharded should preserve in compound."""
        from nabla.sharding.propagation import propagate_sharding
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Input sharded on first dimension
        input_spec = ShardingSpec(mesh, [
            DimSpec(["x"], is_open=False),  # d0 sharded
            DimSpec([], is_open=True)        # d1 unsharded
        ])
        
        # Output initially open
        output_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        # Create rule
        template = OpShardingRuleTemplate(
            [{0: ["d0"], 1: ["d1"]}],
            [{0: ["d0", "d1"]}]
        )
        rule = template.instantiate([(2, 4)], [(8,)])
        
        # Propagate
        changed = propagate_sharding(rule, [input_spec], [output_spec])
        
        assert changed
        # Output should have x (from d0)
        assert output_spec.dim_specs[0].axes == ["x"]
    
    def test_split_distributes_sharding(self):
        """(8,) -> (2, 4) with dim sharded should shard first output dim."""
        from nabla.sharding.propagation import propagate_sharding
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Input compound dimension sharded
        input_spec = ShardingSpec(mesh, [DimSpec(["x"], is_open=False)])
        
        # Outputs initially open
        output_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),
            DimSpec([], is_open=True)
        ])
        
        # Create rule
        template = OpShardingRuleTemplate(
            [{0: ["d0", "d1"]}],
            [{0: ["d0"], 1: ["d1"]}]
        )
        rule = template.instantiate([(8,)], [(2, 4)])
        
        # Propagate
        changed = propagate_sharding(rule, [input_spec], [output_spec])
        
        assert changed
        # First output dim should get x (d0 gets first available axis)
        assert output_spec.dim_specs[0].axes == ["x"]
        # Second dim should remain unsharded (d1 doesn't get axes)
        assert output_spec.dim_specs[1].axes == []


class TestReshapeExecution:
    """Test actual reshape execution with sharding (not just propagation rules)."""
    
    def test_reshape_with_traced_tensors(self):
        """Test that reshape operation actually works with traced tensors."""
        from nabla import Tensor
        from nabla.ops.view import reshape
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Create tensor and shard it
        A = Tensor.ones((8, 4)).trace()
        A.shard(mesh, [
            DimSpec(["x"], is_open=False),  # First dim sharded
            DimSpec([], is_open=True)
        ])
        
        # Reshape (8, 4) -> (2, 4, 4)
        B = reshape(A, (2, 4, 4))
        
        # Verify B has sharding
        assert B._impl.sharding is not None
        
        # Verify shape is correct
        assert tuple(int(d) for d in B.shape) == (2, 4, 4)
    
    def test_reshape_preserves_compound_factor_sharding(self):
        """Test that reshape correctly preserves sharding via compound factors."""
        from nabla import Tensor
        from nabla.ops.view import reshape
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Create (2, 4) tensor with first dim sharded
        A = Tensor.ones((2, 4)).trace()
        A.shard(mesh, [
            DimSpec(["x"], is_open=False),
            DimSpec([], is_open=True)
        ])
        
        # Reshape to (8,) - should preserve sharding via compound factor
        B = reshape(A, (8,))
        
        # B should be sharded (compound factor gets d0's sharding)
        assert B._impl.sharding is not None
        assert B._impl.sharding.dim_specs[0].axes == ["x"]
    
    def test_reshape_split_distributes_sharding(self):
        """Test that splitting a sharded dimension distributes sharding."""
        from nabla import Tensor
        from nabla.ops.view import reshape
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Create (8,) tensor sharded
        A = Tensor.ones((8,)).trace()
        A.shard(mesh, [DimSpec(["x"], is_open=False)])
        
        # Reshape to (2, 4)
        B = reshape(A, (2, 4))
        
        # B should have sharding on first dim (d0 gets the axis)
        assert B._impl.sharding is not None
        assert B._impl.sharding.dim_specs[0].axes == ["x"]


class TestReshapeWithOtherOperations:
    """Test reshape interactions with other operations."""
    
    def test_matmul_after_reshape(self):
        """Test matmul after reshape propagates sharding correctly."""
        from nabla import Tensor
        from nabla.ops.view import reshape
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # A: (8,) sharded -> reshape -> (2, 4) -> matmul with B
        A = Tensor.ones((8,)).trace()
        A.shard(mesh, [DimSpec(["x"], is_open=False)])
        
        # Reshape to (2, 4)
        A_reshaped = reshape(A, (2, 4))
        
        # Verify A_reshaped is sharded on first dim
        assert A_reshaped._impl.sharding.dim_specs[0].axes == ["x"]
        
        # Matmul with B
        B = Tensor.ones((4, 3)).trace()
        C = A_reshaped @ B
        
        # C should be row-sharded (from A_reshaped's row sharding)
        assert C._impl.sharding is not None
        assert C._impl.sharding.dim_specs[0].axes == ["x"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
