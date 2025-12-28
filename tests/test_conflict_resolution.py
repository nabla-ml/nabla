"""Tests for auto-conflict resolution in sharding.

When inputs have conflicting shardings (e.g., A row-sharded, B col-sharded for
elementwise), the system should detect this and either:
1. Auto-resolve to a common sharding (JAX style)
2. Error with a helpful message (PyTorch style)

We implement option 1 (auto-resolve to replicated).
"""

import pytest
from nabla import Tensor, DeviceMesh, DimSpec
from nabla.sharding.spec import ShardingSpec


class TestConflictDetection:
    """Test that conflicts are properly detected."""
    
    def test_detect_conflicting_elementwise_inputs(self):
        """Detect when A is row-sharded and B is col-sharded for elementwise."""
        from nabla.sharding.spmd import detect_sharding_conflict
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # A: row-sharded (dim 0 = x)
        a_spec = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
        
        # B: col-sharded (dim 1 = x)
        b_spec = ShardingSpec(mesh, [DimSpec([]), DimSpec(["x"])])
        
        # For elementwise, both should use same sharding per dimension
        # This is a conflict because same axis "x" appears in different dims
        has_conflict = detect_sharding_conflict(None, [a_spec, b_spec])
        
        assert has_conflict, "Should detect conflict when A[dim0=x] and B[dim1=x]"
    
    def test_no_conflict_when_same_sharding(self):
        """No conflict when both inputs have same sharding."""
        from nabla.sharding.spmd import detect_sharding_conflict
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Both row-sharded
        a_spec = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
        b_spec = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
        
        has_conflict = detect_sharding_conflict(None, [a_spec, b_spec])
        
        assert not has_conflict, "No conflict when both have same sharding"
    
    def test_no_conflict_one_replicated(self):
        """No conflict when one input is replicated."""
        from nabla.sharding.spmd import detect_sharding_conflict
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # A sharded, B replicated
        a_spec = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
        b_spec = ShardingSpec(mesh, [DimSpec([]), DimSpec([])])
        
        has_conflict = detect_sharding_conflict(None, [a_spec, b_spec])
        
        assert not has_conflict, "No conflict when one input is replicated"


class TestConflictResolution:
    """Test automatic resolution of conflicts."""
    
    def test_elementwise_conflicting_inputs_auto_resolve(self):
        """Elementwise with conflicting inputs should auto-resolve."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # A: row-sharded
        A = Tensor.ones((4, 4)).trace()
        A.shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        # B: col-sharded (conflict!)
        B = Tensor.ones((4, 4)).trace()
        B.shard(mesh, [DimSpec([]), DimSpec(["x"])])
        
        # Execute addition - should NOT crash!
        # System should either:
        # 1. Reshard to common (replicated) and execute
        # 2. Use propagation to find common ground
        C = A + B
        
        # Verify C exists and has some sharding
        assert C is not None
        # (Implementation will determine what sharding C gets)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
