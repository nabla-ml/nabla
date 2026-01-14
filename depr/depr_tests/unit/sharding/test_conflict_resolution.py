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
