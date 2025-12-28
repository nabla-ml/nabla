"""Rigorous tests for ALL view ops with sharding.

Each view op must:
1. Have sharding_rule that returns correct template
2. Propagate shardings correctly through the op
3. Execute correctly with sharded tensors
"""

import asyncio
import numpy as np
import pytest
from nabla import Tensor, DeviceMesh, DimSpec
from nabla.sharding.spec import ShardingSpec
from nabla.sharding.propagation import (
    unsqueeze_template,
    squeeze_template,
    swap_axes_template,
    broadcast_with_shapes_template,
    reshape_template,
    propagate_sharding,
)


class TestViewOpTemplates:
    """Test that all view op templates are correctly defined."""
    
    def test_unsqueeze_template_basic(self):
        """Unsqueeze (4,) axis=0 -> (1, 4): d0 maps to output d1."""
        template = unsqueeze_template(in_rank=1, axis=0)
        rule = template.instantiate([(4,)], [(1, 4)])
        
        # Input d0 should map to output d1
        assert rule.input_mappings[0] == {0: ["d0"]}
        assert rule.output_mappings[0][1] == ["d0"]
        
    def test_unsqueeze_template_middle(self):
        """Unsqueeze (4, 8) axis=1 -> (4, 1, 8)."""
        template = unsqueeze_template(in_rank=2, axis=1)
        rule = template.instantiate([(4, 8)], [(4, 1, 8)])
        
        # d0 -> d0, new at d1, d1 -> d2
        assert rule.input_mappings[0] == {0: ["d0"], 1: ["d1"]}
        
    def test_squeeze_template_basic(self):
        """Squeeze (1, 4) axis=0 -> (4,): output d0 comes from input d1."""
        template = squeeze_template(in_rank=2, axis=0)
        rule = template.instantiate([(1, 4)], [(4,)])
        
        # Input d0 is squeezed, input d1 maps to output d0
        assert rule.output_mappings[0] == {0: ["d0"]}
        
    def test_squeeze_template_middle(self):
        """Squeeze (4, 1, 8) axis=1 -> (4, 8)."""
        template = squeeze_template(in_rank=3, axis=1)
        rule = template.instantiate([(4, 1, 8)], [(4, 8)])
        
        # d0 stays, d1 squeezed, d2 -> d1
        assert rule.output_mappings[0] == {0: ["d0"], 1: ["d1"]}
        
    def test_swap_axes_template(self):
        """SwapAxes (4, 8) axis1=0, axis2=1 -> (8, 4)."""
        template = swap_axes_template(rank=2, axis1=0, axis2=1)
        rule = template.instantiate([(4, 8)], [(8, 4)])
        
        # Input d0 -> output d1, input d1 -> output d0
        assert rule.input_mappings[0] == {0: ["d0"], 1: ["d1"]}
        assert rule.output_mappings[0] == {0: ["d1"], 1: ["d0"]}


class TestViewOpPropagation:
    """Test sharding propagation through view ops."""
    
    def test_unsqueeze_preserves_sharding(self):
        """Unsqueeze: sharded input -> sharded output (shifted dim)."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Input sharded on dim0
        in_spec = ShardingSpec(mesh, [DimSpec(["x"], is_open=False)])
        out_spec = ShardingSpec(mesh, [DimSpec([], is_open=True), DimSpec([], is_open=True)])
        
        template = unsqueeze_template(in_rank=1, axis=0)
        rule = template.instantiate([(4,)], [(1, 4)])
        
        propagate_sharding(rule, [in_spec], [out_spec])
        
        # Sharding should be on output dim1 (shifted from input dim0)
        assert out_spec.dim_specs[1].axes == ["x"]
        
    def test_squeeze_preserves_sharding(self):
        """Squeeze: sharded input dim -> sharded output dim (shifted)."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Input sharded on dim1 (dim0 is the one to squeeze)
        in_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=False),  # dim0 to squeeze
            DimSpec(["x"], is_open=False)  # sharded
        ])
        out_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        template = squeeze_template(in_rank=2, axis=0)
        rule = template.instantiate([(1, 4)], [(4,)])
        
        propagate_sharding(rule, [in_spec], [out_spec])
        
        # Output dim0 should be sharded (was input dim1)
        assert out_spec.dim_specs[0].axes == ["x"]
        
    def test_swap_axes_swaps_sharding(self):
        """SwapAxes: sharding moves with the swapped dimension."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Input sharded on dim0
        in_spec = ShardingSpec(mesh, [
            DimSpec(["x"], is_open=False),  # sharded
            DimSpec([], is_open=False)
        ])
        out_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),
            DimSpec([], is_open=True)
        ])
        
        template = swap_axes_template(rank=2, axis1=0, axis2=1)
        rule = template.instantiate([(4, 8)], [(8, 4)])
        
        propagate_sharding(rule, [in_spec], [out_spec])
        
        # Sharding should now be on output dim1 (swapped from input dim0)
        assert out_spec.dim_specs[1].axes == ["x"]
        assert out_spec.dim_specs[0].axes == []


class TestViewOpExecution:
    """Test actual execution of view ops with sharded tensors."""
    
    def test_unsqueeze_execution(self):
        """Unsqueeze with sharded tensor."""
        from nabla.ops.view import unsqueeze
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        A = Tensor.ones((4, 8)).trace()
        A.shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        # Unsqueeze at axis 0
        B = unsqueeze(A, axis=0)
        
        assert tuple(int(d) for d in B.shape) == (1, 4, 8)
        assert B._impl.sharding is not None
        # Sharding should be on dim1 (shifted from dim0)
        assert B._impl.sharding.dim_specs[1].axes == ["x"]
        
    def test_squeeze_execution(self):
        """Squeeze with sharded tensor."""
        from nabla.ops.view import squeeze
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        A = Tensor.ones((1, 4, 8)).trace()
        A.shard(mesh, [DimSpec([]), DimSpec(["x"]), DimSpec([])])
        
        # Squeeze axis 0
        B = squeeze(A, axis=0)
        
        assert tuple(int(d) for d in B.shape) == (4, 8)
        assert B._impl.sharding is not None
        # Sharding should be on dim0 (was dim1 before squeeze)
        assert B._impl.sharding.dim_specs[0].axes == ["x"]
        
    def test_swap_axes_execution(self):
        """SwapAxes with sharded tensor."""
        from nabla.ops.view import swap_axes
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        A = Tensor.ones((4, 8)).trace()
        A.shard(mesh, [DimSpec(["x"]), DimSpec([])])  # Row sharded
        
        # Swap axes
        B = swap_axes(A, 0, 1)
        
        assert tuple(int(d) for d in B.shape) == (8, 4)
        assert B._impl.sharding is not None
        # Sharding should be on dim1 (was dim0 before swap)
        assert B._impl.sharding.dim_specs[1].axes == ["x"]
        
    def test_broadcast_execution(self):
        """Broadcast with sharded tensor."""
        from nabla.ops.view import broadcast_to
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        A = Tensor.ones((4, 1)).trace()
        A.shard(mesh, [DimSpec(["x"]), DimSpec([])])  # Row sharded
        
        # Broadcast to (4, 8)
        B = broadcast_to(A, (4, 8))
        
        assert tuple(int(d) for d in B.shape) == (4, 8)
        assert B._impl.sharding is not None
        # Row sharding should be preserved
        assert B._impl.sharding.dim_specs[0].axes == ["x"]


class TestViewOpChains:
    """Test chains of view ops with sharding."""
    
    def test_unsqueeze_then_matmul(self):
        """Unsqueeze followed by matmul."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        A = Tensor.ones((4,)).trace()
        A.shard(mesh, [DimSpec(["x"])])  # Sharded on single dim
        
        # Unsqueeze to (1, 4) then matmul with (4, 2)
        from nabla.ops.view import unsqueeze
        A_2d = unsqueeze(A, axis=0)
        
        B = Tensor.ones((4, 2)).trace()
        C = A_2d @ B
        
        assert tuple(int(d) for d in C.shape) == (1, 2)
        assert C._impl.sharding is not None
        
    def test_reshape_then_swap(self):
        """Reshape followed by swap_axes."""
        from nabla.ops.view import reshape, swap_axes
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        A = Tensor.ones((8,)).trace()
        A.shard(mesh, [DimSpec(["x"])])
        
        # Reshape to (2, 4)
        B = reshape(A, (2, 4))
        
        # Swap axes
        C = swap_axes(B, 0, 1)
        
        assert tuple(int(d) for d in C.shape) == (4, 2)
        assert C._impl.sharding is not None
class TestViewOpsNumerical:
    """Numerical validation tests for view ops with sharding."""
    
    def test_unsqueeze_numerical(self):
        """Verify unsqueeze produces correct numerical values when sharded."""
        from nabla.ops.view import unsqueeze
        
        print("\n=== Test: Unsqueeze Numerical Validation ===")
        
        mesh = DeviceMesh("m", (2,), ("x",))
        
        # Create array with distinct rows for verification
        np_A = np.arange(8, dtype=np.float32).reshape(4, 2)  # [[0,1], [2,3], [4,5], [6,7]]
        expected = np_A.reshape(1, 4, 2)  # Unsqueeze at axis 0
        
        A = Tensor.from_dlpack(np_A).trace()
        A.shard(mesh, [DimSpec(["x"]), DimSpec([])])  # Shard rows
        
        B = unsqueeze(A, axis=0)
        
        # Evaluate
        asyncio.run(B.realize)
        
        # Verify numerically
        assert B._impl.is_realized
        
        # Check shape
        assert tuple(int(d) for d in B.shape) == (1, 4, 2)
        
        # Check values - should be same as input, just reshaped
        shards = [s.to_numpy() for s in B._impl._storages]
        reconstructed = np.concatenate(shards, axis=1)  # Concat on sharded dim
        assert np.allclose(reconstructed, expected), f"Mismatch: {reconstructed} vs {expected}"
        
        print("    ✓ Unsqueeze numerical values verified.")
    
    def test_squeeze_numerical(self):
        """Verify squeeze produces correct numerical values when sharded."""
        from nabla.ops.view import squeeze
        
        print("\n=== Test: Squeeze Numerical Validation ===")
        
        mesh = DeviceMesh("m", (2,), ("x",))
        
        # Create array with shape (1, 4, 2)
        np_A = np.arange(8, dtype=np.float32).reshape(1, 4, 2)
        expected = np_A.reshape(4, 2)  # Squeeze axis 0
        
        A = Tensor.from_dlpack(np_A).trace()
        A.shard(mesh, [DimSpec([]), DimSpec(["x"]), DimSpec([])])  # Shard middle dim
        
        B = squeeze(A, axis=0)
        
        # Evaluate
        asyncio.run(B.realize)
        
        # Check shape
        assert tuple(int(d) for d in B.shape) == (4, 2)
        
        # Check values
        shards = [s.to_numpy() for s in B._impl._storages]
        reconstructed = np.concatenate(shards, axis=0)  # Concat on sharded dim (was dim 1, now dim 0)
        assert np.allclose(reconstructed, expected), f"Mismatch: {reconstructed} vs {expected}"
        
        print("    ✓ Squeeze numerical values verified.")
    
    def test_swap_axes_numerical(self):
        """Verify swap_axes produces correct numerical values when sharded."""
        from nabla.ops.view import swap_axes
        
        print("\n=== Test: SwapAxes Numerical Validation ===")
        
        mesh = DeviceMesh("m", (2,), ("x",))
        
        # Create matrix with distinct values
        np_A = np.arange(8, dtype=np.float32).reshape(4, 2)  # [[0,1], [2,3], [4,5], [6,7]]
        expected = np_A.T  # Transpose: [[0,2,4,6], [1,3,5,7]]
        
        A = Tensor.from_dlpack(np_A).trace()
        A.shard(mesh, [DimSpec(["x"]), DimSpec([])])  # Shard rows
        
        B = swap_axes(A, 0, 1)
        
        # Evaluate
        asyncio.run(B.realize)
        
        # Check shape
        assert tuple(int(d) for d in B.shape) == (2, 4)
        
        # After swap_axes, sharding moves with the dimension
        # So if rows were sharded, now columns are sharded
        shards = [s.to_numpy() for s in B._impl._storages]
        reconstructed = np.concatenate(shards, axis=1)  # Concat on new sharded dim (was rows, now cols)
        assert np.allclose(reconstructed, expected), f"Mismatch: {reconstructed} vs {expected}"
        
        print("    ✓ SwapAxes numerical values verified.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
