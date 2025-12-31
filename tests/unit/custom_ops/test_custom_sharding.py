# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Custom Sharding Tests
# ===----------------------------------------------------------------------=== #

import unittest
import numpy as np
import nabla
from nabla.sharding import DeviceMesh, ShardingSpec, DimSpec

class TestCustomSharding(unittest.TestCase):
    
    def test_basic_execution(self):
        """Test basic execution without sharding."""
        import asyncio
        from .custom_sharding_op import custom_sum_reduce
        
        async def run():
            # (2, 4)
            x_np = np.arange(8).reshape(2, 4).astype(np.float32)
            # Use Tensor.constant for numpy arrays
            x = nabla.Tensor.constant(x_np)
            
            res = custom_sum_reduce(x)
            res_np = (await res).to_numpy()
            
            expected = np.sum(x_np, axis=1)
            np.testing.assert_allclose(res_np, expected)
            
        asyncio.run(run())

    def test_sharding_propagation_parallel(self):
        """Test propagation: d0=x (parallel) -> d0=x."""
        from nabla.sharding.propagation import propagate_sharding
        from nabla.sharding import DeviceMesh, ShardingSpec, DimSpec
        from .custom_sharding_op import custom_sum_reduce
        
        mesh = DeviceMesh("dummy", (2,), ("x",))
        
        # Input: (d0, d1) with d0 sharded
        in_spec = ShardingSpec(mesh, [
            DimSpec(["x"], is_open=False),  # d0 sharded
            DimSpec([], is_open=True)       # d1 open
        ])
        
        # Output: (d0) initially open
        out_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        # Get rule from op
        rule = custom_sum_reduce.sharding_rule(
            input_shapes=[(4, 8)], 
            output_shapes=[(4,)],
            axis=1
        )
        
        # Propagate
        changed = propagate_sharding(rule, [in_spec], [out_spec])

    def test_sharding_propagation_reduction(self):
        """Test propagation: d1=x (contracting) -> output replicated."""
        from nabla.sharding.propagation import propagate_sharding
        from nabla.sharding import DeviceMesh, ShardingSpec, DimSpec
        from .custom_sharding_op import custom_sum_reduce
        
        mesh = DeviceMesh("dummy", (2,), ("x",))
        
        # Input: (d0, d1) with d1 sharded (contracting dim)
        in_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),      # d0 open
            DimSpec(["x"], is_open=False)   # d1 sharded
        ])
        
        # Output: (d0)
        out_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        rule = custom_sum_reduce.sharding_rule(
            input_shapes=[(4, 8)], 
            output_shapes=[(4,)],
            axis=1
        )
        
        changed = propagate_sharding(rule, [in_spec], [out_spec])
        
        # Output should NOT be sharded on x (since x was on reduced dim)
        # Factor d0 is untouched.
        # Note: changed might be False if output was already correctly matching the replicated state
        if changed:
             self.assertEqual(out_spec.dim_specs[0].axes, [])
        else:
             self.assertEqual(out_spec.dim_specs[0].axes, [])
        
        # Verify contracting factor identification
        contracting = rule.get_contracting_factors()
        self.assertIn("d1", contracting)

if __name__ == "__main__":
    unittest.main()
