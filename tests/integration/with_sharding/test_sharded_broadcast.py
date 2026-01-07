"""Test sharding with broadcasting scenarios.

These tests verify the interaction between:
1. Sharded tensors with batch_dims (e.g., from vmap)
2. Unsharded scalars/tensors that need broadcasting
3. The correct flow through propagation and resharding
"""

import unittest
import numpy as np

from nabla.core.tensor import Tensor
from nabla.sharding.spec import DeviceMesh, ShardingSpec, DimSpec
from nabla.ops.communication import shard
from nabla.utils.debug import xpr


class TestShardedBroadcast(unittest.TestCase):
    """Test broadcasting between sharded and unsharded tensors."""
    
    def setUp(self):
        """Create a 2x2 mesh for testing."""
        self.mesh = DeviceMesh("test", (2, 2), ("x", "y"))
    
    def test_sharded_plus_scalar(self):
        """Test: sharded tensor + scalar constant."""
        # Create a 4x4 tensor, shard on both dims
        data = np.arange(16).reshape(4, 4).astype(np.float32)
        x = Tensor.from_dlpack(data).trace()
        
        # Shard: dim 0 by "x", dim 1 by "y"
        x_sharded = shard(x, self.mesh, [DimSpec(["x"]), DimSpec(["y"])])
        
        print("\n=== Test: sharded_plus_scalar ===")
        print(f"x_sharded.shape: {x_sharded.shape}")
        print(f"x_sharded sharding: {x_sharded.sharding}")
        print(f"x_sharded._impl._values count: {len(x_sharded._impl._values) if x_sharded._impl._values else 0}")
        
        # Add scalar
        result = x_sharded + 1.0
        
        print(f"\nresult.shape: {result.shape}")
        print(f"result sharding: {result.sharding}")
        print(f"result._impl._values count: {len(result._impl._values) if result._impl._values else 0}")
        
        # Debug trace
        print("\n--- Computation Graph ---")
        print(xpr(result))
        
        # Verify output is still sharded correctly
        self.assertIsNotNone(result.sharding)
        self.assertEqual(len(result._impl._values), 4)  # 2x2 mesh = 4 shards
    
    def test_sharded_with_batch_dims_plus_scalar(self):
        """Test: sharded tensor with batch_dims + scalar.
        
        This simulates what happens inside vmap when we have:
        - A tensor that came from vmap (has batch_dims > 0)
        - That tensor is also sharded
        - We add a scalar to it
        """
        # Create a tensor with batch dimension
        # Physical shape: (batch=2, H=4, W=4), batch_dims=1
        data = np.arange(32).reshape(2, 4, 4).astype(np.float32)
        x = Tensor.from_dlpack(data).trace()
        
        # Simulate batch_dims=1 (as if from vmap)
        x._impl.batch_dims = 1
        
        # Shard the LOGICAL dims (H, W) - physical dims 1 and 2
        # batch dim stays replicated
        x_sharded = shard(x, self.mesh, [
            DimSpec([]),       # batch dim: replicated
            DimSpec(["x"]),    # H dim: sharded on "x"
            DimSpec(["y"]),    # W dim: sharded on "y"
        ])
        
        print("\n=== Test: sharded_with_batch_dims_plus_scalar ===")
        print(f"x_sharded.shape (logical): {x_sharded.shape}")
        print(f"x_sharded.batch_dims: {x_sharded.batch_dims}")
        print(f"x_sharded sharding: {x_sharded.sharding}")
        
        # Add scalar - this should broadcast correctly
        result = x_sharded + 1.0
        
        print(f"\nresult.shape: {result.shape}")
        print(f"result.batch_dims: {result.batch_dims}")
        print(f"result sharding: {result.sharding}")
        
        # Debug trace
        print("\n--- Computation Graph ---")
        print(xpr(result))
        
        # Verify
        self.assertEqual(result.batch_dims, 1)
        self.assertIsNotNone(result.sharding)
    
    def test_sharded_plus_unsharded_tensor(self):
        """Test: sharded tensor + unsharded tensor with same shape."""
        # Create two 4x4 tensors
        data_a = np.arange(16).reshape(4, 4).astype(np.float32)
        data_b = np.ones((4, 4), dtype=np.float32) * 10
        
        a = Tensor.from_dlpack(data_a).trace()
        b = Tensor.from_dlpack(data_b).trace()
        
        # Shard only 'a'
        a_sharded = shard(a, self.mesh, [DimSpec(["x"]), DimSpec(["y"])])
        
        print("\n=== Test: sharded_plus_unsharded_tensor ===")
        print(f"a_sharded sharding: {a_sharded.sharding}")
        print(f"b sharding: {b.sharding}")
        
        # Add them - 'b' should get sharded to match 'a'
        result = a_sharded + b
        
        print(f"\nresult sharding: {result.sharding}")
        print(f"result._impl._values count: {len(result._impl._values) if result._impl._values else 0}")
        
        # Debug trace
        print("\n--- Computation Graph ---")
        print(xpr(result))
        
        # Verify both now contribute to sharded output
        self.assertIsNotNone(result.sharding)
        self.assertEqual(len(result._impl._values), 4)
    
    def test_2d_mesh_allgather_reshard(self):
        """Test resharding on 2D mesh - the bug case.
        
        If we have a tensor sharded on both mesh axes and want to 
        all-gather to replicated, this should produce correct data.
        """
        import asyncio
        
        # Create 4x4 tensor, shard on 2x2 mesh
        data = np.arange(16).reshape(4, 4).astype(np.float32)
        x = Tensor.from_dlpack(data).trace()
        
        # Shard on both dims
        x_sharded = shard(x, self.mesh, [DimSpec(["x"]), DimSpec(["y"])])
        
        print("\n=== Test: 2d_mesh_allgather_reshard ===")
        print(f"Original data:\n{data}")
        print(f"x_sharded._impl._values count: {len(x_sharded._impl._values) if x_sharded._impl._values else 0}")
        
        # Print each shard's data (via graph inspection)
        if x_sharded._impl._values:
            for i, val in enumerate(x_sharded._impl._values):
                print(f"  Shard {i} shape: {val.type.shape}")
        
        # Reshard to fully replicated
        from nabla.ops.communication import reshard
        x_replicated = reshard(x_sharded, self.mesh, [DimSpec([]), DimSpec([])])
        
        print(f"\nAfter reshard to replicated:")
        print(f"x_replicated sharding: {x_replicated.sharding}")
        print(f"x_replicated._impl._values count: {len(x_replicated._impl._values) if x_replicated._impl._values else 0}")
        
        # All shards should have the full 4x4 shape
        if x_replicated._impl._values:
            for i, val in enumerate(x_replicated._impl._values):
                expected_shape = (4, 4)
                actual_shape = tuple(int(d) for d in val.type.shape)
                print(f"  Shard {i} shape: {actual_shape}")
                self.assertEqual(actual_shape, expected_shape, 
                    f"Shard {i} has shape {actual_shape}, expected {expected_shape}")
        
        # CRITICAL: Realize and check actual values
        print("\n--- Realizing tensor to check values ---")
        asyncio.run(x_replicated.realize)
        
        # Check that all shards have the FULL data (not duplicated partial data)
        if x_replicated._impl._storages:
            for i, storage in enumerate(x_replicated._impl._storages):
                shard_data = np.array(storage.to_numpy())
                print(f"  Shard {i} data:\n{shard_data}")
                
                # Each shard should have the full original data
                np.testing.assert_array_almost_equal(
                    shard_data, data,
                    err_msg=f"Shard {i} data doesn't match original!"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
