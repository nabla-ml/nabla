# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import unittest
import numpy as np
from nabla.sharding.spec import DeviceMesh, DimSpec, ShardingSpec, compute_global_shape

class TestShardingAggregation(unittest.TestCase):
    """Tests for global shape computation from shards (Aggregation mode)."""

    def test_basic_sharding(self):
        """Simple 1D sharding case."""
        mesh = DeviceMesh("test", (4,), ("x",))
        # Sharded on 'x'
        sharding = ShardingSpec(mesh, [DimSpec(["x"])])
        
        # Local shapes: [10], [10], [10], [10] -> Global [40]
        shard_shapes = [(10,)] * 4
        global_shape = compute_global_shape((10,), sharding, shard_shapes=shard_shapes)
        self.assertEqual(global_shape, (40,))

    def test_uneven_sharding(self):
        """Uneven 1D sharding case."""
        mesh = DeviceMesh("test", (4,), ("x",))
        sharding = ShardingSpec(mesh, [DimSpec(["x"])])
        
        # Shards: [3], [3], [3], [1] -> Global [10]
        shard_shapes = [(3,), (3,), (3,), (1,)]
        global_shape = compute_global_shape((3,), sharding, shard_shapes=shard_shapes)
        self.assertEqual(global_shape, (10,))

    def test_replication_multi_axis(self):
        """Replication on one axis, sharding on another."""
        # 2x2 mesh
        mesh = DeviceMesh("test", (2, 2), ("x", "y"))
        # Sharded on 'x', Replicated on 'y'
        sharding = ShardingSpec(mesh, [DimSpec(["x"])])
        
        # Devices: (0,0)=10, (0,1)=10 (Replica 1)
        #          (1,0)=5,  (1,1)=5  (Replica 2)
        shard_shapes = [(10,), (10,), (5,), (5,)]
        
        # Sum = 30. Replicas = 2 (along 'y'). Global = 30 / 2 = 15.
        global_shape = compute_global_shape((10,), sharding, shard_shapes=shard_shapes)
        self.assertEqual(global_shape, (15,))

    def test_sub_axis_aggregation_bugfix(self):
        """CRITICAL: Test sub-axis aggregation which failed with old root-axis 0-indexing logic."""
        # Mesh 'x' of size 4
        mesh = DeviceMesh("test", (4,), ("x",))
        
        # Sub-axis x_high = 'x:(1)2' (size 2, splits x into 2 blocks of 2)
        # x_high=0 for devices 0,1; x_high=1 for devices 2,3
        axis_name = "x:(1)2"
        sharding = ShardingSpec(mesh, [DimSpec([axis_name])])
        
        # Shard 0 (x_high=0) size 10: Devices 0, 1
        # Shard 1 (x_high=1) size 10: Devices 2, 3
        # In reality, they are replicated on the remaining part of x.
        shard_shapes = [(10,), (10,), (10,), (10,)]
        
        # Previsous logic would:
        # 1. See sharded_axes = {'x:(1)2'}
        # 2. See other_axes = ['x'] (since 'x' != 'x:(1)2')
        # 3. Sum only if indices['x'] == 0
        # 4. Result: 10 (incorrect, missing the other shard)
        
        # Correct logic:
        # Sum = 40. total_shards = 2. replicas = 2. Global = 40 / 2 = 20.
        global_shape = compute_global_shape((10,), sharding, shard_shapes=shard_shapes)
        self.assertEqual(global_shape, (20,))
        
        # Test with uneven sub-axis shards
        shard_shapes_uneven = [(10,), (10,), (5,), (5,)]
        # Sum = 30. Replicas = 2. Global = 15.
        global_shape_uneven = compute_global_shape((10,), sharding, shard_shapes=shard_shapes_uneven)
        self.assertEqual(global_shape_uneven, (15,))

    def test_partial_dimension(self):
        """Verify partial dimensions are not aggregated (global == local)."""
        mesh = DeviceMesh("test", (4,), ("x",))
        sharding = ShardingSpec(mesh, [DimSpec(["x"], partial=True)])
        
        shard_shapes = [(10,)] * 4
        # Even if sharded on x, because it's partial, global shape is same as local
        global_shape = compute_global_shape((10,), sharding, shard_shapes=shard_shapes)
        self.assertEqual(global_shape, (10,))

if __name__ == "__main__":
    unittest.main()
