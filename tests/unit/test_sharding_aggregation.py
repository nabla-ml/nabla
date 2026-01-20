# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import unittest

from nabla.core.sharding.spec import (
    DeviceMesh,
    DimSpec,
    ShardingSpec,
    compute_global_shape,
)


class TestShardingAggregation(unittest.TestCase):
    """Tests for global shape computation from shards (Aggregation mode)."""

    def test_basic_sharding(self):
        """Simple 1D sharding case."""
        mesh = DeviceMesh("test", (4,), ("x",))

        sharding = ShardingSpec(mesh, [DimSpec(["x"])])

        shard_shapes = [(10,)] * 4
        global_shape = compute_global_shape((10,), sharding, shard_shapes=shard_shapes)
        self.assertEqual(global_shape, (40,))

    def test_uneven_sharding(self):
        """Uneven 1D sharding case."""
        mesh = DeviceMesh("test", (4,), ("x",))
        sharding = ShardingSpec(mesh, [DimSpec(["x"])])

        shard_shapes = [(3,), (3,), (3,), (1,)]
        global_shape = compute_global_shape((3,), sharding, shard_shapes=shard_shapes)
        self.assertEqual(global_shape, (10,))

    def test_replication_multi_axis(self):
        """Replication on one axis, sharding on another."""

        mesh = DeviceMesh("test", (2, 2), ("x", "y"))

        sharding = ShardingSpec(mesh, [DimSpec(["x"])])

        shard_shapes = [(10,), (10,), (5,), (5,)]

        global_shape = compute_global_shape((10,), sharding, shard_shapes=shard_shapes)
        self.assertEqual(global_shape, (15,))

    def test_sub_axis_aggregation_bugfix(self):
        """CRITICAL: Test sub-axis aggregation which failed with old root-axis 0-indexing logic."""

        mesh = DeviceMesh("test", (4,), ("x",))

        axis_name = "x:(1)2"
        sharding = ShardingSpec(mesh, [DimSpec([axis_name])])

        shard_shapes = [(10,), (10,), (10,), (10,)]

        global_shape = compute_global_shape((10,), sharding, shard_shapes=shard_shapes)
        self.assertEqual(global_shape, (20,))

        shard_shapes_uneven = [(10,), (10,), (5,), (5,)]

        global_shape_uneven = compute_global_shape(
            (10,), sharding, shard_shapes=shard_shapes_uneven
        )
        self.assertEqual(global_shape_uneven, (15,))

    def test_partial_dimension(self):
        """Verify partial dimensions are not aggregated (global == local)."""
        mesh = DeviceMesh("test", (4,), ("x",))
        sharding = ShardingSpec(mesh, [DimSpec(["x"], partial=True)])

        shard_shapes = [(10,)] * 4

        global_shape = compute_global_shape((10,), sharding, shard_shapes=shard_shapes)
        self.assertEqual(global_shape, (10,))


if __name__ == "__main__":
    unittest.main()
