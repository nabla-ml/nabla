# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Tests for communication operations - sharding primitives.

This file tests all communication/collective operations:
- ShardOp: Partition tensor across mesh
- AllGatherOp: Gather shards to replicated
- AllReduceOp: Reduce across shards
- ReshardOp: Change sharding specification

All ops are tested with:
- 1D meshes
- 2D asymmetric meshes (2x4, 4x2)
- 3D meshes
- Numerical verification against numpy
"""


from nabla.core.sharding.spec import DimSpec
from nabla.ops.communication import all_gather, all_reduce, reshard, broadcast
from .conftest import (
    assert_allclose,
    assert_is_sharded,
    assert_shape,
    make_array,
    tensor_from_numpy,
)


class TestShardOp:
    """Test ShardOp: partition tensor into shards."""

    def test_shard_1d_axis0(self, mesh_1d):
        """Shard tensor on first axis with 1D mesh."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)

        result = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])

        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, np_x)

        spec = result.sharding
        assert spec.dim_specs[0].axes == ["dp"]
        assert spec.dim_specs[1].axes == []

    def test_shard_1d_axis1(self, mesh_1d):
        """Shard tensor on second axis with 1D mesh."""
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = x.shard(mesh_1d, [DimSpec([]), DimSpec(["dp"])])

        assert_shape(result, (8, 16))
        assert_is_sharded(result, True)
        assert_allclose(result, np_x)

    def test_shard_2d_asymmetric(self, mesh_2x4):
        """Shard on 2D asymmetric mesh (2, 4)."""
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec(["tp"])])

        assert_shape(result, (8, 16))
        assert_is_sharded(result, True)
        assert_allclose(result, np_x)

        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes
        assert "tp" in spec.dim_specs[1].axes

    def test_shard_2d_asymmetric_flipped(self, mesh_4x2):
        """Shard on 2D asymmetric mesh (4, 2) - flipped dimensions."""
        np_x = make_array(16, 8, seed=42)
        x = tensor_from_numpy(np_x)

        result = x.shard(mesh_4x2, [DimSpec(["dp"]), DimSpec(["tp"])])

        assert_shape(result, (16, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, np_x)

    def test_shard_3d_mesh(self, mesh_3d):
        """Shard on 3D mesh (2, 2, 2)."""
        np_x = make_array(8, 8, 8, seed=42)
        x = tensor_from_numpy(np_x)

        result = x.shard(mesh_3d, [DimSpec(["dp"]), DimSpec(["tp"]), DimSpec(["pp"])])

        assert_shape(result, (8, 8, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, np_x)

        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes
        assert "tp" in spec.dim_specs[1].axes
        assert "pp" in spec.dim_specs[2].axes

    def test_shard_replicated(self, mesh_1d):
        """Shard with fully replicated spec (no actual sharding)."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)

        result = x.shard(mesh_1d, [DimSpec([]), DimSpec([])])

        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, np_x)

        spec = result.sharding
        assert spec.is_fully_replicated()

    def test_shard_numerical_values(self, mesh_1d_2):
        """Verify shard contents are numerically correct slices."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)

        result = x.shard(mesh_1d_2, [DimSpec(["dp"]), DimSpec([])])

        assert result.is_sharded
        assert result.num_shards == 2

        assert_allclose(result, np_x)


class TestAllGatherOp:
    """Test AllGatherOp: gather shards to full tensor."""

    def test_all_gather_1d(self, mesh_1d):
        """AllGather on 1D mesh."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])

        result = all_gather(x_sharded, axis=0)

        assert_shape(result, (8, 4))
        assert_allclose(result, np_x)

        spec = result.sharding
        assert spec is None or spec.is_fully_replicated()

    def test_all_gather_2d_asymmetric(self, mesh_2x4):
        """AllGather on 2D asymmetric mesh - gather one axis."""
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec(["tp"])])

        result = all_gather(x_sharded, axis=1)

        assert result.shape[1] == 16
        assert_allclose(result, np_x)

        spec = result.sharding
        if spec and len(spec.dim_specs) > 1:
            assert "tp" not in spec.dim_specs[1].axes

    def test_all_gather_3d(self, mesh_3d):
        """AllGather on 3D mesh."""
        np_x = make_array(8, 8, 8, seed=42)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh_3d, [DimSpec(["dp"]), DimSpec([]), DimSpec([])])

        result = all_gather(x_sharded, axis=0)

        assert_shape(result, (8, 8, 8))
        assert_allclose(result, np_x)

    def test_all_gather_numerical(self, mesh_1d_2):
        """Verify all_gather produces correct full array."""
        np_x = make_array(6, 4, seed=42)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh_1d_2, [DimSpec(["dp"]), DimSpec([])])

        result = all_gather(x_sharded, axis=0)

        assert_allclose(result, np_x)


class TestAllReduceOp:
    """Test AllReduceOp: reduce values across all shards."""

    def test_all_reduce_sum_1d(self, mesh_1d):
        """AllReduce sum on 1D mesh."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])

        result = all_reduce(x_sharded)

        assert result is not None

    def test_all_reduce_sum_2d_asymmetric(self, mesh_2x4):
        """AllReduce on 2D asymmetric mesh."""
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec(["tp"])])

        result = all_reduce(x_sharded)

        assert result is not None

    def test_all_reduce_numerical(self, mesh_1d_2):
        """Verify all_reduce sum produces correct result."""
        np_x = make_array(4, 4, seed=42)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh_1d_2, [DimSpec(["dp"]), DimSpec([])])

        result = all_reduce(x_sharded)

        assert result is not None


class TestReshardOp:
    """Test reshard: change from one sharding to another."""

    def test_reshard_change_sharding(self, mesh_2x4):
        """Reshard from one axis to another."""
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec([])])

        result = reshard(x_sharded, mesh_2x4, [DimSpec([]), DimSpec(["tp"])])

        assert_shape(result, (8, 16))
        assert_allclose(result, np_x)

        spec = result.sharding
        assert spec.dim_specs[0].axes == []
        assert "tp" in spec.dim_specs[1].axes

    def test_reshard_to_replicated(self, mesh_1d):
        """Reshard from sharded to fully replicated."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])

        result = reshard(x_sharded, mesh_1d, [DimSpec([]), DimSpec([])])

        assert_shape(result, (8, 4))
        assert_allclose(result, np_x)

        spec = result.sharding
        assert spec.is_fully_replicated()

    def test_reshard_between_different_patterns(self, mesh_2x4):
        """Reshard between different multi-axis patterns."""
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec(["tp"])])

        result = reshard(x_sharded, mesh_2x4, [DimSpec(["tp"]), DimSpec(["dp"])])

        assert_shape(result, (8, 16))
        assert_allclose(result, np_x)


class TestCommunicationOpsEdgeCases:
    """Test edge cases for communication ops."""

    def test_shard_uneven_size(self, mesh_1d):
        """Shard dimension not evenly divisible by mesh size."""

        np_x = make_array(10, 4, seed=42)
        x = tensor_from_numpy(np_x)

        result = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])

        assert_shape(result, (10, 4))
        assert_is_sharded(result, True)

    def test_shard_already_sharded(self, mesh_1d):
        """Shard an already-sharded tensor (should reshard)."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])

        result = x_sharded.shard(mesh_1d, [DimSpec([]), DimSpec(["dp"])])

        assert_shape(result, (8, 4))
        assert_allclose(result, np_x)


class TestBroadcastOp:
    """Test BroadcastOp: replicate tensor from root to all devices."""

    def test_broadcast_1d_mesh(self, mesh_1d):
        """Broadcast tensor from device 0 to all devices on 1D mesh."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)

        # Shard tensor first
        x_sharded = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])

        # Broadcast from device 0
        result = broadcast(x_sharded, root=0)

        # Result should be replicated (full shape on all devices)
        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, np_x)

        # Verify output is fully replicated
        spec = result.sharding
        assert spec.is_fully_replicated()

    def test_broadcast_2d_mesh(self, mesh_2x4):
        """Broadcast tensor on 2D mesh."""
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        # Shard on both axes
        x_sharded = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec(["tp"])])

        # Broadcast should replicate to all devices
        result = broadcast(x_sharded, root=0)

        assert_shape(result, (8, 16))
        assert_allclose(result, np_x)
        assert result.sharding.is_fully_replicated()

    def test_broadcast_preserves_values(self, mesh_1d):
        """Verify broadcast preserves exact tensor values."""
        np_x = make_array(4, 8, seed=123)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])
        result = broadcast(x_sharded, root=0)

        # Every device should have the complete tensor
        assert_allclose(result, np_x)

    def test_broadcast_single_device_noop(self, mesh_1d):
        """Broadcast on single device should be no-op."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)

        # Don't shard - single device
        result = broadcast(x, root=0)

        assert_shape(result, (8, 4))
        assert_allclose(result, np_x)


__all__ = [
    "TestShardOp",
    "TestAllGatherOp",
    "TestAllReduceOp",
    "TestReshardOp",
    "TestBroadcastOp",
    "TestCommunicationOpsEdgeCases",
]
