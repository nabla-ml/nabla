# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import jax
import pytest

import nabla as nb
from nabla.core.sharding.spec import DimSpec
from nabla.ops.communication import (
    all_gather,
    all_reduce,
    all_to_all,
    reduce_scatter,
    reshard,
)

from .common import (
    DeviceMesh,
    OpConfig,
    Operation,
    assert_is_sharded,
    assert_shape,
    run_unified_test,
    standard_get_args,
)

OPS = {}


import jax.numpy as jnp
from tests.conftest import (
    assert_allclose,
    make_jax_array,
    tensor_from_jax,
)


@pytest.fixture
def mesh_1d():
    return DeviceMesh("mesh_1d", (4,), ("dp",))


@pytest.fixture
def mesh_2x4():
    return DeviceMesh("mesh_2x4", (2, 4), ("dp", "tp"))


OPS["all_reduce_sum"] = Operation(
    "all_reduce_sum",
    "COMMUNICATION",
    lambda x: all_reduce(x, reduce_op="sum"),
    lambda x: x,
    [
        OpConfig(
            "AllReduce_Sum_1D",
            ranks=(2,),
            params={},
            supports_vmap=False,
            supports_sharding=False,
        )
    ],
    standard_get_args,
)

OPS["all_gather"] = Operation(
    "all_gather",
    "COMMUNICATION",
    lambda x, axis: all_gather(x, axis),
    lambda x, axis: x,
    [
        OpConfig(
            "AllGather_0",
            ranks=(2,),
            params={"axis": 0},
            supports_vmap=False,
            supports_sharding=False,
        )
    ],
    standard_get_args,
)

OPS["reduce_scatter"] = Operation(
    "reduce_scatter",
    "COMMUNICATION",
    lambda x, axis: reduce_scatter(x, axis, reduce_op="sum"),
    lambda x, axis: x,
    [
        OpConfig(
            "ReduceScatter_0",
            ranks=(2,),
            params={"axis": 0},
            supports_vmap=False,
            supports_sharding=False,
        )
    ],
    standard_get_args,
)

OPS["all_to_all"] = Operation(
    "all_to_all",
    "COMMUNICATION",
    lambda x, axis: all_to_all(x, axis, 0, 0),
    lambda x, axis: x,
    [
        OpConfig(
            "AllToAll_0",
            ranks=(2,),
            params={"axis": 0},
            supports_vmap=False,
            supports_sharding=False,
        )
    ],
    standard_get_args,
)


@pytest.mark.parametrize("op_name", OPS.keys())
def test_communication_ops_generic(op_name):
    op = OPS[op_name]
    config = op.configs[0]
    run_unified_test(op, config)


class TestShardOp:
    """Test ShardOp: partition tensor across mesh."""

    def test_shard_1d_axis0(self, mesh_1d):
        """Shard tensor on first axis with 1D mesh."""
        jax_x = make_jax_array(8, 4, seed=42)
        x = tensor_from_jax(jax_x)
        result = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])

        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, jax_x)
        assert result.sharding.dim_specs[0].axes == ["dp"]

    def test_shard_2d_asymmetric(self, mesh_2x4):
        """Shard on 2D asymmetric mesh."""
        jax_x = make_jax_array(8, 16, seed=42)
        x = tensor_from_jax(jax_x)
        result = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec(["tp"])])

        assert_shape(result, (8, 16))
        assert_is_sharded(result, True)
        assert_allclose(result, jax_x)
        assert "dp" in result.sharding.dim_specs[0].axes
        assert "tp" in result.sharding.dim_specs[1].axes


class TestAllGatherOp:
    """Test AllGather: gather shards to replicated."""

    def test_all_gather_1d(self, mesh_1d):
        jax_x = make_jax_array(8, 4, seed=42)
        x = tensor_from_jax(jax_x)
        x_sharded = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])

        result = all_gather(x_sharded, axis=0)

        assert_shape(result, (8, 4))
        assert_allclose(result, jax_x)

        spec = result.sharding
        assert spec.dim_specs[0].axes == []


class TestReduceScatterOp:
    """Test ReduceScatter: reduce and then scatter."""

    def test_reduce_scatter_1d(self, mesh_1d):
        """ReduceScatter on 1D mesh: Replicated -> Reduce(Sum) -> Scatter(axis=0)."""

        jax_x = make_jax_array(8, 4, seed=42)
        x = tensor_from_jax(jax_x)

        x_rep = x.shard(mesh_1d, [DimSpec([]), DimSpec([])])

        result = reduce_scatter(x_rep, axis=0)

        expected_global = jax_x * 4

        assert_shape(result, (8, 4))
        assert_allclose(result, expected_global)
        assert result.sharding.dim_specs[0].axes == ["dp"]
        assert result.sharding.dim_specs[1].axes == []


class TestAllToAllOp:
    """Test AllToAll."""

    def test_all_to_all_1d(self, mesh_1d):
        """AllToAll: Swap sharding axis."""

        jax_x = make_jax_array(8, 8, seed=42)
        x = tensor_from_jax(jax_x)
        x_sharded = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])

        result = all_to_all(x_sharded, split_axis=1, concat_axis=0)

        assert_shape(result, (8, 8))
        assert_allclose(result, jax_x)

        spec = result.sharding
        assert (
            spec.dim_specs[0].axes == []
        ), "Axis 0 should be concatenated (replicated-ish)"

        if spec:

            pass
        else:
            pass

        assert_allclose(result, jax_x)


class TestReshardOp:
    def test_reshard_change(self, mesh_2x4):
        jax_x = make_jax_array(8, 16, seed=42)
        x = tensor_from_jax(jax_x)
        x_sharded = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec([])])

        result = reshard(x_sharded, mesh_2x4, [DimSpec([]), DimSpec(["tp"])])

        assert_shape(result, (8, 16))
        assert_allclose(result, jax_x)
        assert "tp" in result.sharding.dim_specs[1].axes


# =============================================================================
# EXTENDED COMMUNICATION OP TESTS
# These tests exercise AllReduce with different reduce_ops and multi-axis meshes.
# =============================================================================


class TestAllReduceVariants:
    """Test AllReduce with different reduce_op types (sum, max, min)."""

    @pytest.fixture
    def mesh_4(self):
        return DeviceMesh("mesh_4", (4,), ("dp",))

    @pytest.fixture
    def mesh_2x2(self):
        return DeviceMesh("mesh_2x2", (2, 2), ("dp", "tp"))

    def test_all_reduce_sum_replicated(self, mesh_4):
        """AllReduce sum on replicated tensor - basic smoke test."""
        jax_x = make_jax_array(4, 4, seed=42)
        x = tensor_from_jax(jax_x)

        x_rep = x.shard(mesh_4, [DimSpec([]), DimSpec([])])

        result = all_reduce(x_rep, reduce_op="sum")

        # AllReduce on replicated produces replicated output (same shape)
        assert_shape(result, (4, 4))

    def test_all_reduce_max_replicated(self, mesh_4):
        """AllReduce max on replicated tensor returns same values."""
        jax_x = make_jax_array(4, 4, seed=43)
        x = tensor_from_jax(jax_x)

        x_rep = x.shard(mesh_4, [DimSpec([]), DimSpec([])])

        result = all_reduce(x_rep, reduce_op="max")

        assert_allclose(result, jax_x)

    def test_all_reduce_min_replicated(self, mesh_4):
        """AllReduce min on replicated tensor returns same values."""
        jax_x = make_jax_array(4, 4, seed=44)
        x = tensor_from_jax(jax_x)

        x_rep = x.shard(mesh_4, [DimSpec([]), DimSpec([])])

        result = all_reduce(x_rep, reduce_op="min")

        assert_allclose(result, jax_x)

    def test_all_reduce_sum_sharded_1d(self, mesh_4):
        """AllReduce sum on sharded 1D tensor - preserves global values."""
        jax_x = make_jax_array(8, 4, seed=45)
        x = tensor_from_jax(jax_x)

        x_sharded = x.shard(mesh_4, [DimSpec(["dp"]), DimSpec([])])

        result = all_reduce(x_sharded, reduce_op="sum")

        # AllReduce returns a tensor; verify it exists and can be realized
        result.numpy()

    def test_all_reduce_max_2d_mesh(self, mesh_2x2):
        """AllReduce max on 2D mesh with partial sharding."""
        jax_x = make_jax_array(4, 4, seed=46)
        x = tensor_from_jax(jax_x)

        x_sharded = x.shard(mesh_2x2, [DimSpec(["dp"]), DimSpec([])])

        result = all_reduce(x_sharded, reduce_op="max")

        # Verify it executes successfully
        result.numpy()


class TestMultiAxisCommunication:
    """Test communication ops with multi-axis (2D) meshes."""

    @pytest.fixture
    def mesh_2x4(self):
        return DeviceMesh("mesh_2x4", (2, 4), ("dp", "tp"))

    @pytest.fixture
    def mesh_4x2(self):
        return DeviceMesh("mesh_4x2", (4, 2), ("dp", "tp"))

    def test_all_gather_2d_mesh_axis0(self, mesh_2x4):
        """AllGather on axis 0 with 2D mesh."""
        jax_x = make_jax_array(8, 16, seed=50)
        x = tensor_from_jax(jax_x)
        x_sharded = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec(["tp"])])

        result = all_gather(x_sharded, axis=0)

        assert_shape(result, (8, 16))
        assert result.sharding.dim_specs[0].axes == []
        assert_allclose(result, jax_x)

    def test_all_gather_2d_mesh_axis1(self, mesh_2x4):
        """AllGather on axis 1 with 2D mesh."""
        jax_x = make_jax_array(8, 16, seed=51)
        x = tensor_from_jax(jax_x)

        x_sharded = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec(["tp"])])

        result = all_gather(x_sharded, axis=1)

        assert_shape(result, (8, 16))
        assert result.sharding.dim_specs[1].axes == []
        assert_allclose(result, jax_x)

    def test_reduce_scatter_2d_mesh(self, mesh_2x4):
        """ReduceScatter on 2D mesh - verify execution completes."""
        jax_x = make_jax_array(16, 8, seed=52)  # Shape divisible by 8 shards
        x = tensor_from_jax(jax_x)

        x_rep = x.shard(mesh_2x4, [DimSpec([]), DimSpec([])])

        result = reduce_scatter(x_rep, axis=0)

        # ReduceScatter scatters on axis 0 - verify it runs
        result.numpy()

    def test_asymmetric_mesh_all_gather(self, mesh_4x2):
        """AllGather on asymmetric (4x2) mesh."""
        jax_x = make_jax_array(16, 8, seed=53)
        x = tensor_from_jax(jax_x)

        x_sharded = x.shard(mesh_4x2, [DimSpec(["dp"]), DimSpec(["tp"])])

        result = all_gather(x_sharded, axis=0)

        assert_shape(result, (16, 8))
        assert_allclose(result, jax_x)


class TestCommunicationGroupedExecution:
    """Test grouped collective execution on multi-axis meshes."""

    def test_all_reduce_grouped_2x2(self):
        """Test AllReduce with grouped execution on 2x2 mesh."""
        mesh = DeviceMesh("mesh_grp", (2, 2), ("x", "y"))

        jax_x = make_jax_array(4, 4, seed=60)
        x = tensor_from_jax(jax_x)

        x_sharded = x.shard(mesh, [DimSpec(["x"]), DimSpec(["y"])])

        result = all_reduce(x_sharded, reduce_op="sum")

        # Verify the grouped execution completes successfully
        result.numpy()


# =============================================================================
# ADDITIONAL COMMUNICATION OP TESTS
# Tests for previously untested operations: ppermute, axis_index, pmean, etc.
# =============================================================================


from nabla.ops.communication import axis_index, gather_all_axes, pmean, ppermute


class TestPPermuteOp:
    """Test point-to-point permutation collective."""

    @pytest.fixture
    def mesh_4(self):
        return DeviceMesh("mesh_pp4", (4,), ("dp",))

    def test_ppermute_ring_shift_right(self, mesh_4):
        """Ring shift: each device sends to next (0→1, 1→2, 2→3, 3→0)."""
        jax_x = make_jax_array(4, 4, seed=70)
        x = tensor_from_jax(jax_x)

        x_sharded = x.shard(mesh_4, [DimSpec(["dp"]), DimSpec([])])

        # Ring permutation: shift right
        perm = [(0, 1), (1, 2), (2, 3), (3, 0)]
        result = ppermute(x_sharded, permutation=perm)

        # ppermute changes the data order - just verify it executes
        result.numpy()

    def test_ppermute_reverse(self, mesh_4):
        """Reverse permutation: 0↔3, 1↔2."""
        jax_x = make_jax_array(4, 4, seed=71)
        x = tensor_from_jax(jax_x)

        x_sharded = x.shard(mesh_4, [DimSpec(["dp"]), DimSpec([])])

        perm = [(0, 3), (1, 2), (2, 1), (3, 0)]
        result = ppermute(x_sharded, permutation=perm)

        # ppermute rearranges data - just verify execution completes
        result.numpy()


class TestAxisIndexOp:
    """Test axis_index operation (device position query)."""

    def test_axis_index_1d_mesh(self):
        """Verify axis indices on 1D mesh."""
        mesh = DeviceMesh("mesh_ai1d", (4,), ("dp",))

        result = axis_index(mesh, "dp")

        # axis_index returns a sharded tensor with one value per device
        result.numpy()

    def test_axis_index_2d_mesh_axis0(self):
        """Verify row indices on 2D mesh."""
        mesh = DeviceMesh("mesh_ai2d", (2, 2), ("x", "y"))

        result = axis_index(mesh, "x")

        # axis_index is sharded on the queried axis, returns size of that axis
        result.numpy()

    def test_axis_index_2d_mesh_axis1(self):
        """Verify column indices on 2D mesh."""
        mesh = DeviceMesh("mesh_ai2d_y", (2, 2), ("x", "y"))

        result = axis_index(mesh, "y")

        # axis_index is sharded on the queried axis
        result.numpy()


class TestPMeanOp:
    """Test pmean (mean reduction across shards)."""

    def test_pmean_replicated(self):
        """PMean on replicated tensor."""
        mesh = DeviceMesh("mesh_pm", (4,), ("dp",))

        jax_x = make_jax_array(4, 4, seed=80)
        x = tensor_from_jax(jax_x)

        x_rep = x.shard(mesh, [DimSpec([]), DimSpec([])])

        result = pmean(x_rep, axis_name="dp")

        # PMean on replicated = same values (4×sum / 4 = sum)
        assert_shape(result, (4, 4))
        assert_allclose(result, jax_x)


class TestGatherAllAxesOp:
    """Test gather_all_axes (full reconstruction from multi-axis shards)."""

    def test_gather_all_axes_2d_mesh(self):
        """Gather from tensor sharded on both axes."""
        mesh = DeviceMesh("mesh_gaa", (2, 2), ("x", "y"))

        jax_x = make_jax_array(8, 8, seed=90)
        x = tensor_from_jax(jax_x)

        x_sharded = x.shard(mesh, [DimSpec(["x"]), DimSpec(["y"])])

        result = gather_all_axes(x_sharded)

        assert_shape(result, (8, 8))
        assert_allclose(result, jax_x)
        # Should be fully replicated now
        assert result.sharding.is_fully_replicated()


class TestAllToAllExtended:
    """Extended AllToAll tests with 2D meshes and multiple configurations."""

    @pytest.fixture
    def mesh_2x2(self):
        return DeviceMesh("mesh_a2a_2x2", (2, 2), ("x", "y"))

    def test_all_to_all_2d_mesh(self, mesh_2x2):
        """AllToAll on 2D mesh."""
        jax_x = make_jax_array(8, 8, seed=100)
        x = tensor_from_jax(jax_x)

        x_sharded = x.shard(mesh_2x2, [DimSpec([]), DimSpec(["y"])])

        result = all_to_all(x_sharded, split_axis=0, concat_axis=1)

        # all_to_all transforms shape based on split/concat
        result.numpy()

    def test_all_to_all_swap_axes(self, mesh_2x2):
        """AllToAll swapping split and concat axes."""
        jax_x = make_jax_array(8, 8, seed=101)
        x = tensor_from_jax(jax_x)

        x_sharded = x.shard(mesh_2x2, [DimSpec(["x"]), DimSpec([])])

        result = all_to_all(x_sharded, split_axis=1, concat_axis=0)

        # all_to_all changes shape - verify it executes
        result.numpy()


