# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import numpy as np
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


def make_array(*shape: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def tensor_from_numpy(arr: np.ndarray) -> nb.Tensor:
    return nb.Tensor.from_dlpack(arr)


def assert_allclose(result: nb.Tensor, expected: np.ndarray, rtol: float = 1e-5):
    np.testing.assert_allclose(result.numpy(), expected, rtol=rtol)


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
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        result = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])

        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, np_x)
        assert result.sharding.dim_specs[0].axes == ["dp"]

    def test_shard_2d_asymmetric(self, mesh_2x4):
        """Shard on 2D asymmetric mesh."""
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        result = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec(["tp"])])

        assert_shape(result, (8, 16))
        assert_is_sharded(result, True)
        assert_allclose(result, np_x)
        assert "dp" in result.sharding.dim_specs[0].axes
        assert "tp" in result.sharding.dim_specs[1].axes


class TestAllGatherOp:
    """Test AllGather: gather shards to replicated."""

    def test_all_gather_1d(self, mesh_1d):
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        x_sharded = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])

        result = all_gather(x_sharded, axis=0)

        assert_shape(result, (8, 4))
        assert_allclose(result, np_x)

        spec = result.sharding
        assert spec.dim_specs[0].axes == []


class TestReduceScatterOp:
    """Test ReduceScatter: reduce and then scatter."""

    def test_reduce_scatter_1d(self, mesh_1d):
        """ReduceScatter on 1D mesh: Replicated -> Reduce(Sum) -> Scatter(axis=0)."""

        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)

        x_rep = x.shard(mesh_1d, [DimSpec([]), DimSpec([])])

        result = reduce_scatter(x_rep, axis=0)

        expected_global = np_x * 4

        assert_shape(result, (8, 4))
        assert_allclose(result, expected_global)
        assert result.sharding.dim_specs[0].axes == ["dp"]
        assert result.sharding.dim_specs[1].axes == []


class TestAllToAllOp:
    """Test AllToAll."""

    def test_all_to_all_1d(self, mesh_1d):
        """AllToAll: Swap sharding axis."""

        np_x = make_array(8, 8, seed=42)
        x = tensor_from_numpy(np_x)
        x_sharded = x.shard(mesh_1d, [DimSpec(["dp"]), DimSpec([])])

        result = all_to_all(x_sharded, split_axis=1, concat_axis=0)

        assert_shape(result, (8, 8))
        assert_allclose(result, np_x)

        spec = result.sharding
        assert (
            spec.dim_specs[0].axes == []
        ), "Axis 0 should be concatenated (replicated-ish)"

        if spec:

            pass
        else:
            pass

        assert_allclose(result, np_x)


class TestReshardOp:
    def test_reshard_change(self, mesh_2x4):
        np_x = make_array(8, 16, seed=42)
        x = tensor_from_numpy(np_x)
        x_sharded = x.shard(mesh_2x4, [DimSpec(["dp"]), DimSpec([])])

        result = reshard(x_sharded, mesh_2x4, [DimSpec([]), DimSpec(["tp"])])

        assert_shape(result, (8, 16))
        assert_allclose(result, np_x)
        assert "tp" in result.sharding.dim_specs[1].axes
