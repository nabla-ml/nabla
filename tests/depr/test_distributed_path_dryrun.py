# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
from max import graph as g
from max.graph import ops

from nabla import DeviceMesh, Tensor
from nabla.core.sharding.spec import DimSpec, ShardingSpec
from nabla.ops.communication import all_gather, all_reduce, reduce_scatter, shard


@pytest.fixture
def distributed_mesh():
    mesh = DeviceMesh("dist_mesh", (4,), ("dp",))
    # We need unique device_refs to trigger is_distributed=True
    # We use actual DeviceRef objects if possible, otherwise strings might fail the set check
    # in the DeviceMesh property depending on how they are compared.
    try:
        from max.graph import DeviceRef  # noqa: F401

        # Create 4 distinct DeviceRef-like objects or actual ones
        # For simulation we can use multiple "Accelerator"-like slots if it's supported
        mesh.device_refs = [f"gpu:{i}" for i in range(4)]
    except ImportError:
        mesh.device_refs = [f"gpu:{i}" for i in range(4)]

    # Verify is_distributed is True
    assert mesh.is_distributed
    return mesh


def test_distributed_all_gather_construction(distributed_mesh):
    """Verify that AllGatherOp constructs its graph logic on a distributed mesh."""
    from max.dtype import DType

    with g.Graph(name="test"):
        x_val = ops.constant(np.zeros((8, 4), dtype=np.float32), DType.float32)

        # Manually create a sharded tensor (symbolic)
        spec = ShardingSpec(distributed_mesh, [DimSpec(["dp"]), DimSpec([])])
        x = Tensor(x_val, sharding=spec)

        # This should trigger the native AllGather path
        result = all_gather(x, axis=0)
        # If we reached here without error, the signatures and buffer logic are sound
        assert result.is_sharded
        assert not result.sharding.dim_specs[0].axes


def test_distributed_all_reduce_construction(distributed_mesh):
    """Verify AllReduceOp construction on distributed mesh."""
    from max.dtype import DType

    with g.Graph(name="test"):
        x_val = ops.constant(np.zeros((8, 4), dtype=np.float32), DType.float32)
        spec = ShardingSpec(distributed_mesh, [DimSpec(["dp"]), DimSpec([])])
        x = Tensor(x_val, sharding=spec)

        result = all_reduce(x, axes=["dp"])
        assert result.is_sharded
        # AllReduce to axes makes it replicated on those axes
        assert not result.sharding.dim_specs[0].axes


def test_distributed_reduce_scatter_construction(distributed_mesh):
    """Verify ReduceScatterOp construction on distributed mesh."""
    from max.dtype import DType

    with g.Graph(name="test"):
        x_val = ops.constant(np.zeros((8, 4), dtype=np.float32), DType.float32)
        spec = ShardingSpec(
            distributed_mesh, [DimSpec([]), DimSpec([])], replicated_axes={"dp"}
        )
        x = Tensor(x_val, sharding=spec)

        # Scatter along dimension 0
        result = reduce_scatter(x, axis=0, mesh_axes=["dp"])
        assert result.is_sharded
        assert result.sharding.dim_specs[0].axes == ["dp"]


def test_distributed_shard_replication_construction(distributed_mesh):
    """Verify ShardOp broadcast construction."""
    from max.dtype import DType

    with g.Graph(name="test"):
        x_val = ops.constant(np.zeros((8, 4), dtype=np.float32), DType.float32)
        # This should trigger distributed_broadcast because spec is fully replicated
        result = shard(x_val, distributed_mesh, [DimSpec([]), DimSpec([])])
        assert len(result.values) == 4


def test_distributed_shard_stack_construction(distributed_mesh):
    """Verify ShardOp shard_and_stack construction."""
    from max.dtype import DType

    with g.Graph(name="test"):
        x_val = ops.constant(np.zeros((8, 4), dtype=np.float32), DType.float32)
        # This should trigger shard_and_stack because it's 1D sharding across mesh
        result = shard(x_val, distributed_mesh, [DimSpec(["dp"]), DimSpec([])])
        assert len(result.values) == 4
        # Local shape should be [2, 4] (8 / 4)
        assert result.values[0].type.shape == (2, 4)
