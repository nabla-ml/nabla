# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest

import nabla as nb

from .common import (
    OpConfig,
    Operation,
    get_test_data_for_shapes,
    run_test_with_consistency_check,
    standard_get_args,
)


def get_creation_args(config: OpConfig):
    return ((), config.params), ((), config.params)


def get_indexing_args(config: OpConfig):
    shapes = config.primal_shapes
    if not shapes:
        return standard_get_args(config)

    x_nb, x_jax = get_test_data_for_shapes((shapes[0],), config)
    x_nb, x_jax = x_nb[0], x_jax[0]

    limit = shapes[0][config.params.get("axis", 0)]
    idx_shape = shapes[1]
    idx_np = np.random.randint(0, limit, size=idx_shape)

    idx_nb = nb.constant(idx_np, dtype=nb.DType.int64)
    idx_jax = jnp.array(idx_np, dtype="int64")

    inputs_nb = [x_nb, idx_nb]
    inputs_jax = [x_jax, idx_jax]

    if len(shapes) > 2:
        u_nb, u_jax = get_test_data_for_shapes((shapes[2],), config)
        inputs_nb.append(u_nb[0])
        inputs_jax.append(u_jax[0])

    return (tuple(inputs_nb), config.params), (tuple(inputs_jax), config.params)


def get_where_args(config: OpConfig):
    shapes = config.primal_shapes
    c_np = np.random.choice([True, False], size=shapes[0])
    c_nb = nb.constant(c_np, dtype=nb.DType.bool)
    c_jax = jnp.array(c_np)

    data_nb, data_jax = get_test_data_for_shapes(shapes[1:], config)

    return ((c_nb, *data_nb), config.params), ((c_jax, *data_jax), config.params)


OPS = {}

OPS["zeros"] = Operation(
    "zeros",
    "CREATION",
    nb.zeros,
    jnp.zeros,
    [OpConfig("R2", params={"shape": (2, 3)})],
    get_creation_args,
)
OPS["ones"] = Operation(
    "ones",
    "CREATION",
    nb.ones,
    jnp.ones,
    [OpConfig("R2", params={"shape": (2, 3)})],
    get_creation_args,
)
OPS["arange"] = Operation(
    "arange",
    "CREATION",
    nb.arange,
    jnp.arange,
    [OpConfig("Range", params={"start": 0, "stop": 5})],
    get_creation_args,
)

OPS["gather"] = Operation(
    "gather",
    "INDEX",
    nb.gather,
    lambda x, i, axis: jnp.take(x, i, axis=axis),
    [OpConfig("Simp", primal_shapes=((4, 4), (2,)), params={"axis": 0})],
    get_indexing_args,
)
OPS["scatter"] = Operation(
    "scatter",
    "INDEX",
    nb.scatter,
    lambda x, i, u, axis: x.at[i].set(u) if axis == 0 else x,
    [OpConfig("Simp", primal_shapes=((4,), (1,), (1,)), params={"axis": 0})],
    get_indexing_args,
)

OPS["where"] = Operation(
    "where",
    "CONTROL",
    nb.where,
    jnp.where,
    [OpConfig("Simp", primal_shapes=((2, 2), (2, 2), (2, 2)))],
    get_where_args,
)


@pytest.mark.parametrize("op_name", OPS.keys())
def test_misc_base(op_name):
    op = OPS[op_name]
    config = op.configs[0]
    (a_nb, k_nb), (a_jax, k_jax) = op.get_args(config)

    nb_fn = partial(op.nabla_fn, **k_nb)
    jax_fn = partial(op.jax_fn, **k_jax)

    run_test_with_consistency_check(
        f"{op_name}_Base", lambda: nb_fn(*a_nb), lambda: jax_fn(*a_jax)
    )


# =============================================================================
# COMPREHENSIVE GATHER/SCATTER TESTS WITH SHARDING AND VMAP
# =============================================================================

from tests.conftest import (
    assert_allclose,
    assert_is_sharded,
    assert_shape,
    make_array,
    replicated,
    shard_on_axis,
    tensor_from_numpy,
)


class TestGatherSharding:
    """Test gather operation with sharding."""

    def test_gather_replicated(self, mesh_1d):
        """Gather from replicated tensor."""
        np_data = make_array(8, 4, seed=42)
        np_indices = np.array([1, 5], dtype=np.int32)

        data = tensor_from_numpy(np_data)
        data_repl = replicated(data, mesh_1d)
        indices = tensor_from_numpy(np_indices)

        result = nb.gather(data_repl, indices, axis=0)
        expected = np_data[np_indices, :]

        assert_shape(result, (2, 4))
        assert_allclose(result, expected)

    def test_gather_sharded_non_gather_axis(self, mesh_1d):
        """Gather from tensor sharded on non-gather axis."""
        np_data = make_array(8, 4, seed=42)
        np_indices = np.array([0, 3, 7], dtype=np.int32)

        data = tensor_from_numpy(np_data)
        data_sharded = shard_on_axis(data, mesh_1d, axis=1)
        indices = tensor_from_numpy(np_indices)

        result = nb.gather(data_sharded, indices, axis=0)
        expected = np_data[np_indices, :]

        assert_shape(result, (3, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)

    def test_gather_3d_middle_axis(self, mesh_1d):
        """Gather from 3D tensor along middle axis."""
        np_data = make_array(4, 8, 6, seed=42)
        np_indices = np.array([0, 3, 7], dtype=np.int32)

        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)

        result = nb.gather(data, indices, axis=1)
        expected = np_data[:, np_indices, :]

        assert_shape(result, (4, 3, 6))
        assert_allclose(result, expected)

    def test_gather_negative_axis(self, mesh_1d):
        """Gather with negative axis."""
        np_data = make_array(4, 8, seed=42)
        np_indices = np.array([1, 3, 5], dtype=np.int32)

        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)

        result = nb.gather(data, indices, axis=-1)
        expected = np_data[:, np_indices]

        assert_shape(result, (4, 3))
        assert_allclose(result, expected)


class TestGatherVmap:
    """Test gather with vmap (automatic batching)."""

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_vmap_gather_axis0(self, batch_size):
        """Vmap over gather with batch in data."""
        np_data = make_array(batch_size, 8, 4, seed=42)
        np_indices = np.array([0, 3, 7], dtype=np.int32)

        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)

        def fn(x):
            return nb.gather(x, indices, axis=0)

        result = nb.vmap(fn)(data)
        expected = np_data[:, np_indices, :]

        assert_shape(result, (batch_size, 3, 4))
        assert_allclose(result, expected)

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_vmap_gather_batched_indices(self, batch_size):
        """Vmap with both data and indices batched."""
        np_data = make_array(batch_size, 8, seed=42)
        np_indices = np.array(
            [[0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 2, 3]][:batch_size], dtype=np.int32
        )

        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)

        def fn(x, idx):
            return nb.gather(x, idx, axis=0)

        result = nb.vmap(fn)(data, indices)

        expected = np.array([np_data[i, np_indices[i]] for i in range(batch_size)])

        assert_shape(result, (batch_size, 3))
        assert_allclose(result, expected)


class TestScatterSharding:
    """Test scatter operation with sharding."""

    def test_scatter_replicated(self, mesh_1d):
        """Scatter into replicated tensor."""
        np_data = make_array(8, 4, seed=42)
        np_indices = np.array([2, 5], dtype=np.int32)
        np_updates = make_array(2, 4, seed=43)

        data = tensor_from_numpy(np_data)
        data_repl = replicated(data, mesh_1d)
        indices = tensor_from_numpy(np_indices)
        updates = tensor_from_numpy(np_updates)

        result = nb.scatter(data_repl, indices, updates, axis=0)

        expected = np_data.copy()
        expected[np_indices, :] = np_updates

        assert_shape(result, (8, 4))
        assert_allclose(result, expected)

    def test_scatter_2d_axis1(self, mesh_1d):
        """Scatter into 2D tensor along axis 1."""
        np_data = make_array(4, 8, seed=42)
        np_indices = np.array([1, 5, 7], dtype=np.int32)
        np_updates = make_array(4, 3, seed=43)

        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)
        updates = tensor_from_numpy(np_updates)

        result = nb.scatter(data, indices, updates, axis=1)

        expected = np_data.copy()
        expected[:, np_indices] = np_updates

        assert_shape(result, (4, 8))
        assert_allclose(result, expected)

    def test_scatter_sharded_non_scatter_axis(self, mesh_1d):
        """Scatter with tensor sharded on non-scatter axis."""
        np_data = make_array(8, 4, seed=42)
        np_indices = np.array([1, 3], dtype=np.int32)
        np_updates = make_array(2, 4, seed=43)

        data = tensor_from_numpy(np_data)
        data_sharded = shard_on_axis(data, mesh_1d, axis=1)
        indices = tensor_from_numpy(np_indices)
        updates = tensor_from_numpy(np_updates)

        result = nb.scatter(data_sharded, indices, updates, axis=0)

        expected = np_data.copy()
        expected[np_indices, :] = np_updates

        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


class TestGatherScatterRoundTrip:
    """Test gather followed by scatter produces expected results."""

    def test_gather_scatter_identity(self):
        """Gather then scatter back to same positions."""
        np_data = make_array(8, 4, seed=42)
        np_indices = np.array([1, 3, 5], dtype=np.int32)

        data = tensor_from_numpy(np_data)
        indices = tensor_from_numpy(np_indices)

        gathered = nb.gather(data, indices, axis=0)

        buffer = tensor_from_numpy(np.zeros((8, 4), dtype=np.float32))
        result = nb.scatter(buffer, indices, gathered, axis=0)

        expected = np.zeros((8, 4), dtype=np.float32)
        expected[np_indices, :] = np_data[np_indices, :]

        assert_shape(result, (8, 4))
        assert_allclose(result, expected)

    def test_gather_scatter_with_sharding(self, mesh_1d):
        """Gather-scatter round trip with sharding."""
        np_data = make_array(8, 4, seed=42)
        np_indices = np.array([0, 2, 6], dtype=np.int32)

        data = tensor_from_numpy(np_data)
        data_sharded = shard_on_axis(data, mesh_1d, axis=1)
        indices = tensor_from_numpy(np_indices)

        gathered = nb.gather(data_sharded, indices, axis=0)

        buffer = nb.zeros((8, 4))
        buffer_sharded = shard_on_axis(buffer, mesh_1d, axis=1)
        result = nb.scatter(buffer_sharded, indices, gathered, axis=0)

        expected = np.zeros((8, 4), dtype=np.float32)
        expected[np_indices, :] = np_data[np_indices, :]

        assert_shape(result, (8, 4))
        assert_allclose(result, expected)

