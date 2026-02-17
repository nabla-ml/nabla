# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""
Tests for creation ops (zeros, ones, arange, hann_window, triu, tril) and
comprehensive gather/scatter sharding/vmap tests.

Forward/VJP/JVP/vmap for gather, scatter, and where are covered by
test_unified.py via unified_registry.py.
"""

from functools import partial

import jax.numpy as jnp
import pytest

import nabla as nb
from nabla import P

from .common import (
    OpConfig,
    Operation,
    run_test_with_consistency_check,
    standard_get_args,
)


def get_creation_args(config: OpConfig):
    return ((), config.params), ((), config.params)


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
OPS["hann_window"] = Operation(
    "hann_window",
    "CREATION",
    nb.hann_window,
    lambda window_length, **kwargs: jnp.hanning(window_length),
    [OpConfig("Hann", params={"window_length": 8, "periodic": False})],
    get_creation_args,
)
OPS["triu"] = Operation(
    "triu",
    "CREATION",
    nb.triu,
    jnp.triu,
    [OpConfig("Triu", ranks=(2,), params={"k": 0})],
    standard_get_args,
)
OPS["tril"] = Operation(
    "tril",
    "CREATION",
    nb.tril,
    jnp.tril,
    [OpConfig("Tril", ranks=(2,), params={"k": 0})],
    standard_get_args,
)


@pytest.mark.parametrize("op_name", OPS.keys())
def test_creation_ops(op_name):
    """Test creation ops (not covered by unified test matrix)."""
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

from .common import (
    assert_allclose,
    assert_is_sharded,
    assert_shape,
    make_jax_array,
    replicated,
    shard_on_axis,
    tensor_from_jax,
)


class TestGatherSharding:
    """Test gather operation with sharding."""

    def test_gather_replicated(self, mesh_1d):
        """Gather from replicated tensor."""
        jax_data = make_jax_array(8, 4, seed=42)
        jax_indices = jnp.array([1, 5], dtype=jnp.int32)

        data = tensor_from_jax(jax_data)
        data_repl = replicated(data, mesh_1d)
        indices = tensor_from_jax(jax_indices)

        result = nb.gather(data_repl, indices, axis=0)
        expected = jax_data[jax_indices, :]

        assert_shape(result, (2, 4))
        assert_allclose(result, expected)

    def test_gather_sharded_non_gather_axis(self, mesh_1d):
        """Gather from tensor sharded on non-gather axis."""
        jax_data = make_jax_array(8, 4, seed=42)
        jax_indices = jnp.array([0, 3, 7], dtype=jnp.int32)

        data = tensor_from_jax(jax_data)
        data_sharded = shard_on_axis(data, mesh_1d, axis=1)
        indices = tensor_from_jax(jax_indices)

        result = nb.gather(data_sharded, indices, axis=0)
        expected = jax_data[jax_indices, :]

        assert_shape(result, (3, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)

    def test_gather_3d_middle_axis(self, mesh_1d):
        """Gather from 3D tensor along middle axis."""
        jax_data = make_jax_array(4, 8, 6, seed=42)
        jax_indices = jnp.array([0, 3, 7], dtype=jnp.int32)

        data = tensor_from_jax(jax_data)
        indices = tensor_from_jax(jax_indices)

        result = nb.gather(data, indices, axis=1)
        expected = jax_data[:, jax_indices, :]

        assert_shape(result, (4, 3, 6))
        assert_allclose(result, expected)

    def test_gather_negative_axis(self, mesh_1d):
        """Gather with negative axis."""
        jax_data = make_jax_array(4, 8, seed=42)
        jax_indices = jnp.array([1, 3, 5], dtype=jnp.int32)

        data = tensor_from_jax(jax_data)
        indices = tensor_from_jax(jax_indices)

        result = nb.gather(data, indices, axis=-1)
        expected = jax_data[:, jax_indices]

        assert_shape(result, (4, 3))
        assert_allclose(result, expected)


class TestGatherVmap:
    """Test gather with vmap (automatic batching)."""

    @pytest.mark.xfail(reason="vmap(gather) returns wrong shape — pre-existing issue")
    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_vmap_gather_axis0(self, batch_size):
        """Vmap over gather with batch in data."""
        jax_data = make_jax_array(batch_size, 8, 4, seed=42)
        jax_indices = jnp.array([0, 3, 7], dtype=jnp.int32)

        data = tensor_from_jax(jax_data)
        indices = tensor_from_jax(jax_indices)

        def fn(x):
            return nb.gather(x, indices, axis=0)

        result = nb.vmap(fn)(data)
        expected = jax_data[:, jax_indices, :]

        assert_shape(result, (batch_size, 3, 4))
        assert_allclose(result, expected)

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_vmap_gather_batched_indices(self, batch_size):
        """Vmap with both data and indices batched."""
        jax_data = make_jax_array(batch_size, 8, seed=42)
        jax_indices = jnp.array(
            [[0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 2, 3]][:batch_size], dtype=jnp.int32
        )

        data = tensor_from_jax(jax_data)
        indices = tensor_from_jax(jax_indices)

        def fn(x, idx):
            return nb.gather(x, idx, axis=0)

        result = nb.vmap(fn)(data, indices)

        expected = jnp.array([jax_data[i, jax_indices[i]] for i in range(batch_size)])

        assert_shape(result, (batch_size, 3))
        assert_allclose(result, expected)


class TestScatterSharding:
    """Test scatter operation with sharding."""

    def test_scatter_replicated(self, mesh_1d):
        """Scatter into replicated tensor."""
        jax_data = make_jax_array(8, 4, seed=42)
        jax_indices = jnp.array([2, 5], dtype=jnp.int32)
        jax_updates = make_jax_array(2, 4, seed=43)

        data = tensor_from_jax(jax_data)
        data_repl = replicated(data, mesh_1d)
        indices = tensor_from_jax(jax_indices)
        updates = tensor_from_jax(jax_updates)

        result = nb.scatter(data_repl, indices, updates, axis=0)

        expected = jax_data.at[jax_indices].set(jax_updates)

        assert_shape(result, (8, 4))
        assert_allclose(result, expected)

    def test_scatter_2d_axis1(self, mesh_1d):
        """Scatter into 2D tensor along axis 1."""
        jax_data = make_jax_array(4, 8, seed=42)
        jax_indices = jnp.array([1, 5, 7], dtype=jnp.int32)
        jax_updates = make_jax_array(4, 3, seed=43)

        data = tensor_from_jax(jax_data)
        indices = tensor_from_jax(jax_indices)
        updates = tensor_from_jax(jax_updates)

        result = nb.scatter(data, indices, updates, axis=1)

        expected = jax_data.at[:, jax_indices].set(jax_updates)

        assert_shape(result, (4, 8))
        assert_allclose(result, expected)

    def test_scatter_sharded_non_scatter_axis(self, mesh_1d):
        """Scatter with tensor sharded on non-scatter axis."""
        jax_data = make_jax_array(8, 4, seed=42)
        jax_indices = jnp.array([1, 3], dtype=jnp.int32)
        jax_updates = make_jax_array(2, 4, seed=43)

        data = tensor_from_jax(jax_data)
        data_sharded = shard_on_axis(data, mesh_1d, axis=1)
        indices = tensor_from_jax(jax_indices)
        updates = tensor_from_jax(jax_updates)

        result = nb.scatter(data_sharded, indices, updates, axis=0)

        expected = jax_data.at[jax_indices].set(jax_updates)

        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


class TestGatherScatterRoundTrip:
    """Test gather followed by scatter produces expected results."""

    def test_gather_scatter_identity(self):
        """Gather then scatter back to same positions."""
        jax_data = make_jax_array(8, 4, seed=42)
        jax_indices = jnp.array([1, 3, 5], dtype=jnp.int32)

        data = tensor_from_jax(jax_data)
        indices = tensor_from_jax(jax_indices)

        gathered = nb.gather(data, indices, axis=0)

        buffer = tensor_from_jax(jnp.zeros((8, 4), dtype=jnp.float32))
        result = nb.scatter(buffer, indices, gathered, axis=0)

        expected = jnp.zeros((8, 4), dtype="float32")
        expected = expected.at[jax_indices].set(jax_data[jax_indices, :])

        assert_shape(result, (8, 4))
        assert_allclose(result, expected)

    def test_gather_scatter_with_sharding(self, mesh_1d):
        """Gather-scatter round trip with sharding."""
        jax_data = make_jax_array(8, 4, seed=42)
        jax_indices = jnp.array([0, 2, 6], dtype=jnp.int32)

        data = tensor_from_jax(jax_data)
        data_sharded = shard_on_axis(data, mesh_1d, axis=1)
        indices = tensor_from_jax(jax_indices)

        gathered = nb.gather(data_sharded, indices, axis=0)

        buffer = nb.zeros((8, 4))
        buffer_sharded = shard_on_axis(buffer, mesh_1d, axis=1)
        result = nb.scatter(buffer_sharded, indices, gathered, axis=0)

        expected = jnp.zeros((8, 4), dtype=jnp.float32)
        expected = expected.at[jax_indices].set(jax_data[jax_indices, :])

        assert_shape(result, (8, 4))
        assert_allclose(result, expected)


class TestRandomSharding:
    """Test that random ops produce independent values on different shards."""

    def test_uniform_sharding_independence(self, mesh_1d):
        """Uniform on 1D mesh should have different values per shard."""
        shape = (8,)
        # Create a sharded uniform tensor
        res = nb.uniform(shape, low=0.0, high=1.0)
        res_sharded = res.shard(mesh_1d, P("dp"))

        # Realize it
        val = res_sharded.numpy()

        # Split into shards (manually for 1D mesh)
        mid = len(val) // 2
        shard0 = val[:mid]
        shard1 = val[mid:]

        # They should NOT be equal (very low probability they are exactly same)
        assert not jnp.allclose(shard0, shard1)

        # But they should be in range
        assert jnp.all(val >= 0.0) and jnp.all(val <= 1.0)
