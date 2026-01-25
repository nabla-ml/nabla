# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Tests for physical operations (foundation layer) ported from tests/unit."""

import jax
import jax.numpy as jnp
import pytest

from nabla import (
    Tensor,
    broadcast_to_physical,
    mean_physical,
    reduce_sum_physical,
    squeeze_physical,
    unsqueeze_physical,
)
from nabla.core.sharding.spec import DeviceMesh

from .common import (
    DeviceMesh,
    assert_is_sharded,
    assert_physical_shape,
    shard_on_axis,
)


from .common import (
    assert_allclose,
    assert_shape,
    make_jax_array,
    make_positive_jax_array,
    tensor_from_jax,
)


class TestReduceSumPhysical:
    """Test reduce_sum_physical on various shapes and axes."""

    @pytest.mark.parametrize(
        "shape,axis",
        [
            ((8,), 0),
            ((4, 8), 0),
            ((4, 8), 1),
            ((4, 8), -1),
            ((2, 4, 8), 0),
            ((2, 4, 8), 1),
            ((2, 4, 8), 2),
        ],
    )
    def test_reduce_axis(self, shape, axis):
        """Test reduction along specific axis."""
        jax_x = make_jax_array(*shape, seed=42)
        x = tensor_from_jax(jax_x)

        result = reduce_sum_physical(x, axis=axis)
        expected = jnp.sum(jax_x, axis=axis)

        dummy_expected = jnp.sum(jax_x, axis=axis)
        assert_shape(result, dummy_expected.shape)
        assert_allclose(result, expected)

    def test_sharded_reduce_non_sharded_axis(self, mesh_1d):
        """Reduce on axis that is NOT sharded - should work without gather."""
        jax_x = make_jax_array(8, 4, seed=42)
        x = tensor_from_jax(jax_x)
        x = shard_on_axis(x, mesh_1d, axis=0)

        result = reduce_sum_physical(x, axis=1)
        expected = jnp.sum(jax_x, axis=1)

        assert_shape(result, expected.shape)
        assert_is_sharded(result, True)
        assert_allclose(result, expected)

    def test_sharded_reduce_sharded_axis(self, mesh_1d):
        """Reduce on axis that IS sharded - requires all_reduce."""
        jax_x = make_jax_array(8, 4, seed=42)
        x = tensor_from_jax(jax_x)
        x = shard_on_axis(x, mesh_1d, axis=0)

        result = reduce_sum_physical(x, axis=0)
        expected = jnp.sum(jax_x, axis=0)

        assert_shape(result, expected.shape)
        assert_allclose(result, expected)


class TestMeanPhysical:
    """Test mean_physical on various shapes and axes."""

    @pytest.mark.parametrize(
        "shape,axis",
        [
            ((8,), 0),
            ((4, 8), 0),
            ((4, 8), 1),
            ((2, 4, 8), 1),
        ],
    )
    def test_mean_axis(self, shape, axis):
        jax_x = make_jax_array(*shape, seed=42)
        x = tensor_from_jax(jax_x)

        result = mean_physical(x, axis=axis)
        expected = jnp.mean(jax_x, axis=axis)

        assert_shape(result, expected.shape)
        assert_allclose(result, expected)

    def test_sharded_mean_non_sharded_axis(self, mesh_1d):
        """Mean on non-sharded axis."""
        jax_x = make_jax_array(8, 4, seed=42)
        x = tensor_from_jax(jax_x)
        x = shard_on_axis(x, mesh_1d, axis=0)

        result = mean_physical(x, axis=1)
        expected = jnp.mean(jax_x, axis=1)

        assert_shape(result, expected.shape)
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


class TestSqueezePhysical:
    """Test squeeze_physical on various shapes."""

    @pytest.mark.parametrize(
        "shape,axis,expected_shape",
        [
            ((1, 8), 0, (8,)),
            ((4, 1), 1, (4,)),
            ((4, 1, 8), 1, (4, 8)),
            ((1, 4, 8), 0, (4, 8)),
        ],
    )
    def test_squeeze_axis(self, shape, axis, expected_shape):
        jax_x = make_jax_array(*shape, seed=42)
        x = tensor_from_jax(jax_x)

        result = squeeze_physical(x, axis=axis)
        expected = jnp.squeeze(jax_x, axis=axis)

        assert_shape(result, expected_shape)
        assert_allclose(result, expected)

    def test_sharded_squeeze_non_sharded_dim(self, mesh_1d):
        """Squeeze a non-sharded dimension."""

        jax_x = make_jax_array(4, 1, 8, seed=42)
        x = tensor_from_jax(jax_x)
        x = shard_on_axis(x, mesh_1d, axis=0)

        result = squeeze_physical(x, axis=1)
        expected = jnp.squeeze(jax_x, axis=1)

        assert_shape(result, (4, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


class TestUnsqueezePhysical:
    """Test unsqueeze_physical on various shapes."""

    @pytest.mark.parametrize(
        "shape,axis,expected_shape",
        [
            ((8,), 0, (1, 8)),
            ((8,), 1, (8, 1)),
            ((4, 8), 0, (1, 4, 8)),
            ((4, 8), 1, (4, 1, 8)),
        ],
    )
    def test_unsqueeze_axis(self, shape, axis, expected_shape):
        jax_x = make_jax_array(*shape, seed=42)
        x = tensor_from_jax(jax_x)

        result = unsqueeze_physical(x, axis=axis)
        expected = jnp.expand_dims(jax_x, axis=axis)

        assert_shape(result, expected_shape)
        assert_allclose(result, expected)

    def test_sharded_unsqueeze_before_sharded(self, mesh_1d):
        """Unsqueeze before sharded dimension - sharded dim index shifts."""

        jax_x = make_jax_array(4, 8, seed=42)
        x = tensor_from_jax(jax_x)
        x = shard_on_axis(x, mesh_1d, axis=0)

        result = unsqueeze_physical(x, axis=0)
        expected = jnp.expand_dims(jax_x, axis=0)

        assert_shape(result, (1, 4, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


class TestBroadcastToPhysical:
    """Test broadcast_to_physical."""

    @pytest.mark.parametrize(
        "shape,target_shape",
        [
            ((1,), (4,)),
            ((8,), (4, 8)),
            ((1, 8), (4, 8)),
            ((4, 1), (4, 8)),
        ],
    )
    def test_broadcast(self, shape, target_shape):
        jax_x = make_jax_array(*shape, seed=42)
        x = tensor_from_jax(jax_x)

        result = broadcast_to_physical(x, target_shape)
        expected = jnp.broadcast_to(jax_x, target_shape)

        assert_physical_shape(result, target_shape)
        assert_allclose(result, expected)

    def test_broadcast_preserves_sharding(self, mesh_1d):
        """Broadcast should preserve sharding on non-broadcast dims."""

        jax_x = make_jax_array(4, 1, seed=42)
        x = tensor_from_jax(jax_x)
        x = shard_on_axis(x, mesh_1d, axis=0)

        result = broadcast_to_physical(x, (4, 8))
        expected = jnp.broadcast_to(jax_x, (4, 8))

        assert_shape(result, (4, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)
