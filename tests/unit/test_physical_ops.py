# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Tests for physical operations - the foundation layer.

Physical ops work on full physical shapes (no batch_dims awareness).
These must pass before testing logical ops or vmap transforms.

Physical ops tested:
- reduce_sum_physical: Sum reduction on physical axis
- mean_physical: Mean reduction on physical axis
- squeeze_physical: Remove dim of size 1 at physical axis
- unsqueeze_physical: Add dim of size 1 at physical axis
- broadcast_to_physical: Broadcast to physical target shape

All physical ops should also work correctly with sharding propagation.
"""

import numpy as np
import pytest

from nabla import (
    broadcast_to_physical,
    mean_physical,
    reduce_sum_physical,
    squeeze_physical,
    unsqueeze_physical,
)
from .conftest import (
    assert_allclose,
    assert_is_sharded,
    assert_physical_shape,
    assert_shape,
    make_array,
    shard_on_axis,
    tensor_from_numpy,
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
            ((2, 4, 8), -2),
        ],
    )
    def test_reduce_axis(self, shape, axis):
        """Test reduction along specific axis."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = reduce_sum_physical(x, axis=axis)
        expected = np.sum(np_x, axis=axis)

        assert_shape(result, expected.shape)
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "shape,axis",
        [
            ((4, 8), 0),
            ((4, 8), 1),
            ((2, 4, 8), 1),
        ],
    )
    def test_keepdims(self, shape, axis):
        """Test reduction with keepdims=True."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = reduce_sum_physical(x, axis=axis, keepdims=True)
        expected = np.sum(np_x, axis=axis, keepdims=True)

        assert_shape(result, expected.shape)
        assert_allclose(result, expected)

    def test_sharded_reduce_non_sharded_axis(self, mesh_1d):
        """Reduce on axis that is NOT sharded - should work without gather."""

        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)

        result = reduce_sum_physical(x, axis=1)
        expected = np.sum(np_x, axis=1)

        assert_shape(result, expected.shape)
        assert_is_sharded(result, True)
        assert_allclose(result, expected)

    def test_sharded_reduce_sharded_axis(self, mesh_1d):
        """Reduce on axis that IS sharded - requires all_reduce."""

        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)

        result = reduce_sum_physical(x, axis=0)
        expected = np.sum(np_x, axis=0)

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
            ((2, 4, 8), 0),
            ((2, 4, 8), 1),
            ((2, 4, 8), 2),
        ],
    )
    def test_mean_axis(self, shape, axis):
        """Test mean along specific axis."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = mean_physical(x, axis=axis)
        expected = np.mean(np_x, axis=axis)

        assert_shape(result, expected.shape)
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "shape,axis",
        [
            ((4, 8), 0),
            ((4, 8), 1),
        ],
    )
    def test_keepdims(self, shape, axis):
        """Test mean with keepdims=True."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = mean_physical(x, axis=axis, keepdims=True)
        expected = np.mean(np_x, axis=axis, keepdims=True)

        assert_shape(result, expected.shape)
        assert_allclose(result, expected)

    def test_sharded_mean_non_sharded_axis(self, mesh_1d):
        """Mean on non-sharded axis."""
        np_x = make_array(8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)

        result = mean_physical(x, axis=1)
        expected = np.mean(np_x, axis=1)

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
            ((4, 8, 1), 2, (4, 8)),
            ((4, 8, 1), -1, (4, 8)),
        ],
    )
    def test_squeeze_axis(self, shape, axis, expected_shape):
        """Test squeeze at specific axis."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = squeeze_physical(x, axis=axis)
        expected = np.squeeze(np_x, axis=axis)

        assert_shape(result, expected_shape)
        assert_allclose(result, expected)

    def test_sharded_squeeze_non_sharded_dim(self, mesh_1d):
        """Squeeze a non-sharded dimension."""

        np_x = make_array(4, 1, 8, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)

        result = squeeze_physical(x, axis=1)
        expected = np.squeeze(np_x, axis=1)

        assert_shape(result, (4, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)

    def test_sharded_squeeze_after_sharded_dim(self, mesh_1d):
        """Squeeze a dim that comes after the sharded dim."""

        np_x = make_array(4, 8, 1, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)

        result = squeeze_physical(x, axis=2)
        expected = np.squeeze(np_x, axis=2)

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
            ((4, 8), 2, (4, 8, 1)),
            ((4, 8), -1, (4, 8, 1)),
        ],
    )
    def test_unsqueeze_axis(self, shape, axis, expected_shape):
        """Test unsqueeze at specific axis."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = unsqueeze_physical(x, axis=axis)
        expected = np.expand_dims(np_x, axis=axis)

        assert_shape(result, expected_shape)
        assert_allclose(result, expected)

    def test_sharded_unsqueeze_before_sharded(self, mesh_1d):
        """Unsqueeze before sharded dimension - sharded dim index shifts."""

        np_x = make_array(4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)

        result = unsqueeze_physical(x, axis=0)
        expected = np.expand_dims(np_x, axis=0)

        assert_shape(result, (1, 4, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)

    def test_sharded_unsqueeze_after_sharded(self, mesh_1d):
        """Unsqueeze after sharded dimension."""

        np_x = make_array(4, 8, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)

        result = unsqueeze_physical(x, axis=1)
        expected = np.expand_dims(np_x, axis=1)

        assert_shape(result, (4, 1, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


class TestBroadcastToPhysical:
    """Test broadcast_to_physical on various shapes."""

    @pytest.mark.parametrize(
        "shape,target_shape",
        [
            ((1,), (4,)),
            ((8,), (4, 8)),
            ((1, 8), (4, 8)),
            ((4, 1), (4, 8)),
            ((1, 1, 8), (2, 4, 8)),
            ((1, 4, 1), (2, 4, 8)),
        ],
    )
    def test_broadcast(self, shape, target_shape):
        """Test broadcast to target shape."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = broadcast_to_physical(x, target_shape)
        expected = np.broadcast_to(np_x, target_shape)

        assert_physical_shape(result, target_shape)
        assert_allclose(result, expected)

    def test_broadcast_preserves_sharding(self, mesh_1d):
        """Broadcast should preserve sharding on non-broadcast dims."""

        np_x = make_array(4, 1, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)

        result = broadcast_to_physical(x, (4, 8))
        expected = np.broadcast_to(np_x, (4, 8))

        assert_shape(result, (4, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)

    def test_broadcast_add_leading_dim_sharded(self, mesh_1d):
        """Broadcast by adding leading dim to sharded tensor."""

        np_x = make_array(4, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axis(x, mesh_1d, axis=0)

        result = broadcast_to_physical(x, (2, 4))
        expected = np.broadcast_to(np_x, (2, 4))

        assert_physical_shape(result, (2, 4))
        assert_allclose(result, expected)


class TestPhysicalOpsEdgeCases:
    """Test edge cases for physical ops."""

    def test_reduce_sum_single_element(self):
        """Reduce a single-element tensor."""
        np_x = make_array(1, seed=42)
        x = tensor_from_numpy(np_x)

        result = reduce_sum_physical(x, axis=0)
        expected = np.sum(np_x, axis=0)

        assert_allclose(result, expected)

    def test_mean_single_element(self):
        """Mean of single-element tensor."""
        np_x = make_array(1, seed=42)
        x = tensor_from_numpy(np_x)

        result = mean_physical(x, axis=0)
        expected = np.mean(np_x, axis=0)

        assert_allclose(result, expected)

    def test_squeeze_multiple_ones(self):
        """Squeeze when there are multiple dims of size 1."""
        np_x = make_array(1, 4, 1, 8, 1, seed=42)
        x = tensor_from_numpy(np_x)

        result = squeeze_physical(x, axis=0)
        expected = np.squeeze(np_x, axis=0)

        assert_shape(result, (4, 1, 8, 1))
        assert_allclose(result, expected)

    def test_broadcast_no_change(self):
        """Broadcast to same shape (no-op)."""
        np_x = make_array(4, 8, seed=42)
        x = tensor_from_numpy(np_x)

        result = broadcast_to_physical(x, (4, 8))
        expected = np.broadcast_to(np_x, (4, 8))

        assert_shape(result, (4, 8))
        assert_allclose(result, expected)
