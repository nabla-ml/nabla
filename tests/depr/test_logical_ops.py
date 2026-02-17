# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Tests for logical operations - no batch_dims, establishing baseline correctness.

Logical ops work on tensors without batch_dims. This establishes that the
core operations produce correct numerical results before we test vmap.

Ops tested:
- Binary: add, sub, mul, div (and broadcasting)
- Unary: relu, sigmoid, tanh, exp, neg
- Reduction: reduce_sum, mean
- View: reshape, squeeze, unsqueeze, swap_axes, broadcast_to, moveaxis

All ops are tested against numpy reference implementations.
"""

import numpy as np
import pytest

from nabla import (
    add,
    broadcast_to,
    div,
    exp,
    mean,
    moveaxis,
    mul,
    neg,
    reduce_sum,
    relu,
    reshape,
    sigmoid,
    squeeze,
    sub,
    swap_axes,
    tanh,
    unsqueeze,
)

from .conftest import (
    assert_allclose,
    assert_is_sharded,
    assert_shape,
    make_array,
    make_positive_array,
    replicated,
    shard_on_axis,
    tensor_from_numpy,
)


class TestBinaryOpsBasic:
    """Test binary ops with same-shape inputs."""

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_add(self, shape):
        """Test add with various shapes."""
        np_a = make_array(*shape, seed=42)
        np_b = make_array(*shape, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        result = add(a, b)

        assert_shape(result, shape)
        assert_allclose(result, np_a + np_b)

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_sub(self, shape):
        """Test sub with various shapes."""
        np_a = make_array(*shape, seed=42)
        np_b = make_array(*shape, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        result = sub(a, b)

        assert_shape(result, shape)
        assert_allclose(result, np_a - np_b)

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_mul(self, shape):
        """Test mul with various shapes."""
        np_a = make_array(*shape, seed=42)
        np_b = make_array(*shape, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        result = mul(a, b)

        assert_shape(result, shape)
        assert_allclose(result, np_a * np_b)

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_div(self, shape):
        """Test div with various shapes."""
        np_a = make_array(*shape, seed=42)
        np_b = make_positive_array(*shape, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        result = div(a, b)

        assert_shape(result, shape)
        assert_allclose(result, np_a / np_b)


class TestBinaryOpsBroadcasting:
    """Test binary ops with broadcasting."""

    @pytest.mark.parametrize(
        "shape_a,shape_b,expected_shape",
        [
            ((4, 8), (8,), (4, 8)),
            ((8,), (4, 8), (4, 8)),
            ((4, 1), (1, 8), (4, 8)),
            ((2, 4, 8), (8,), (2, 4, 8)),
            ((2, 4, 8), (4, 8), (2, 4, 8)),
            ((2, 1, 8), (4, 1), (2, 4, 8)),
        ],
    )
    def test_add_broadcast(self, shape_a, shape_b, expected_shape):
        """Test add with broadcasting."""
        np_a = make_array(*shape_a, seed=42)
        np_b = make_array(*shape_b, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        result = add(a, b)

        assert_shape(result, expected_shape)
        assert_allclose(result, np_a + np_b)

    def test_mul_broadcast(self):
        """Test mul with broadcasting."""
        np_a = make_array(4, 8, seed=42)
        np_b = make_array(8, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        result = mul(a, b)

        assert_shape(result, (4, 8))
        assert_allclose(result, np_a * np_b)


class TestBinaryOpsSharding:
    """Test binary ops with sharding."""

    def test_add_both_sharded_same(self, mesh_1d):
        """Add two tensors sharded the same way."""
        np_a = make_array(8, 4, seed=42)
        np_b = make_array(8, 4, seed=43)

        a = shard_on_axis(tensor_from_numpy(np_a), mesh_1d, axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh_1d, axis=0)
        result = add(a, b)

        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, np_a + np_b)

    def test_add_sharded_replicated(self, mesh_1d):
        """Add sharded tensor with replicated tensor."""
        np_a = make_array(8, 4, seed=42)
        np_b = make_array(8, 4, seed=43)

        a = shard_on_axis(tensor_from_numpy(np_a), mesh_1d, axis=0)
        b = replicated(tensor_from_numpy(np_b), mesh_1d)
        result = add(a, b)

        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, np_a + np_b)

    def test_add_broadcast_sharded(self, mesh_1d):
        """Add with broadcasting where one is sharded."""
        np_a = make_array(8, 4, seed=42)
        np_b = make_array(4, seed=43)

        a = shard_on_axis(tensor_from_numpy(np_a), mesh_1d, axis=0)
        b = tensor_from_numpy(np_b)
        result = add(a, b)

        assert_shape(result, (8, 4))
        assert_allclose(result, np_a + np_b)


class TestUnaryOpsBasic:
    """Test unary ops with various shapes."""

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_relu(self, shape):
        """Test relu."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = relu(x)
        expected = np.maximum(np_x, 0)

        assert_shape(result, shape)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_sigmoid(self, shape):
        """Test sigmoid."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = sigmoid(x)
        expected = 1 / (1 + np.exp(-np_x))

        assert_shape(result, shape)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_tanh(self, shape):
        """Test tanh."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = tanh(x)
        expected = np.tanh(np_x)

        assert_shape(result, shape)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_exp(self, shape):
        """Test exp."""

        np_x = make_array(*shape, seed=42) * 0.5
        x = tensor_from_numpy(np_x)

        result = exp(x)
        expected = np.exp(np_x)

        assert_shape(result, shape)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_neg(self, shape):
        """Test neg."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = neg(x)
        expected = -np_x

        assert_shape(result, shape)
        assert_allclose(result, expected)


class TestUnaryOpsSharding:
    """Test unary ops with sharding."""

    def test_relu_sharded(self, mesh_1d):
        """Relu on sharded tensor."""
        np_x = make_array(8, 4, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh_1d, axis=0)

        result = relu(x)
        expected = np.maximum(np_x, 0)

        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)

    def test_sigmoid_sharded(self, mesh_1d):
        """Sigmoid on sharded tensor."""
        np_x = make_array(8, 4, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh_1d, axis=0)

        result = sigmoid(x)
        expected = 1 / (1 + np.exp(-np_x))

        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


class TestReductionOps:
    """Test reduction ops."""

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
    def test_reduce_sum(self, shape, axis):
        """Test reduce_sum along axis."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = reduce_sum(x, axis=axis)
        expected = np.sum(np_x, axis=axis)

        assert_shape(result, expected.shape)
        assert_allclose(result, expected)

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
    def test_mean(self, shape, axis):
        """Test mean along axis."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = mean(x, axis=axis)
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
    def test_reduce_sum_keepdims(self, shape, axis):
        """Test reduce_sum with keepdims=True."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = reduce_sum(x, axis=axis, keepdims=True)
        expected = np.sum(np_x, axis=axis, keepdims=True)

        assert_shape(result, expected.shape)
        assert_allclose(result, expected)


class TestReductionOpsSharding:
    """Test reduction ops with sharding."""

    def test_reduce_sum_non_sharded_axis(self, mesh_1d):
        """Reduce sum on non-sharded axis."""
        np_x = make_array(8, 4, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh_1d, axis=0)

        result = reduce_sum(x, axis=1)
        expected = np.sum(np_x, axis=1)

        assert_shape(result, expected.shape)
        assert_is_sharded(result, True)
        assert_allclose(result, expected)

    def test_reduce_sum_sharded_axis(self, mesh_1d):
        """Reduce sum on sharded axis - requires all-reduce."""
        np_x = make_array(8, 4, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh_1d, axis=0)

        result = reduce_sum(x, axis=0)
        expected = np.sum(np_x, axis=0)

        assert_shape(result, expected.shape)
        assert_allclose(result, expected)


class TestReshape:
    """Test reshape."""

    @pytest.mark.parametrize(
        "shape,new_shape",
        [
            ((4, 8), (32,)),
            ((32,), (4, 8)),
            ((2, 4, 8), (8, 8)),
            ((8, 8), (2, 4, 8)),
            ((4, 8), (8, 4)),
        ],
    )
    def test_reshape(self, shape, new_shape):
        """Test reshape."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = reshape(x, new_shape)
        expected = np_x.reshape(new_shape)

        assert_shape(result, new_shape)
        assert_allclose(result, expected)


class TestSqueeze:
    """Test squeeze."""

    @pytest.mark.parametrize(
        "shape,axis,expected_shape",
        [
            ((1, 8), 0, (8,)),
            ((4, 1), 1, (4,)),
            ((4, 1, 8), 1, (4, 8)),
            ((1, 4, 8), 0, (4, 8)),
        ],
    )
    def test_squeeze(self, shape, axis, expected_shape):
        """Test squeeze at axis."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = squeeze(x, axis=axis)
        expected = np.squeeze(np_x, axis=axis)

        assert_shape(result, expected_shape)
        assert_allclose(result, expected)


class TestUnsqueeze:
    """Test unsqueeze."""

    @pytest.mark.parametrize(
        "shape,axis,expected_shape",
        [
            ((8,), 0, (1, 8)),
            ((8,), 1, (8, 1)),
            ((4, 8), 0, (1, 4, 8)),
            ((4, 8), 1, (4, 1, 8)),
            ((4, 8), 2, (4, 8, 1)),
        ],
    )
    def test_unsqueeze(self, shape, axis, expected_shape):
        """Test unsqueeze at axis."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = unsqueeze(x, axis=axis)
        expected = np.expand_dims(np_x, axis=axis)

        assert_shape(result, expected_shape)
        assert_allclose(result, expected)


class TestSwapAxes:
    """Test swap_axes."""

    @pytest.mark.parametrize(
        "shape,axis1,axis2,expected_shape",
        [
            ((4, 8), 0, 1, (8, 4)),
            ((2, 4, 8), 0, 1, (4, 2, 8)),
            ((2, 4, 8), 0, 2, (8, 4, 2)),
            ((2, 4, 8), 1, 2, (2, 8, 4)),
        ],
    )
    def test_swap_axes(self, shape, axis1, axis2, expected_shape):
        """Test swap_axes."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = swap_axes(x, axis1, axis2)
        expected = np.swapaxes(np_x, axis1, axis2)

        assert_shape(result, expected_shape)
        assert_allclose(result, expected)


class TestMoveaxis:
    """Test moveaxis."""

    @pytest.mark.parametrize(
        "shape,source,destination,expected_shape",
        [
            ((4, 8), 0, 1, (8, 4)),
            ((4, 8), 1, 0, (8, 4)),
            ((2, 4, 8), 0, 2, (4, 8, 2)),
            ((2, 4, 8), 2, 0, (8, 2, 4)),
        ],
    )
    def test_moveaxis(self, shape, source, destination, expected_shape):
        """Test moveaxis."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = moveaxis(x, source, destination)
        expected = np.moveaxis(np_x, source, destination)

        assert_shape(result, expected_shape)
        assert_allclose(result, expected)


class TestBroadcastTo:
    """Test broadcast_to."""

    @pytest.mark.parametrize(
        "shape,target_shape",
        [
            ((8,), (4, 8)),
            ((1, 8), (4, 8)),
            ((4, 1), (4, 8)),
            ((1, 4, 1), (2, 4, 8)),
        ],
    )
    def test_broadcast_to(self, shape, target_shape):
        """Test broadcast_to."""
        np_x = make_array(*shape, seed=42)
        x = tensor_from_numpy(np_x)

        result = broadcast_to(x, target_shape)
        expected = np.broadcast_to(np_x, target_shape)

        assert_shape(result, target_shape)
        assert_allclose(result, expected)


class TestViewOpsSharding:
    """Test view ops with sharding."""

    def test_reshape_sharded(self, mesh_1d):
        """Reshape a sharded tensor."""
        np_x = make_array(8, 4, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh_1d, axis=0)

        result = reshape(x, (4, 8))
        expected = np_x.reshape(4, 8)

        assert_shape(result, (4, 8))
        assert_allclose(result, expected)

    def test_swap_axes_sharded(self, mesh_1d):
        """Swap axes of sharded tensor."""
        np_x = make_array(8, 4, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh_1d, axis=0)

        result = swap_axes(x, 0, 1)
        expected = np.swapaxes(np_x, 0, 1)

        assert_shape(result, (4, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)
