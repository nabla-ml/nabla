# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest

from nabla import DeviceMesh, P, add, div, mul, sub
from nabla.core.sharding.spec import DimSpec
from tests.conftest import (
    assert_allclose,
    assert_is_sharded,
    assert_shape,
    make_array,
    make_positive_array,
    replicated,
    shard_on_axis,
    tensor_from_numpy,
)


class TestBinaryOpsSameSharding:
    """Test binary ops where both inputs have identical sharding."""

    @pytest.mark.parametrize(
        "op,numpy_op",
        [
            (add, lambda a, b: a + b),
            (sub, lambda a, b: a - b),
            (mul, lambda a, b: a * b),
        ],
    )
    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("dp",)),
            ((4,), ("dp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_binary_same_sharding_axis0(self, op, numpy_op, mesh_shape, mesh_axes):
        """Binary op with both inputs sharded on axis 0."""
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        np_a = make_array(8, 16, seed=42)
        np_b = make_array(8, 16, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)

        a_sharded = a.shard(mesh, P(mesh_axes[0], None))
        b_sharded = b.shard(mesh, P(mesh_axes[0], None))

        result = op(a_sharded, b_sharded)
        expected = numpy_op(np_a, np_b)

        assert_shape(result, (8, 16))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "op,numpy_op",
        [
            (add, lambda a, b: a + b),
            (mul, lambda a, b: a * b),
        ],
    )
    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((4,), ("tp",)),
        ],
    )
    def test_binary_same_sharding_axis1(self, op, numpy_op, mesh_shape, mesh_axes):
        """Binary op with both inputs sharded on axis 1 (tensor parallel)."""
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        np_a = make_array(8, 16, seed=42)
        np_b = make_array(8, 16, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)

        a_sharded = a.shard(mesh, P(None, mesh_axes[0]))
        b_sharded = b.shard(mesh, P(None, mesh_axes[0]))

        result = op(a_sharded, b_sharded)
        expected = numpy_op(np_a, np_b)

        assert_shape(result, (8, 16))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)

    def test_div_same_sharding(self, mesh_1d):
        """Div with both inputs sharded identically."""
        np_a = make_array(8, 4, seed=42)
        np_b = make_positive_array(8, 4, seed=43)

        a = shard_on_axis(tensor_from_numpy(np_a), mesh_1d, axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh_1d, axis=0)

        result = div(a, b)
        expected = np_a / np_b

        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


class TestBinaryOpsShardedReplicated:
    """Test binary ops where one input is sharded and one is replicated."""

    @pytest.mark.parametrize(
        "op,numpy_op",
        [
            (add, lambda a, b: a + b),
            (mul, lambda a, b: a * b),
        ],
    )
    def test_sharded_plus_replicated(self, op, numpy_op, mesh_1d):
        """Sharded input + replicated input."""
        np_a = make_array(8, 4, seed=42)
        np_b = make_array(8, 4, seed=43)

        a = shard_on_axis(tensor_from_numpy(np_a), mesh_1d, axis=0)
        b = replicated(tensor_from_numpy(np_b), mesh_1d)

        result = op(a, b)
        expected = numpy_op(np_a, np_b)

        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "op,numpy_op",
        [
            (add, lambda a, b: a + b),
            (mul, lambda a, b: a * b),
        ],
    )
    def test_replicated_plus_sharded(self, op, numpy_op, mesh_1d):
        """Replicated input + sharded input (order flipped)."""
        np_a = make_array(8, 4, seed=42)
        np_b = make_array(8, 4, seed=43)

        a = replicated(tensor_from_numpy(np_a), mesh_1d)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh_1d, axis=0)

        result = op(a, b)
        expected = numpy_op(np_a, np_b)

        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)

    def test_sharded_plus_unsharded(self, mesh_1d):
        """Sharded input + completely unsharded (no mesh) input."""
        np_a = make_array(8, 4, seed=42)
        np_b = make_array(8, 4, seed=43)

        a = shard_on_axis(tensor_from_numpy(np_a), mesh_1d, axis=0)
        b = tensor_from_numpy(np_b)

        result = add(a, b)
        expected = np_a + np_b

        assert_shape(result, (8, 4))
        assert_allclose(result, expected)


class TestBinaryOpsMismatchedSharding:
    """Test binary ops where inputs have different sharding - requires implicit resharding."""

    def test_different_sharded_axes(self, mesh_2d):
        """Inputs sharded on different axes - should work via resharding."""
        np_a = make_array(8, 8, seed=42)
        np_b = make_array(8, 8, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)

        specs_a = [DimSpec(["dp"], is_open=False), DimSpec([], is_open=True)]
        specs_b = [DimSpec([], is_open=True), DimSpec(["tp"], is_open=False)]

        a_sharded = a.shard(mesh_2d, specs_a)
        b_sharded = b.shard(mesh_2d, specs_b)

        result = add(a_sharded, b_sharded)
        expected = np_a + np_b

        assert_shape(result, (8, 8))
        assert_allclose(result, expected)

    def test_different_mesh_axes_same_dim(self):
        """Inputs sharded on same tensor dim but different mesh axes."""
        mesh = DeviceMesh("mesh", (2, 2), ("dp", "tp"))

        np_a = make_array(8, 4, seed=42)
        np_b = make_array(8, 4, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)

        a_sharded = a.shard(mesh, P("dp", None))
        b_sharded = b.shard(mesh, P("tp", None))

        result = add(a_sharded, b_sharded)
        expected = np_a + np_b

        assert_shape(result, (8, 4))
        assert_allclose(result, expected)


class TestBinaryOpsBroadcastSharding:
    """Test binary ops with broadcasting and sharding combined."""

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("dp",)),
            ((4,), ("dp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_broadcast_sharded_with_vector(self, mesh_shape, mesh_axes):
        """Broadcast: (8, 16) sharded + (16,) -> (8, 16) sharded."""
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        np_a = make_array(8, 16, seed=42)
        np_b = make_array(16, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)

        a_sharded = a.shard(mesh, P(mesh_axes[0], None))

        result = add(a_sharded, b)
        expected = np_a + np_b

        assert_shape(result, (8, 16))
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((4,), ("tp",)),
        ],
    )
    def test_broadcast_with_sharded_vector(self, mesh_shape, mesh_axes):
        """Broadcast: (8, 16) + (16,) sharded -> proper broadcast with sharding."""
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        np_a = make_array(8, 16, seed=42)
        np_b = make_array(16, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)

        b_sharded = b.shard(mesh, P(mesh_axes[0]))

        result = add(a, b_sharded)
        expected = np_a + np_b

        assert_shape(result, (8, 16))
        assert_allclose(result, expected)

    def test_broadcast_both_sharded_compatible(self):
        """Both inputs sharded on compatible dims during broadcast."""
        mesh = DeviceMesh("mesh", (2,), ("tp",))

        np_a = make_array(8, 16, seed=42)
        np_b = make_array(16, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)

        a_sharded = a.shard(mesh, P(None, "tp"))
        b_sharded = b.shard(mesh, P("tp"))

        result = add(a_sharded, b_sharded)
        expected = np_a + np_b

        assert_shape(result, (8, 16))
        assert_allclose(result, expected)


class TestBinaryOpsMultiAxisSharding:
    """Test binary ops with multi-axis (2D mesh) sharding."""

    def test_2d_mesh_both_axes_sharded(self, mesh_2d):
        """Both tensor axes sharded on 2D mesh."""
        np_a = make_array(8, 8, seed=42)
        np_b = make_array(8, 8, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)

        a_sharded = a.shard(mesh_2d, P("dp", "tp"))
        b_sharded = b.shard(mesh_2d, P("dp", "tp"))

        result = add(a_sharded, b_sharded)
        expected = np_a + np_b

        assert_shape(result, (8, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)

    def test_3d_tensor_on_2d_mesh(self, mesh_2d):
        """3D tensor with 2 axes sharded on 2D mesh."""
        np_a = make_array(4, 8, 16, seed=42)
        np_b = make_array(4, 8, 16, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)

        a_sharded = a.shard(mesh_2d, P("dp", None, "tp"))
        b_sharded = b.shard(mesh_2d, P("dp", None, "tp"))

        result = mul(a_sharded, b_sharded)
        expected = np_a * np_b

        assert_shape(result, (4, 8, 16))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


class TestShardingPropagationVerification:
    """Verify that output sharding matches expected based on input shardings."""

    def test_output_inherits_sharding_from_inputs(self, mesh_1d):
        """Verify output has correct sharding spec after binary op."""
        np_a = make_array(8, 4, seed=42)
        np_b = make_array(8, 4, seed=43)

        a = shard_on_axis(tensor_from_numpy(np_a), mesh_1d, axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh_1d, axis=0)

        result = add(a, b)

        assert result.is_sharded
        assert result.sharding is not None

        sharding = result.sharding
        assert len(sharding.dim_specs) >= 2

        assert "dp" in sharding.dim_specs[0].axes or sharding.dim_specs[0].axes == [
            "dp"
        ]

    def test_output_sharding_matches_2d_mesh_input(self, mesh_2d):
        """Verify output sharding with 2D mesh input."""
        np_a = make_array(8, 8, seed=42)
        np_b = make_array(8, 8, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)

        a_sharded = a.shard(mesh_2d, P("dp", "tp"))
        b_sharded = b.shard(mesh_2d, P("dp", "tp"))

        result = mul(a_sharded, b_sharded)

        assert result.is_sharded
        sharding = result.sharding

        assert sharding.dim_specs[0].axes
        assert sharding.dim_specs[1].axes


__all__ = [
    "TestBinaryOpsSameSharding",
    "TestBinaryOpsShardedReplicated",
    "TestBinaryOpsMismatchedSharding",
    "TestBinaryOpsBroadcastSharding",
    "TestBinaryOpsMultiAxisSharding",
    "TestShardingPropagationVerification",
]
