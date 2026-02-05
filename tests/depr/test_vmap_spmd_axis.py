# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Tests for vmap with spmd_axis_name parameter.

This file tests the CRITICAL composability question:
    Does spmd_axis_name (batch dim sharding) compose correctly with
    logical dim sharding inside the vmapped function?

Test hierarchy:
1. TestSpmdAxisNameBasic - vmap with spmd_axis_name alone
2. TestSpmdAxisWithLogicalSharding - THE CRITICAL COMPOSABILITY TESTS
3. TestSpmdAxisNestedVmap - Nested vmap with spmd_axis_name
"""

import numpy as np
import pytest

from nabla import DeviceMesh, P, add, matmul, reduce_sum, relu, vmap
from nabla.core import trace
from .conftest import (
    assert_allclose,
    assert_shape,
    make_array,
    tensor_from_numpy,
)


class TestSpmdAxisNameBasic:
    """Test vmap with spmd_axis_name parameter alone (no logical sharding)."""

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes,spmd_axis",
        [
            ((2,), ("dp",), "dp"),
            ((4,), ("dp",), "dp"),
            ((2, 4), ("dp", "tp"), "dp"),
        ],
    )
    def test_spmd_basic_relu(self, mesh_shape, mesh_axes, spmd_axis):
        """vmap with spmd_axis_name on unary op."""
        batch = 8
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        np_x = make_array(batch, 16, seed=42)
        x = tensor_from_numpy(np_x)

        vmapped_relu = vmap(relu, spmd_axis_name=spmd_axis)

        print(f"\n{'='*60}")
        print("TEST: test_spmd_basic_relu")
        print(f"mesh_shape={mesh_shape}, mesh_axes={mesh_axes}, spmd_axis={spmd_axis}")
        print(f"{'='*60}")
        t = trace(vmapped_relu, x)
        print(t)
        print(f"{'='*60}\n")

        result = vmapped_relu(x)
        expected = np.maximum(np_x, 0)

        assert_shape(result, (batch, 16))
        assert_allclose(result, expected)

        if result.sharding:
            spec = result.sharding

            assert (
                spmd_axis in spec.dim_specs[0].axes
            ), f"Expected batch dim sharded on {spmd_axis}, got {spec.dim_specs[0]}"

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes,spmd_axis",
        [
            ((2,), ("dp",), "dp"),
            ((2, 4), ("dp", "tp"), "dp"),
        ],
    )
    def test_spmd_basic_add(self, mesh_shape, mesh_axes, spmd_axis):
        """vmap with spmd_axis_name on binary op."""
        batch = 8
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        np_a = make_array(batch, 16, seed=42)
        np_b = make_array(batch, 16, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)

        result = vmap(add, spmd_axis_name=spmd_axis)(a, b)
        expected = np_a + np_b

        assert_shape(result, (batch, 16))
        assert_allclose(result, expected)

        if result.sharding:
            assert spmd_axis in result.sharding.dim_specs[0].axes

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes,spmd_axis",
        [
            ((2,), ("dp",), "dp"),
            ((2, 4), ("dp", "tp"), "dp"),
        ],
    )
    def test_spmd_basic_matmul(self, mesh_shape, mesh_axes, spmd_axis):
        """vmap with spmd_axis_name on matmul."""
        batch = 4
        M, K, N = 8, 16, 12
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        np_x = make_array(batch, M, K, seed=42)
        np_w = make_array(K, N, seed=43)

        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)

        result = vmap(matmul, in_axes=(0, None), spmd_axis_name=spmd_axis)(x, w)
        expected = np_x @ np_w

        assert_shape(result, (batch, M, N))
        assert_allclose(result, expected, rtol=1e-4)

        if result.sharding:
            assert spmd_axis in result.sharding.dim_specs[0].axes

    def test_spmd_asymmetric_mesh(self, mesh_2x4):
        """vmap with spmd_axis_name on asymmetric mesh."""
        batch = 8

        np_x = make_array(batch, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(relu, spmd_axis_name="dp")(x)
        expected = np.maximum(np_x, 0)

        assert_shape(result, (batch, 16))
        assert_allclose(result, expected)

        if result.sharding:
            assert "dp" in result.sharding.dim_specs[0].axes


class TestSpmdAxisWithLogicalSharding:
    """Test the critical composability: spmd_axis_name + logical dim sharding.

    This is THE KEY question: when we use spmd_axis_name to shard the batch dim,
    AND we shard logical dims inside the vmapped function, do they compose correctly?

    Expected behavior:
    - Physical shape has batch dim at position 0
    - Batch dim sharded on spmd_axis_name
    - Logical dims sharded per user annotation
    - Result: multi-axis sharding on physical tensor
    """

    def test_spmd_plus_logical_sharding_relu(self, mesh_2x4):
        """Batch dim sharded on dp, logical dim on tp."""
        batch = 8
        mesh = mesh_2x4

        np_x = make_array(batch, 16, seed=42)
        x = tensor_from_numpy(np_x)

        @vmap(spmd_axis_name="dp", mesh=mesh)
        def f(row):
            row_sharded = row.shard(mesh, P("tp"))
            return relu(row_sharded)

        print(f"\n{'='*60}")
        print("TEST: test_spmd_plus_logical_sharding_relu")
        print("CRITICAL: batch dim on 'dp' + logical dim on 'tp'")
        print("mesh_shape=(2, 4), mesh_axes=('dp', 'tp')")
        print("Expected: multi-axis sharding [dp, tp] on output")
        print(f"{'='*60}")
        t = trace(f, x)
        print(t)
        print(f"{'='*60}\n")

        result = f(x)
        expected = np.maximum(np_x, 0)

        assert_shape(result, (batch, 16))
        assert_allclose(result, expected)

        spec = result.sharding
        assert spec is not None, "Result should have sharding"

        assert (
            "dp" in spec.dim_specs[0].axes
        ), f"Batch dim should be sharded on dp, got {spec.dim_specs[0]}"

        assert (
            "tp" in spec.dim_specs[1].axes
        ), f"Features dim should be sharded on tp, got {spec.dim_specs[1]}"

    def test_spmd_plus_logical_sharding_add(self, mesh_2x4):
        """Binary op with batch sharded on dp, logical on tp."""
        batch = 8
        mesh = mesh_2x4

        np_a = make_array(batch, 16, seed=42)
        np_b = make_array(batch, 16, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)

        @vmap(spmd_axis_name="dp", mesh=mesh)
        def f(row_a, row_b):

            row_a_s = row_a.shard(mesh, P("tp"))
            row_b_s = row_b.shard(mesh, P("tp"))
            return add(row_a_s, row_b_s)

        result = f(a, b)
        expected = np_a + np_b

        assert_shape(result, (batch, 16))
        assert_allclose(result, expected)

        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes
        assert "tp" in spec.dim_specs[1].axes

    def test_spmd_plus_logical_matmul_column_parallel(self, mesh_2x4):
        """Matmul with batch sharded, weight column-parallel."""
        batch = 4
        M, K, N = 8, 16, 32
        mesh = mesh_2x4

        np_x = make_array(batch, M, K, seed=42)
        np_w = make_array(K, N, seed=43)

        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)

        @vmap(in_axes=(0, None), spmd_axis_name="dp", mesh=mesh)
        def f(xi, wi):

            wi_sharded = wi.shard(mesh, P(None, "tp"))
            return matmul(xi, wi_sharded)

        result = f(x, w)
        expected = np_x @ np_w

        assert_shape(result, (batch, M, N))
        assert_allclose(result, expected, rtol=1e-4)

        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes
        assert "tp" in spec.dim_specs[2].axes

    def test_spmd_plus_logical_matmul_row_parallel(self, mesh_2x4):
        """Matmul with batch sharded, weight row-parallel."""
        batch = 4
        M, K, N = 8, 32, 16
        mesh = mesh_2x4

        np_x = make_array(batch, M, K, seed=42)
        np_w = make_array(K, N, seed=43)

        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)

        @vmap(in_axes=(0, None), spmd_axis_name="dp", mesh=mesh)
        def f(xi, wi):

            xi_sharded = xi.shard(mesh, P(None, "tp"))
            wi_sharded = wi.shard(mesh, P("tp", None))
            return matmul(xi_sharded, wi_sharded)

        print(f"\n{'='*60}")
        print("TEST: test_spmd_plus_logical_matmul_row_parallel")
        print("CRITICAL: Contracting dim K is sharded on 'tp'")
        print("Expected: AllReduce should appear in trace!")
        print("mesh_shape=(2, 4), mesh_axes=('dp', 'tp')")
        print(f"{'='*60}")
        t = trace(f, x, w)
        print(t)

        trace_str = str(t)
        if "all_reduce" in trace_str.lower():
            print("✅ AllReduce detected in trace - CORRECT!")
        else:
            print("❌ WARNING: No AllReduce detected - may be INCORRECT!")
        print(f"{'='*60}\n")

        result = f(x, w)
        expected = np_x @ np_w

        assert_shape(result, (batch, M, N))
        assert_allclose(result, expected, rtol=1e-4)

        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes

    def test_spmd_plus_logical_reduction(self, mesh_2x4):
        """Reduction op with batch and logical sharding."""
        batch = 8
        mesh = mesh_2x4

        np_x = make_array(batch, 16, seed=42)
        x = tensor_from_numpy(np_x)

        @vmap(spmd_axis_name="dp", mesh=mesh)
        def f(row):
            row_sharded = row.shard(mesh, P("tp"))
            return reduce_sum(row_sharded, axis=0)

        print(f"\n{'='*60}")
        print("TEST: test_spmd_plus_logical_reduction")
        print("CRITICAL: Reducing sharded axis (tp)")
        print("Expected: AllReduce should appear after local reduce_sum!")
        print("mesh_shape=(2, 4), mesh_axes=('dp', 'tp')")
        print(f"{'='*60}")
        t = trace(f, x)
        print(t)
        trace_str = str(t)
        if "all_reduce" in trace_str.lower():
            print("✅ AllReduce detected - CORRECT for reducing sharded axis!")
        else:
            print("❌ WARNING: No AllReduce detected - reduction may be INCORRECT!")
        print(f"{'='*60}\n")

        result = f(x)
        expected = np.sum(np_x, axis=1)

        assert_shape(result, (batch,))
        assert_allclose(result, expected)

        spec = result.sharding
        if spec:
            assert "dp" in spec.dim_specs[0].axes

    def test_spmd_plus_logical_3d_mesh(self, mesh_3d_2x4x2):
        """With 3D mesh: batch on dp, features on tp."""
        batch = 8
        mesh = mesh_3d_2x4x2

        np_x = make_array(batch, 16, seed=42)
        x = tensor_from_numpy(np_x)

        @vmap(spmd_axis_name="dp", mesh=mesh)
        def f(row):
            row_sharded = row.shard(mesh, P("tp"))
            return relu(row_sharded)

        result = f(x)
        expected = np.maximum(np_x, 0)

        assert_shape(result, (batch, 16))
        assert_allclose(result, expected)

        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes
        assert "tp" in spec.dim_specs[1].axes


class TestSpmdAxisNestedVmap:
    """Test nested vmap with spmd_axis_name."""

    def test_nested_vmap_spmd_outer(self, mesh_2x4):
        """Nested vmap: spmd_axis_name on outer vmap only."""
        outer_batch = 4
        inner_batch = 6
        features = 8
        mesh = mesh_2x4

        np_x = make_array(outer_batch, inner_batch, features, seed=42)
        x = tensor_from_numpy(np_x)

        @vmap(spmd_axis_name="dp", mesh=mesh)
        def outer(batch_x):

            return vmap(relu)(batch_x)

        result = outer(x)
        expected = np.maximum(np_x, 0)

        assert_shape(result, (outer_batch, inner_batch, features))
        assert_allclose(result, expected)

        spec = result.sharding
        if spec:
            assert "dp" in spec.dim_specs[0].axes

    def test_nested_vmap_spmd_with_logical(self, mesh_2x4):
        """Nested vmap: spmd_axis_name + logical sharding."""
        outer_batch = 4
        inner_batch = 6
        features = 16
        mesh = mesh_2x4

        np_x = make_array(outer_batch, inner_batch, features, seed=42)
        x = tensor_from_numpy(np_x)

        @vmap(spmd_axis_name="dp", mesh=mesh)
        def outer(batch_x):
            @vmap
            def inner(row):
                row_sharded = row.shard(mesh, P("tp"))
                return relu(row_sharded)

            return inner(batch_x)

        result = outer(x)
        expected = np.maximum(np_x, 0)

        assert_shape(result, (outer_batch, inner_batch, features))
        assert_allclose(result, expected)

        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes
        assert "tp" in spec.dim_specs[2].axes

    def test_nested_vmap_both_spmd(self, mesh_3d_2x4x2):
        """Nested vmap: both with spmd_axis_name on different axes."""
        outer_batch = 4
        inner_batch = 8
        features = 16
        mesh = mesh_3d_2x4x2

        np_x = make_array(outer_batch, inner_batch, features, seed=42)
        x = tensor_from_numpy(np_x)

        @vmap(spmd_axis_name="dp", mesh=mesh)
        def outer(batch_x):
            @vmap(spmd_axis_name="pp", mesh=mesh)
            def inner(row):
                row_sharded = row.shard(mesh, P("tp"))
                return relu(row_sharded)

            return inner(batch_x)

        result = outer(x)
        expected = np.maximum(np_x, 0)

        assert_shape(result, (outer_batch, inner_batch, features))
        assert_allclose(result, expected)

        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes
        assert "pp" in spec.dim_specs[1].axes
        assert "tp" in spec.dim_specs[2].axes


__all__ = [
    "TestSpmdAxisNameBasic",
    "TestSpmdAxisWithLogicalSharding",
    "TestSpmdAxisNestedVmap",
]
