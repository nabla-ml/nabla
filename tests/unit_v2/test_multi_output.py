# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import jax.numpy as jnp
import jax.numpy as jnp
import pytest

import nabla as nb

from .common import OpConfig, Operation, run_unified_test, standard_get_args


def jax_split_wrapper(x, num_splits, axis=0):

    return jnp.split(x, num_splits, axis=axis)


def jax_chunk_wrapper(x, chunks, axis=0):

    return jnp.split(x, chunks, axis=axis)


def jax_unbind_wrapper(x, axis=0):

    return tuple(
        jnp.squeeze(s, axis=axis) for s in jnp.split(x, x.shape[axis], axis=axis)
    )


OPS = {}

OPS["split"] = Operation(
    "split",
    "MULTI_OUTPUT",
    nb.split,
    jax_split_wrapper,
    [
        OpConfig(
            "Split_2",
            ranks=(2,),
            params={"num_splits": 2, "axis": 0},
            primal_shapes=((4, 4),),
        ),
        OpConfig(
            "Split_Axis1",
            ranks=(2,),
            params={"num_splits": 2, "axis": 1},
            primal_shapes=((4, 4),),
        ),
    ],
    standard_get_args,
)

OPS["chunk"] = Operation(
    "chunk",
    "MULTI_OUTPUT",
    nb.chunk,
    jax_chunk_wrapper,
    [
        OpConfig(
            "Chunk_2",
            ranks=(2,),
            params={"chunks": 2, "axis": 0},
            primal_shapes=((4, 4),),
        ),
    ],
    standard_get_args,
)

OPS["unbind"] = Operation(
    "unbind",
    "MULTI_OUTPUT",
    nb.unbind,
    jax_unbind_wrapper,
    [
        OpConfig(
            "Unbind_Axis0", ranks=(2,), params={"axis": 0}, primal_shapes=((4, 4),)
        ),
    ],
    standard_get_args,
)


@pytest.mark.parametrize("op_name", OPS.keys())
@pytest.mark.parametrize("config_idx", [0, 1])
def test_multi_output_ops(op_name, config_idx):
    op = OPS[op_name]
    if config_idx >= len(op.configs):
        return
    config = op.configs[config_idx]
    run_unified_test(op, config)


# =============================================================================
# COMPREHENSIVE MULTI-OUTPUT SHARDING AND VMAP TESTS
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


class TestSplitSharding:
    """Test split operation with sharding."""

    def test_split_sharded_non_split_axis(self, mesh_1d):
        """Split with sharding on non-split axis."""
        jax_x = make_jax_array(16, 8, seed=42)
        x = tensor_from_jax(jax_x)

        x_sharded = shard_on_axis(x, mesh_1d, axis=0)

        results = nb.split(x_sharded, num_splits=2, axis=1)
        expected = jnp.split(jax_x, 2, axis=1)

        assert len(results) == 2
        for r, e in zip(results, expected, strict=False):
            assert_shape(r, (16, 4))
            assert_allclose(r, e)
            assert_is_sharded(r, True)

    def test_split_sharded_split_axis(self, mesh_1d):
        """Split with sharding on the split axis itself."""
        jax_x = make_jax_array(16, 8, seed=42)
        x = tensor_from_jax(jax_x)

        x_sharded = shard_on_axis(x, mesh_1d, axis=0)

        results = nb.split(x_sharded, num_splits=2, axis=0)
        expected = jnp.split(jax_x, 2, axis=0)

        assert len(results) == 2
        for r, e in zip(results, expected, strict=False):
            assert_shape(r, (8, 8))
            assert_allclose(r, e)

    def test_split_multi_axis_mesh(self, mesh_2x4):
        """Split on 2D mesh with sharding on both axes."""
        jax_x = make_jax_array(16, 16, seed=42)
        x = tensor_from_jax(jax_x)

        mesh = mesh_2x4
        from nabla.core.sharding.spec import DimSpec

        x_sharded = x.shard(mesh, [DimSpec(["dp"]), DimSpec(["tp"])])

        results = nb.split(x_sharded, num_splits=4, axis=0)

        assert len(results) == 4
        for r in results:
            assert_shape(r, (4, 16))
            assert_is_sharded(r, True)


class TestChunkSharding:
    """Test chunk operation with sharding."""

    def test_chunk_sharded(self, mesh_1d):
        """Chunk with sharding."""
        jax_x = make_jax_array(12, 8, seed=42)
        x = tensor_from_jax(jax_x)

        x_sharded = shard_on_axis(x, mesh_1d, axis=0)

        results = nb.chunk(x_sharded, chunks=3, axis=0)
        expected = jnp.array_split(jax_x, 3, axis=0)

        assert len(results) == 3
        for r, e in zip(results, expected, strict=False):
            assert_allclose(r, e)

    def test_chunk_uneven_division(self, mesh_1d):
        """Chunk with uneven division on sharded tensor."""
        jax_x = make_jax_array(10, 8, seed=42)
        x = tensor_from_jax(jax_x)

        x_sharded = shard_on_axis(x, mesh_1d, axis=1)

        results = nb.chunk(x_sharded, chunks=3, axis=0)
        expected = jnp.array_split(jax_x, 3, axis=0)

        assert len(results) == 3
        for r, e in zip(results, expected, strict=False):
            assert_allclose(r, e)
            assert_is_sharded(r, True)


class TestUnbindSharding:
    """Test unbind operation with sharding."""

    def test_unbind_sharded(self, mesh_1d):
        """Unbind with sharding."""
        jax_x = make_jax_array(4, 8, seed=42)
        x = tensor_from_jax(jax_x)

        x_sharded = shard_on_axis(x, mesh_1d, axis=1)

        results = nb.unbind(x_sharded, axis=0)

        assert len(results) == 4
        for i, r in enumerate(results):
            assert_shape(r, (8,))
            assert_allclose(r, jax_x[i])
            assert_is_sharded(r, True)

    def test_unbind_3d_sharded_middle_axis(self, mesh_1d):
        """Unbind 3D tensor with sharding on non-unbind axis."""
        jax_x = make_jax_array(4, 2, 8, seed=42)
        x = tensor_from_jax(jax_x)

        x_sharded = shard_on_axis(x, mesh_1d, axis=1)

        # Shard on axis 1, unbind on axis 0 (different axes)
        results = nb.unbind(x_sharded, axis=0)

        assert len(results) == 4
        for i, r in enumerate(results):
            print(
                f"DEBUG: result[{i}] shape={tuple(int(d) for d in r.shape)}, expected=(2, 8)"
            )
            assert_shape(r, (2, 8))  # Unbinding axis 0 from (4,2,8) gives (2,8)
            assert_allclose(r, jax_x[i, :, :])
            assert_is_sharded(r, True)  # Preserves sharding from axis 1


class TestMinMaxOp:
    """Test minmax operation (dict output)."""

    def test_minmax_basic(self):
        """Basic minmax without sharding."""
        jax_x = make_jax_array(8, 4, seed=42)
        x = tensor_from_jax(jax_x)

        result = nb.minmax(x)

        assert isinstance(result, dict)
        assert "min" in result
        assert "max" in result

        assert_allclose(result["min"], jnp.min(jax_x))
        assert_allclose(result["max"], jnp.max(jax_x))

    def test_minmax_sharded(self, mesh_1d):
        """Minmax with sharded input."""
        jax_x = make_jax_array(8, 4, seed=42)
        x = tensor_from_jax(jax_x)

        x_sharded = shard_on_axis(x, mesh_1d, axis=0)

        result = nb.minmax(x_sharded)

        assert isinstance(result, dict)
        assert_allclose(result["min"], jnp.min(jax_x))
        assert_allclose(result["max"], jnp.max(jax_x))


class TestVmapMultiOutput:
    """Test vmap with multi-output operations."""

    def test_vmap_split(self):
        """Split inside vmap."""
        batch = 4
        jax_x = make_jax_array(batch, 16, seed=42)
        x = tensor_from_jax(jax_x)

        def f(row):
            return nb.split(row, num_splits=2, axis=0)

        results = nb.vmap(f)(x)

        assert len(results) == 2
        for r in results:
            assert_shape(r, (batch, 8))

        expected_parts = jnp.split(jax_x, 2, axis=1)
        for r, e in zip(results, expected_parts, strict=False):
            assert_allclose(r, e)

    def test_vmap_chunk(self):
        """Chunk inside vmap."""
        batch = 4
        jax_x = make_jax_array(batch, 10, seed=42)
        x = tensor_from_jax(jax_x)

        def f(row):
            return nb.chunk(row, chunks=3, axis=0)

        results = nb.vmap(f)(x)

        assert len(results) == 3
        expected_parts = [jnp.array_split(jax_x, 3, axis=1)[i] for i in range(3)]
        for r, e in zip(results, expected_parts, strict=False):
            assert_allclose(r, e)

    def test_vmap_unbind(self):
        """Unbind inside vmap."""
        batch = 4
        jax_x = make_jax_array(batch, 3, 8, seed=42)
        x = tensor_from_jax(jax_x)

        def f(batch_x):
            return nb.unbind(batch_x, axis=0)

        results = nb.vmap(f)(x)

        assert len(results) == 3
        for i, r in enumerate(results):
            assert_shape(r, (batch, 8))
            assert_allclose(r, jax_x[:, i, :])

    def test_vmap_split_with_sharding(self, mesh_2x4):
        """Split inside vmap with spmd_axis_name + logical sharding."""
        batch = 8
        mesh = mesh_2x4

        jax_x = make_jax_array(batch, 16, seed=42)
        x = tensor_from_jax(jax_x)

        @nb.vmap(spmd_axis_name="dp")
        def f(row):
            row_sharded = shard_on_axis(row, mesh, axis=0, mesh_axis=1)
            return nb.split(row_sharded, num_splits=2, axis=0)

        results = f(x)

        assert len(results) == 2
        for r in results:
            assert_shape(r, (batch, 8))


class TestMultiOutputEdgeCases:
    """Test edge cases for multi-output ops."""

    def test_split_single_part(self):
        """Split into single part (essentially a no-op)."""
        jax_x = make_jax_array(8, 4, seed=42)
        x = tensor_from_jax(jax_x)

        results = nb.split(x, num_splits=1, axis=0)

        assert len(results) == 1
        assert_allclose(results[0], jax_x)

    def test_chunk_more_chunks_than_elements(self):
        """Chunk with more chunks than elements."""
        jax_x = make_jax_array(2, 4, seed=42)
        x = tensor_from_jax(jax_x)

        results = nb.chunk(x, chunks=5, axis=0)

        assert len(results) == 2
        for i, r in enumerate(results):
            assert_allclose(r, jax_x[i : i + 1])

    def test_unbind_single_element(self):
        """Unbind along dimension of size 1."""
        jax_x = make_jax_array(1, 8, seed=42)
        x = tensor_from_jax(jax_x)

        results = nb.unbind(x, axis=0)

        assert len(results) == 1
        assert_shape(results[0], (8,))
        assert_allclose(results[0], jax_x[0])
