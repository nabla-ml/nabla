# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Comprehensive stress tests for vmap + communication operations.

Focuses on:
- AllToAll with complex axis mappings and vmap interactions.
- AllReduce robustness.
- Error handling for invalid communication configs.
"""

import pytest
import jax
import jax.numpy as jnp
import nabla as nb
from nabla.core.sharding.spec import DeviceMesh, P
from nabla.ops.communication import all_to_all, all_reduce, all_gather
from .common import (
    assert_allclose,
    tensor_from_jax,
)


class TestVmapCommunicationComprehensive:

    def test_vmap_all_to_all_negative_axes(self):
        """Test all_to_all with negative axis indices inside vmap."""
        batch, h1, h2 = 2, 8, 8
        mesh = DeviceMesh("mesh_neg_a2a", (2,), ("tp",))

        def f(x):
            x_sharded = x.shard(mesh, P("tp", None))
            return all_to_all(x_sharded, split_axis=-1, concat_axis=-2)

        np_x = jax.random.normal(
            jax.random.PRNGKey(101), (batch, h1, h2), dtype=jnp.float32
        )
        x = tensor_from_jax(np_x)

        # vmap adds a leading dimension, so inside f we see (h1, h2)
        result = nb.vmap(f)(x)

        # Expected: logical shape is preserved (8, 8)
        assert tuple(int(d) for d in result.shape) == (batch, h1, h2)

        # Check sharding moved to the last axis (h2)
        spec = result.sharding
        assert spec.dim_specs[0].is_replicated()  # batch
        assert spec.dim_specs[1].is_replicated()  # h1 (was sharded, now gathered)
        assert spec.dim_specs[2].axes == ["tp"]  # h2 (was replicated, now split)

    def test_vmap_all_to_all_batch_interaction(self):
        """Test all_to_all where split axis interacts with vmapped batch dim."""
        batch, seq, hidden = 2, 4, 8
        mesh = DeviceMesh("mesh_a2a_batch", (2,), ("tp",))

        def f(x):
            x_sharded = x.shard(mesh, P("tp", None))
            return all_to_all(x_sharded, split_axis=1, concat_axis=0)

        np_x = jax.random.normal(
            jax.random.PRNGKey(102), (batch, seq, hidden), dtype=jnp.float32
        )
        x = tensor_from_jax(np_x)

        result = nb.vmap(f)(x)

        # Verify output spec
        spec = result.sharding
        assert spec.dim_specs[0].is_replicated()
        assert spec.dim_specs[1].is_replicated()  # seq
        assert spec.dim_specs[2].axes == ["tp"]  # hidden

    def test_all_to_all_invalid_split(self):
        """Test error when split axis is not divisible by mesh size."""
        # 1D mesh with 2 devices
        mesh = DeviceMesh("mesh_invalid", (2,), ("tp",))

        # Dimension size 3 cannot be split by 2 evenly
        h1, h2 = 3, 4

        def f(x):
            x_sharded = x.shard(mesh, P("tp", None))
            return all_to_all(x_sharded, split_axis=0, concat_axis=1)

        np_x = jax.random.normal(jax.random.PRNGKey(103), (h1, h2), dtype=jnp.float32)
        x = tensor_from_jax(np_x)

        # This should fail at runtime (or trace time if shapes are known)
        with pytest.raises(ValueError, match="divisible"):
            # We don't need vmap to trigger this, straightforward call is enough
            f(x)

    def test_nested_vmap_all_gather(self):
        """Test nested vmap with all_gather."""
        # Shape: (B1, B2, H)
        b1, b2, h = 2, 2, 4
        mesh = DeviceMesh("mesh_nested", (2,), ("tp",))

        def inner(x):
            # x shape: (H,)
            # Shard H
            x_s = x.shard(mesh, P("tp"))
            return all_gather(x_s, axis=0)

        def outer(x):
            # x shape: (B2, H)
            return nb.vmap(inner)(x)

        np_x = jax.random.normal(
            jax.random.PRNGKey(104), (b1, b2, h), dtype=jnp.float32
        )
        x = tensor_from_jax(np_x)

        result = nb.vmap(outer)(x)

        assert_allclose(result, np_x)
        # All axes should be replicated finally
        assert result.sharding.is_fully_replicated()

    def test_all_reduce_prod(self):
        """Test AllReduce with 'prod' op."""
        mesh = DeviceMesh("mesh_prod", (2,), ("tp",))
        shape = (4, 4)

        def f(x):
            x_s = x.shard(mesh, P("tp", None))
            return all_reduce(x_s, reduce_op="prod")

        np_x = jax.random.uniform(
            jax.random.PRNGKey(105), shape, minval=0.5, maxval=2.0
        )
        x = tensor_from_jax(np_x)

        result = f(x)
        expected = np_x[:2, :] * np_x[2:, :]
        assert_allclose(result, expected)

        assert tuple(int(d) for d in result.shape) == (2, 4)
