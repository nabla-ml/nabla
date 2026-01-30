"""
Unit tests for VJP (backward pass) with sharded tensors.

These tests catch bugs where shape computation goes wrong during gradient
computation with sharded/vmapped tensors.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nabla as nb
from nabla import ops
from nabla.core.sharding import DeviceMesh, DimSpec, PartitionSpec as P
from nabla.ops import communication
from nabla.ops.control_flow import where
from nabla.ops.creation import zeros_like
from nabla.ops.reduction import reduce_sum
from nabla.transforms import vmap
from tests.unit_v2.common import (
    assert_allclose,
    assert_shape,
    make_jax_array,
    shard_on_axis,
    tensor_from_jax,
)


@pytest.fixture
def mesh_4():
    return DeviceMesh("mesh_4", (4,), ("dp",))


class TestWhereVJPSharded:
    """Test gradient through where with sharded tensors."""

    def test_where_grad_basic(self):
        """Gradient through where without sharding - baseline."""
        jax_cond = jnp.array([True, False, True, False], dtype=bool)
        jax_x = make_jax_array(4, seed=42)
        jax_y = make_jax_array(4, seed=43)

        def loss_fn(x):
            return jnp.sum(jnp.where(jax_cond, x, jax_y))

        expected_grad = jax.grad(loss_fn)(jax_x)

        # Nabla
        cond_t = tensor_from_jax(jax_cond)
        x = tensor_from_jax(jax_x)
        y = tensor_from_jax(jax_y)

        def nb_loss(x_in):
            return reduce_sum(where(cond_t, x_in, y))

        grad_x = nb.grad(nb_loss)(x)

        assert_shape(grad_x, (4,))
        assert_allclose(grad_x, expected_grad)

    def test_where_grad_sharded(self, mesh_4):
        """Gradient through where with sharded inputs."""
        jax_cond = jnp.array([[True], [False], [True], [False]], dtype=bool)
        jax_x = make_jax_array(4, 4, seed=42)
        jax_y = make_jax_array(4, 4, seed=43)

        def jax_loss_fn(x):
            return jnp.sum(jnp.where(jax_cond, x, jax_y))

        expected_grad = jax.grad(jax_loss_fn)(jax_x)

        # Nabla with sharding
        cond_t = tensor_from_jax(jax_cond)
        x = tensor_from_jax(jax_x)
        y = tensor_from_jax(jax_y)

        cond_sharded = shard_on_axis(cond_t, mesh_4, axis=0)
        x_sharded = shard_on_axis(x, mesh_4, axis=0)
        y_sharded = shard_on_axis(y, mesh_4, axis=0)

        def nb_loss(x_in):
            return reduce_sum(where(cond_sharded, x_in, y_sharded))

        grad_x = nb.grad(nb_loss)(x_sharded)

        assert_shape(grad_x, (4, 4))
        assert_allclose(grad_x, expected_grad)


class TestPPermuteVJP:
    """Test gradient through ppermute operation."""

    def test_ppermute_grad_simple(self, mesh_4):
        """Gradient through ppermute with identity permutation."""
        jax_x = make_jax_array(4, 4, seed=42)

        # Identity permutation
        perm = [(i, i) for i in range(4)]

        x = tensor_from_jax(jax_x)
        x_sharded = shard_on_axis(x, mesh_4, axis=0)

        def nb_loss(x_in):
            shifted = communication.ppermute(x_in, perm)
            return reduce_sum(shifted)

        grad_x = nb.grad(nb_loss)(x_sharded)

        # Identity permutation should give gradient of ones
        expected_grad = np.ones_like(jax_x)

        assert_shape(grad_x, (4, 4))
        assert_allclose(grad_x, expected_grad)

    def test_ppermute_grad_ring_shift(self, mesh_4):
        """Gradient through ppermute with ring shift."""
        jax_x = make_jax_array(4, 4, seed=42)

        # Ring shift right: i -> (i+1) % 4
        perm = [(i, (i + 1) % 4) for i in range(4)]

        x = tensor_from_jax(jax_x)
        x_sharded = shard_on_axis(x, mesh_4, axis=0)

        def nb_loss(x_in):
            shifted = communication.ppermute(x_in, perm)
            return reduce_sum(shifted)

        grad_x = nb.grad(nb_loss)(x_sharded)

        # ppermute VJP should apply inverse permutation
        # Sum gradient is ones, inverse permutation shifts it
        expected_grad = np.ones_like(jax_x)

        assert_shape(grad_x, (4, 4))
        assert_allclose(grad_x, expected_grad)


class TestVmapPPermuteWhereGrad:
    """Test the specific pattern from pipeline parallelism:
    vmap(compute) -> ppermute -> where -> reduce_sum -> grad
    
    This is the pattern that fails in test_pp_grad2.py.
    """

    def test_vmap_ppermute_where_reduce_grad(self, mesh_4):
        """
        The critical bug pattern:
        1. vmap creates batch_dims on output
        2. ppermute shuffles data between shards
        3. where uses ppermute output
        4. reduce_sum reduces over sharded axis
        5. grad backward reconstructs shapes incorrectly
        """
        # Setup similar to pipeline step
        jax_x = make_jax_array(4, 4, 8, seed=42)  # [stages, batch, features]
        jax_mask = jnp.array(np.eye(4, 1).reshape(4, 1, 1).astype(bool))  # Stage 0 mask

        x = tensor_from_jax(jax_x)
        mask = tensor_from_jax(jax_mask)

        spec = [DimSpec.from_raw(d) for d in P("dp", None, None)]
        x_sharded = ops.shard(x, mesh_4, spec).realize()
        mask_sharded = ops.shard(mask, mesh_4, spec).realize()

        perm = [(i, (i + 1) % 4) for i in range(4)]

        # vmapped identity (like stage_compute but simpler)
        def identity(x_in):
            return x_in

        vmapped_fn = vmap(identity, in_axes=0, out_axes=0, spmd_axis_name="dp", mesh=mesh_4)

        def loss_fn(x_in):
            computed = vmapped_fn(x_in)
            shifted = communication.ppermute(computed, perm)
            # The bug: where sees shifted with wrong shape
            selected = where(mask_sharded, shifted, zeros_like(shifted))
            reduced = reduce_sum(selected, axis=0)
            return reduce_sum(reduced)

        # This should NOT crash with shape mismatch
        grad_x = nb.grad(loss_fn)(x_sharded)

        # Gradient should have correct shape
        assert_shape(grad_x, (4, 4, 8))


class TestReduceSumVJPSharded:
    """Test reduce_sum VJP with sharded tensors."""

    def test_reduce_sum_grad_sharded_axis(self, mesh_4):
        """Gradient through reduce_sum where reduction axis is sharded."""
        jax_x = make_jax_array(4, 4, seed=42)

        def jax_loss(x):
            return jnp.sum(jnp.sum(x, axis=0, keepdims=True))

        expected_grad = jax.grad(jax_loss)(jax_x)

        x = tensor_from_jax(jax_x)
        x_sharded = shard_on_axis(x, mesh_4, axis=0)

        def nb_loss(x_in):
            reduced = reduce_sum(x_in, axis=0, keepdims=True)
            return reduce_sum(reduced)

        grad_x = nb.grad(nb_loss)(x_sharded)

        assert_shape(grad_x, (4, 4))
        assert_allclose(grad_x, expected_grad)

    def test_reduce_sum_where_chain_grad(self, mesh_4):
        """Gradient through where -> reduce_sum chain with sharding."""
        jax_x = make_jax_array(4, 4, seed=42)
        jax_mask = jnp.array([[True], [False], [False], [False]], dtype=bool)

        def jax_loss(x):
            selected = jnp.where(jax_mask, x, jnp.zeros_like(x))
            reduced = jnp.sum(selected, axis=0, keepdims=True)
            return jnp.sum(reduced)

        expected_grad = jax.grad(jax_loss)(jax_x)

        x = tensor_from_jax(jax_x)
        mask = tensor_from_jax(jax_mask)
        x_sharded = shard_on_axis(x, mesh_4, axis=0)
        mask_sharded = shard_on_axis(mask, mesh_4, axis=0)

        def nb_loss(x_in):
            selected = where(mask_sharded, x_in, zeros_like(x_in))
            reduced = reduce_sum(selected, axis=0, keepdims=True)
            return reduce_sum(reduced)

        grad_x = nb.grad(nb_loss)(x_sharded)

        assert_shape(grad_x, (4, 4))
        assert_allclose(grad_x, expected_grad)
