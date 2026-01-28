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
            # Input shape inside vmap: (h1, h2)
            # Shard on h1 (axis 0)
            x_sharded = x.shard(mesh, P("tp", None))
            # Swap axis -1 (h2) and -2 (h1)
            # Note: all_to_all split_axis and concat_axis are relative to the *local* tensor?
            # Or logical? They are logical axes of the input.
            # split_axis=0 (h1), concat_axis=1 (h2)
            return all_to_all(x_sharded, split_axis=-2, concat_axis=-1)

        np_x = jax.random.normal(jax.random.PRNGKey(101), (batch, h1, h2), dtype=jnp.float32)
        x = tensor_from_jax(np_x)

        # vmap adds a leading dimension, so inside f we see (h1, h2)
        # We assume standard vmap behavior.
        result = nb.vmap(f)(x)

        # Expected: The values are effectively transposed if we view it globally?
        # Actually all_to_all is a distributed transpose.
        # If we split on h1 and concat on h2, we move the sharding from h1 to h2.
        # The values themselves are just moved around.
        # For a simple test, we just check shape and reconstruction.
        
        assert tuple(int(d) for d in result.shape) == (batch, h1, h2)
        assert_allclose(result, np_x)
        
        # Check sharding moved to the last axis (h2)
        # Result shape (batch, h1, h2).
        # Inside f: output is sharded on axis 1 (h2).
        # Outside vmap: output is sharded on axis 2 (h2) ? No, vmap adds axis 0.
        # So spec should be (None, None, "tp")
        spec = result.sharding
        assert spec.dim_specs[0].is_replicated() # batch (vmapped axis was not sharded)
        assert spec.dim_specs[1].is_replicated() # h1 (was sharded, now split)
        assert spec.dim_specs[2].axes == ["tp"]  # h2 (was replicated, now concatenated)

    def test_vmap_all_to_all_batch_interaction(self):
        """Test all_to_all where split axis interacts with vmapped batch dim?
        
        Actually, vmap isolates the batch dim. We can't easily cross it *inside* the function 
        unless we use specific collective primitives that are vmap-aware or use `lax.all_to_all`.
        
        But here we test that `all_to_all` inside `vmap` correctly ignores/shifts based on the hidden batch dim.
        """
        batch, seq, hidden = 2, 4, 8
        mesh = DeviceMesh("mesh_a2a_batch", (2,), ("tp",))
        
        def f(x):
            # x shape: (seq, hidden)
            # Shard on seq
            x_sharded = x.shard(mesh, P("tp", None))
            # Split seq (0), concat hidden (1)
            return all_to_all(x_sharded, split_axis=0, concat_axis=1)

        np_x = jax.random.normal(jax.random.PRNGKey(102), (batch, seq, hidden), dtype=jnp.float32)
        x = tensor_from_jax(np_x)

        result = nb.vmap(f)(x)
        
        # Verify output spec
        # Input to f: (seq, hidden). vmap adds batch at 0.
        # Inside f: split=0 (seq), concat=1 (hidden).
        # Physical execution should see batch_dims=1.
        # So physical split=1, concat=2.
        
        spec = result.sharding
        assert spec.dim_specs[0].is_replicated()
        assert spec.dim_specs[1].is_replicated() # seq
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
            
        np_x = jax.random.normal(jax.random.PRNGKey(104), (b1, b2, h), dtype=jnp.float32)
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
            
        np_x = jax.random.uniform(jax.random.PRNGKey(105), shape, minval=0.5, maxval=2.0)
        x = tensor_from_jax(np_x)
        
        result = f(x)
        
        # Simulated all_reduce prod: each value is x * x (since we shard input, 
        # but all_reduce sums across devices. Wait.
        # If we shard input, each device has a chunk.
        # AllReduce reduces ACROSS devices.
        # If input is [A, B] on devices [0, 1].
        # Result on both is A * B.
        # But here x is sharded. So dev0 has x[0:2], dev1 has x[2:4].
        # Result is x[0:2] * x[2:4] (elementwise).
        
        # Let's verify expectations manually.
        # Input x split into x1, x2.
        # Result = x1 * x2.
        expected = np_x[:2, :] * np_x[2:, :]
        # But wait, result shape is (2, 4) because it preserves the sharded shape?
        # No, AllReduce produces a full tensor? 
        # Standard AllReduce expects same-shape inputs on all devices and produces same-shape output.
        # If we feed sharded tensor, we are saying "Device 0 has this partial data, Device 1 has that".
        # If it's "sharded", usually we mean it's distinct parts of a larger tensor.
        # BUT AllReduce treats them as "contributions to a sum".
        # So functionally: output = shard0 op shard1 op ...
        
        assert_allclose(result, expected)
        # Output sharding:
        # The axis that was sharded (0) is now "reduced".
        # In Nabla, AllReduce keeps the sharding spec but clears partials?
        # If we feed a sharded tensor (partitioned data), AllReduce-ing it usually implies 
        # we are mixing data. 
        # If the input was "TP sharded", it means each device holds a slice.
        # If we AllReduce, we overlap them.
        # The result effectively has the shape of a shard.
        
        assert result.shape == (2, 4)

