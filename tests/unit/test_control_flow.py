# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Comprehensive control flow operation tests with sharding and vmap.

Tests for where, cond, while_loop, and scan operations:
1. Basic functionality (correctness)
2. Sharding propagation through branches/loops
3. vmap integration with control flow
4. Edge cases (nested conditions, sharded loop state)
"""

import jax
import jax.numpy as jnp
import pytest

import nabla as nb
from nabla.ops.control_flow import cond, scan, where, while_loop

from .common import (
    assert_allclose,
    assert_shape,
    make_jax_array,
    shard_on_axis,
    tensor_from_jax,
)


class TestWhereOp:
    """Test where operation: element-wise conditional selection."""

    def test_where_basic(self):
        """Basic where without sharding."""
        cond_jax = jnp.array([True, False, True, False], dtype=bool)
        jax_x = make_jax_array(4, seed=42)
        jax_y = make_jax_array(4, seed=43)

        cond_t = tensor_from_jax(cond_jax)
        x = tensor_from_jax(jax_x)
        y = tensor_from_jax(jax_y)

        result = where(cond_t, x, y)
        expected = jnp.where(cond_jax, jax_x, jax_y)

        assert_shape(result, (4,))
        assert_allclose(result, expected)

    def test_where_2d(self):
        """Where on 2D tensors."""
        key = jax.random.PRNGKey(100)
        cond_jax = jax.random.choice(key, jnp.array([True, False]), shape=(4, 8))
        jax_x = make_jax_array(4, 8, seed=42)
        jax_y = make_jax_array(4, 8, seed=43)

        cond_t = tensor_from_jax(cond_jax)
        x = tensor_from_jax(jax_x)
        y = tensor_from_jax(jax_y)

        result = where(cond_t, x, y)
        expected = jnp.where(cond_jax, jax_x, jax_y)

        assert_shape(result, (4, 8))
        assert_allclose(result, expected)

    def test_where_sharded_inputs(self, mesh_1d):
        """Where with sharded x and y tensors."""
        key = jax.random.PRNGKey(101)
        cond_jax = jax.random.choice(key, jnp.array([True, False]), shape=(8, 4))
        jax_x = make_jax_array(8, 4, seed=42)
        jax_y = make_jax_array(8, 4, seed=43)

        cond_t = tensor_from_jax(cond_jax)
        x = tensor_from_jax(jax_x)
        y = tensor_from_jax(jax_y)

        x_sharded = shard_on_axis(x, mesh_1d, axis=0)
        y_sharded = shard_on_axis(y, mesh_1d, axis=0)

        result = where(cond_t, x_sharded, y_sharded)
        expected = jnp.where(cond_jax, jax_x, jax_y)

        assert_shape(result, (8, 4))
        assert_allclose(result, expected)

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_where_vmap(self, batch_size):
        """vmap(where) with batched inputs."""
        key = jax.random.PRNGKey(102)
        cond_jax = jax.random.choice(
            key, jnp.array([True, False]), shape=(batch_size, 8)
        )
        jax_x = make_jax_array(batch_size, 8, seed=42)
        jax_y = make_jax_array(batch_size, 8, seed=43)

        cond_t = tensor_from_jax(cond_jax)
        x = tensor_from_jax(jax_x)
        y = tensor_from_jax(jax_y)

        def fn(c, x_val, y_val):
            return where(c, x_val, y_val)

        result = nb.vmap(fn)(cond_t, x, y)
        expected = jnp.where(cond_jax, jax_x, jax_y)

        assert_shape(result, (batch_size, 8))
        assert_allclose(result, expected)


class TestCondOp:
    """Test cond operation: conditional branching."""

    def test_cond_basic_true(self):
        """Cond with true predicate."""

        def true_fn(x):
            return x + 1.0

        def false_fn(x):
            return x - 1.0

        jax_x = jnp.array(5.0, dtype=jnp.float32)
        x = tensor_from_jax(jax_x)

        pred = nb.constant(True, dtype=nb.DType.bool)
        result = cond(pred, true_fn, false_fn, x)

        assert_allclose(result, jnp.array(6.0, dtype=jnp.float32))

    def test_cond_basic_false(self):
        """Cond with false predicate."""

        def true_fn(x):
            return x + 1.0

        def false_fn(x):
            return x - 1.0

        jax_x = jnp.array(5.0, dtype=jnp.float32)
        x = tensor_from_jax(jax_x)

        pred = nb.constant(False, dtype=nb.DType.bool)
        result = cond(pred, true_fn, false_fn, x)

        assert_allclose(result, jnp.array(4.0, dtype=jnp.float32))

    def test_cond_pytree_output(self):
        """Cond returning tuple (pytree)."""

        def true_fn(x):
            return (x, x + 1.0)

        def false_fn(x):
            return (x, x - 1.0)

        jax_x = jnp.array(5.0, dtype=jnp.float32)
        x = tensor_from_jax(jax_x)

        pred = nb.constant(True, dtype=nb.DType.bool)
        result = cond(pred, true_fn, false_fn, x)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert_allclose(result[0], jnp.array(5.0, dtype=jnp.float32))
        assert_allclose(result[1], jnp.array(6.0, dtype=jnp.float32))

    def test_cond_sharding_in_branch(self, mesh_1d):
        """Cond with sharding applied inside branch."""

        def true_fn(x):
            x_sharded = shard_on_axis(x, mesh_1d, axis=0)
            return nb.reduce_sum(x_sharded, axis=0)

        def false_fn(x):
            return nb.reduce_sum(x, axis=0)

        jax_x = make_jax_array(8, 4, seed=42)
        x = tensor_from_jax(jax_x)

        pred = nb.constant(True, dtype=nb.DType.bool)
        result = cond(pred, true_fn, false_fn, x)

        expected = jnp.sum(jax_x, axis=0)
        assert_shape(result, (4,))
        assert_allclose(result, expected)
        """Cond with sharding applied inside branch."""

        def true_fn(x):
            x_sharded = shard_on_axis(x, mesh_1d, axis=0)
            return nb.reduce_sum(x_sharded, axis=0)

        def false_fn(x):
            return nb.reduce_sum(x, axis=0)

        jax_x = make_jax_array(8, 4, seed=42)
        x = tensor_from_jax(jax_x)

        pred = nb.constant(True, dtype=nb.DType.bool)
        result = cond(pred, true_fn, false_fn, x)

        expected = jnp.sum(jax_x, axis=0)
        assert_shape(result, (4,))
        assert_allclose(result, expected)


class TestWhileLoopOp:
    """Test while_loop operation: iterative loops."""

    def test_while_loop_counter(self):
        """Basic while loop counting to limit."""

        def cond_fn(i):
            limit = nb.constant(10, dtype=nb.DType.int32)
            return nb.less(i, limit)

        def body_fn(i):
            return i + 1

        i_init = nb.constant(0, dtype=nb.DType.int32)
        result = while_loop(cond_fn, body_fn, i_init)

        assert_allclose(result, jnp.array(10, dtype=jnp.int32))

    def test_while_loop_accumulator(self):
        """While loop with accumulator (sum 1+2+...+10)."""

        def cond_fn(state):
            i, _ = state
            limit = nb.constant(11, dtype=nb.DType.int32)
            return nb.less(i, limit)

        def body_fn(state):
            i, acc = state
            return (i + 1, acc + i)

        init_state = (
            nb.constant(1, dtype=nb.DType.int32),
            nb.constant(0, dtype=nb.DType.int32),
        )
        result_i, result_acc = while_loop(cond_fn, body_fn, init_state)

        assert_allclose(result_i, jnp.array(11, dtype=jnp.int32))
        assert_allclose(result_acc, jnp.array(55, dtype=jnp.int32))

    def test_while_loop_sharded_state(self, mesh_1d):
        """While loop with sharded loop-carried state."""

        def cond_fn(state):
            i, _ = state
            limit = nb.constant(5, dtype=nb.DType.int32)
            return nb.less(i, limit)

        def body_fn(state):
            i, x = state
            return (i + 1, x + 1.0)

        jax_x = make_jax_array(8, 4, seed=42)
        x = tensor_from_jax(jax_x)
        x_sharded = shard_on_axis(x, mesh_1d, axis=0)

        init_state = (nb.constant(0, dtype=nb.DType.int32), x_sharded)
        result_i, result_x = while_loop(cond_fn, body_fn, init_state)

        expected_x = jax_x + 5.0
        assert_allclose(result_i, jnp.array(5, dtype=jnp.int32))
        assert_shape(result_x, (8, 4))
        assert_allclose(result_x, expected_x)


class TestScanOp:
    """Test scan operation: sequential iteration with outputs."""

    def test_scan_cumsum(self):
        """Scan implementing cumulative sum."""

        def f(carry, x):
            new_carry = carry + x
            return new_carry, new_carry

        jax_xs = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        xs = tensor_from_jax(jax_xs)
        init = nb.constant(0.0, dtype=nb.DType.float32)

        final_carry, ys = scan(f, init, xs)

        expected_carry = 10.0
        expected_ys = jnp.array([1.0, 3.0, 6.0, 10.0], dtype=jnp.float32)

        assert_allclose(final_carry, jnp.array(expected_carry, dtype=jnp.float32))
        assert_shape(ys, (4,))
        assert_allclose(ys, expected_ys)

    def test_scan_cumulative_product(self):
        """Scan implementing cumulative product."""

        def f(carry, x):
            new_carry = carry * x
            return new_carry, new_carry

        jax_xs = jnp.array([2.0, 3.0, 4.0], dtype=jnp.float32)
        xs = tensor_from_jax(jax_xs)
        init = nb.constant(1.0, dtype=nb.DType.float32)

        final_carry, ys = scan(f, init, xs)

        expected_ys = jnp.array([2.0, 6.0, 24.0], dtype=jnp.float32)

        assert_allclose(final_carry, jnp.array(24.0, dtype=jnp.float32))
        assert_allclose(ys, expected_ys)

    def test_scan_2d_input(self):
        """Scan over 2D tensor accumulating columns."""

        def f(carry, x):
            # x is a row vector (8,), carry is also (8,)
            s = carry + x  # Element-wise add
            return s, s

        jax_xs = make_jax_array(4, 8, seed=42)
        xs = tensor_from_jax(jax_xs)
        init = nb.zeros((8,))

        final_carry, ys = scan(f, init, xs)

        # Expected: cumulative sum of rows
        expected_final = jnp.sum(jax_xs, axis=0)
        assert_shape(final_carry, (8,))
        assert_shape(ys, (4, 8))
        assert_allclose(final_carry, expected_final, rtol=1e-4)

    def test_scan_sharded_xs(self, mesh_1d):
        """Scan with sharded xs sequence."""

        def f(carry, x):
            new_carry = carry + x
            return new_carry, new_carry

        jax_xs = make_jax_array(8, 4, seed=42)
        xs = tensor_from_jax(jax_xs)
        xs_sharded = shard_on_axis(xs, mesh_1d, axis=1)

        init = nb.zeros((4,))
        init_sharded = shard_on_axis(init, mesh_1d, axis=0)

        final_carry, ys = scan(f, init_sharded, xs_sharded)

        expected_final = jnp.sum(jax_xs, axis=0)
        assert_shape(final_carry, (4,))
        assert_shape(ys, (8, 4))
        assert_allclose(final_carry, expected_final, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_scan_nested_carry(self, batch_size):
        """Scan with nested tuple carry (state tracking)."""

        def f(carry, x):
            count, total = carry
            new_count = count + 1
            new_total = total + x
            return (new_count, new_total), new_total

        jax_xs = make_jax_array(batch_size, seed=42)
        xs = tensor_from_jax(jax_xs)

        init_count = nb.constant(0, dtype=nb.DType.int32)
        init_total = nb.constant(0.0, dtype=nb.DType.float32)

        (final_count, final_total), ys = scan(f, (init_count, init_total), xs)

        expected_total = jnp.sum(jax_xs)
        assert_allclose(final_count, jnp.array(batch_size, dtype=jnp.int32))
        assert_allclose(final_total, expected_total, rtol=1e-4)
        assert_shape(ys, (batch_size,))


class TestControlFlowEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_cond_with_multiple_operands(self):
        """Cond with multiple operands."""

        def true_fn(x, y):
            return x + y

        def false_fn(x, y):
            return x - y

        jax_x = jnp.array(10.0, dtype=jnp.float32)
        jax_y = jnp.array(3.0, dtype=jnp.float32)

        x = tensor_from_jax(jax_x)
        y = tensor_from_jax(jax_y)

        pred = nb.constant(True, dtype=nb.DType.bool)
        result = cond(pred, true_fn, false_fn, x, y)

        assert_allclose(result, jnp.array(13.0, dtype=jnp.float32))

    def test_while_loop_early_exit(self):
        """While loop with early exit condition."""

        def cond_fn(state):
            i, val = state
            limit = nb.constant(100, dtype=nb.DType.int32)
            threshold = nb.constant(50.0, dtype=nb.DType.float32)
            return nb.logical_and(nb.less(i, limit), nb.less(val, threshold))

        def body_fn(state):
            i, val = state
            return (i + 1, val + 5.0)

        init_state = (
            nb.constant(0, dtype=nb.DType.int32),
            nb.constant(0.0, dtype=nb.DType.float32),
        )
        result_i, result_val = while_loop(cond_fn, body_fn, init_state)

        assert_allclose(result_i, jnp.array(10, dtype=jnp.int32))
        assert_allclose(result_val, jnp.array(50.0, dtype=jnp.float32))


__all__ = [
    "TestWhereOp",
    "TestCondOp",
    "TestWhileLoopOp",
    "TestScanOp",
    "TestControlFlowEdgeCases",
]
