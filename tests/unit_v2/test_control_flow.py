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

import numpy as np
import pytest

import nabla as nb
from nabla.ops.control_flow import cond, scan, where, while_loop


from . common import (
    MESH_CONFIGS,
    DeviceMesh,
)

from tests.conftest import (
    assert_allclose,
    assert_shape,
    make_array,
    replicated,
    shard_on_axis,
    tensor_from_numpy,
)


class TestWhereOp:
    """Test where operation: element-wise conditional selection."""

    def test_where_basic(self):
        """Basic where without sharding."""
        cond_np = np.array([True, False, True, False], dtype=bool)
        x_np = make_array(4, seed=42)
        y_np = make_array(4, seed=43)

        cond_t = tensor_from_numpy(cond_np)
        x = tensor_from_numpy(x_np)
        y = tensor_from_numpy(y_np)

        result = where(cond_t, x, y)
        expected = np.where(cond_np, x_np, y_np)

        assert_shape(result, (4,))
        assert_allclose(result, expected)

    def test_where_2d(self):
        """Where on 2D tensors."""
        cond_np = np.random.choice([True, False], size=(4, 8))
        x_np = make_array(4, 8, seed=42)
        y_np = make_array(4, 8, seed=43)

        cond_t = tensor_from_numpy(cond_np)
        x = tensor_from_numpy(x_np)
        y = tensor_from_numpy(y_np)

        result = where(cond_t, x, y)
        expected = np.where(cond_np, x_np, y_np)

        assert_shape(result, (4, 8))
        assert_allclose(result, expected)

    def test_where_sharded_inputs(self, mesh_1d):
        """Where with sharded x and y tensors."""
        cond_np = np.random.choice([True, False], size=(8, 4))
        x_np = make_array(8, 4, seed=42)
        y_np = make_array(8, 4, seed=43)

        cond_t = tensor_from_numpy(cond_np)
        x = tensor_from_numpy(x_np)
        y = tensor_from_numpy(y_np)

        x_sharded = shard_on_axis(x, mesh_1d, axis=0)
        y_sharded = shard_on_axis(y, mesh_1d, axis=0)

        result = where(cond_t, x_sharded, y_sharded)
        expected = np.where(cond_np, x_np, y_np)

        assert_shape(result, (8, 4))
        assert_allclose(result, expected)

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_where_vmap(self, batch_size):
        """vmap(where) with batched inputs."""
        cond_np = np.random.choice([True, False], size=(batch_size, 8))
        x_np = make_array(batch_size, 8, seed=42)
        y_np = make_array(batch_size, 8, seed=43)

        cond_t = tensor_from_numpy(cond_np)
        x = tensor_from_numpy(x_np)
        y = tensor_from_numpy(y_np)

        def fn(c, x_val, y_val):
            return where(c, x_val, y_val)

        result = nb.vmap(fn)(cond_t, x, y)
        expected = np.where(cond_np, x_np, y_np)

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

        x_np = np.array(5.0, dtype=np.float32)
        x = tensor_from_numpy(x_np)

        pred = nb.constant(True, dtype=nb.DType.bool)
        result = cond(pred, true_fn, false_fn, x)

        assert_allclose(result, np.array(6.0, dtype=np.float32))

    def test_cond_basic_false(self):
        """Cond with false predicate."""

        def true_fn(x):
            return x + 1.0

        def false_fn(x):
            return x - 1.0

        x_np = np.array(5.0, dtype=np.float32)
        x = tensor_from_numpy(x_np)

        pred = nb.constant(False, dtype=nb.DType.bool)
        result = cond(pred, true_fn, false_fn, x)

        assert_allclose(result, np.array(4.0, dtype=np.float32))

    def test_cond_pytree_output(self):
        """Cond returning tuple (pytree)."""

        def true_fn(x):
            return (x, x + 1.0)

        def false_fn(x):
            return (x, x - 1.0)

        x_np = np.array(5.0, dtype=np.float32)
        x = tensor_from_numpy(x_np)

        pred = nb.constant(True, dtype=nb.DType.bool)
        result = cond(pred, true_fn, false_fn, x)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert_allclose(result[0], np.array(5.0, dtype=np.float32))
        assert_allclose(result[1], np.array(6.0, dtype=np.float32))

    def test_cond_sharding_in_branch(self, mesh_1d):
        """Cond with sharding applied inside branch."""

        def true_fn(x):
            x_sharded = shard_on_axis(x, mesh_1d, axis=0)
            return nb.reduce_sum(x_sharded, axis=0)

        def false_fn(x):
            return nb.reduce_sum(x, axis=0)

        x_np = make_array(8, 4, seed=42)
        x = tensor_from_numpy(x_np)

        pred = nb.constant(True, dtype=nb.DType.bool)
        result = cond(pred, true_fn, false_fn, x)

        expected = np.sum(x_np, axis=0)
        assert_shape(result, (4,))
        assert_allclose(result, expected)
        """Cond with sharding applied inside branch."""

        def true_fn(x):
            x_sharded = shard_on_axis(x, mesh_1d, axis=0)
            return nb.reduce_sum(x_sharded, axis=0)

        def false_fn(x):
            return nb.reduce_sum(x, axis=0)

        x_np = make_array(8, 4, seed=42)
        x = tensor_from_numpy(x_np)

        pred = nb.constant(True, dtype=nb.DType.bool)
        result = cond(pred, true_fn, false_fn, x)

        expected = np.sum(x_np, axis=0)
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

        assert_allclose(result, np.array(10, dtype=np.int32))

    def test_while_loop_accumulator(self):
        """While loop with accumulator (sum 1+2+...+10)."""

        def cond_fn(state):
            i, _ = state
            limit = nb.constant(11, dtype=nb.DType.int32)
            return nb.less(i, limit)

        def body_fn(state):
            i, acc = state
            return (i + 1, acc + i)

        init_state = (nb.constant(1, dtype=nb.DType.int32), nb.constant(0, dtype=nb.DType.int32))
        result_i, result_acc = while_loop(cond_fn, body_fn, init_state)

        assert_allclose(result_i, np.array(11, dtype=np.int32))
        assert_allclose(result_acc, np.array(55, dtype=np.int32))

    def test_while_loop_sharded_state(self, mesh_1d):
        """While loop with sharded loop-carried state."""

        def cond_fn(state):
            i, _ = state
            limit = nb.constant(5, dtype=nb.DType.int32)
            return nb.less(i, limit)

        def body_fn(state):
            i, x = state
            return (i + 1, x + 1.0)

        x_np = make_array(8, 4, seed=42)
        x = tensor_from_numpy(x_np)
        x_sharded = shard_on_axis(x, mesh_1d, axis=0)

        init_state = (nb.constant(0, dtype=nb.DType.int32), x_sharded)
        result_i, result_x = while_loop(cond_fn, body_fn, init_state)

        expected_x = x_np + 5.0
        assert_allclose(result_i, np.array(5, dtype=np.int32))
        assert_shape(result_x, (8, 4))
        assert_allclose(result_x, expected_x)


class TestScanOp:
    """Test scan operation: sequential iteration with outputs."""

    def test_scan_cumsum(self):
        """Scan implementing cumulative sum."""

        def f(carry, x):
            new_carry = carry + x
            return new_carry, new_carry

        xs_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        xs = tensor_from_numpy(xs_np)
        init = nb.constant(0.0, dtype=nb.DType.float32)

        final_carry, ys = scan(f, init, xs)

        expected_carry = 10.0
        expected_ys = np.array([1.0, 3.0, 6.0, 10.0], dtype=np.float32)

        assert_allclose(final_carry, np.array(expected_carry, dtype=np.float32))
        assert_shape(ys, (4,))
        assert_allclose(ys, expected_ys)

    def test_scan_cumulative_product(self):
        """Scan implementing cumulative product."""

        def f(carry, x):
            new_carry = carry * x
            return new_carry, new_carry

        xs_np = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        xs = tensor_from_numpy(xs_np)
        init = nb.constant(1.0, dtype=nb.DType.float32)

        final_carry, ys = scan(f, init, xs)

        expected_ys = np.array([2.0, 6.0, 24.0], dtype=np.float32)

        assert_allclose(final_carry, np.array(24.0, dtype=np.float32))
        assert_allclose(ys, expected_ys)

    def test_scan_2d_input(self):
        """Scan over 2D tensor accumulating columns."""

        def f(carry, x):
            # x is a row vector (8,), carry is also (8,)
            s = carry + x  # Element-wise add
            return s, s

        xs_np = make_array(4, 8, seed=42)
        xs = tensor_from_numpy(xs_np)
        init = nb.zeros((8,))

        final_carry, ys = scan(f, init, xs)

        # Expected: cumulative sum of rows
        expected_final = np.sum(xs_np, axis=0)
        assert_shape(final_carry, (8,))
        assert_shape(ys, (4, 8))
        assert_allclose(final_carry, expected_final, rtol=1e-4)

    def test_scan_sharded_xs(self, mesh_1d):
        """Scan with sharded xs sequence."""

        def f(carry, x):
            new_carry = carry + x
            return new_carry, new_carry

        xs_np = make_array(8, 4, seed=42)
        xs = tensor_from_numpy(xs_np)
        xs_sharded = shard_on_axis(xs, mesh_1d, axis=1)

        init = nb.zeros((4,))
        init_sharded = shard_on_axis(init, mesh_1d, axis=0)

        final_carry, ys = scan(f, init_sharded, xs_sharded)

        expected_final = np.sum(xs_np, axis=0)
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

        xs_np = make_array(batch_size, seed=42)
        xs = tensor_from_numpy(xs_np)

        init_count = nb.constant(0, dtype=nb.DType.int32)
        init_total = nb.constant(0.0, dtype=nb.DType.float32)

        (final_count, final_total), ys = scan(f, (init_count, init_total), xs)

        expected_total = np.sum(xs_np)
        assert_allclose(final_count, np.array(batch_size, dtype=np.int32))
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

        x_np = np.array(10.0, dtype=np.float32)
        y_np = np.array(3.0, dtype=np.float32)

        x = tensor_from_numpy(x_np)
        y = tensor_from_numpy(y_np)

        pred = nb.constant(True, dtype=nb.DType.bool)
        result = cond(pred, true_fn, false_fn, x, y)

        assert_allclose(result, np.array(13.0, dtype=np.float32))

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

        init_state = (nb.constant(0, dtype=nb.DType.int32), nb.constant(0.0, dtype=nb.DType.float32))
        result_i, result_val = while_loop(cond_fn, body_fn, init_state)

        assert_allclose(result_i, np.array(10, dtype=np.int32))
        assert_allclose(result_val, np.array(50.0, dtype=np.float32))


__all__ = [
    "TestWhereOp",
    "TestCondOp",
    "TestWhileLoopOp",
    "TestScanOp",
    "TestControlFlowEdgeCases",
]
