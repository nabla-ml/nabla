# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import unittest

import numpy as np
from max.dtype import DType

from nabla import ops
from nabla.core import tensor
from nabla.ops import control_flow


class TestControlFlow(unittest.TestCase):
    def setUp(self):

        pass

    def test_where(self):
        cond = tensor.Tensor.from_dlpack(np.array([True, False, True], dtype=bool))
        x = tensor.Tensor.from_dlpack(np.array([1, 2, 3], dtype=np.float32))
        y = tensor.Tensor.from_dlpack(np.array([10, 20, 30], dtype=np.float32))

        res = control_flow.where(cond, x, y)
        res_np = res.to_numpy()

        expected = np.array([1, 20, 3], dtype=np.float32)
        np.testing.assert_array_equal(res_np, expected)

    def test_cond_basic(self):
        def true_fn(x):
            return x + 1.0

        def false_fn(x):
            return x - 1.0

        x = tensor.Tensor.constant(5.0, dtype=DType.float32)

        pred_true = tensor.Tensor.constant(True, dtype=DType.bool)
        res_true = control_flow.cond(pred_true, true_fn, false_fn, x)
        self.assertEqual(res_true.to_numpy(), 6.0)

        pred_false = tensor.Tensor.constant(False, dtype=DType.bool)
        res_false = control_flow.cond(pred_false, true_fn, false_fn, x)
        self.assertEqual(res_false.to_numpy(), 4.0)

    def test_cond_pytree(self):
        def true_fn(x):
            return (x, x + 1.0)

        def false_fn(x):
            return (x, x - 1.0)

        x = tensor.Tensor.constant(5.0, dtype=DType.float32)
        pred = tensor.Tensor.constant(True, dtype=DType.bool)

        res = control_flow.cond(pred, true_fn, false_fn, x)
        self.assertTrue(isinstance(res, tuple))
        self.assertEqual(res[0].to_numpy(), 5.0)
        self.assertEqual(res[1].to_numpy(), 6.0)

    def test_while_loop_basic(self):

        def cond_fn(i):
            limit = tensor.Tensor.constant(10, dtype=DType.int32)
            return ops.comparison.less(i, limit)

        def body_fn(i):
            return i + 1

        i_init = tensor.Tensor.constant(0, dtype=DType.int32)

        res = control_flow.while_loop(cond_fn, body_fn, i_init)
        self.assertEqual(res.to_numpy(), 10)

    def test_scan_cumsum(self):

        def f(carry, x):

            new_carry = carry + x
            return new_carry, new_carry

        xs = tensor.Tensor.from_dlpack(np.array([1, 2, 3, 4], dtype=np.float32))
        init = tensor.Tensor.constant(0.0, dtype=DType.float32)

        final_carry, stacked_ys = control_flow.scan(f, init, xs)

        self.assertEqual(final_carry.to_numpy(), 10.0)
        np.testing.assert_array_equal(
            stacked_ys.to_numpy(), np.array([1, 3, 6, 10], dtype=np.float32)
        )

    def test_scan_nested(self):

        pass


if __name__ == "__main__":
    unittest.main()
