# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
import numpy as np
import nabla
from nabla import add, sub, mul, div, matmul

def test_add(op_verifier):
    op_verifier.verify(op_fn=add, numpy_fn=np.add, input_shapes=[(4, 4), (4, 4)])

def test_add_broadcast(op_verifier):
    op_verifier.verify(op_fn=add, numpy_fn=np.add, input_shapes=[(4, 4), (1, 4)])

def test_sub(op_verifier):
    op_verifier.verify(op_fn=sub, numpy_fn=np.subtract, input_shapes=[(128,), (128,)])

def test_mul(op_verifier):
    op_verifier.verify(op_fn=mul, numpy_fn=np.multiply, input_shapes=[(2, 3), (2, 3)])

def test_div(op_verifier):
    op_verifier.verify(
        op_fn=div,
        numpy_fn=np.divide,
        input_shapes=[(10, 10), (10, 10)],
        input_data_fn=lambda s: np.abs(np.random.standard_normal(s)) + 0.1
    )

def test_matmul(op_verifier):
    op_verifier.verify(op_fn=matmul, numpy_fn=np.matmul, input_shapes=[(4, 8), (8, 4)])
