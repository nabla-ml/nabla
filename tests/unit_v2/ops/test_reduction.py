# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
import numpy as np
import nabla
from nabla import reduce_sum, mean

def test_reduce_sum_last_axis(op_verifier):
    op_verifier.verify(op_fn=reduce_sum, numpy_fn=np.sum, input_shapes=[(4, 8, 4)], axis=-1)

def test_reduce_sum_axis_0(op_verifier):
    op_verifier.verify(op_fn=reduce_sum, numpy_fn=np.sum, input_shapes=[(4, 8)], axis=0)

def test_reduce_sum_keepdims(op_verifier):
    op_verifier.verify(op_fn=reduce_sum, numpy_fn=np.sum, input_shapes=[(4, 8)], axis=1, keepdims=True)

def test_mean(op_verifier):
    op_verifier.verify(op_fn=mean, numpy_fn=np.mean, input_shapes=[(4, 8)], axis=1)
