# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
import numpy as np
import nabla
from nabla import relu, neg, exp, tanh, sigmoid, abs, softmax

def test_relu(op_verifier):
    op_verifier.verify(op_fn=relu, numpy_fn=lambda x: np.maximum(x, 0), input_shapes=[(8, 8)])

def test_neg(op_verifier):
    op_verifier.verify(op_fn=neg, numpy_fn=np.negative, input_shapes=[(5, 5)])

def test_exp(op_verifier):
    op_verifier.verify(op_fn=exp, numpy_fn=np.exp, input_shapes=[(4, 4)])

def test_tanh(op_verifier):
    op_verifier.verify(op_fn=tanh, numpy_fn=np.tanh, input_shapes=[(4, 4)])

def test_sigmoid(op_verifier):
    sigmoid_np = lambda x: 1 / (1 + np.exp(-x))
    op_verifier.verify(op_fn=sigmoid, numpy_fn=sigmoid_np, input_shapes=[(4, 4)])

def test_abs(op_verifier):
    op_verifier.verify(op_fn=abs, numpy_fn=np.abs, input_shapes=[(4, 4)])

def test_softmax(op_verifier):
    def softmax_np(x, axis=-1):
        ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return ex / np.sum(ex, axis=axis, keepdims=True)
    op_verifier.verify(op_fn=softmax, numpy_fn=softmax_np, input_shapes=[(4, 8)], axis=-1)
