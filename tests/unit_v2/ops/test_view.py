# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
import numpy as np
import nabla
from nabla.core import Tensor
from nabla.transforms.vmap import vmap
from nabla.ops.view import reshape, swap_axes, unsqueeze, squeeze, broadcast_to, concatenate, stack

class TestViewOps:
    def test_reshape(self):
        x_np = np.random.randn(2, 3).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        y = reshape(x, (6,))
        np.testing.assert_allclose(y.to_numpy(), x_np.reshape(6))
        
    def test_swap_axes(self):
        x_np = np.random.randn(2, 3).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        y = swap_axes(x, 0, 1)
        np.testing.assert_allclose(y.to_numpy(), x_np.transpose(1, 0))
        
    def test_unsqueeze(self):
        x_np = np.random.randn(2).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        y = unsqueeze(x, axis=0)
        np.testing.assert_allclose(y.to_numpy(), np.expand_dims(x_np, 0))
        
    def test_squeeze(self):
        x_np = np.random.randn(1, 2).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        y = squeeze(x, axis=0)
        np.testing.assert_allclose(y.to_numpy(), np.squeeze(x_np, 0))
        
    def test_broadcast_to(self):
        x_np = np.random.randn(2, 1).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        y = broadcast_to(x, (2, 3))
        np.testing.assert_allclose(y.to_numpy(), np.broadcast_to(x_np, (2, 3)))

    def test_concatenate(self):
        x_np = np.random.randn(2, 2).astype(np.float32)
        y_np = np.random.randn(2, 2).astype(np.float32)
        x, y = Tensor.from_dlpack(x_np), Tensor.from_dlpack(y_np)
        res = concatenate([x, y], axis=0)
        np.testing.assert_allclose(res.to_numpy(), np.concatenate([x_np, y_np], axis=0))

    def test_stack(self):
        x_np = np.random.randn(2, 2).astype(np.float32)
        y_np = np.random.randn(2, 2).astype(np.float32)
        x, y = Tensor.from_dlpack(x_np), Tensor.from_dlpack(y_np)
        res = stack([x, y], axis=0)
        np.testing.assert_allclose(res.to_numpy(), np.stack([x_np, y_np], axis=0))

    def test_vmap_reshape(self):
        B, M, N = 2, 3, 4
        x_np = np.random.randn(B, M, N).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        vmapped = vmap(lambda t: reshape(t, (M*N,)))
        y = vmapped(x)
        np.testing.assert_allclose(y.to_numpy(), x_np.reshape(B, M*N))
