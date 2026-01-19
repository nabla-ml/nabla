# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
import numpy as np
import nabla
from nabla import vmap, add
from nabla.core import Tensor

def test_nested_vmap():
    vmapped = vmap(vmap(lambda x, y: x + y))
    B1, B2, Shape = 2, 3, (4, 4)
    x_np = np.random.randn(B1, B2, *Shape).astype(np.float32)
    y_np = np.random.randn(B1, B2, *Shape).astype(np.float32)
    x, y = Tensor.from_dlpack(x_np), Tensor.from_dlpack(y_np)
    res = vmapped(x, y)
    np.testing.assert_allclose(res.to_numpy(), x_np + y_np, rtol=1e-5)

def test_vmap_broadcasting_with_batch():
    vmapped = vmap(lambda x, y: x + y, in_axes=(0, None))
    B, M = 5, 10
    x_np = np.random.randn(B, M).astype(np.float32)
    y_np = np.random.randn(M).astype(np.float32)
    x, y = Tensor.from_dlpack(x_np), Tensor.from_dlpack(y_np)
    res = vmapped(x, y)
    np.testing.assert_allclose(res.to_numpy(), x_np + y_np, rtol=1e-5)
