# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for Module forward correctness — validated against PyTorch."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng, nb_from_np


# ===----------------------------------------------------------------------=== #
# Linear forward
# ===----------------------------------------------------------------------=== #


class TestLinearForward:
    """Linear module forward pass compared against PyTorch."""

    @pytest.fixture()
    def _weights(self):
        rng = make_rng(303)
        w_np = rng.normal(size=(4, 3)).astype(np.float32)
        b_np = rng.normal(size=(1, 3)).astype(np.float32)
        x_np = rng.normal(size=(9, 4)).astype(np.float32)
        return x_np, w_np, b_np

    def test_linear_forward_vs_pytorch(self, _weights):
        torch = pytest.importorskip("torch")
        x_np, w_np, b_np = _weights

        model = nb.nn.Linear(4, 3)
        model.weight = nb_from_np(w_np, requires_grad=True)
        model.bias = nb_from_np(b_np, requires_grad=True)

        y_nb = model(nb.Tensor.from_dlpack(x_np))
        y_pt = torch.from_numpy(x_np) @ torch.from_numpy(w_np) + torch.from_numpy(b_np)

        nb.testing.assert_allclose(y_nb, y_pt, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("batch,in_f,out_f", [(5, 4, 3), (1, 8, 2), (32, 16, 16)])
    def test_linear_shapes(self, batch, in_f, out_f):
        model = nb.nn.Linear(in_f, out_f)
        x = nb.Tensor.from_dlpack(make_rng(42).normal(size=(batch, in_f)).astype(np.float32))
        y = model(x)
        assert tuple(int(d) for d in y.shape) == (batch, out_f)

    def test_linear_no_bias(self):
        torch = pytest.importorskip("torch")
        rng = make_rng(50)
        w_np = rng.normal(size=(4, 3)).astype(np.float32)
        x_np = rng.normal(size=(5, 4)).astype(np.float32)

        model = nb.nn.Linear(4, 3, bias=False)
        model.weight = nb_from_np(w_np, requires_grad=True)

        y_nb = model(nb.Tensor.from_dlpack(x_np))
        y_pt = torch.from_numpy(x_np) @ torch.from_numpy(w_np)

        nb.testing.assert_allclose(y_nb, y_pt, rtol=1e-5, atol=1e-6)
        assert model.bias is None


# ===----------------------------------------------------------------------=== #
# ReLU module forward
# ===----------------------------------------------------------------------=== #


class TestReLUForward:
    def test_relu_module_vs_pytorch(self):
        torch = pytest.importorskip("torch")
        rng = make_rng(60)
        x_np = rng.normal(size=(10, 5)).astype(np.float32)

        relu = nb.nn.ReLU()
        y_nb = relu(nb.Tensor.from_dlpack(x_np))
        y_pt = torch.nn.functional.relu(torch.from_numpy(x_np))

        nb.testing.assert_allclose(y_nb, y_pt, rtol=1e-6, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# GELU module forward
# ===----------------------------------------------------------------------=== #


class TestGELUForward:
    def test_gelu_module_vs_pytorch(self):
        torch = pytest.importorskip("torch")
        rng = make_rng(70)
        x_np = rng.normal(size=(10, 5)).astype(np.float32)

        gelu = nb.nn.GELU()
        y_nb = gelu(nb.Tensor.from_dlpack(x_np))
        y_pt = torch.nn.functional.gelu(torch.from_numpy(x_np))

        nb.testing.assert_allclose(y_nb, y_pt, rtol=1e-4, atol=1e-5)


# ===----------------------------------------------------------------------=== #
# Sequential forward
# ===----------------------------------------------------------------------=== #


class TestSequentialForward:
    def test_sequential_linear_relu_linear_vs_pytorch(self):
        torch = pytest.importorskip("torch")
        rng = make_rng(80)

        x_np = rng.normal(size=(8, 4)).astype(np.float32)
        w1_np = rng.normal(size=(4, 6)).astype(np.float32)
        b1_np = rng.normal(size=(1, 6)).astype(np.float32)
        w2_np = rng.normal(size=(6, 3)).astype(np.float32)
        b2_np = rng.normal(size=(1, 3)).astype(np.float32)

        model = nb.nn.Sequential(
            nb.nn.Linear(4, 6),
            nb.nn.ReLU(),
            nb.nn.Linear(6, 3),
        )
        getattr(model, "0").weight = nb_from_np(w1_np, requires_grad=True)
        getattr(model, "0").bias = nb_from_np(b1_np, requires_grad=True)
        getattr(model, "2").weight = nb_from_np(w2_np, requires_grad=True)
        getattr(model, "2").bias = nb_from_np(b2_np, requires_grad=True)

        y_nb = model(nb.Tensor.from_dlpack(x_np))

        # PyTorch reference (note: nabla uses y = x @ W + b, not F.linear)
        h = torch.nn.functional.relu(
            torch.from_numpy(x_np) @ torch.from_numpy(w1_np) + torch.from_numpy(b1_np)
        )
        y_pt = h @ torch.from_numpy(w2_np) + torch.from_numpy(b2_np)

        nb.testing.assert_allclose(y_nb, y_pt, rtol=1e-5, atol=1e-6)
