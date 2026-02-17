# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for Module backward / gradient computation — validated against PyTorch."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng, nb_from_np


# ===----------------------------------------------------------------------=== #
# nb.grad with Module input
# ===----------------------------------------------------------------------=== #


class TestGradWithModule:
    """Functional nb.grad over a Module, validated against PyTorch autograd."""

    def test_grad_returns_module_shaped_grads(self):
        """nb.grad(loss_fn)(model) returns a Module-shaped gradient tree."""
        rng = make_rng(101)
        model = nb.nn.Linear(4, 2)
        model.weight = nb_from_np(
            rng.normal(size=(4, 2)).astype(np.float32), requires_grad=True
        )
        model.bias = nb_from_np(
            rng.normal(size=(1, 2)).astype(np.float32), requires_grad=True
        )

        x = nb.Tensor.from_dlpack(rng.normal(size=(12, 4)).astype(np.float32))
        y = nb.Tensor.from_dlpack(rng.normal(size=(12, 2)).astype(np.float32))

        def loss_fn(m):
            return nb.nn.functional.mse_loss(m(x), y)

        grads = nb.grad(loss_fn)(model)
        assert isinstance(grads, nb.nn.Linear)
        assert isinstance(grads.weight, nb.Tensor)
        assert isinstance(grads.bias, nb.Tensor)

    def test_grad_values_vs_pytorch(self):
        """Gradient values from nb.grad match PyTorch autograd."""
        torch = pytest.importorskip("torch")
        rng = make_rng(102)

        w_np = rng.normal(size=(4, 2)).astype(np.float32)
        b_np = rng.normal(size=(1, 2)).astype(np.float32)
        x_np = rng.normal(size=(12, 4)).astype(np.float32)
        y_np = rng.normal(size=(12, 2)).astype(np.float32)

        # Nabla
        model = nb.nn.Linear(4, 2)
        model.weight = nb_from_np(w_np.copy(), requires_grad=True)
        model.bias = nb_from_np(b_np.copy(), requires_grad=True)
        x = nb.Tensor.from_dlpack(x_np)
        y = nb.Tensor.from_dlpack(y_np)

        def loss_fn(m):
            return nb.nn.functional.mse_loss(m(x), y)

        grads = nb.grad(loss_fn)(model)

        # PyTorch reference
        w_pt = torch.tensor(w_np, requires_grad=True)
        b_pt = torch.tensor(b_np, requires_grad=True)
        pred_pt = torch.from_numpy(x_np) @ w_pt + b_pt
        loss_pt = torch.mean((pred_pt - torch.from_numpy(y_np)) ** 2)
        loss_pt.backward()

        nb.testing.assert_allclose(grads.weight, w_pt.grad, rtol=1e-4, atol=1e-5)
        nb.testing.assert_allclose(grads.bias, b_pt.grad, rtol=1e-4, atol=1e-5)


# ===----------------------------------------------------------------------=== #
# value_and_grad + SGD update
# ===----------------------------------------------------------------------=== #


class TestValueAndGradSGD:
    """One-step SGD update using value_and_grad, validated against PyTorch."""

    def test_one_step_sgd_vs_pytorch(self):
        torch = pytest.importorskip("torch")
        rng = make_rng(404)
        lr = 1e-2

        w_np = rng.normal(size=(4, 3)).astype(np.float32)
        b_np = rng.normal(size=(1, 3)).astype(np.float32)
        x_np = rng.normal(size=(10, 4)).astype(np.float32)
        y_np = rng.normal(size=(10, 3)).astype(np.float32)

        # --- Nabla ---
        model = nb.nn.Linear(4, 3)
        model.weight = nb_from_np(w_np.copy(), requires_grad=True)
        model.bias = nb_from_np(b_np.copy(), requires_grad=True)
        x, y = nb.Tensor.from_dlpack(x_np), nb.Tensor.from_dlpack(y_np)

        def loss_fn(m):
            return nb.nn.functional.mse_loss(m(x), y)

        loss, grads = nb.value_and_grad(loss_fn, argnums=0)(model)
        w_new = model.weight - lr * grads.weight
        b_new = model.bias - lr * grads.bias

        # --- PyTorch reference ---
        w_pt = torch.tensor(w_np, requires_grad=True)
        b_pt = torch.tensor(b_np, requires_grad=True)
        pred_pt = torch.from_numpy(x_np) @ w_pt + b_pt
        loss_pt = torch.mean((pred_pt - torch.from_numpy(y_np)) ** 2)
        loss_pt.backward()
        w_ref = w_np - lr * w_pt.grad.numpy()
        b_ref = b_np - lr * b_pt.grad.numpy()

        nb.testing.assert_allclose(w_new, w_ref, rtol=1e-4, atol=1e-5)
        nb.testing.assert_allclose(b_new, b_ref, rtol=1e-4, atol=1e-5)


# ===----------------------------------------------------------------------=== #
# PyTorch-style .backward()
# ===----------------------------------------------------------------------=== #


class TestImperativeBackward:
    """Module.backward() and Module.zero_grad(), validated against PyTorch."""

    def test_backward_matches_pytorch_autograd(self):
        torch = pytest.importorskip("torch")
        rng = make_rng(505)

        w_np = rng.normal(size=(4, 2)).astype(np.float32)
        b_np = rng.normal(size=(1, 2)).astype(np.float32)
        x_np = rng.normal(size=(12, 4)).astype(np.float32)
        y_np = rng.normal(size=(12, 2)).astype(np.float32)

        # --- Nabla imperative ---
        model = nb.nn.Linear(4, 2)
        model.weight = nb_from_np(w_np.copy(), requires_grad=True)
        model.bias = nb_from_np(b_np.copy(), requires_grad=True)
        x, y = nb.Tensor.from_dlpack(x_np), nb.Tensor.from_dlpack(y_np)

        model.zero_grad()
        loss = nb.nn.functional.mse_loss(model(x), y)
        loss.backward()

        assert model.weight.grad is not None
        assert model.bias.grad is not None

        # --- PyTorch reference ---
        w_pt = torch.tensor(w_np, requires_grad=True)
        b_pt = torch.tensor(b_np, requires_grad=True)
        pred_pt = torch.from_numpy(x_np) @ w_pt + b_pt
        loss_pt = torch.mean((pred_pt - torch.from_numpy(y_np)) ** 2)
        loss_pt.backward()

        nb.testing.assert_allclose(model.weight.grad, w_pt.grad, rtol=1e-4, atol=1e-5)
        nb.testing.assert_allclose(model.bias.grad, b_pt.grad, rtol=1e-4, atol=1e-5)

    def test_backward_and_functional_grad_agree(self):
        """Imperative .backward() and functional nb.grad give same results."""
        torch = pytest.importorskip("torch")
        rng = make_rng(506)

        w_np = rng.normal(size=(4, 2)).astype(np.float32)
        b_np = rng.normal(size=(1, 2)).astype(np.float32)
        x_np = rng.normal(size=(12, 4)).astype(np.float32)
        y_np = rng.normal(size=(12, 2)).astype(np.float32)

        # Imperative path
        model = nb.nn.Linear(4, 2)
        model.weight = nb_from_np(w_np.copy(), requires_grad=True)
        model.bias = nb_from_np(b_np.copy(), requires_grad=True)
        x, y = nb.Tensor.from_dlpack(x_np), nb.Tensor.from_dlpack(y_np)

        model.zero_grad()
        loss = nb.nn.functional.mse_loss(model(x), y)
        loss.backward()
        gw_back = model.weight.grad
        gb_back = model.bias.grad

        # Functional path (fresh model with same weights)
        model2 = nb.nn.Linear(4, 2)
        model2.weight = nb_from_np(w_np.copy(), requires_grad=True)
        model2.bias = nb_from_np(b_np.copy(), requires_grad=True)

        def loss_fn(m):
            return nb.nn.functional.mse_loss(m(x), y)

        grads_fun = nb.grad(loss_fn)(model2)

        nb.testing.assert_allclose(gw_back, grads_fun.weight, rtol=1e-6, atol=1e-6)
        nb.testing.assert_allclose(gb_back, grads_fun.bias, rtol=1e-6, atol=1e-6)

    def test_zero_grad_clears_gradients(self):
        rng = make_rng(510)
        model = nb.nn.Linear(4, 2)
        x = nb.Tensor.from_dlpack(rng.normal(size=(6, 4)).astype(np.float32))
        y = nb.Tensor.from_dlpack(rng.normal(size=(6, 2)).astype(np.float32))

        loss = nb.nn.functional.mse_loss(model(x), y)
        loss.backward()
        assert model.weight.grad is not None

        model.zero_grad()
        assert model.weight.grad is None
        assert model.bias.grad is None
