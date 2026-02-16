# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for Module execution policies (auto-realize forward/backward)."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng


# ===----------------------------------------------------------------------=== #
# Auto-realize toplevel forward
# ===----------------------------------------------------------------------=== #


class TestAutoRealizeForward:
    def test_auto_realize_produces_real_output(self):
        prev = nb.nn.Module.get_execution_policy()
        nb.nn.Module.set_execution_policy(auto_realize_toplevel_forward=True)
        try:
            rng = make_rng(707)
            x_np = rng.normal(size=(4, 3)).astype(np.float32)
            model = nb.nn.Sequential(
                nb.nn.Linear(3, 5),
                nb.nn.ReLU(),
                nb.nn.Linear(5, 2),
            )
            nb.realize_all(*list(model.parameters()))

            out = model(nb.Tensor.from_dlpack(x_np))
            assert out.real
        finally:
            nb.nn.Module.set_execution_policy(**prev)


# ===----------------------------------------------------------------------=== #
# Auto-realize backward grads
# ===----------------------------------------------------------------------=== #


class TestAutoRealizeBackward:
    def test_auto_realize_grads_are_real(self):
        prev = nb.nn.Module.get_execution_policy()
        nb.nn.Module.set_execution_policy(auto_realize_backward_grads=True)
        try:
            rng = make_rng(808)
            x_np = rng.normal(size=(16, 4)).astype(np.float32)
            y_np = rng.normal(size=(16, 2)).astype(np.float32)

            model = nb.nn.Linear(4, 2)
            x = nb.Tensor.from_dlpack(x_np)
            y = nb.Tensor.from_dlpack(y_np)

            model.zero_grad()
            loss = nb.nn.functional.mse_loss(model(x), y)
            model.backward(loss)

            grads = [p.grad for p in model.parameters()]
            assert all(g is not None for g in grads)
            assert all(g.real for g in grads if g is not None)
        finally:
            nb.nn.Module.set_execution_policy(**prev)

    def test_policy_roundtrips(self):
        prev = nb.nn.Module.get_execution_policy()
        try:
            nb.nn.Module.set_execution_policy(
                auto_realize_toplevel_forward=True,
                auto_realize_backward_grads=True,
            )
            pol = nb.nn.Module.get_execution_policy()
            assert pol["auto_realize_toplevel_forward"] is True
            assert pol["auto_realize_backward_grads"] is True

            nb.nn.Module.set_execution_policy(
                auto_realize_toplevel_forward=False,
                auto_realize_backward_grads=False,
            )
            pol2 = nb.nn.Module.get_execution_policy()
            assert pol2["auto_realize_toplevel_forward"] is False
            assert pol2["auto_realize_backward_grads"] is False
        finally:
            nb.nn.Module.set_execution_policy(**prev)
