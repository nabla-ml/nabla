# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""End-to-end training proof for adapted max.nn modules."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng


max_nn = pytest.importorskip("max.nn")


class AdaptedMaxMLP(nb.nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        adapted = nb.nn.adapt_max_nn_core(max_nn)
        Linear = adapted["Linear"]
        self.fc1 = Linear(in_features, hidden_features)
        self.fc2 = Linear(hidden_features, out_features)

    def forward(self, x: nb.Tensor) -> nb.Tensor:
        hidden = nb.relu(self.fc1(x))
        return self.fc2(hidden)


class TestAdaptedMaxTraining:
    def test_adapted_max_mlp_trains_with_stateful_nabla_optimizer(self):
        rng = make_rng(20260216)

        x_np = rng.normal(size=(128, 5)).astype(np.float32)
        w1_np = rng.normal(size=(5, 16)).astype(np.float32)
        b1_np = rng.normal(size=(1, 16)).astype(np.float32)
        w2_np = rng.normal(size=(16, 3)).astype(np.float32)
        b2_np = rng.normal(size=(1, 3)).astype(np.float32)

        y_np = np.maximum(x_np @ w1_np + b1_np, 0.0) @ w2_np + b2_np

        x = nb.Tensor.from_dlpack(x_np)
        y = nb.Tensor.from_dlpack(y_np)

        model = AdaptedMaxMLP(in_features=5, hidden_features=16, out_features=3)
        optimizer = nb.nn.optim.AdamW(model, lr=2e-2, weight_decay=0.0)

        def loss_fn(m):
            return nb.nn.functional.mse_loss(m(x), y)

        start_loss = float(loss_fn(model).to_numpy())

        current_model = model
        for _ in range(30):
            _, grads = nb.value_and_grad(loss_fn, argnums=0, realize=False)(current_model)
            current_model = optimizer.step(grads)

        end_loss = float(loss_fn(current_model).to_numpy())

        assert end_loss < start_loss
        assert end_loss < (0.6 * start_loss)
