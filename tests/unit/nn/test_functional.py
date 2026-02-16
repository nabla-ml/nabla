# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb


def test_linear_forward_values_and_lazy_realization():
    rng = np.random.default_rng(11)
    x_np = rng.normal(size=(5, 4)).astype(np.float32)
    w_np = rng.normal(size=(4, 3)).astype(np.float32)
    b_np = rng.normal(size=(1, 3)).astype(np.float32)

    x = nb.Tensor.from_dlpack(x_np)
    w = nb.Tensor.from_dlpack(w_np)
    b = nb.Tensor.from_dlpack(b_np)

    y = nb.nn.functional.linear(x, w, b)
    assert not y.real
    assert tuple(int(d) for d in y.shape) == (5, 3)

    y_ref = x_np @ w_np + b_np
    nb.testing.assert_allclose(y, y_ref, rtol=1e-5, atol=1e-6)


def test_mse_loss_values_and_lazy_realization():
    rng = np.random.default_rng(17)
    pred_np = rng.normal(size=(8, 2)).astype(np.float32)
    target_np = rng.normal(size=(8, 2)).astype(np.float32)

    pred = nb.Tensor.from_dlpack(pred_np)
    target = nb.Tensor.from_dlpack(target_np)

    loss = nb.nn.functional.mse_loss(pred, target)
    assert not loss.real
    assert tuple(int(d) for d in loss.shape) == tuple()

    loss_ref = np.mean((pred_np - target_np) ** 2)
    nb.testing.assert_allclose(loss, loss_ref, rtol=1e-5, atol=1e-6)


def test_initializers_statistics_and_dtype():
    shape = (512, 256)
    xavier = nb.nn.functional.xavier_normal(shape)
    he = nb.nn.functional.he_normal(shape)

    assert tuple(int(d) for d in xavier.shape) == shape
    assert tuple(int(d) for d in he.shape) == shape
    assert xavier.dtype == nb.DType.float32
    assert he.dtype == nb.DType.float32

    nb.testing.batch_realize(xavier, he)
    x_np = np.asarray(xavier)
    h_np = np.asarray(he)
    x_target = np.sqrt(2.0 / (shape[0] + shape[1]))
    h_target = np.sqrt(2.0 / shape[0])

    assert abs(float(np.mean(x_np))) < 0.02
    assert abs(float(np.mean(h_np))) < 0.02
    assert abs(float(np.std(x_np)) - x_target) < 0.02
    assert abs(float(np.std(h_np)) - h_target) < 0.02


def test_cross_entropy_matches_jax_when_available():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    rng = np.random.default_rng(23)
    logits_np = rng.normal(size=(6, 4)).astype(np.float32)
    targets_idx = rng.integers(0, 4, size=(6,))
    targets_np = np.eye(4, dtype=np.float32)[targets_idx]

    logits = nb.Tensor.from_dlpack(logits_np)
    targets = nb.Tensor.from_dlpack(targets_np)

    loss_nb = nb.nn.functional.cross_entropy_loss(logits, targets)
    assert not loss_nb.real

    log_probs_jax = jax.nn.log_softmax(jnp.asarray(logits_np), axis=-1)
    loss_jax = -jnp.sum(jnp.asarray(targets_np) * log_probs_jax) / logits_np.shape[0]

    nb.testing.assert_allclose(loss_nb, loss_jax, rtol=1e-5, atol=1e-6)


def test_linear_matches_torch_when_available():
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(29)
    x_np = rng.normal(size=(7, 3)).astype(np.float32)
    w_np = rng.normal(size=(3, 5)).astype(np.float32)
    b_np = rng.normal(size=(1, 5)).astype(np.float32)

    x = nb.Tensor.from_dlpack(x_np)
    w = nb.Tensor.from_dlpack(w_np)
    b = nb.Tensor.from_dlpack(b_np)
    y_nb = nb.nn.functional.linear(x, w, b)
    assert not y_nb.real

    y_torch = torch.from_numpy(x_np) @ torch.from_numpy(w_np) + torch.from_numpy(b_np)
    nb.testing.assert_allclose(y_nb, y_torch, rtol=1e-5, atol=1e-6)
