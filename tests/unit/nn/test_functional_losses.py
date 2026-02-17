# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for loss functions in nabla.nn.functional — validated against JAX."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng


# ===----------------------------------------------------------------------=== #
# MSE loss
# ===----------------------------------------------------------------------=== #


class TestMSELoss:
    @pytest.mark.parametrize("shape", [(8, 2), (4, 5), (1, 10)])
    def test_mse_loss_vs_jax(self, shape):
        jnp = pytest.importorskip("jax.numpy")

        rng = make_rng(17)
        pred_np = rng.normal(size=shape).astype(np.float32)
        target_np = rng.normal(size=shape).astype(np.float32)

        loss_nb = nb.nn.functional.mse_loss(
            nb.Tensor.from_dlpack(pred_np),
            nb.Tensor.from_dlpack(target_np),
        )
        loss_jax = jnp.mean((jnp.asarray(pred_np) - jnp.asarray(target_np)) ** 2)

        nb.testing.assert_allclose(loss_nb, loss_jax, rtol=1e-5, atol=1e-6)

    def test_mse_loss_is_scalar(self):
        rng = make_rng(18)
        pred = nb.Tensor.from_dlpack(rng.normal(size=(4, 2)).astype(np.float32))
        target = nb.Tensor.from_dlpack(rng.normal(size=(4, 2)).astype(np.float32))
        loss = nb.nn.functional.mse_loss(pred, target)
        assert tuple(int(d) for d in loss.shape) == ()

    def test_mse_loss_grad_vs_jax(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        rng = make_rng(19)
        pred_np = rng.normal(size=(6, 3)).astype(np.float32)
        target_np = rng.normal(size=(6, 3)).astype(np.float32)
        target = nb.Tensor.from_dlpack(target_np)

        def nb_loss(p):
            return nb.nn.functional.mse_loss(p, target)

        g_nb = nb.grad(nb_loss)(nb.Tensor.from_dlpack(pred_np).requires_grad_(True))

        def jax_loss(p):
            return jnp.mean((p - jnp.asarray(target_np)) ** 2)

        g_jax = jax.grad(jax_loss)(jnp.asarray(pred_np))
        nb.testing.assert_allclose(g_nb, g_jax, rtol=1e-5, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# Cross-entropy loss
# ===----------------------------------------------------------------------=== #


class TestCrossEntropyLoss:
    def test_cross_entropy_vs_jax(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        rng = make_rng(23)
        logits_np = rng.normal(size=(6, 4)).astype(np.float32)
        targets_idx = rng.integers(0, 4, size=(6,))
        targets_np = np.eye(4, dtype=np.float32)[targets_idx]

        loss_nb = nb.nn.functional.cross_entropy_loss(
            nb.Tensor.from_dlpack(logits_np),
            nb.Tensor.from_dlpack(targets_np),
        )

        log_probs_jax = jax.nn.log_softmax(jnp.asarray(logits_np), axis=-1)
        loss_jax = (
            -jnp.sum(jnp.asarray(targets_np) * log_probs_jax) / logits_np.shape[0]
        )

        nb.testing.assert_allclose(loss_nb, loss_jax, rtol=1e-5, atol=1e-6)

    def test_cross_entropy_grad_vs_jax(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        rng = make_rng(24)
        logits_np = rng.normal(size=(8, 5)).astype(np.float32)
        targets_idx = rng.integers(0, 5, size=(8,))
        targets_np = np.eye(5, dtype=np.float32)[targets_idx]
        targets = nb.Tensor.from_dlpack(targets_np)

        def nb_loss(logits):
            return nb.nn.functional.cross_entropy_loss(logits, targets)

        g_nb = nb.grad(nb_loss)(nb.Tensor.from_dlpack(logits_np).requires_grad_(True))

        def jax_loss(logits):
            lp = jax.nn.log_softmax(logits, axis=-1)
            return -jnp.sum(jnp.asarray(targets_np) * lp) / logits.shape[0]

        g_jax = jax.grad(jax_loss)(jnp.asarray(logits_np))
        nb.testing.assert_allclose(g_nb, g_jax, rtol=1e-4, atol=1e-5)
