# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for functional activations exposed via nabla.nn.functional — validated
against JAX equivalents."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng


# ===----------------------------------------------------------------------=== #
# Forward correctness for each activation
# ===----------------------------------------------------------------------=== #


class TestActivationForward:
    """Each nn.functional activation matches its JAX counterpart."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.rng = make_rng(40)
        self.x_np = self.rng.normal(size=(10, 5)).astype(np.float32)
        self.x = nb.Tensor.from_dlpack(self.x_np)

    def test_relu_vs_jax(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        y_nb = nb.nn.functional.relu(self.x)
        y_jax = jax.nn.relu(jnp.asarray(self.x_np))
        nb.testing.assert_allclose(y_nb, y_jax, rtol=1e-6, atol=1e-6)

    def test_gelu_vs_jax(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        y_nb = nb.nn.functional.gelu(self.x)
        y_jax = jax.nn.gelu(jnp.asarray(self.x_np))
        # GELU implementations may use slightly different approximations
        nb.testing.assert_allclose(y_nb, y_jax, rtol=5e-3, atol=5e-4)

    def test_sigmoid_vs_jax(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        y_nb = nb.nn.functional.sigmoid(self.x)
        y_jax = jax.nn.sigmoid(jnp.asarray(self.x_np))
        nb.testing.assert_allclose(y_nb, y_jax, rtol=1e-5, atol=1e-6)

    def test_tanh_vs_jax(self):
        jnp = pytest.importorskip("jax.numpy")
        y_nb = nb.nn.functional.tanh(self.x)
        y_jax = jnp.tanh(jnp.asarray(self.x_np))
        nb.testing.assert_allclose(y_nb, y_jax, rtol=1e-5, atol=1e-6)

    def test_silu_vs_jax(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        y_nb = nb.nn.functional.silu(self.x)
        y_jax = jax.nn.silu(jnp.asarray(self.x_np))
        nb.testing.assert_allclose(y_nb, y_jax, rtol=1e-5, atol=1e-6)

    def test_softmax_vs_jax(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        y_nb = nb.nn.functional.softmax(self.x)
        y_jax = jax.nn.softmax(jnp.asarray(self.x_np), axis=-1)
        nb.testing.assert_allclose(y_nb, y_jax, rtol=1e-5, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# Gradient correctness for activations
# ===----------------------------------------------------------------------=== #


class TestActivationGrad:
    """Gradient through activations matches JAX."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.rng = make_rng(41)
        self.x_np = self.rng.normal(size=(8, 4)).astype(np.float32)

    def _check_grad(self, nb_fn, jax_fn):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        x = nb.Tensor.from_dlpack(self.x_np).requires_grad_(True)

        def nb_loss(t):
            return nb.mean(nb_fn(t))

        g_nb = nb.grad(nb_loss)(x)

        def jax_loss(t):
            return jnp.mean(jax_fn(t))

        g_jax = jax.grad(jax_loss)(jnp.asarray(self.x_np))
        nb.testing.assert_allclose(g_nb, g_jax, rtol=1e-4, atol=1e-5)

    def test_relu_grad(self):
        jax = pytest.importorskip("jax")
        self._check_grad(nb.nn.functional.relu, jax.nn.relu)

    def test_sigmoid_grad(self):
        jax = pytest.importorskip("jax")
        self._check_grad(nb.nn.functional.sigmoid, jax.nn.sigmoid)

    def test_tanh_grad(self):
        jnp = pytest.importorskip("jax.numpy")
        self._check_grad(nb.nn.functional.tanh, jnp.tanh)

    def test_silu_grad(self):
        jax = pytest.importorskip("jax")
        self._check_grad(nb.nn.functional.silu, jax.nn.silu)

    def test_gelu_grad(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        # Use relaxed tolerance due to GELU approximation differences
        x = nb.Tensor.from_dlpack(self.x_np).requires_grad_(True)

        def nb_loss(t):
            return nb.mean(nb.nn.functional.gelu(t))

        g_nb = nb.grad(nb_loss)(x)

        def jax_loss(t):
            return jnp.mean(jax.nn.gelu(t))

        g_jax = jax.grad(jax_loss)(jnp.asarray(self.x_np))
        nb.testing.assert_allclose(g_nb, g_jax, rtol=5e-3, atol=5e-4)
