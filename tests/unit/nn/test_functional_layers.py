# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for nabla.nn.functional.linear — validated against JAX."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng

# ===----------------------------------------------------------------------=== #
# Forward correctness
# ===----------------------------------------------------------------------=== #


class TestFunctionalLinear:
    @pytest.mark.parametrize("batch,in_f,out_f", [(5, 4, 3), (1, 8, 2), (16, 3, 7)])
    def test_linear_forward_vs_jax(self, batch, in_f, out_f):
        jax = pytest.importorskip("jax")  # noqa: F841
        jnp = pytest.importorskip("jax.numpy")

        rng = make_rng(29)
        x_np = rng.normal(size=(batch, in_f)).astype(np.float32)
        w_np = rng.normal(size=(in_f, out_f)).astype(np.float32)
        b_np = rng.normal(size=(1, out_f)).astype(np.float32)

        y_nb = nb.nn.functional.linear(
            nb.Tensor.from_dlpack(x_np),
            nb.Tensor.from_dlpack(w_np),
            nb.Tensor.from_dlpack(b_np),
        )
        y_jax = jnp.asarray(x_np) @ jnp.asarray(w_np) + jnp.asarray(b_np)

        nb.testing.assert_allclose(y_nb, y_jax, rtol=1e-5, atol=1e-6)

    def test_linear_no_bias_vs_jax(self):
        jnp = pytest.importorskip("jax.numpy")

        rng = make_rng(30)
        x_np = rng.normal(size=(5, 4)).astype(np.float32)
        w_np = rng.normal(size=(4, 3)).astype(np.float32)

        y_nb = nb.nn.functional.linear(
            nb.Tensor.from_dlpack(x_np),
            nb.Tensor.from_dlpack(w_np),
            bias=None,
        )
        y_jax = jnp.asarray(x_np) @ jnp.asarray(w_np)
        nb.testing.assert_allclose(y_nb, y_jax, rtol=1e-5, atol=1e-6)

    def test_linear_lazy_realization(self):
        rng = make_rng(11)
        x = nb.Tensor.from_dlpack(rng.normal(size=(5, 4)).astype(np.float32))
        w = nb.Tensor.from_dlpack(rng.normal(size=(4, 3)).astype(np.float32))
        b = nb.Tensor.from_dlpack(rng.normal(size=(1, 3)).astype(np.float32))

        y = nb.nn.functional.linear(x, w, b)
        assert not y.real
        assert tuple(int(d) for d in y.shape) == (5, 3)


# ===----------------------------------------------------------------------=== #
# Gradient through linear via JAX
# ===----------------------------------------------------------------------=== #


class TestFunctionalLinearGrad:
    def test_grad_through_linear_vs_jax(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        rng = make_rng(31)
        x_np = rng.normal(size=(5, 4)).astype(np.float32)
        w_np = rng.normal(size=(4, 3)).astype(np.float32)
        b_np = rng.normal(size=(1, 3)).astype(np.float32)

        # Nabla
        x = nb.Tensor.from_dlpack(x_np)
        w = nb.Tensor.from_dlpack(w_np).requires_grad_(True)
        b = nb.Tensor.from_dlpack(b_np).requires_grad_(True)

        def nb_loss(w_t, b_t):
            return nb.mean(nb.nn.functional.linear(x, w_t, b_t))

        gw_nb, gb_nb = nb.grad(nb_loss, argnums=(0, 1))(w, b)

        # JAX reference
        def jax_loss(w_j, b_j):
            return jnp.mean(jnp.asarray(x_np) @ w_j + b_j)

        gw_jax, gb_jax = jax.grad(jax_loss, argnums=(0, 1))(
            jnp.asarray(w_np), jnp.asarray(b_np)
        )

        nb.testing.assert_allclose(gw_nb, gw_jax, rtol=1e-5, atol=1e-6)
        nb.testing.assert_allclose(gb_nb, gb_jax, rtol=1e-5, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# Functional layer_norm
# ===----------------------------------------------------------------------=== #


class TestFunctionalLayerNorm:
    @pytest.mark.parametrize("shape,normalized", [((6, 8), (8,)), ((3, 4, 5), (4, 5))])
    def test_layer_norm_forward_vs_jax(self, shape, normalized):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        rng = make_rng(51)
        x_np = rng.normal(size=shape).astype(np.float32)
        w_np = rng.normal(size=normalized).astype(np.float32)
        b_np = rng.normal(size=normalized).astype(np.float32)
        eps = 1e-5

        x = nb.Tensor.from_dlpack(x_np)
        w = nb.Tensor.from_dlpack(w_np)
        b = nb.Tensor.from_dlpack(b_np)
        axis = tuple(range(-len(normalized), 0))

        y_nb = nb.nn.functional.layer_norm(x, weight=w, bias=b, eps=eps, axis=axis)

        mu = jnp.mean(jnp.asarray(x_np), axis=axis, keepdims=True)
        centered = jnp.asarray(x_np) - mu
        var = jnp.mean(centered * centered, axis=axis, keepdims=True)
        y_jax = centered * jax.lax.rsqrt(var + eps)
        y_jax = y_jax * jnp.asarray(w_np) + jnp.asarray(b_np)

        nb.testing.assert_allclose(y_nb, y_jax, rtol=1e-5, atol=1e-6)

    def test_layer_norm_grad_vs_jax(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        rng = make_rng(52)
        x_np = rng.normal(size=(5, 7)).astype(np.float32)
        w_np = rng.normal(size=(7,)).astype(np.float32)
        b_np = rng.normal(size=(7,)).astype(np.float32)
        eps = 1e-5

        x = nb.Tensor.from_dlpack(x_np).requires_grad_(True)
        w = nb.Tensor.from_dlpack(w_np).requires_grad_(True)
        b = nb.Tensor.from_dlpack(b_np).requires_grad_(True)

        def nb_loss(x_t, w_t, b_t):
            y = nb.nn.functional.layer_norm(x_t, weight=w_t, bias=b_t, eps=eps, axis=-1)
            return nb.mean(y)

        gx_nb, gw_nb, gb_nb = nb.grad(nb_loss, argnums=(0, 1, 2))(x, w, b)

        def jax_loss(x_t, w_t, b_t):
            mu = jnp.mean(x_t, axis=-1, keepdims=True)
            centered = x_t - mu
            var = jnp.mean(centered * centered, axis=-1, keepdims=True)
            y = centered * jax.lax.rsqrt(var + eps)
            y = y * w_t + b_t
            return jnp.mean(y)

        gx_jax, gw_jax, gb_jax = jax.grad(jax_loss, argnums=(0, 1, 2))(
            jnp.asarray(x_np), jnp.asarray(w_np), jnp.asarray(b_np)
        )

        nb.testing.assert_allclose(gx_nb, gx_jax, rtol=1e-4, atol=1e-5)
        nb.testing.assert_allclose(gw_nb, gw_jax, rtol=1e-4, atol=1e-5)
        nb.testing.assert_allclose(gb_nb, gb_jax, rtol=1e-4, atol=1e-5)
