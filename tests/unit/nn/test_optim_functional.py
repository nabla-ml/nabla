# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for functional optimizer primitives — validated against JAX (optax-style
manual math for AdamW, SGD)."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng


# ===----------------------------------------------------------------------=== #
# adamw_step (single-tensor, lowest level)
# ===----------------------------------------------------------------------=== #


class TestAdamWStep:
    def test_adamw_step_output_shapes(self):
        rng = make_rng(41)
        p = nb.Tensor.from_dlpack(rng.normal(size=(4, 3)).astype(np.float32))
        g = nb.Tensor.from_dlpack(rng.normal(size=(4, 3)).astype(np.float32))
        m = nb.zeros_like(p)
        v = nb.zeros_like(p)

        p2, m2, v2 = nb.nn.optim.adamw_step(
            p, g, m, v, 1, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0
        )
        for t in (p2, m2, v2):
            assert tuple(int(d) for d in t.shape) == (4, 3)

    def test_adamw_step_vs_jax_manual(self):
        """Verify single step against manually computing AdamW in JAX."""
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        rng = make_rng(42)
        p_np = rng.normal(size=(4, 3)).astype(np.float32)
        g_np = rng.normal(size=(4, 3)).astype(np.float32)
        lr, beta1, beta2, eps, wd = 1e-2, 0.9, 0.999, 1e-8, 1e-2

        p2, m2, v2 = nb.nn.optim.adamw_step(
            nb.Tensor.from_dlpack(p_np),
            nb.Tensor.from_dlpack(g_np),
            nb.zeros_like(nb.Tensor.from_dlpack(p_np)),
            nb.zeros_like(nb.Tensor.from_dlpack(p_np)),
            1,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=wd,
        )

        # JAX reference (manual AdamW)
        m_jax = (1 - beta1) * jnp.asarray(g_np)
        v_jax = (1 - beta2) * (jnp.asarray(g_np) ** 2)
        m_hat = m_jax / (1 - beta1)
        v_hat = v_jax / (1 - beta2)
        update = m_hat / (jnp.sqrt(v_hat) + eps) + wd * jnp.asarray(p_np)
        p_ref = jnp.asarray(p_np) - lr * update

        nb.testing.assert_allclose(m2, m_jax, rtol=1e-5, atol=1e-6)
        nb.testing.assert_allclose(v2, v_jax, rtol=1e-5, atol=1e-6)
        nb.testing.assert_allclose(p2, p_ref, rtol=1e-5, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# sgd_step (single-tensor)
# ===----------------------------------------------------------------------=== #


class TestSGDStep:
    """sgd_step validated against PyTorch SGD for consistency."""

    def test_sgd_step_no_momentum(self):
        torch = pytest.importorskip("torch")
        rng = make_rng(50)
        p_np = rng.normal(size=(4, 3)).astype(np.float32)
        g_np = rng.normal(size=(4, 3)).astype(np.float32)
        lr = 0.1

        p2, buf = nb.nn.optim.sgd_step(
            nb.Tensor.from_dlpack(p_np),
            nb.Tensor.from_dlpack(g_np),
            momentum_buffer=None,
            lr=lr,
        )

        # PyTorch reference
        p_pt = torch.nn.Parameter(torch.from_numpy(p_np.copy()))
        p_pt.grad = torch.from_numpy(g_np.copy())
        opt = torch.optim.SGD([p_pt], lr=lr)
        opt.step()

        nb.testing.assert_allclose(p2, p_pt, rtol=1e-5, atol=1e-6)
        assert buf is None

    def test_sgd_step_with_momentum(self):
        rng = make_rng(51)
        p_np = rng.normal(size=(4, 3)).astype(np.float32)
        g_np = rng.normal(size=(4, 3)).astype(np.float32)
        lr, mu = 0.01, 0.9

        p2, buf = nb.nn.optim.sgd_step(
            nb.Tensor.from_dlpack(p_np),
            nb.Tensor.from_dlpack(g_np),
            momentum_buffer=None,
            lr=lr,
            momentum=mu,
        )

        # First step: buffer = grad, update = buffer, p = p - lr * update
        buf_ref = g_np  # first step momentum buffer equals grad
        p_ref = p_np - lr * buf_ref

        nb.testing.assert_allclose(p2, p_ref, rtol=1e-5, atol=1e-6)
        assert buf is not None
        nb.testing.assert_allclose(buf, buf_ref, rtol=1e-5, atol=1e-6)

    def test_sgd_step_with_weight_decay(self):
        torch = pytest.importorskip("torch")
        rng = make_rng(52)
        p_np = rng.normal(size=(3, 4)).astype(np.float32)
        g_np = rng.normal(size=(3, 4)).astype(np.float32)
        lr, wd = 0.01, 0.1

        p2, _ = nb.nn.optim.sgd_step(
            nb.Tensor.from_dlpack(p_np),
            nb.Tensor.from_dlpack(g_np),
            momentum_buffer=None,
            lr=lr,
            weight_decay=wd,
        )

        # Reference: update = grad + wd*p, p_new = p - lr * update
        update = g_np + wd * p_np
        p_ref = p_np - lr * update
        nb.testing.assert_allclose(p2, p_ref, rtol=1e-5, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# adamw_init / adamw_update (functional pytree-level)
# ===----------------------------------------------------------------------=== #


class TestAdamWFunctionalPytree:
    def test_adamw_init_creates_zero_state(self):
        rng = make_rng(60)
        params = {
            "w": nb.Tensor.from_dlpack(rng.normal(size=(3, 3)).astype(np.float32))
        }
        state = nb.nn.optim.adamw_init(params)
        assert state["step"] == 0
        assert "m" in state and "v" in state
        np.testing.assert_allclose(np.asarray(state["m"]["w"]), 0.0)
        np.testing.assert_allclose(np.asarray(state["v"]["w"]), 0.0)

    def test_adamw_update_single_step_vs_jax(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        rng = make_rng(47)
        p_np = rng.normal(size=(3, 3)).astype(np.float32)
        g_np = rng.normal(size=(3, 3)).astype(np.float32)
        lr = 5e-3

        params = {"w": nb.Tensor.from_dlpack(p_np)}
        grads = {"w": nb.Tensor.from_dlpack(g_np)}
        state = nb.nn.optim.adamw_init(params)
        new_params, new_state = nb.nn.optim.adamw_update(params, grads, state, lr=lr)

        assert new_state["step"] == 1

        # JAX reference
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = (1 - beta1) * jnp.asarray(g_np)
        v = (1 - beta2) * (jnp.asarray(g_np) ** 2)
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
        p_ref = jnp.asarray(p_np) - lr * (m_hat / (jnp.sqrt(v_hat) + eps))

        nb.testing.assert_allclose(new_state["m"]["w"], m, rtol=1e-5, atol=1e-6)
        nb.testing.assert_allclose(new_state["v"]["w"], v, rtol=1e-5, atol=1e-6)
        nb.testing.assert_allclose(new_params["w"], p_ref, rtol=1e-5, atol=1e-6)

    def test_functional_and_stateful_agree(self):
        """adamw_update and AdamW.step give identical results for one step."""
        rng = make_rng(53)
        params = {
            "w": nb.Tensor.from_dlpack(rng.normal(size=(4, 2)).astype(np.float32)),
            "b": nb.Tensor.from_dlpack(rng.normal(size=(1, 2)).astype(np.float32)),
        }
        grads = {
            "w": nb.Tensor.from_dlpack(rng.normal(size=(4, 2)).astype(np.float32)),
            "b": nb.Tensor.from_dlpack(rng.normal(size=(1, 2)).astype(np.float32)),
        }

        # Stateful
        opt = nb.nn.optim.AdamW(params, lr=1e-2)
        p_stateful = opt.step(grads)

        # Functional
        state = nb.nn.optim.adamw_init(params)
        p_functional, _ = nb.nn.optim.adamw_update(params, grads, state, lr=1e-2)

        nb.testing.assert_allclose(
            p_stateful["w"], p_functional["w"], rtol=1e-5, atol=1e-6
        )
        nb.testing.assert_allclose(
            p_stateful["b"], p_functional["b"], rtol=1e-5, atol=1e-6
        )
