# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb


def test_functional_adamw_step_values_and_lazy_realization():
    rng = np.random.default_rng(41)
    p_np = rng.normal(size=(4, 3)).astype(np.float32)
    g_np = rng.normal(size=(4, 3)).astype(np.float32)

    p = nb.Tensor.from_dlpack(p_np)
    g = nb.Tensor.from_dlpack(g_np)
    m = nb.zeros_like(p)
    v = nb.zeros_like(p)

    p2, m2, v2 = nb.nn.optim.adamw_step(
        p,
        g,
        m,
        v,
        1,
        lr=1e-2,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-2,
    )
    assert not p2.real
    assert not m2.real
    assert not v2.real

    assert tuple(int(d) for d in p2.shape) == (4, 3)
    assert tuple(int(d) for d in m2.shape) == (4, 3)
    assert tuple(int(d) for d in v2.shape) == (4, 3)

    beta1, beta2 = 0.9, 0.999
    m_ref = (1.0 - beta1) * g_np
    v_ref = (1.0 - beta2) * (g_np * g_np)
    m_hat = m_ref / (1.0 - beta1**1)
    v_hat = v_ref / (1.0 - beta2**1)
    update = m_hat / (np.sqrt(v_hat) + 1e-8) + 1e-2 * p_np
    p_ref = p_np - 1e-2 * update

    nb.realize_all(p2, m2, v2)
    np.testing.assert_allclose(m2.to_numpy(), m_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(v2.to_numpy(), v_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(p2.to_numpy(), p_ref, rtol=1e-5, atol=1e-6)


def test_stateful_adamw_with_pytree_params():
    rng = np.random.default_rng(43)
    params = {
        "w": nb.Tensor.from_dlpack(rng.normal(size=(5, 2)).astype(np.float32)),
        "b": nb.Tensor.from_dlpack(rng.normal(size=(1, 2)).astype(np.float32)),
    }
    grads = {
        "w": nb.Tensor.from_dlpack(rng.normal(size=(5, 2)).astype(np.float32)),
        "b": nb.Tensor.from_dlpack(rng.normal(size=(1, 2)).astype(np.float32)),
    }

    opt = nb.nn.optim.AdamW(params, lr=1e-2)
    new_params = opt.step(grads)

    assert isinstance(new_params, dict)
    assert not new_params["w"].real
    assert not new_params["b"].real
    assert tuple(int(d) for d in new_params["w"].shape) == (5, 2)
    assert tuple(int(d) for d in new_params["b"].shape) == (1, 2)

    nb.realize_all(new_params["w"], new_params["b"])


def test_functional_adamw_update_compat():
    rng = np.random.default_rng(47)
    params = {"w": nb.Tensor.from_dlpack(rng.normal(size=(3, 3)).astype(np.float32))}
    grads = {"w": nb.Tensor.from_dlpack(rng.normal(size=(3, 3)).astype(np.float32))}

    state = nb.nn.optim.adamw_init(params)
    new_params, new_state = nb.nn.optim.adamw_update(params, grads, state, lr=5e-3)

    assert not new_params["w"].real
    assert tuple(int(d) for d in new_params["w"].shape) == (3, 3)
    assert new_state["step"] == 1


def test_functional_and_stateful_adamw_match_for_one_step():
    rng = np.random.default_rng(53)
    params = {
        "w": nb.Tensor.from_dlpack(rng.normal(size=(4, 2)).astype(np.float32)),
        "b": nb.Tensor.from_dlpack(rng.normal(size=(1, 2)).astype(np.float32)),
    }
    grads = {
        "w": nb.Tensor.from_dlpack(rng.normal(size=(4, 2)).astype(np.float32)),
        "b": nb.Tensor.from_dlpack(rng.normal(size=(1, 2)).astype(np.float32)),
    }

    opt_stateful = nb.nn.optim.AdamW(params, lr=1e-2)
    params_stateful = opt_stateful.step(grads)

    state = nb.nn.optim.adamw_init(params)
    params_functional, _ = nb.nn.optim.adamw_update(params, grads, state, lr=1e-2)

    nb.realize_all(
        params_stateful["w"],
        params_stateful["b"],
        params_functional["w"],
        params_functional["b"],
    )
    np.testing.assert_allclose(
        params_stateful["w"].to_numpy(),
        params_functional["w"].to_numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        params_stateful["b"].to_numpy(),
        params_functional["b"].to_numpy(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_adamw_step_matches_torch_when_available():
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(59)
    p_np = rng.normal(size=(3, 4)).astype(np.float32)
    g_np = rng.normal(size=(3, 4)).astype(np.float32)

    p = nb.Tensor.from_dlpack(p_np)
    g = nb.Tensor.from_dlpack(g_np)
    m = nb.zeros_like(p)
    v = nb.zeros_like(p)

    p2, _, _ = nb.nn.optim.adamw_step(
        p,
        g,
        m,
        v,
        1,
        lr=1e-2,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-2,
    )

    p_t = torch.nn.Parameter(torch.from_numpy(p_np.copy()))
    p_t.grad = torch.from_numpy(g_np.copy())
    opt = torch.optim.AdamW(
        [p_t],
        lr=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    )
    opt.step()

    np.testing.assert_allclose(
        p2.to_numpy(),
        p_t.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
