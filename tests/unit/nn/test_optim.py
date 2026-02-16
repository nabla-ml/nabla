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

    nb.testing.batch_realize(m2, v2, p2)
    nb.testing.assert_allclose(m2, m_ref, rtol=1e-5, atol=1e-6, realize=False)
    nb.testing.assert_allclose(v2, v_ref, rtol=1e-5, atol=1e-6, realize=False)
    nb.testing.assert_allclose(p2, p_ref, rtol=1e-5, atol=1e-6, realize=False)


def test_stateful_adamw_with_pytree_params():
    torch = pytest.importorskip("torch")

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
    assert new_params["w"].real
    assert new_params["b"].real
    assert tuple(int(d) for d in new_params["w"].shape) == (5, 2)
    assert tuple(int(d) for d in new_params["b"].shape) == (1, 2)

    p_w = torch.nn.Parameter(torch.from_numpy(np.array(params["w"], copy=True)))
    p_b = torch.nn.Parameter(torch.from_numpy(np.array(params["b"], copy=True)))
    p_w.grad = torch.from_numpy(np.array(grads["w"], copy=True))
    p_b.grad = torch.from_numpy(np.array(grads["b"], copy=True))
    opt_t = torch.optim.AdamW([p_w, p_b], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    opt_t.step()

    nb.testing.batch_realize(new_params["w"], new_params["b"])
    nb.testing.assert_allclose(new_params["w"], p_w, rtol=1e-5, atol=1e-6, realize=False)
    nb.testing.assert_allclose(new_params["b"], p_b, rtol=1e-5, atol=1e-6, realize=False)

    st_w = opt_t.state[p_w]
    st_b = opt_t.state[p_b]
    nb.testing.batch_realize(opt.m["w"], opt.v["w"], opt.m["b"], opt.v["b"])
    nb.testing.assert_allclose(opt.m["w"], st_w["exp_avg"], rtol=1e-5, atol=1e-6, realize=False)
    nb.testing.assert_allclose(opt.v["w"], st_w["exp_avg_sq"], rtol=1e-5, atol=1e-6, realize=False)
    nb.testing.assert_allclose(opt.m["b"], st_b["exp_avg"], rtol=1e-5, atol=1e-6, realize=False)
    nb.testing.assert_allclose(opt.v["b"], st_b["exp_avg_sq"], rtol=1e-5, atol=1e-6, realize=False)


def test_functional_adamw_update_compat():
    rng = np.random.default_rng(47)
    params = {"w": nb.Tensor.from_dlpack(rng.normal(size=(3, 3)).astype(np.float32))}
    grads = {"w": nb.Tensor.from_dlpack(rng.normal(size=(3, 3)).astype(np.float32))}

    state = nb.nn.optim.adamw_init(params)
    new_params, new_state = nb.nn.optim.adamw_update(params, grads, state, lr=5e-3)

    assert new_params["w"].real
    assert tuple(int(d) for d in new_params["w"].shape) == (3, 3)
    assert new_state["step"] == 1

    g_np = np.asarray(grads["w"])
    p_np = np.asarray(params["w"])
    m_ref = 0.1 * g_np
    v_ref = 0.001 * (g_np * g_np)
    m_hat = m_ref / (1.0 - 0.9)
    v_hat = v_ref / (1.0 - 0.999)
    p_ref = p_np - 5e-3 * (m_hat / (np.sqrt(v_hat) + 1e-8))

    nb.testing.batch_realize(new_state["m"]["w"], new_state["v"]["w"], new_params["w"])
    nb.testing.assert_allclose(new_state["m"]["w"], m_ref, rtol=1e-5, atol=1e-6, realize=False)
    nb.testing.assert_allclose(new_state["v"]["w"], v_ref, rtol=1e-5, atol=1e-6, realize=False)
    nb.testing.assert_allclose(new_params["w"], p_ref, rtol=1e-5, atol=1e-6, realize=False)


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

    nb.testing.batch_realize(params_stateful["w"], params_stateful["b"], params_functional["w"], params_functional["b"])
    nb.testing.assert_allclose(params_stateful["w"], params_functional["w"], rtol=1e-5, atol=1e-6, realize=False)
    nb.testing.assert_allclose(params_stateful["b"], params_functional["b"], rtol=1e-5, atol=1e-6, realize=False)


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

    nb.testing.assert_allclose(p2, p_t, rtol=1e-5, atol=1e-6)


def test_optimizer_execution_policy_can_disable_auto_realization():
    prev = nb.nn.optim.Optimizer.get_execution_policy()
    nb.nn.optim.Optimizer.set_execution_policy(
        auto_realize_updated_params=False,
        auto_realize_updated_state=False,
    )
    try:
        rng = np.random.default_rng(61)
        params = {"w": nb.Tensor.from_dlpack(rng.normal(size=(3, 2)).astype(np.float32))}
        grads = {"w": nb.Tensor.from_dlpack(rng.normal(size=(3, 2)).astype(np.float32))}

        opt = nb.nn.optim.AdamW(params, lr=1e-2)
        p_new = opt.step(grads)
        assert not p_new["w"].real

        state = nb.nn.optim.adamw_init(params)
        p_new2, state2 = nb.nn.optim.adamw_update(params, grads, state, lr=1e-2)
        assert not p_new2["w"].real
        assert state2["step"] == 1

        g_np = np.asarray(grads["w"])
        p_np = np.asarray(params["w"])
        m_ref = 0.1 * g_np
        v_ref = 0.001 * (g_np * g_np)
        p_ref = p_np - 1e-2 * ((m_ref / 0.1) / (np.sqrt(v_ref / 0.001) + 1e-8))

        nb.testing.batch_realize(p_new["w"], p_new2["w"])
        nb.testing.assert_allclose(p_new["w"], p_ref, rtol=1e-5, atol=1e-6, realize=False)
        nb.testing.assert_allclose(p_new2["w"], p_ref, rtol=1e-5, atol=1e-6, realize=False)
    finally:
        nb.nn.optim.Optimizer.set_execution_policy(**prev)
