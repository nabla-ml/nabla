# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for stateful AdamW optimizer — validated against PyTorch."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng


# ===----------------------------------------------------------------------=== #
# AdamW stateful vs PyTorch
# ===----------------------------------------------------------------------=== #


class TestAdamWStateful:
    def test_one_step_vs_pytorch(self):
        torch = pytest.importorskip("torch")
        rng = make_rng(43)

        w_np = rng.normal(size=(5, 2)).astype(np.float32)
        b_np = rng.normal(size=(1, 2)).astype(np.float32)
        gw_np = rng.normal(size=(5, 2)).astype(np.float32)
        gb_np = rng.normal(size=(1, 2)).astype(np.float32)

        params = {
            "w": nb.Tensor.from_dlpack(w_np),
            "b": nb.Tensor.from_dlpack(b_np),
        }
        grads = {
            "w": nb.Tensor.from_dlpack(gw_np),
            "b": nb.Tensor.from_dlpack(gb_np),
        }

        opt_nb = nb.nn.optim.AdamW(params, lr=1e-2)
        new_params = opt_nb.step(grads)

        # PyTorch reference
        p_w = torch.nn.Parameter(torch.from_numpy(w_np.copy()))
        p_b = torch.nn.Parameter(torch.from_numpy(b_np.copy()))
        p_w.grad = torch.from_numpy(gw_np.copy())
        p_b.grad = torch.from_numpy(gb_np.copy())
        opt_pt = torch.optim.AdamW(
            [p_w, p_b], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        opt_pt.step()

        nb.testing.assert_allclose(new_params["w"], p_w, rtol=1e-5, atol=1e-6)
        nb.testing.assert_allclose(new_params["b"], p_b, rtol=1e-5, atol=1e-6)

        # Also check internal state matches PyTorch
        st_w = opt_pt.state[p_w]
        st_b = opt_pt.state[p_b]
        nb.testing.assert_allclose(opt_nb.m["w"], st_w["exp_avg"], rtol=1e-5, atol=1e-6)
        nb.testing.assert_allclose(opt_nb.v["w"], st_w["exp_avg_sq"], rtol=1e-5, atol=1e-6)
        nb.testing.assert_allclose(opt_nb.m["b"], st_b["exp_avg"], rtol=1e-5, atol=1e-6)
        nb.testing.assert_allclose(opt_nb.v["b"], st_b["exp_avg_sq"], rtol=1e-5, atol=1e-6)

    def test_adamw_step_with_weight_decay_vs_pytorch(self):
        torch = pytest.importorskip("torch")
        rng = make_rng(59)
        p_np = rng.normal(size=(3, 4)).astype(np.float32)
        g_np = rng.normal(size=(3, 4)).astype(np.float32)

        p2, _, _ = nb.nn.optim.adamw_step(
            nb.Tensor.from_dlpack(p_np),
            nb.Tensor.from_dlpack(g_np),
            nb.zeros_like(nb.Tensor.from_dlpack(p_np)),
            nb.zeros_like(nb.Tensor.from_dlpack(p_np)),
            1,
            lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2,
        )

        p_pt = torch.nn.Parameter(torch.from_numpy(p_np.copy()))
        p_pt.grad = torch.from_numpy(g_np.copy())
        opt = torch.optim.AdamW([p_pt], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
        opt.step()

        nb.testing.assert_allclose(p2, p_pt, rtol=1e-5, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# Optimizer execution policy
# ===----------------------------------------------------------------------=== #


class TestOptimizerExecutionPolicy:
    def test_disable_auto_realization(self):
        prev = nb.nn.optim.Optimizer.get_execution_policy()
        nb.nn.optim.Optimizer.set_execution_policy(
            auto_realize_updated_params=False,
            auto_realize_updated_state=False,
        )
        try:
            rng = make_rng(61)
            params = {"w": nb.Tensor.from_dlpack(rng.normal(size=(3, 2)).astype(np.float32))}
            grads = {"w": nb.Tensor.from_dlpack(rng.normal(size=(3, 2)).astype(np.float32))}

            opt = nb.nn.optim.AdamW(params, lr=1e-2)
            p_new = opt.step(grads)
            assert not p_new["w"].real

            state = nb.nn.optim.adamw_init(params)
            p_new2, state2 = nb.nn.optim.adamw_update(params, grads, state, lr=1e-2)
            assert not p_new2["w"].real
            assert state2["step"] == 1
        finally:
            nb.nn.optim.Optimizer.set_execution_policy(**prev)

    def test_policy_roundtrip(self):
        prev = nb.nn.optim.Optimizer.get_execution_policy()
        try:
            nb.nn.optim.Optimizer.set_execution_policy(
                auto_realize_updated_params=False,
                auto_realize_updated_state=False,
            )
            pol = nb.nn.optim.Optimizer.get_execution_policy()
            assert pol["auto_realize_updated_params"] is False
            assert pol["auto_realize_updated_state"] is False
        finally:
            nb.nn.optim.Optimizer.set_execution_policy(**prev)
