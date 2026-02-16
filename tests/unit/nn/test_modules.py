# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb


def test_module_is_pytree_and_has_tensor_leaves():
    model = nb.nn.Linear(4, 3)
    leaves = nb.tree_leaves(model)
    tensor_leaves = [x for x in leaves if isinstance(x, nb.Tensor)]
    assert len(tensor_leaves) == 2


def test_grad_with_module_input():
    rng = np.random.default_rng(101)
    model = nb.nn.Linear(4, 2)

    w_np = rng.normal(size=(4, 2)).astype(np.float32)
    b_np = rng.normal(size=(1, 2)).astype(np.float32)
    x_np = rng.normal(size=(12, 4)).astype(np.float32)
    y_np = rng.normal(size=(12, 2)).astype(np.float32)

    model.weight = nb.Tensor.from_dlpack(w_np).requires_grad_(True)
    model.bias = nb.Tensor.from_dlpack(b_np).requires_grad_(True)

    x = nb.Tensor.from_dlpack(x_np)
    y = nb.Tensor.from_dlpack(y_np)

    def loss_fn(m: nb.nn.Module):
        pred = m(x)
        return nb.nn.functional.mse_loss(pred, y)

    grads = nb.grad(loss_fn, realize=False)(model)
    assert isinstance(grads, nb.nn.Linear)
    assert isinstance(grads.weight, nb.Tensor)
    assert isinstance(grads.bias, nb.Tensor)
    assert not grads.weight.real
    assert not grads.bias.real

    pred_np = x_np @ w_np + b_np
    dloss_dpred = 2.0 * (pred_np - y_np) / np.prod(pred_np.shape)
    grad_w_ref = x_np.T @ dloss_dpred
    grad_b_ref = np.sum(dloss_dpred, axis=0, keepdims=True)

    nb.realize_all(grads.weight, grads.bias)
    np.testing.assert_allclose(grads.weight.to_numpy(), grad_w_ref, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(grads.bias.to_numpy(), grad_b_ref, rtol=1e-4, atol=1e-5)


def test_vmap_with_module_argument_axis_none():
    rng = np.random.default_rng(202)
    model = nb.nn.Linear(4, 3)
    x_np = rng.normal(size=(7, 5, 4)).astype(np.float32)
    x = nb.Tensor.from_dlpack(x_np)

    def f(m, xb):
        return m(xb)

    batched = nb.vmap(f, in_axes=(None, 0), out_axes=0)
    y = batched(model, x)
    assert not y.real
    assert tuple(int(d) for d in y.shape) == (7, 5, 3)

    y_manual = np.stack([model(nb.Tensor.from_dlpack(x_np[i])).to_numpy() for i in range(x_np.shape[0])], axis=0)
    np.testing.assert_allclose(y.to_numpy(), y_manual, rtol=1e-5, atol=1e-6)


def test_compile_with_module_argument_cache_hits():
    model = nb.nn.Linear(4, 3)
    x = nb.Tensor.from_dlpack(np.random.randn(6, 4).astype(np.float32))

    @nb.compile
    def compiled(m, x_in):
        return m(x_in)

    out1 = compiled(model, x)
    out2 = compiled(model, x)

    np.testing.assert_allclose(out1.to_numpy(), out2.to_numpy(), rtol=1e-6, atol=1e-6)
    assert compiled.stats.misses == 1
    assert compiled.stats.hits >= 1


def test_module_forward_matches_jax_when_available():
    jnp = pytest.importorskip("jax.numpy")

    rng = np.random.default_rng(303)
    x_np = rng.normal(size=(9, 4)).astype(np.float32)
    w_np = rng.normal(size=(4, 3)).astype(np.float32)
    b_np = rng.normal(size=(1, 3)).astype(np.float32)

    model = nb.nn.Linear(4, 3)
    model.weight = nb.Tensor.from_dlpack(w_np).requires_grad_(True)
    model.bias = nb.Tensor.from_dlpack(b_np).requires_grad_(True)

    y_nb = model(nb.Tensor.from_dlpack(x_np))
    assert not y_nb.real
    y_jax = jnp.asarray(x_np) @ jnp.asarray(w_np) + jnp.asarray(b_np)

    np.testing.assert_allclose(y_nb.to_numpy(), np.asarray(y_jax), rtol=1e-5, atol=1e-6)


def test_one_step_sgd_update_matches_jax_grad_when_available():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    rng = np.random.default_rng(404)
    x_np = rng.normal(size=(10, 4)).astype(np.float32)
    y_np = rng.normal(size=(10, 3)).astype(np.float32)
    w_np = rng.normal(size=(4, 3)).astype(np.float32)
    b_np = rng.normal(size=(1, 3)).astype(np.float32)
    lr = 1e-2

    model = nb.nn.Linear(4, 3)
    model.weight = nb.Tensor.from_dlpack(w_np).requires_grad_(True)
    model.bias = nb.Tensor.from_dlpack(b_np).requires_grad_(True)

    x = nb.Tensor.from_dlpack(x_np)
    y = nb.Tensor.from_dlpack(y_np)

    def loss_fn(m: nb.nn.Module):
        pred = m(x)
        return nb.nn.functional.mse_loss(pred, y)

    loss, grads = nb.value_and_grad(loss_fn, argnums=0, realize=False)(model)
    assert not loss.real
    assert not grads.weight.real
    assert not grads.bias.real

    w_new = model.weight - lr * grads.weight
    b_new = model.bias - lr * grads.bias
    nb.realize_all(loss, w_new, b_new, grads.weight, grads.bias)

    def jax_loss(w, b):
        pred = jnp.asarray(x_np) @ w + b
        return jnp.mean((pred - jnp.asarray(y_np)) ** 2)

    gw_jax, gb_jax = jax.grad(jax_loss, argnums=(0, 1))(jnp.asarray(w_np), jnp.asarray(b_np))
    w_new_ref = w_np - lr * np.asarray(gw_jax)
    b_new_ref = b_np - lr * np.asarray(gb_jax)

    np.testing.assert_allclose(grads.weight.to_numpy(), np.asarray(gw_jax), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(grads.bias.to_numpy(), np.asarray(gb_jax), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(w_new.to_numpy(), w_new_ref, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(b_new.to_numpy(), b_new_ref, rtol=1e-4, atol=1e-5)


def test_pytorch_style_backward_matches_functional_grad():
    rng = np.random.default_rng(505)
    x_np = rng.normal(size=(12, 4)).astype(np.float32)
    y_np = rng.normal(size=(12, 2)).astype(np.float32)
    w_np = rng.normal(size=(4, 2)).astype(np.float32)
    b_np = rng.normal(size=(1, 2)).astype(np.float32)

    model = nb.nn.Linear(4, 2)
    model.weight = nb.Tensor.from_dlpack(w_np).requires_grad_(True)
    model.bias = nb.Tensor.from_dlpack(b_np).requires_grad_(True)

    x = nb.Tensor.from_dlpack(x_np)
    y = nb.Tensor.from_dlpack(y_np)

    out = model(x)
    loss = nb.nn.functional.mse_loss(out, y)
    assert not out.real
    assert not loss.real

    model.zero_grad()
    loss.backward()

    grad_w_back = model.weight.grad
    grad_b_back = model.bias.grad
    assert grad_w_back is not None
    assert grad_b_back is not None

    def loss_fn(m: nb.nn.Module):
        return nb.nn.functional.mse_loss(m(x), y)

    grads_fun = nb.grad(loss_fn, realize=True)(model)

    np.testing.assert_allclose(
        grad_w_back.to_numpy(),
        grads_fun.weight.to_numpy(),
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        grad_b_back.to_numpy(),
        grads_fun.bias.to_numpy(),
        rtol=1e-4,
        atol=1e-5,
    )


def test_module_value_and_grad_with_adamw_update_decreases_loss():
    rng = np.random.default_rng(606)
    x_np = rng.normal(size=(32, 5)).astype(np.float32)
    w_true = rng.normal(size=(5, 3)).astype(np.float32)
    b_true = rng.normal(size=(1, 3)).astype(np.float32)
    y_np = x_np @ w_true + b_true

    x = nb.Tensor.from_dlpack(x_np)
    y = nb.Tensor.from_dlpack(y_np)

    model = nb.nn.Linear(5, 3)
    opt = nb.nn.optim.adamw_init(model)

    def loss_fn(m: nb.nn.Module):
        return nb.nn.functional.mse_loss(m(x), y)

    start = loss_fn(model)
    assert not start.real
    start_v = float(start.to_numpy())

    current_model = model
    current_opt = opt
    for _ in range(8):
        loss, grads = nb.value_and_grad(loss_fn, argnums=0, realize=False)(current_model)
        current_model, current_opt = nb.nn.optim.adamw_update(
            current_model,
            grads,
            current_opt,
            lr=1e-2,
            weight_decay=0.0,
        )

        to_realize = [loss]
        to_realize.extend(t for t in nb.tree_leaves(current_model) if isinstance(t, nb.Tensor))
        to_realize.extend(t for t in nb.tree_leaves(current_opt) if isinstance(t, nb.Tensor))
        nb.realize_all(*to_realize)

    end = loss_fn(current_model)
    assert not end.real
    end_v = float(end.to_numpy())
    assert end_v < start_v


def test_module_policy_auto_realize_toplevel_forward_only():
    prev = nb.nn.Module.get_execution_policy()
    nb.nn.Module.set_execution_policy(auto_realize_toplevel_forward=True)
    try:
        rng = np.random.default_rng(707)
        x_np = rng.normal(size=(4, 3)).astype(np.float32)
        model = nb.nn.Sequential(
            nb.nn.Linear(3, 5),
            nb.nn.ReLU(),
            nb.nn.Linear(5, 2),
        )

        out = model(nb.Tensor.from_dlpack(x_np))
        assert out.real
    finally:
        nb.nn.Module.set_execution_policy(**prev)


def test_module_policy_auto_realize_backward_grads():
    prev = nb.nn.Module.get_execution_policy()
    nb.nn.Module.set_execution_policy(auto_realize_backward_grads=True)
    try:
        rng = np.random.default_rng(808)
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
