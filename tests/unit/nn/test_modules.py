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

    rng = np.random.default_rng(12)
    x_np = rng.normal(size=(5, 4)).astype(np.float32)
    x = nb.Tensor.from_dlpack(x_np)

    y_ref = model(x).to_numpy()
    flat, treedef = nb.tree_flatten(model)
    rebuilt = nb.tree_unflatten(treedef, flat)
    y_rebuilt = rebuilt(x).to_numpy()
    np.testing.assert_allclose(y_rebuilt, y_ref, rtol=1e-6, atol=1e-6)


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

    nb.testing.batch_realize(grads.weight, grads.bias)
    nb.testing.assert_allclose(grads.weight, grad_w_ref, rtol=1e-4, atol=1e-5, realize=False)
    nb.testing.assert_allclose(grads.bias, grad_b_ref, rtol=1e-4, atol=1e-5, realize=False)


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

    w = model.weight.to_numpy()
    b = model.bias.to_numpy() if model.bias is not None else 0.0
    y_manual = np.stack([(x_np[i] @ w) + b for i in range(x_np.shape[0])], axis=0)
    nb.testing.assert_allclose(y, y_manual, rtol=1e-5, atol=1e-6)


def test_compile_with_module_argument_cache_hits():
    rng = np.random.default_rng(3031)
    model = nb.nn.Linear(4, 3)
    x_np = rng.normal(size=(6, 4)).astype(np.float32)
    x = nb.Tensor.from_dlpack(x_np)

    @nb.compile
    def compiled(m, x_in):
        return m(x_in)

    out1 = compiled(model, x)
    out2 = compiled(model, x)

    w = model.weight.to_numpy()
    b = model.bias.to_numpy() if model.bias is not None else 0.0
    ref = x_np @ w + b
    nb.testing.batch_realize(out1, out2)
    nb.testing.assert_allclose(out1, out2, rtol=1e-6, atol=1e-6, realize=False)
    nb.testing.assert_allclose(out1, ref, rtol=1e-5, atol=1e-6, realize=False)
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

    nb.testing.assert_allclose(y_nb, y_jax, rtol=1e-5, atol=1e-6)


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
    def jax_loss(w, b):
        pred = jnp.asarray(x_np) @ w + b
        return jnp.mean((pred - jnp.asarray(y_np)) ** 2)

    gw_jax, gb_jax = jax.grad(jax_loss, argnums=(0, 1))(jnp.asarray(w_np), jnp.asarray(b_np))
    w_new_ref = w_np - lr * np.asarray(gw_jax)
    b_new_ref = b_np - lr * np.asarray(gb_jax)

    nb.testing.batch_realize(grads.weight, grads.bias, w_new, b_new)
    nb.testing.assert_allclose(grads.weight, gw_jax, rtol=1e-4, atol=1e-5, realize=False)
    nb.testing.assert_allclose(grads.bias, gb_jax, rtol=1e-4, atol=1e-5, realize=False)
    nb.testing.assert_allclose(w_new, w_new_ref, rtol=1e-4, atol=1e-5, realize=False)
    nb.testing.assert_allclose(b_new, b_new_ref, rtol=1e-4, atol=1e-5, realize=False)


def test_pytorch_style_backward_matches_functional_grad():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

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

    def jax_loss(w, b):
        pred = jnp.asarray(x_np) @ w + b
        return jnp.mean((pred - jnp.asarray(y_np)) ** 2)

    gw_jax, gb_jax = jax.grad(jax_loss, argnums=(0, 1))(jnp.asarray(w_np), jnp.asarray(b_np))
    gw_ref = np.asarray(gw_jax)
    gb_ref = np.asarray(gb_jax)

    nb.testing.batch_realize(grad_w_back, grads_fun.weight, grad_b_back, grads_fun.bias)
    nb.testing.assert_allclose(grad_w_back, gw_ref, rtol=1e-4, atol=1e-5, realize=False)
    nb.testing.assert_allclose(grads_fun.weight, gw_ref, rtol=1e-4, atol=1e-5, realize=False)
    nb.testing.assert_allclose(grad_b_back, gb_ref, rtol=1e-4, atol=1e-5, realize=False)
    nb.testing.assert_allclose(grads_fun.bias, gb_ref, rtol=1e-4, atol=1e-5, realize=False)


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

    w_ref = model.weight.to_numpy().copy()
    b_ref = model.bias.to_numpy().copy()
    m_w = np.zeros_like(w_ref)
    v_w = np.zeros_like(w_ref)
    m_b = np.zeros_like(b_ref)
    v_b = np.zeros_like(b_ref)

    current_model = model
    current_opt = opt
    for step in range(1, 9):
        loss, grads = nb.value_and_grad(loss_fn, argnums=0, realize=False)(current_model)
        current_model, current_opt = nb.nn.optim.adamw_update(
            current_model,
            grads,
            current_opt,
            lr=1e-2,
            weight_decay=0.0,
        )

        pred_ref = x_np @ w_ref + b_ref
        dloss_dpred = 2.0 * (pred_ref - y_np) / np.prod(pred_ref.shape)
        g_w = x_np.T @ dloss_dpred
        g_b = np.sum(dloss_dpred, axis=0, keepdims=True)

        beta1, beta2 = 0.9, 0.999
        m_w = beta1 * m_w + (1.0 - beta1) * g_w
        v_w = beta2 * v_w + (1.0 - beta2) * (g_w * g_w)
        m_b = beta1 * m_b + (1.0 - beta1) * g_b
        v_b = beta2 * v_b + (1.0 - beta2) * (g_b * g_b)

        m_w_hat = m_w / (1.0 - beta1**step)
        v_w_hat = v_w / (1.0 - beta2**step)
        m_b_hat = m_b / (1.0 - beta1**step)
        v_b_hat = v_b / (1.0 - beta2**step)

        w_ref = w_ref - 1e-2 * (m_w_hat / (np.sqrt(v_w_hat) + 1e-8))
        b_ref = b_ref - 1e-2 * (m_b_hat / (np.sqrt(v_b_hat) + 1e-8))

    end = loss_fn(current_model)
    assert not end.real
    end_v = float(end.to_numpy())
    assert end_v < start_v

    nb.testing.batch_realize(current_model.weight, current_model.bias)
    nb.testing.assert_allclose(current_model.weight, w_ref, rtol=2e-4, atol=2e-5, realize=False)
    nb.testing.assert_allclose(current_model.bias, b_ref, rtol=2e-4, atol=2e-5, realize=False)


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

        nb.realize_all(*list(model.parameters()))

        w1, b1, w2, b2 = nb.Tensor.to_numpy_all(
            getattr(model, "0").weight,
            getattr(model, "0").bias,
            getattr(model, "2").weight,
            getattr(model, "2").bias,
        )

        out = model(nb.Tensor.from_dlpack(x_np))
        assert out.real
        h = np.maximum((x_np @ w1) + b1, 0.0)
        out_ref = (h @ w2) + b2
        nb.testing.assert_allclose(out, out_ref, rtol=1e-5, atol=1e-6, realize=False)
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

        pred = (x_np @ model.weight.to_numpy()) + model.bias.to_numpy()
        dloss_dpred = 2.0 * (pred - y_np) / np.prod(pred.shape)
        gw_ref = x_np.T @ dloss_dpred
        gb_ref = np.sum(dloss_dpred, axis=0, keepdims=True)
        nb.testing.assert_allclose(model.weight.grad, gw_ref, rtol=1e-4, atol=1e-5, realize=False)
        nb.testing.assert_allclose(model.bias.grad, gb_ref, rtol=1e-4, atol=1e-5, realize=False)
    finally:
        nb.nn.Module.set_execution_policy(**prev)


def test_state_dict_load_state_dict_roundtrip_nested():
    rng = np.random.default_rng(909)
    model_src = nb.nn.Sequential(
        nb.nn.Linear(4, 6),
        nb.nn.ReLU(),
        nb.nn.Linear(6, 3),
    )
    model_dst = nb.nn.Sequential(
        nb.nn.Linear(4, 6),
        nb.nn.ReLU(),
        nb.nn.Linear(6, 3),
    )

    x_np = rng.normal(size=(11, 4)).astype(np.float32)
    x = nb.Tensor.from_dlpack(x_np)

    y_src = model_src(x)

    first = getattr(model_dst, "0")
    first.weight = first.weight + 0.123
    y_dst_before = model_dst(x)

    model_dst.load_state_dict(model_src.state_dict())
    y_dst_after = model_dst(x)

    src_np = np.asarray(y_src)
    before_np = np.asarray(y_dst_before)
    after_np = np.asarray(y_dst_after)

    assert not np.allclose(src_np, before_np, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(src_np, after_np, rtol=1e-6, atol=1e-6)


def test_compile_dynamic_dims_with_module_input():
    rng = np.random.default_rng(910)
    model = nb.nn.Linear(4, 3, bias=False)

    @nb.compile(dynamic_dims={0: {0: "batch"}})
    def compiled(x_in):
        return model(x_in)

    x1_np = rng.normal(size=(5, 4)).astype(np.float32)
    x2_np = rng.normal(size=(9, 4)).astype(np.float32)
    x1 = nb.Tensor.from_dlpack(x1_np)
    x2 = nb.Tensor.from_dlpack(x2_np)

    y1 = compiled(x1)
    y2 = compiled(x2)

    w = model.weight.to_numpy()
    y1_ref = x1_np @ w
    y2_ref = x2_np @ w
    nb.testing.batch_realize(y1, y2)
    nb.testing.assert_allclose(y1, y1_ref, rtol=1e-5, atol=1e-6, realize=False)
    nb.testing.assert_allclose(y2, y2_ref, rtol=1e-5, atol=1e-6, realize=False)
    assert compiled.stats.misses == 1
    assert compiled.stats.hits >= 1


def test_module_forward_with_sharded_input_if_available():
    rng = np.random.default_rng(911)
    x_np = rng.normal(size=(6, 4)).astype(np.float32)
    model = nb.nn.Linear(4, 3)

    try:
        mesh = nb.DeviceMesh("nn_mesh_tp", (2,), ("tp",))
        x_sharded = nb.Tensor.from_dlpack(x_np).shard(mesh, nb.P(None, "tp"))
        y_sharded = model(x_sharded)
    except Exception as exc:
        pytest.skip(f"nn sharding path unavailable in this environment: {exc}")

    w = model.weight.to_numpy()
    b = model.bias.to_numpy() if model.bias is not None else 0.0
    y_ref = (x_np @ w) + b
    gathered = y_sharded.gather()
    nb.testing.assert_allclose(gathered, y_ref, rtol=1e-4, atol=1e-5)
