# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for using Modules with nabla transforms: vmap, compile, value_and_grad."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng, nb_from_np


# ===----------------------------------------------------------------------=== #
# vmap with Module
# ===----------------------------------------------------------------------=== #


class TestVmapWithModule:
    def test_vmap_broadcasts_module_over_batch(self):
        """vmap(f, in_axes=(None, 0)) broadcasts the module across batch dim."""
        rng = make_rng(202)
        model = nb.nn.Linear(4, 3)
        x_np = rng.normal(size=(7, 5, 4)).astype(np.float32)
        x = nb.Tensor.from_dlpack(x_np)

        def f(m, xb):
            return m(xb)

        batched = nb.vmap(f, in_axes=(None, 0), out_axes=0)
        y = batched(model, x)
        assert tuple(int(d) for d in y.shape) == (7, 5, 3)

        w = model.weight.to_numpy()
        b = model.bias.to_numpy() if model.bias is not None else 0.0
        y_ref = np.stack([(x_np[i] @ w) + b for i in range(x_np.shape[0])], axis=0)
        nb.testing.assert_allclose(y, y_ref, rtol=1e-5, atol=1e-6)


# ===----------------------------------------------------------------------=== #
# compile with Module
# ===----------------------------------------------------------------------=== #


class TestCompileWithModule:
    def test_compile_caches_after_first_call(self):
        rng = make_rng(3031)
        model = nb.nn.Linear(4, 3)
        x = nb.Tensor.from_dlpack(rng.normal(size=(6, 4)).astype(np.float32))

        @nb.compile
        def compiled(m, x_in):
            return m(x_in)

        out1 = compiled(model, x)
        out2 = compiled(model, x)

        nb.testing.assert_allclose(out1, out2, rtol=1e-6, atol=1e-6)
        assert compiled.stats.misses == 1
        assert compiled.stats.hits >= 1

    def test_compile_dynamic_dims(self):
        rng = make_rng(910)
        model = nb.nn.Linear(4, 3, bias=False)

        @nb.compile(dynamic_dims={0: {0: "batch"}})
        def compiled(x_in):
            return model(x_in)

        x1 = nb.Tensor.from_dlpack(rng.normal(size=(5, 4)).astype(np.float32))
        x2 = nb.Tensor.from_dlpack(rng.normal(size=(9, 4)).astype(np.float32))

        y1 = compiled(x1)
        y2 = compiled(x2)

        w = model.weight.to_numpy()
        nb.testing.assert_allclose(y1, x1.to_numpy() @ w, rtol=1e-5, atol=1e-6)
        nb.testing.assert_allclose(y2, x2.to_numpy() @ w, rtol=1e-5, atol=1e-6)
        assert compiled.stats.misses == 1
        assert compiled.stats.hits >= 1


# ===----------------------------------------------------------------------=== #
# value_and_grad + training loop
# ===----------------------------------------------------------------------=== #


class TestValueAndGradTraining:
    def test_adamw_training_loop_decreases_loss(self):
        """Multi-step training with value_and_grad + adamw_update reduces loss."""
        rng = make_rng(606)
        x_np = rng.normal(size=(32, 5)).astype(np.float32)
        w_true = rng.normal(size=(5, 3)).astype(np.float32)
        b_true = rng.normal(size=(1, 3)).astype(np.float32)
        y_np = x_np @ w_true + b_true

        x, y = nb.Tensor.from_dlpack(x_np), nb.Tensor.from_dlpack(y_np)
        model = nb.nn.Linear(5, 3)
        opt = nb.nn.optim.adamw_init(model)

        def loss_fn(m):
            return nb.nn.functional.mse_loss(m(x), y)

        start_v = float(loss_fn(model).to_numpy())

        current_model, current_opt = model, opt
        for _ in range(8):
            loss, grads = nb.value_and_grad(loss_fn, argnums=0, realize=False)(current_model)
            current_model, current_opt = nb.nn.optim.adamw_update(
                current_model, grads, current_opt, lr=1e-2, weight_decay=0.0
            )

        end_v = float(loss_fn(current_model).to_numpy())
        assert end_v < start_v


# ===----------------------------------------------------------------------=== #
# Sharded forward (soft skip)
# ===----------------------------------------------------------------------=== #


class TestShardedModule:
    def test_forward_with_sharded_input(self):
        rng = make_rng(911)
        x_np = rng.normal(size=(6, 4)).astype(np.float32)
        model = nb.nn.Linear(4, 3)

        try:
            mesh = nb.DeviceMesh("nn_mesh_tp", (2,), ("tp",))
            x_sharded = nb.Tensor.from_dlpack(x_np).shard(mesh, nb.P(None, "tp"))
            y_sharded = model(x_sharded)
        except Exception as exc:
            pytest.skip(f"Sharding unavailable: {exc}")

        w = model.weight.to_numpy()
        b = model.bias.to_numpy() if model.bias is not None else 0.0
        y_ref = (x_np @ w) + b
        gathered = y_sharded.gather()
        nb.testing.assert_allclose(gathered, y_ref, rtol=1e-4, atol=1e-5)
