# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for QLoRA quantization and qlora_linear."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng


# ===----------------------------------------------------------------------=== #
# quantize / dequantize NF4
# ===----------------------------------------------------------------------=== #


class TestQuantizeNF4:
    def test_quantize_returns_expected_keys(self):
        rng = make_rng(80)
        w = nb.Tensor.from_dlpack(rng.normal(size=(8, 6)).astype(np.float32))
        qw = nb.nn.finetune.quantize_nf4(w, block_size=8)

        assert "indices" in qw
        assert "scales" in qw
        assert "original_shape" in qw
        assert qw["original_shape"] == (8, 6)

    def test_dequantize_shape_matches_original(self):
        rng = make_rng(81)
        w = nb.Tensor.from_dlpack(rng.normal(size=(12, 8)).astype(np.float32))
        qw = nb.nn.finetune.quantize_nf4(w, block_size=8)
        w_deq = nb.nn.finetune.dequantize_nf4(qw)
        assert tuple(int(d) for d in w_deq.shape) == (12, 8)

    def test_dequantize_approximately_recovers_original(self):
        """NF4 quantization is lossy, but should be within reasonable error."""
        rng = make_rng(82)
        w_np = rng.normal(size=(16, 16)).astype(np.float32)
        w = nb.Tensor.from_dlpack(w_np)
        qw = nb.nn.finetune.quantize_nf4(w, block_size=8)
        w_deq = nb.nn.finetune.dequantize_nf4(qw)
        w_deq_np = np.asarray(w_deq)

        # NF4 has 16 levels — expect modest reconstruction error
        rel_err = np.abs(w_deq_np - w_np).mean() / (np.abs(w_np).mean() + 1e-8)
        assert rel_err < 0.3  # generous bound for 4-bit


# ===----------------------------------------------------------------------=== #
# qlora_linear
# ===----------------------------------------------------------------------=== #


class TestQLoRALinear:
    def test_qlora_linear_forward_deterministic(self):
        rng = make_rng(83)
        x_np = rng.normal(size=(4, 10)).astype(np.float32)
        w_np = rng.normal(size=(10, 6)).astype(np.float32)

        x = nb.Tensor.from_dlpack(x_np)
        w = nb.Tensor.from_dlpack(w_np)
        qw = nb.nn.finetune.quantize_nf4(w, block_size=8)
        adapter = nb.nn.finetune.init_lora_adapter(w, rank=3, init_std=0.01)

        y1 = nb.nn.finetune.qlora_linear(x, qw, adapter, alpha=8.0)
        y2 = nb.nn.finetune.qlora_linear(x, qw, adapter, alpha=8.0)

        nb.testing.assert_allclose(y1, y2, rtol=1e-6, atol=1e-6)

    def test_qlora_training_reduces_loss(self):
        """End-to-end QLoRA training loop reduces MSE loss."""
        rng = make_rng(2027)
        n, in_dim, out_dim = 64, 12, 7
        x_np = rng.normal(size=(n, in_dim)).astype(np.float32)
        w_np = rng.normal(size=(in_dim, out_dim)).astype(np.float32)
        u = rng.normal(size=(in_dim, 2)).astype(np.float32)
        v = rng.normal(size=(2, out_dim)).astype(np.float32)
        y_np = (x_np @ (w_np + 0.2 * (u @ v))).astype(np.float32)

        x = nb.Tensor.from_dlpack(x_np)
        y = nb.Tensor.from_dlpack(y_np)
        w = nb.Tensor.from_dlpack(w_np)

        qw = nb.nn.finetune.quantize_nf4(w, block_size=8)
        adapter = nb.nn.finetune.init_lora_adapter(w, rank=3, init_std=0.01)
        opt = nb.nn.optim.adamw_init(adapter)

        def loss_fn(lora_p):
            pred = nb.nn.finetune.qlora_linear(x, qw, lora_p, alpha=8.0)
            err = pred - y
            return nb.mean(err * err)

        initial = float(loss_fn(adapter).to_numpy())

        for _ in range(12):
            _, grads = nb.value_and_grad(loss_fn, realize=False)(adapter)
            adapter, opt = nb.nn.optim.adamw_update(
                adapter, grads, opt, lr=2e-2, weight_decay=0.0
            )

        final = float(loss_fn(adapter).to_numpy())
        assert final < initial
