# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from pathlib import Path

import numpy as np

import nabla as nb


def _make_data(seed: int = 0, n: int = 48, in_dim: int = 10, out_dim: int = 6):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, in_dim)).astype(np.float32)
    w = rng.normal(size=(in_dim, out_dim)).astype(np.float32)
    u = rng.normal(size=(in_dim, 2)).astype(np.float32)
    v = rng.normal(size=(2, out_dim)).astype(np.float32)
    y = x @ (w + 0.2 * (u @ v))
    return x, y.astype(np.float32), w.astype(np.float32)


def test_lora_merge_unmerge_roundtrip():
    weight = nb.Tensor.from_dlpack(np.random.randn(12, 10).astype(np.float32))
    adapter = nb.init_lora_adapter(weight, rank=4, init_std=0.02)

    merged = nb.merge_lora_weight(weight, adapter, alpha=8.0)
    restored = nb.unmerge_lora_weight(merged, adapter, alpha=8.0)

    np.testing.assert_allclose(restored.to_numpy(), weight.to_numpy(), rtol=1e-5, atol=1e-5)


def test_lora_training_reduces_loss():
    x_np, y_np, w_np = _make_data(seed=123)
    x = nb.Tensor.from_dlpack(x_np)
    y = nb.Tensor.from_dlpack(y_np)
    w = nb.Tensor.from_dlpack(w_np)

    adapter = nb.init_lora_adapter(w, rank=4, init_std=0.01)
    opt = nb.adamw_init(adapter)

    def loss_fn(lora_p):
        pred = nb.lora_linear(x, w, lora_p, alpha=8.0)
        err = pred - y
        return nb.mean(err * err)

    initial = float(loss_fn(adapter).to_numpy())

    for _ in range(25):
        loss, grads = nb.value_and_grad(loss_fn)(adapter)
        adapter, opt = nb.adamw_update(adapter, grads, opt, lr=3e-2, weight_decay=0.0)

    final = float(loss_fn(adapter).to_numpy())
    assert final < initial * 0.8, f"Expected meaningful loss drop, got {initial} -> {final}"


def test_lora_checkpoint_roundtrip(tmp_path: Path):
    x_np, y_np, w_np = _make_data(seed=77)
    x = nb.Tensor.from_dlpack(x_np)
    y = nb.Tensor.from_dlpack(y_np)
    w = nb.Tensor.from_dlpack(w_np)

    adapter = nb.init_lora_adapter(w, rank=4, init_std=0.01)
    opt = nb.adamw_init(adapter)

    def loss_fn(lora_p):
        pred = nb.lora_linear(x, w, lora_p, alpha=8.0)
        err = pred - y
        return nb.mean(err * err)

    for _ in range(8):
        loss, grads = nb.value_and_grad(loss_fn)(adapter)
        adapter, opt = nb.adamw_update(adapter, grads, opt, lr=2e-2, weight_decay=0.0)

    ckpt = tmp_path / "lora_ckpt"
    nb.save_finetune_checkpoint(ckpt, lora_params=adapter, optimizer_state=opt, metadata={"tag": "unit"})

    lora_template = nb.init_lora_adapter(w, rank=4, init_std=0.01)
    opt_template = nb.adamw_init(lora_template)

    loaded_lora, loaded_opt, _ = nb.load_finetune_checkpoint(
        ckpt,
        lora_template=lora_template,
        optimizer_template=opt_template,
    )

    pred_ref = nb.lora_linear(x, w, adapter, alpha=8.0)
    pred_loaded = nb.lora_linear(x, w, loaded_lora, alpha=8.0)

    np.testing.assert_allclose(pred_ref.to_numpy(), pred_loaded.to_numpy(), rtol=1e-5, atol=1e-5)
    assert loaded_opt is not None
    assert loaded_opt["step"] == opt["step"]


def test_nf4_quantization_and_qlora_linear():
    rng = np.random.default_rng(2026)
    x_np = rng.normal(size=(16, 8)).astype(np.float32)
    w_np = rng.normal(size=(8, 5)).astype(np.float32)

    x = nb.Tensor.from_dlpack(x_np)
    w = nb.Tensor.from_dlpack(w_np)

    qweight = nb.quantize_nf4(w, block_size=8)
    w_recon = nb.dequantize_nf4(qweight)

    rel_err = np.linalg.norm(w_recon.to_numpy() - w_np) / (np.linalg.norm(w_np) + 1e-8)
    assert rel_err < 0.25

    adapter = nb.init_lora_adapter(w, rank=2, init_std=0.01)
    out = nb.qlora_linear(x, qweight, adapter, alpha=6.0)
    assert tuple(int(d) for d in out.shape) == (16, 5)
