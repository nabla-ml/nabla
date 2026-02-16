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
    rng = np.random.default_rng(7071)
    weight_np = rng.normal(size=(12, 10)).astype(np.float32)
    weight = nb.Tensor.from_dlpack(weight_np)
    adapter = nb.nn.finetune.init_lora_adapter(weight, rank=4, init_std=0.02)

    merged = nb.nn.finetune.merge_lora_weight(weight, adapter, alpha=8.0)
    restored = nb.nn.finetune.unmerge_lora_weight(merged, adapter, alpha=8.0)

    assert not merged.real
    assert not restored.real

    a_np = adapter["A"].to_numpy()
    b_np = adapter["B"].to_numpy()
    delta_np = (a_np @ b_np) * (8.0 / a_np.shape[1])
    merged_ref = weight_np + delta_np
    restored_ref = merged_ref - delta_np

    nb.testing.batch_realize(merged, restored)
    nb.testing.assert_allclose(merged, merged_ref, rtol=1e-5, atol=1e-5, realize=False)
    nb.testing.assert_allclose(restored, restored_ref, rtol=1e-5, atol=1e-5, realize=False)


def test_qlora_training_and_checkpoint_roundtrip(tmp_path: Path):
    x_np, y_np, w_np = _make_data(seed=2027, n=64, in_dim=12, out_dim=7)
    x = nb.Tensor.from_dlpack(x_np)
    y = nb.Tensor.from_dlpack(y_np)
    w = nb.Tensor.from_dlpack(w_np)

    qweight = nb.nn.finetune.quantize_nf4(w, block_size=8)
    adapter = nb.nn.finetune.init_lora_adapter(w, rank=3, init_std=0.01)
    opt = nb.nn.optim.adamw_init(adapter)

    w_deq_np = nb.nn.finetune.dequantize_nf4(qweight).to_numpy()
    a0 = adapter["A"].to_numpy()
    b0 = adapter["B"].to_numpy()
    delta0 = (a0 @ b0) * (8.0 / a0.shape[1])
    pred0_ref = x_np @ (w_deq_np + delta0)
    pred0_nb = nb.nn.finetune.qlora_linear(x, qweight, adapter, alpha=8.0)
    nb.testing.assert_allclose(pred0_nb, pred0_ref, rtol=1e-5, atol=1e-5)

    def loss_fn(lora_p):
        pred = nb.nn.finetune.qlora_linear(x, qweight, lora_p, alpha=8.0)
        err = pred - y
        return nb.mean(err * err)

    initial_loss = loss_fn(adapter)
    assert not initial_loss.real
    initial = float(initial_loss.to_numpy())

    for _ in range(12):
        loss, grads = nb.value_and_grad(loss_fn, realize=False)(adapter)
        adapter, opt = nb.nn.optim.adamw_update(adapter, grads, opt, lr=2e-2, weight_decay=0.0)

    final_loss = loss_fn(adapter)
    assert not final_loss.real
    final = float(final_loss.to_numpy())
    assert final < initial

    a_last = adapter["A"].to_numpy()
    b_last = adapter["B"].to_numpy()
    delta_last = (a_last @ b_last) * (8.0 / a_last.shape[1])
    pred_last_ref = x_np @ (w_deq_np + delta_last)
    pred_last_nb = nb.nn.finetune.qlora_linear(x, qweight, adapter, alpha=8.0)
    nb.testing.assert_allclose(pred_last_nb, pred_last_ref, rtol=1e-5, atol=1e-5)

    ckpt = tmp_path / "qlora_lora_ckpt"
    nb.nn.finetune.save_finetune_checkpoint(
        ckpt,
        lora_params=adapter,
        optimizer_state=opt,
        metadata={"tag": "qlora-unit"},
    )

    lora_template = nb.nn.finetune.init_lora_adapter(w, rank=3, init_std=0.01)
    opt_template = nb.nn.optim.adamw_init(lora_template)
    loaded_lora, loaded_opt, _ = nb.nn.finetune.load_finetune_checkpoint(
        ckpt,
        lora_template=lora_template,
        optimizer_template=opt_template,
    )

    pred_ref = nb.nn.finetune.qlora_linear(x, qweight, adapter, alpha=8.0)
    pred_loaded = nb.nn.finetune.qlora_linear(x, qweight, loaded_lora, alpha=8.0)

    nb.testing.batch_realize(pred_ref, pred_loaded)
    nb.testing.assert_allclose(pred_ref, pred_loaded, rtol=1e-5, atol=1e-5, realize=False)
    assert loaded_opt is not None
    assert loaded_opt["step"] == opt["step"]
