# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from pathlib import Path
import time

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
    print(f"[DEBUG] initial_loss={initial:.6f}")

    for step in range(25):
        t0 = time.perf_counter()
        loss, grads = nb.value_and_grad(loss_fn, realize=False)(adapter)
        t1 = time.perf_counter()

        new_adapter, new_opt = nb.adamw_update(
            adapter,
            grads,
            opt,
            lr=3e-2,
            weight_decay=0.0,
        )
        t2 = time.perf_counter()

        to_realize = [loss]
        to_realize.extend(t for t in nb.tree_leaves(grads) if isinstance(t, nb.Tensor))
        to_realize.extend(
            t for t in nb.tree_leaves(new_adapter) if isinstance(t, nb.Tensor)
        )
        to_realize.extend(t for t in nb.tree_leaves(new_opt) if isinstance(t, nb.Tensor))

        nb.realize_all(*to_realize)
        loss_val = float(loss.to_numpy())
        t3 = time.perf_counter()

        adapter, opt = new_adapter, new_opt
        print(
            f"[DEBUG] step={step + 1:02d} loss={loss_val:.6f} "
            f"grad_ms={(t1 - t0) * 1000:.2f} "
            f"opt_ms={(t2 - t1) * 1000:.2f} "
            f"realize_ms={(t3 - t2) * 1000:.2f}"
        )

    final = float(loss_fn(adapter).to_numpy())
    print(f"[DEBUG] final_loss={final:.6f}")
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


def test_qlora_training_and_checkpoint_roundtrip(tmp_path: Path):
    x_np, y_np, w_np = _make_data(seed=2027, n=64, in_dim=12, out_dim=7)
    x = nb.Tensor.from_dlpack(x_np)
    y = nb.Tensor.from_dlpack(y_np)
    w = nb.Tensor.from_dlpack(w_np)

    qweight = nb.quantize_nf4(w, block_size=8)
    adapter = nb.init_lora_adapter(w, rank=3, init_std=0.01)
    opt = nb.adamw_init(adapter)

    def loss_fn(lora_p):
        pred = nb.qlora_linear(x, qweight, lora_p, alpha=8.0)
        err = pred - y
        return nb.mean(err * err)

    initial = float(loss_fn(adapter).to_numpy())

    for _ in range(20):
        loss, grads = nb.value_and_grad(loss_fn, realize=False)(adapter)
        new_adapter, new_opt = nb.adamw_update(
            adapter, grads, opt, lr=2e-2, weight_decay=0.0
        )

        to_realize = [loss]
        to_realize.extend(t for t in nb.tree_leaves(grads) if isinstance(t, nb.Tensor))
        to_realize.extend(
            t for t in nb.tree_leaves(new_adapter) if isinstance(t, nb.Tensor)
        )
        to_realize.extend(t for t in nb.tree_leaves(new_opt) if isinstance(t, nb.Tensor))
        nb.realize_all(*to_realize)

        adapter, opt = new_adapter, new_opt

    final = float(loss_fn(adapter).to_numpy())
    assert final < initial * 0.8, f"Expected meaningful loss drop, got {initial} -> {final}"

    ckpt = tmp_path / "qlora_lora_ckpt"
    nb.save_finetune_checkpoint(
        ckpt,
        lora_params=adapter,
        optimizer_state=opt,
        metadata={"tag": "qlora-unit"},
    )

    lora_template = nb.init_lora_adapter(w, rank=3, init_std=0.01)
    opt_template = nb.adamw_init(lora_template)
    loaded_lora, loaded_opt, _ = nb.load_finetune_checkpoint(
        ckpt,
        lora_template=lora_template,
        optimizer_template=opt_template,
    )

    pred_ref = nb.qlora_linear(x, qweight, adapter, alpha=8.0)
    pred_loaded = nb.qlora_linear(x, qweight, loaded_lora, alpha=8.0)

    np.testing.assert_allclose(
        pred_ref.to_numpy(), pred_loaded.to_numpy(), rtol=1e-5, atol=1e-5
    )
    assert loaded_opt is not None
    assert loaded_opt["step"] == opt["step"]


def test_qlora_repeated_forward_stable():
    rng = np.random.default_rng(2028)
    x_np = rng.normal(size=(8, 6)).astype(np.float32)
    w_np = rng.normal(size=(6, 4)).astype(np.float32)

    x = nb.Tensor.from_dlpack(x_np)
    w = nb.Tensor.from_dlpack(w_np)
    adapter = nb.init_lora_adapter(w, rank=2, init_std=0.01)
    qweight = nb.quantize_nf4(w, block_size=8)

    outputs = []
    for _ in range(4):
        out = nb.qlora_linear(x, qweight, adapter, alpha=6.0)
        nb.realize_all(out)
        outputs.append(out.to_numpy())

    for i in range(1, len(outputs)):
        np.testing.assert_allclose(outputs[0], outputs[i], rtol=1e-6, atol=1e-6)


def test_qlora_resume_training_improves_loss(tmp_path: Path):
    x_np, y_np, w_np = _make_data(seed=2029, n=96, in_dim=14, out_dim=9)
    x = nb.Tensor.from_dlpack(x_np)
    y = nb.Tensor.from_dlpack(y_np)
    w = nb.Tensor.from_dlpack(w_np)

    qweight = nb.quantize_nf4(w, block_size=8)
    adapter = nb.init_lora_adapter(w, rank=4, init_std=0.01)
    opt = nb.adamw_init(adapter)

    def loss_fn(lora_p):
        pred = nb.qlora_linear(x, qweight, lora_p, alpha=10.0)
        err = pred - y
        return nb.mean(err * err)

    def run_steps(adapter_p, opt_state, steps: int):
        for _ in range(steps):
            loss, grads = nb.value_and_grad(loss_fn, realize=False)(adapter_p)
            new_adapter, new_opt = nb.adamw_update(
                adapter_p, grads, opt_state, lr=1.5e-2, weight_decay=0.0
            )
            to_realize = [loss]
            to_realize.extend(
                t for t in nb.tree_leaves(grads) if isinstance(t, nb.Tensor)
            )
            to_realize.extend(
                t for t in nb.tree_leaves(new_adapter) if isinstance(t, nb.Tensor)
            )
            to_realize.extend(
                t for t in nb.tree_leaves(new_opt) if isinstance(t, nb.Tensor)
            )
            nb.realize_all(*to_realize)
            adapter_p, opt_state = new_adapter, new_opt
        return adapter_p, opt_state

    initial = float(loss_fn(adapter).to_numpy())
    adapter, opt = run_steps(adapter, opt, 12)
    mid = float(loss_fn(adapter).to_numpy())

    ckpt = tmp_path / "qlora_resume_ckpt"
    nb.save_finetune_checkpoint(
        ckpt,
        lora_params=adapter,
        optimizer_state=opt,
        metadata={"phase": "resume"},
    )

    lora_template = nb.init_lora_adapter(w, rank=4, init_std=0.01)
    opt_template = nb.adamw_init(lora_template)
    loaded_lora, loaded_opt, _ = nb.load_finetune_checkpoint(
        ckpt,
        lora_template=lora_template,
        optimizer_template=opt_template,
    )

    assert loaded_opt is not None
    step_before = int(loaded_opt["step"])

    loaded_lora, loaded_opt = run_steps(loaded_lora, loaded_opt, 12)
    final = float(loss_fn(loaded_lora).to_numpy())

    assert mid < initial, f"Expected mid loss to improve: {initial} -> {mid}"
    assert final < mid, f"Expected resumed training to improve: {mid} -> {final}"
    assert int(loaded_opt["step"]) == step_before + 12


def test_compiled_qlora_captured_qweight_cache_hits():
    rng = np.random.default_rng(2030)
    x_np = rng.normal(size=(8, 6)).astype(np.float32)
    w_np = rng.normal(size=(6, 4)).astype(np.float32)

    x = nb.Tensor.from_dlpack(x_np)
    w = nb.Tensor.from_dlpack(w_np)
    adapter = nb.init_lora_adapter(w, rank=2, init_std=0.01)
    qweight = nb.quantize_nf4(w, block_size=8)

    @nb.compile
    def compiled_qlora(x_in, a, b):
        return nb.qlora_linear(x_in, qweight, {"A": a, "B": b}, alpha=6.0)

    out1 = compiled_qlora(x, adapter["A"], adapter["B"])
    out2 = compiled_qlora(x, adapter["A"], adapter["B"])

    np.testing.assert_allclose(out1.to_numpy(), out2.to_numpy(), rtol=1e-6, atol=1e-6)
    assert compiled_qlora.stats.misses == 1
    assert compiled_qlora.stats.hits >= 1
