# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""QLoRA fine-tuning MVP in pure Nabla.

This v1 uses NF4 quantization with uint8 index storage (unpacked) and trains
LoRA adapters on top of frozen quantized weights.
"""

# %% [markdown]
# # Example 11: QLoRA Fine-Tuning MVP
#
# This example mirrors LoRA fine-tuning with quantized base weights:
# - NF4 quantization of frozen weights
# - LoRA adapter training on quantized weights
# - quick quality checks (loss drop + quantization error)

# %%

from __future__ import annotations

import numpy as np

import nabla as nb


def make_regression_data(
    n_samples: int, in_dim: int, out_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(321)
    x = rng.normal(size=(n_samples, in_dim)).astype(np.float32)

    w_base = rng.normal(size=(in_dim, out_dim)).astype(np.float32) * 0.4
    u = rng.normal(size=(in_dim, 4)).astype(np.float32)
    v = rng.normal(size=(4, out_dim)).astype(np.float32)
    delta = 0.30 * (u @ v)

    y = x @ (w_base + delta)
    return x, y.astype(np.float32), w_base.astype(np.float32)


def main() -> None:
    in_dim, out_dim = 64, 32
    rank = 8
    alpha = 16.0
    learning_rate = 2e-2
    steps = 120

    x_np, y_np, w_base_np = make_regression_data(
        n_samples=512, in_dim=in_dim, out_dim=out_dim
    )

    x = nb.Tensor.from_dlpack(x_np)
    y = nb.Tensor.from_dlpack(y_np)
    frozen_weight = nb.Tensor.from_dlpack(w_base_np)

    qweight = nb.nn.finetune.quantize_nf4(frozen_weight, block_size=64)
    dense_recon = nb.nn.finetune.dequantize_nf4(qweight)
    quant_rel_err = float(
        np.linalg.norm(dense_recon.to_numpy() - frozen_weight.to_numpy())
        / (np.linalg.norm(frozen_weight.to_numpy()) + 1e-8)
    )
    print(f"NF4 relative reconstruction error: {quant_rel_err:.4f}")

    lora_params = nb.nn.finetune.init_lora_adapter(
        frozen_weight, rank=rank, init_std=0.01
    )
    opt_state = nb.nn.optim.adamw_init(lora_params)

    def loss_fn(adapter, batch_x, batch_y):
        pred = nb.nn.finetune.qlora_linear(
            batch_x,
            qweight,
            adapter,
            alpha=alpha,
            compute_dtype=nb.DType.float32,
        )
        diff = pred - batch_y
        return nb.mean(diff * diff)

    def train_step(adapter, optimizer_state, batch_x, batch_y):
        loss, grads = nb.value_and_grad(loss_fn, argnums=0, realize=False)(
            adapter, batch_x, batch_y
        )
        new_adapter, new_state = nb.nn.optim.adamw_update(
            adapter,
            grads,
            optimizer_state,
            lr=learning_rate,
            weight_decay=0.0,
        )
        to_realize = [loss]
        to_realize.extend(t for t in nb.tree_leaves(grads) if isinstance(t, nb.Tensor))
        to_realize.extend(
            t for t in nb.tree_leaves(new_adapter) if isinstance(t, nb.Tensor)
        )
        to_realize.extend(
            t for t in nb.tree_leaves(new_state) if isinstance(t, nb.Tensor)
        )
        nb.realize_all(*to_realize)
        return loss, new_adapter, new_state

    initial_loss = float(loss_fn(lora_params, x, y).to_numpy())
    print(f"Initial loss: {initial_loss:.6f}")

    for step in range(steps):
        loss, lora_params, opt_state = train_step(lora_params, opt_state, x, y)
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1:>3d}: loss={float(loss.to_numpy()):.6f}")

    final_loss = float(loss_fn(lora_params, x, y).to_numpy())
    print(f"Final loss:   {final_loss:.6f}")

    if final_loss >= initial_loss:
        raise RuntimeError("QLoRA training did not reduce loss.")

    print("✅ QLoRA MVP finished successfully.")


if __name__ == "__main__":
    main()
