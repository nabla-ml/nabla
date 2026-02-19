# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""LoRA fine-tuning MVP in pure Nabla.

This example trains only a LoRA adapter on top of a frozen linear weight and
verifies checkpoint round-trip correctness.
"""

# %% [markdown]
# # Example 10: LoRA Fine-Tuning MVP
#
# This example shows a minimal parameter-efficient fine-tuning workflow:
# - keep base weights frozen
# - train only LoRA adapters
# - save and reload a finetune checkpoint

# %%

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

import nabla as nb

# %% [markdown]
# ## 1. Synthetic Data Helper

# %%


def make_regression_data(
    n_samples: int, in_dim: int, out_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, in_dim)).astype(np.float32)

    w_base = rng.normal(size=(in_dim, out_dim)).astype(np.float32) * 0.5
    u = rng.normal(size=(in_dim, 4)).astype(np.float32)
    v = rng.normal(size=(4, out_dim)).astype(np.float32)
    delta = 0.35 * (u @ v)

    y = x @ (w_base + delta)
    return x, y.astype(np.float32), w_base.astype(np.float32)


# %% [markdown]
# ## 2. Train Adapter and Validate Checkpoint

# %%


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

    lora_params = nb.nn.finetune.init_lora_adapter(
        frozen_weight, rank=rank, init_std=0.01
    )
    opt_state = nb.nn.optim.adamw_init(lora_params)

    def loss_fn(adapter, batch_x, batch_y):
        pred = nb.nn.finetune.lora_linear(batch_x, frozen_weight, adapter, alpha=alpha)
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

    ckpt_dir = Path(".tmp_lora_ckpt")
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    nb.nn.finetune.save_finetune_checkpoint(
        ckpt_dir,
        lora_params=lora_params,
        optimizer_state=opt_state,
        metadata={"alpha": alpha, "rank": rank},
    )

    lora_template = nb.nn.finetune.init_lora_adapter(
        frozen_weight, rank=rank, init_std=0.01
    )
    opt_template = nb.nn.optim.adamw_init(lora_template)

    loaded_lora, loaded_opt, meta = nb.nn.finetune.load_finetune_checkpoint(
        ckpt_dir,
        lora_template=lora_template,
        optimizer_template=opt_template,
    )

    original_pred = nb.nn.finetune.lora_linear(
        x, frozen_weight, lora_params, alpha=alpha
    )
    loaded_pred = nb.nn.finetune.lora_linear(x, frozen_weight, loaded_lora, alpha=alpha)

    max_diff = np.max(np.abs(original_pred.to_numpy() - loaded_pred.to_numpy()))
    print(f"Checkpoint step: {loaded_opt['step'] if loaded_opt else 'N/A'}")
    print(f"Checkpoint max prediction diff: {max_diff:.8f}")
    print(f"Checkpoint metadata keys: {sorted(meta.get('user_metadata', {}).keys())}")

    if final_loss >= initial_loss:
        raise RuntimeError("LoRA training did not reduce loss.")
    if max_diff > 1e-5:
        raise RuntimeError(f"Checkpoint roundtrip mismatch too large: {max_diff}")

    print("✅ LoRA MVP finished successfully.")


if __name__ == "__main__":
    main()
