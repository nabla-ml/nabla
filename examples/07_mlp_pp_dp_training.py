# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""2D parallelism: pipeline + data parallel.

Demonstrates: 2D mesh (dp=2, pp=4), sharding weights on pp axis, sharding data on dp axis,
combined PP+DP gradient computation compared against JAX.
"""

# %% [markdown]
# # Example 7: 2D Parallel Training (PP + DP)
#
# This example extends pipeline parallelism with data parallelism:
# - a 2D device mesh (`dp`, `pp`)
# - sharded parameters and sharded batches
# - correctness checks against a JAX baseline

# %%

import numpy as np
from max.dtype import DType

import nabla as nb
from nabla import ops
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.ops import communication
from nabla.transforms import vmap

# --- Project Constants ---
DP_SIZE = 2
PP_SIZE = 4
TOTAL_DEVICES = DP_SIZE * PP_SIZE

MICRO_BATCHES = 4
MICRO_BATCH_SIZE = 4  # Total batch size per step
DIM = 16

# %% [markdown]
# ## 1. Define 2D Pipeline Helpers
#
# These functions implement one pp+dp pipeline step and the loop over micro-batches.

# %%


def stage_compute(x, w, b):
    return ops.relu(ops.matmul(x, w) + b)


def pipeline_step(
    current_state, fresh_input, weight_stack, bias_stack, mask_0, step_fn, perm
):
    """Single 2D pipeline step: compute -> shift -> extract -> inject."""
    computed = step_fn(current_state, weight_stack, bias_stack)
    shifted = communication.ppermute(computed, perm)
    res_part = ops.where(mask_0, shifted, ops.zeros_like(shifted))
    result = ops.reduce_sum(res_part, axis=0)
    next_state = ops.where(mask_0, fresh_input, shifted)
    return next_state, result


def pipeline_loop(
    padded_inputs,
    weight_stack,
    bias_stack,
    current_state,
    mask_0,
    step_fn,
    perm,
    total_steps,
):
    results = []
    for t in range(total_steps):
        # Fetch Input
        start_idx = (t, 0, 0)
        slice_size = (1, MICRO_BATCH_SIZE, DIM)
        fraction = ops.slice_tensor(padded_inputs, start=start_idx, size=slice_size)
        fresh = ops.squeeze(fraction, axis=0)

        current_state, res = pipeline_step(
            current_state, fresh, weight_stack, bias_stack, mask_0, step_fn, perm
        )
        results.append(res)
    return ops.stack(results, axis=0), current_state


# %% [markdown]
# ## 2. Run 2D Gradient Check
#
# Build sharded inputs/weights and compare gradients with a JAX reference.

# %%


def test_pp_dp_grad():
    mesh = DeviceMesh("2d", (DP_SIZE, PP_SIZE), ("dp", "pp"))
    print(f"Running 2D Parallelism Test (DP={DP_SIZE}, PP={PP_SIZE})")

    np.random.seed(42)

    w_np = np.random.randn(PP_SIZE, DIM, DIM).astype(np.float32)
    b_np = np.random.randn(PP_SIZE, DIM).astype(np.float32)
    total_steps = MICRO_BATCHES + PP_SIZE
    x_np = np.random.randn(MICRO_BATCHES, MICRO_BATCH_SIZE, DIM).astype(np.float32)
    y_np = np.random.randn(MICRO_BATCHES, MICRO_BATCH_SIZE, DIM).astype(np.float32)

    # Weights sharded on 'pp', replicated on 'dp'
    w_spec = [DimSpec.from_raw("pp"), None, None]
    b_spec = [DimSpec.from_raw("pp"), None]

    # Data sharded on 'dp'
    x_padded_np = np.concatenate(
        [x_np, np.zeros((PP_SIZE, MICRO_BATCH_SIZE, DIM), dtype=np.float32)], axis=0
    )
    x_spec = [None, DimSpec.from_raw("dp"), None]

    w_sharded = ops.shard(nb.Tensor.from_dlpack(w_np), mesh, w_spec).realize()
    b_sharded = ops.shard(nb.Tensor.from_dlpack(b_np), mesh, b_spec).realize()
    x_sharded = ops.shard(nb.Tensor.from_dlpack(x_padded_np), mesh, x_spec).realize()
    y_sharded = ops.shard(nb.Tensor.from_dlpack(y_np), mesh, x_spec).realize()

    state_spec = [DimSpec.from_raw("pp"), DimSpec.from_raw("dp"), None]
    state_sharded = ops.shard(
        nb.zeros((PP_SIZE, MICRO_BATCH_SIZE, DIM), dtype=DType.float32),
        mesh,
        state_spec,
    ).realize()

    mask_np = np.eye(PP_SIZE, 1).reshape(PP_SIZE, 1, 1).astype(bool)
    mask_spec = [DimSpec.from_raw("pp"), None, None]
    mask_sharded = ops.shard(nb.Tensor.from_dlpack(mask_np), mesh, mask_spec).realize()

    idx = mesh.axis_names.index("pp")
    size = mesh.shape[idx]
    perm = []
    for dp in range(DP_SIZE):
        for src_pp in range(PP_SIZE):
            src = dp * PP_SIZE + src_pp
            dst = dp * PP_SIZE + (src_pp + 1) % size
            perm.append((src, dst))

    step_fn = vmap(
        stage_compute, in_axes=(0, 0, 0), out_axes=0, spmd_axis_name="pp", mesh=mesh
    )

    def pipeline_loss(inputs, weights, biases, state, mask, targets):
        stream_outputs, _ = pipeline_loop(
            inputs, weights, biases, state, mask, step_fn, perm, total_steps
        )
        indices = ops.arange(PP_SIZE, PP_SIZE + MICRO_BATCHES, dtype=DType.int64)
        valid_preds = ops.gather(stream_outputs, indices, axis=0)
        diff = valid_preds - targets
        return ops.mean(diff * diff)

    print("Computing 2D Parallel Gradients...")
    from nabla.core.autograd import value_and_grad

    grad_fn = value_and_grad(pipeline_loss, argnums=(1, 2))
    (loss_nb, (w_grad, b_grad)) = grad_fn(
        x_sharded, w_sharded, b_sharded, state_sharded, mask_sharded, y_sharded
    )
    print(f"Nabla Loss: {loss_nb.item():.6f}")

    w_grad_np = w_grad.to_numpy()
    b_grad_np = b_grad.to_numpy()

    print("Running JAX Reference...")
    import jax
    import jax.numpy as jnp

    def jax_ref(pw, pb, px, py):
        def apply(curr, w, b):
            return jax.nn.relu(curr @ w + b)

        preds = []
        for i in range(MICRO_BATCHES):
            a = px[i]
            for w, b in zip(pw, pb, strict=False):
                a = apply(a, w, b)
            preds.append(a)
        preds = jnp.stack(preds)
        return jnp.mean((preds - py) ** 2)

    jax_val_grad_fn = jax.value_and_grad(jax_ref, argnums=(0, 1))
    loss_jax, (w_ref, b_ref) = jax_val_grad_fn(w_np, b_np, x_np, y_np)
    print(f"JAX Loss:   {loss_jax:.6f}")

    # 7. Compare
    print("Nabla Weights Grad Sample:", w_grad_np[0, 0, :3])
    print("JAX Weights Grad Sample:  ", w_ref[0, 0, :3])

    w_diff = np.max(np.abs(w_grad_np - w_ref))
    b_diff = np.max(np.abs(b_grad_np - b_ref))
    print(f"Max 2D Diff - Weights: {w_diff:.6f}, Bias: {b_diff:.6f}")

    if w_diff < 5e-4 and b_diff < 5e-4:
        print("✅ SUCCESS: 2D Parallel Gradients Match")
    else:
        print("❌ FAILURE")


if __name__ == "__main__":
    test_pp_dp_grad()
