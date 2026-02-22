# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import numpy as np
from max.dtype import DType

import nabla as nb
from nabla import ops
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.core.sharding import PartitionSpec as P
from nabla.ops import communication
from nabla.transforms import vmap

# --- Project Constants ---
STAGES = 4
MICRO_BATCHES = 8
MICRO_BATCH_SIZE = 4
DIM = 16


def stage_compute(x, w):
    """Simple MLP layer: ReLU(X @ W)."""
    return ops.relu(ops.matmul(x, w))


def pipeline_step(current_state, fresh_input, weight_stack, mask_0, step_fn, perm):
    """Single step of the GPipe pipeline: Compute -> Shift -> Extract -> Inject."""
    # Compute sharded step across all stages
    computed = step_fn(current_state, weight_stack)

    # Shift activations to the right (circular: i -> i+1)
    shifted = communication.ppermute(computed, perm)

    # Extract result from Stage 0 (which holds the wrapped Stage N-1 output)
    res_part = ops.where(mask_0, shifted, ops.zeros_like(shifted))
    result = ops.reduce_sum(res_part, axis=0)

    # Inject fresh input into Stage 0, overwriting the wrapped data
    next_state = ops.where(mask_0, fresh_input, shifted)

    return next_state, result


def pipeline_loop(
    padded_inputs, weight_stack, current_state, mask_0, step_fn, perm, total_steps
):
    """Unrolled GPipe execution loop."""
    results = []

    for t in range(total_steps):
        # A. Fetch Input (Slice + Squeeze)
        start_idx = (t, 0, 0)
        slice_size = (1, MICRO_BATCH_SIZE, DIM)
        fraction = ops.slice_tensor(padded_inputs, start=start_idx, size=slice_size)
        fresh = ops.squeeze(fraction, axis=0)

        current_state, res = pipeline_step(
            current_state, fresh, weight_stack, mask_0, step_fn, perm
        )
        results.append(res)

    return ops.stack(results, axis=0), current_state


def test_pp_grad_clean():
    # 1. Setup Mesh
    mesh = DeviceMesh("pp", (STAGES,), ("stage",))
    print(f"Running Clean GPipe Grads Test on Mesh: {mesh}")

    np.random.seed(42)

    # 2. Data Setup
    w_np = np.random.randn(STAGES, DIM, DIM).astype(np.float32)
    x_np = np.random.randn(MICRO_BATCHES, MICRO_BATCH_SIZE, DIM).astype(np.float32)
    y_np = np.random.randn(MICRO_BATCHES, MICRO_BATCH_SIZE, DIM).astype(np.float32)

    w_spec = [DimSpec.from_raw(d) for d in P("stage", None, None)]
    w_sharded = ops.shard(nb.Tensor.from_dlpack(w_np), mesh, w_spec).realize()

    # Pad inputs with zeros for flush phase (Total steps = MB + STAGES)
    padding = np.zeros((STAGES, MICRO_BATCH_SIZE, DIM), dtype=np.float32)
    x_padded_nb = nb.Tensor.from_dlpack(np.concatenate([x_np, padding], axis=0))
    y_nb = nb.Tensor.from_dlpack(y_np)

    # Initial State (Zeros)
    state_sharded = ops.shard(
        nb.zeros((STAGES, MICRO_BATCH_SIZE, DIM), dtype=DType.float32), mesh, w_spec
    ).realize()

    # Injection Mask (Stage 0)
    mask_np = np.eye(STAGES, 1).reshape(STAGES, 1, 1).astype(bool)
    mask_0_sharded = ops.shard(nb.Tensor.from_dlpack(mask_np), mesh, w_spec).realize()

    # 3. Communication & VMap Setup
    idx = mesh.axis_names.index("stage")
    size = mesh.shape[idx]
    perm = [(i, (i + 1) % size) for i in range(size)]

    # Auto-vectorize the stage calculation over the 'stage' axis
    step_fn = vmap(
        stage_compute, in_axes=(0, 0), out_axes=0, spmd_axis_name="stage", mesh=mesh
    )

    # 4. Define Loss Function for Grad
    def pipeline_loss(inputs, weights, state, mask, targets):
        total_steps = MICRO_BATCHES + STAGES
        stream_outputs, _ = pipeline_loop(
            inputs, weights, state, mask, step_fn, perm, total_steps
        )

        # Slice valid range [STAGES : STAGES+MB]
        indices = ops.arange(STAGES, STAGES + MICRO_BATCHES, dtype=DType.int64)
        valid_preds = ops.gather(stream_outputs, indices, axis=0)

        # MSE Loss
        diff = valid_preds - targets
        return ops.mean(diff * diff)

    # 5. Compute Gradients
    print("Computing Gradients...")
    from nabla.transforms import grad

    grad_fn = grad(pipeline_loss, argnums=1)  # weights is second argument

    w_grad_sharded = grad_fn(
        x_padded_nb, w_sharded, state_sharded, mask_0_sharded, y_nb
    )
    w_grad_np = w_grad_sharded.to_numpy()

    # 6. Verify against JAX
    print("Running Reference (JAX)...")
    import jax
    import jax.numpy as jnp

    # Force float32 consistency
    jax.config.update("jax_enable_x64", False)

    def jax_ref(params, x, y):
        def apply(curr, w):
            return jax.nn.relu(curr @ w)

        preds = []
        for i in range(MICRO_BATCHES):
            a = x[i]
            for w in params:
                a = apply(a, w)
            preds.append(a)
        preds = jnp.stack(preds)
        return jnp.mean((preds - y) ** 2)

    grad_ref_fn = jax.jit(jax.grad(jax_ref, argnums=0))
    w_grad_ref = grad_ref_fn(w_np, x_np, y_np)

    # 7. Compare
    diff = np.max(np.abs(w_grad_np - w_grad_ref))
    print(f"Max Grad Diff: {diff:.6f}")

    if diff < 5e-4:
        print("✅ SUCCESS: Gradients Match")
    else:
        print("❌ FAILURE: Gradients Mismatch")
        print("NB Grad Sample:", w_grad_np[0, 0, :5])
        print("Ref Grad Sample:", w_grad_ref[0, 0, :5])


if __name__ == "__main__":
    test_pp_grad_clean()
