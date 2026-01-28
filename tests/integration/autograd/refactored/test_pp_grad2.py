# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import numpy as np
import nabla as nb
from nabla import ops
from nabla.core.sharding import DeviceMesh, PartitionSpec as P, DimSpec
from nabla.transforms import vmap
from nabla.ops import communication
from max.dtype import DType

# --- Project Constants ---
STAGES = 4
MICRO_BATCHES = 8
MICRO_BATCH_SIZE = 4
DIM = 16

def stage_compute(x, w, b):
    """Simple MLP layer with bias: ReLU(X @ W + B)."""
    return ops.relu(ops.matmul(x, w) + b)

def pipeline_step(current_state, fresh_input, weight_stack, bias_stack, mask_0, step_fn, perm):
    """Single step of the GPipe pipeline with bias handling."""
    # Compute sharded step across all stages (vmapped)
    computed = step_fn(current_state, weight_stack, bias_stack)
    
    # Shift activations to the right (circular: i -> i+1)
    shifted = communication.ppermute(computed, perm)
    
    # Extract result from Stage 0 (which holds the wrapped Stage N-1 output)
    res_part = ops.where(mask_0, shifted, ops.zeros_like(shifted))
    result = ops.reduce_sum(res_part, axis=0)

    # Inject fresh input into Stage 0, overwriting the wrapped data
    next_state = ops.where(mask_0, fresh_input, shifted)
    
    return next_state, result

def pipeline_loop(padded_inputs, weight_stack, bias_stack, current_state, mask_0, step_fn, perm, total_steps):
    """Unrolled GPipe execution loop."""
    results = []
    
    for t in range(total_steps):
        # A. Fetch Input (Slice + Squeeze)
        start_idx = (t, 0, 0)
        slice_size = (1, MICRO_BATCH_SIZE, DIM)
        fraction = ops.slice_tensor(padded_inputs, start=start_idx, size=slice_size)
        fresh = ops.squeeze(fraction, axis=0)

        current_state, res = pipeline_step(current_state, fresh, weight_stack, bias_stack, mask_0, step_fn, perm)
        results.append(res)

    return ops.stack(results, axis=0), current_state

def test_pp_grad_with_bias():
    # 1. Setup Mesh
    mesh = DeviceMesh("pp", (STAGES,), ("stage",))
    print(f"Running GPipe Grads Test (Weights + Bias) on Mesh: {mesh}")

    np.random.seed(42)
    
    # 2. Data Setup
    # Weights: (Stages, In, Out)
    w_np = np.random.randn(STAGES, DIM, DIM).astype(np.float32)
    # Bias: (Stages, Out) -> We add a 1 for broadcasting compatibility if needed, 
    # but vmap handles the first dim as the parallel axis.
    b_np = np.random.randn(STAGES, DIM).astype(np.float32)
    
    x_np = np.random.randn(MICRO_BATCHES, MICRO_BATCH_SIZE, DIM).astype(np.float32)
    y_np = np.random.randn(MICRO_BATCHES, MICRO_BATCH_SIZE, DIM).astype(np.float32)

    w_spec = [DimSpec.from_raw(d) for d in P("stage", None, None)]
    b_spec = [DimSpec.from_raw(d) for d in P("stage", None)]
    
    w_sharded = ops.shard(nb.Tensor.from_dlpack(w_np), mesh, w_spec).realize()
    b_sharded = ops.shard(nb.Tensor.from_dlpack(b_np), mesh, b_spec).realize()

    # Pad inputs with zeros for flush phase (Total steps = MB + STAGES)
    padding = np.zeros((STAGES, MICRO_BATCH_SIZE, DIM), dtype=np.float32)
    x_padded_nb = nb.Tensor.from_dlpack(np.concatenate([x_np, padding], axis=0))
    y_nb = nb.Tensor.from_dlpack(y_np) 

    # Initial State (Zeros) - used as the carry across pipeline steps
    state_sharded = ops.shard(nb.zeros((STAGES, MICRO_BATCH_SIZE, DIM), dtype=DType.float32), mesh, w_spec).realize()
    
    # Injection Mask (Stage 0)
    mask_np = np.eye(STAGES, 1).reshape(STAGES, 1, 1).astype(bool)
    mask_0_sharded = ops.shard(nb.Tensor.from_dlpack(mask_np), mesh, w_spec).realize()

    # 3. Communication & VMap Setup
    idx = mesh.axis_names.index("stage")
    size = mesh.shape[idx]
    perm = [(i, (i + 1) % size) for i in range(size)]
    
    # Auto-vectorize the stage calculation over the 'stage' axis
    # in_axes=(0, 0, 0) means x, w, and b are all sharded/vmapped over dim 0
    step_fn = vmap(stage_compute, in_axes=(0, 0, 0), out_axes=0, spmd_axis_name="stage", mesh=mesh)

    # 4. Define Loss Function for Grad
    def pipeline_loss(inputs, weights, biases, state, mask, targets):
        total_steps = MICRO_BATCHES + STAGES
        stream_outputs, _ = pipeline_loop(inputs, weights, biases, state, mask, step_fn, perm, total_steps)

        # Slice valid range [STAGES : STAGES+MB] where results start emerging
        indices = ops.arange(STAGES, STAGES + MICRO_BATCHES, dtype=DType.int64)
        valid_preds = ops.gather(stream_outputs, indices, axis=0)

        # MSE Loss
        diff = valid_preds - targets
        return ops.mean(diff * diff)

    # 5. Compute Gradients
    print("Computing Gradients (W, B)...")
    from nabla.core.autograd import grad
    
    # We differentiate w.r.t. weights (arg 1) and biases (arg 2)
    grad_fn = grad(pipeline_loss, argnums=(1, 2))
    
    w_grad_sharded, b_grad_sharded = grad_fn(x_padded_nb, w_sharded, b_sharded, state_sharded, mask_0_sharded, y_nb)
    
    w_grad_np = w_grad_sharded.to_numpy()
    b_grad_np = b_grad_sharded.to_numpy()

    # 6. Verify against JAX
    print("Running Reference (JAX)...")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", False)
    
    def jax_ref(params_w, params_b, x, y):
        def apply(curr, w, b): return jax.nn.relu(curr @ w + b)
        preds = []
        for i in range(MICRO_BATCHES):
            a = x[i]
            # Sequential application through stages
            for w, b in zip(params_w, params_b): 
                a = apply(a, w, b)
            preds.append(a)
        preds = jnp.stack(preds)
        return jnp.mean((preds - y)**2)

    grad_ref_fn = jax.jit(jax.grad(jax_ref, argnums=(0, 1)))
    w_grad_ref, b_grad_ref = grad_ref_fn(w_np, b_np, x_np, y_np)
    
    # 7. Compare
    w_diff = np.max(np.abs(w_grad_np - w_grad_ref))
    b_diff = np.max(np.abs(b_grad_np - b_grad_ref))
    
    print(f"Max Weight Grad Diff: {w_diff:.6f}")
    print(f"Max Bias Grad Diff:   {b_diff:.6f}")
    
    passed = (w_diff < 5e-4) and (b_diff < 5e-4)
    
    if passed:
        print("✅ SUCCESS: Gradients Match")
    else:
        print("❌ FAILURE: Gradients Mismatch")
        if w_diff >= 5e-4:
            print("Weight Grad Mismatch!")
        if b_diff >= 5e-4:
            print("Bias Grad Mismatch!")

if __name__ == "__main__":
    test_pp_grad_with_bias()
