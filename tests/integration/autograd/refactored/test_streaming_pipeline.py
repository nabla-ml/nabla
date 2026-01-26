
import numpy as np
import pytest
import nabla as nb
from nabla import ops
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.transforms import vmap
from nabla.ops import communication
from .utils import HAS_JAX

np.random.seed(42)

def get_pp_permutation(mesh):
    stage_idx = mesh.axis_names.index("stage")
    stage_size = mesh.shape[stage_idx]
    total = len(mesh.devices)
    perm = []
    for src in range(total):
        coords = list(mesh.get_coordinate(src, ax) for ax in mesh.axis_names)
        coords[stage_idx] = (coords[stage_idx] + 1) % stage_size
        dst = next(d for d in range(total) if list(mesh.get_coordinate(d, ax) for ax in mesh.axis_names) == coords)
        perm.append((src, dst))
    return perm

@pytest.mark.parametrize("dp_size", [2, 4])
def test_streaming_pipeline_shift_register(dp_size):
    print(f"\n=== Testing Streaming Pipeline (Shift Register) [DP={dp_size}] ===")

    # Mesh Setup
    mesh = DeviceMesh("pp_dp", (4, dp_size), ("stage", "data")) # 4 Stages as in user example? User said STAGES=4.
    STAGES = 4
    MICRO_BATCHES = 4 # Reduced for speed, logic holds.
    MICRO_BATCH_SIZE = 2
    DIM = 8

    # Data Gen
    # Stream of inputs: (MICRO_BATCHES, MICRO_BATCH_SIZE, DIM)
    # We pad the specific inputs with zeros to flush the pipe.
    # Total steps = MICRO_BATCHES + STAGES - 1
    # But for strict comparison, let's just run M steps and check logic.
    # User's code runs PADDED input.
    
    x_stream_np = np.random.randn(MICRO_BATCHES, MICRO_BATCH_SIZE, DIM).astype(np.float32)
    targets_np = np.random.randn(MICRO_BATCHES, MICRO_BATCH_SIZE, DIM).astype(np.float32)
    w_np = np.random.randn(STAGES, DIM, DIM).astype(np.float32) * 0.1
    
    # -----------------------------------------------------------------------
    # 1. Nabla Implementation
    # -----------------------------------------------------------------------
    
    # Weights Sharding: (Stage, Dim, Dim)
    # Shard on Stage, Replicate on Data (implied)
    w_nb = nb.Tensor.from_dlpack(w_np)
    w_nb = ops.shard(w_nb, mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])])
    
    # Input Stream not sharded yet (on Host effectively, or replicated)
    x_stream_nb = nb.Tensor.from_dlpack(x_stream_np)
    targets_nb = nb.Tensor.from_dlpack(targets_np)

    # Helper: Dense Layer
    def dense_layer(x, w):
        return ops.relu(nb.matmul(x, w))

    # Kernel: Single Stage Logic
    # Vmap over Data (Batch) dim
    # Inputs: x=(B, D), w=(D, D) -> y=(B, D)
    layer_batched = vmap(dense_layer, in_axes=(0, None), out_axes=0, spmd_axis_name="data", mesh=mesh)

    # Kernel: Distributed Step (Compute + Shift)
    # Vmap over Stage dim
    # Inputs: x_state=(Stage, B, D), w_all=(Stage, D, D)
    layer_stage_batched = vmap(layer_batched, in_axes=(0, 0), out_axes=0, spmd_axis_name="stage", mesh=mesh)

    forward_perm = get_pp_permutation(mesh)

    def compute_and_shift(curr_state, weights):
        # 1. Compute
        y = layer_stage_batched(curr_state, weights)
        # 2. Shift (Ring)
        shifted = communication.ppermute(y, forward_perm)
        return shifted

    def pipeline_forward(x_stream, weights):
        # Init State: Zeros everywhere
        # (STAGES, B, D)
        state = ops.zeros((STAGES, MICRO_BATCH_SIZE, DIM), dtype=x_stream.dtype, device=mesh.devices[0])
        state = ops.shard(state, mesh, [DimSpec(["stage"]), DimSpec(["data"]), DimSpec([])])
        
        all_outputs = []
        
        # Unrolled Loop over Microbatches (Time)
        # We need to loop for len(x_stream).
        # We assume x_stream is indexable (on CPU/Host for control flow) or we unroll.
        # Since 'x_stream' is a Tensor, we iterate range and slice it.
        
        steps = int(x_stream.shape[0])
        
        # Fixed indices
        idx_0 = ops.constant(np.array([0], dtype=np.int32)) # Index 0
        
        for t in range(steps):
            # A. Compute + Shift
            shifted_state = compute_and_shift(state, weights)
            
            # B. Capture Output (Popped from index 0 after wrap)
            # Use gather(axis=0) to slice. 
            # Note: Gather preserves rank? gather(x, [0]) -> (1, B, D).
            # We want (B, D). Squeeze.
            popped = ops.squeeze(ops.gather(shifted_state, idx_0, axis=0), 0)
            all_outputs.append(popped)
            
            # C. Inject Fresh Input
            # Slice stream at t
            idx_t = ops.constant(np.array([t], dtype=np.int32))
            fresh_mb = ops.squeeze(ops.gather(x_stream, idx_t, axis=0), 0)
            
            # Since fresh_mb is (B, D) and state is (Stage, B, D), 
            # we need to inject at index 0.
            # However, `scatter` usually expects updates to match the gathered shape?
            # ops.scatter(state, indices, updates, axis=0)
            # if indices=(1,), updates should be (1, B, D).
            # So unsqueeze fresh_mb.
            fresh_mb_expanded = ops.unsqueeze(fresh_mb, 0)
            
            # Overwrite index 0
            state = ops.scatter(shifted_state, idx_0, fresh_mb_expanded, axis=0)
            
        # Stack outputs
        # List of (B, D) -> (Time, B, D)
        return ops.stack(all_outputs, axis=0)

    def loss_fn(x, w, t):
        # Pad inputs (flush)
        padding = ops.zeros((STAGES - 1, MICRO_BATCH_SIZE, DIM), dtype=x.dtype, device=x.device)
        padded_x = ops.concatenate([x, padding], axis=0)
        
        # Run
        stream_out = pipeline_forward(padded_x, w)
        
        # Valid preds: [STAGES-1 : -1]
        # (Simulated slicing)
        # We know the valid range via integer steps.
        valid_start = STAGES - 1
        
        # We want MICRO_BATCHES items starting from valid_start.
        # Safe integer arithmetic:
        indices_list = list(range(valid_start, valid_start + MICRO_BATCHES))
        valid_indices_np = np.array(indices_list, dtype=np.int32)
        
        valid_indices_nb = nb.Tensor.from_dlpack(valid_indices_np)
        
        valid_preds = ops.gather(stream_out, valid_indices_nb, axis=0)
        
        # MSE
        diff = valid_preds - t
        return ops.mean(diff * diff)

    # Nabla Grad
    traced_loss = nb.core.graph.tracing.trace(loss_fn, x_stream_nb, w_nb, targets_nb)
    cot = nb.Tensor.from_dlpack(np.array(1.0, dtype=np.float32))
    grads = nb.core.autograd.backward_on_trace(traced_loss, cot)
    
    g_w_nb = grads[w_nb].to_numpy()
    g_x_nb = grads[x_stream_nb].to_numpy()

    # -----------------------------------------------------------------------
    # 2. JAX Reference (Exact User Code)
    # -----------------------------------------------------------------------
    if HAS_JAX:
        import jax
        import jax.numpy as jnp
        from jax import vmap as jax_vmap, lax
        # Fake shard_map by loop (to avoid complex mesh setup in test)
        # Or rely on simple vmap reference as before, but modeling standard behavior.
        
        # Since we use `lax.ppermute` logic, we can simulate it sequentially.
        # Or just implement the EXACT logic using loops in JAX (no ppermute needed for logic check).
        
        def ref_dense(x, w):
            return jax.nn.relu(jnp.dot(x, w))
            
        def ref_pipeline(all_inputs, W_bg):
             # Initialize state (S, B, D)
             pipe_state = jnp.zeros((STAGES, MICRO_BATCH_SIZE, DIM))
             
             # Unroll scan
             outs = []
             curr = pipe_state
             
             for i in range(all_inputs.shape[0]):
                 fresh = all_inputs[i]
                 
                 # 1. Compute Layer on each stage
                 # Map over stages
                 # y[s] = layer(curr[s], w[s])
                 y = jax_vmap(ref_dense)(curr, W_bg)
                 
                 # 2. Shift Right (Ring)
                 # [0, 1, 2, 3] -> [3, 0, 1, 2]
                 shifted = jnp.roll(y, shift=1, axis=0)
                 
                 # 3. Capture index 0 (Logic: Wrap around popped)
                 popped = shifted[0]
                 
                 # 4. Inject
                 next_state = shifted.at[0].set(fresh)
                 
                 curr = next_state
                 outs.append(popped)
                 
             return jnp.stack(outs)

        def ref_loss(x, w, t):
            padding = jnp.zeros((STAGES - 1, MICRO_BATCH_SIZE, DIM))
            padded_x = jnp.concatenate([x, padding], axis=0)
            
            stream_out = ref_pipeline(padded_x, w)
            
            # Slice valid [STAGES-1 : STAGES-1 + M]
            valid_preds = stream_out[STAGES-1 : STAGES-1 + MICRO_BATCHES]
            
            return jnp.mean((valid_preds - t) ** 2)
            
        grad_ref_fn = jax.grad(ref_loss, argnums=(0, 1))
        gx_ref, gw_ref = grad_ref_fn(x_stream_np, w_np, targets_np)
        
        # Check
        np.testing.assert_allclose(g_w_nb, gw_ref, rtol=1e-4, atol=1e-4) # Batched, loose tol
        np.testing.assert_allclose(g_x_nb, gx_ref, rtol=1e-4, atol=1e-4)
        print("✓ SUCCESS: Nabla Streaming Pipeline matches JAX Reference")

if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
