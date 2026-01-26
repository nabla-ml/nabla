import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
jax.config.update("jax_enable_x64", True) 

import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

# --- Configuration ---
STAGES = 4
MICRO_BATCHES = 32
MICRO_BATCH_SIZE = 4
DIM = 16
STASH_SIZE = 8  # Constant Memory (2 * STAGES)

# --- Model ---
def dense_layer(x, w):
    return jax.nn.relu(x @ w)

# --- Production 1F1B Step ---
def step_1f1b_fn(state, inputs, constants):
    """
    Executes one step. Now handles Targets as a traveling packet.
    """
    # Unpack State
    grads, stash, w_idx, r_idx = state
    
    # Unpack Inputs (Now includes Targets!)
    # fwd_in:  Data arriving from Left
    # fwd_tgt: Targets arriving from Left
    # fwd_msk: Validity mask
    fwd_in, fwd_tgt, fwd_msk, bwd_in, bwd_msk = inputs
    
    weights, stage_idx = constants
    is_last_stage = (stage_idx == STAGES - 1)

    # --- A. Forward Pass ---
    # 1. Compute Layer
    fwd_out_raw, vjp_fn = jax.vjp(dense_layer, fwd_in, weights)
    
    # 2. Compute Loss (If Last Stage)
    # The target arriving here (fwd_tgt) is the correct one for this data (fwd_in)
    # because they traveled the ring together.
    loss_grad = (fwd_out_raw - fwd_tgt) / MICRO_BATCHES
    
    # 3. Mux Output
    # If Last Stage: Output is Gradient. Else: Output is Activation.
    fwd_out = jnp.where(is_last_stage, loss_grad, fwd_out_raw)
    
    # 4. Mask Bubbles
    fwd_out = jnp.where(fwd_msk, fwd_out, 0)

    # 5. Stash Input (Const memory ring buffer)
    x_input = (fwd_in,)
    stash = jax.lax.dynamic_update_slice(stash, x_input[0][None, ...], (0, w_idx[0], 0, 0))
    w_idx = (w_idx + 1) % STASH_SIZE

    # --- B. Backward Pass ---
    # 1. Mux Gradient Input
    grad_in = jnp.where(is_last_stage, fwd_out, bwd_in)
    do_bwd = jnp.where(is_last_stage, fwd_msk, bwd_msk)

    # 2. Pop Stash
    x_saved = jax.lax.dynamic_slice(stash, (0, r_idx[0], 0, 0), (1, 1, MICRO_BATCH_SIZE, DIM))[0]
    r_idx = (r_idx + 1) % STASH_SIZE

    # 3. Compute Grads
    _, vjp_fn_reconstructed = jax.vjp(dense_layer, x_saved, weights)
    d_x, d_w = vjp_fn_reconstructed(grad_in)

    # 4. Accumulate
    grads += jnp.where(do_bwd, d_w, 0)
    d_x = jnp.where(do_bwd, d_x, 0)

    # --- C. Communication (The Ring Shift) ---
    right_perm = [(i, (i + 1) % STAGES) for i in range(STAGES)]
    left_perm  = [(i, (i - 1) % STAGES) for i in range(STAGES)]

    # Shift Data Right
    fwd_out_shifted = lax.ppermute(fwd_out, axis_name='stage', perm=right_perm)
    # Shift Targets Right (Keep them synced with Data!)
    fwd_tgt_shifted = lax.ppermute(fwd_tgt, axis_name='stage', perm=right_perm)
    # Shift Mask Right
    fwd_msk_shifted = lax.ppermute(fwd_msk, axis_name='stage', perm=right_perm)
    
    # Shift Gradients Left
    bwd_out_shifted = lax.ppermute(d_x, axis_name='stage', perm=left_perm)
    bwd_msk_shifted = lax.ppermute(do_bwd, axis_name='stage', perm=left_perm)

    return (grads, stash, w_idx, r_idx), \
           (fwd_out_shifted, fwd_tgt_shifted, fwd_msk_shifted, bwd_out_shifted, bwd_msk_shifted)

# --- Sharding Definition ---
mesh = Mesh(mesh_utils.create_device_mesh((STAGES,)), axis_names=('stage',))

step_1f1b = shard_map(
    step_1f1b_fn, 
    mesh=mesh,
    in_specs=(
        # State
        (P('stage',None,None), P('stage',None,None,None), P('stage'), P('stage')), 
        # Inputs: Fwd_Data, Fwd_Tgt, Fwd_Mask, Bwd_Grad, Bwd_Mask
        (P('stage',None,None), P('stage',None,None), P('stage',None), P('stage',None,None), P('stage',None)), 
        # Constants
        (P('stage',None,None), P('stage')) 
    ),
    out_specs=(
        # State
        (P('stage',None,None), P('stage',None,None,None), P('stage'), P('stage')), 
        # Outputs
        (P('stage',None,None), P('stage',None,None), P('stage',None), P('stage',None,None), P('stage',None)) 
    )
)

# --- Training Loop ---
def train_1f1b(inputs, targets, weights):
    TOTAL_STEPS = MICRO_BATCHES + 2 * STAGES
    pad_len = TOTAL_STEPS - MICRO_BATCHES

    # 1. Prepare Streams (Clean Aligned Data)
    # Both Inputs and Targets are just simple streams. 
    # No complex time-shifting needed here!
    stream_in = jnp.concatenate([inputs, jnp.zeros((pad_len, STAGES, MICRO_BATCH_SIZE, DIM))])
    stream_tgt = jnp.concatenate([targets, jnp.zeros((pad_len, STAGES, MICRO_BATCH_SIZE, DIM))])
    stream_mask = jnp.concatenate([jnp.ones((MICRO_BATCHES, STAGES, 1)), jnp.zeros((pad_len, STAGES, 1))])

    # 2. Init State
    delays = 2 * (STAGES - 1 - jnp.arange(STAGES))
    init_state = (
        jnp.zeros_like(weights),
        jnp.zeros((STAGES, STASH_SIZE, MICRO_BATCH_SIZE, DIM)),
        jnp.zeros((STAGES,), dtype=jnp.int64),
        ((0 - delays) % STASH_SIZE).astype(jnp.int64)
    )

    # 3. Pipeline Carriers (Tokens)
    # Now includes a slot for Targets
    init_carry_io = (
        jnp.zeros((STAGES, MICRO_BATCH_SIZE, DIM)), # Fwd Data
        jnp.zeros((STAGES, MICRO_BATCH_SIZE, DIM)), # Fwd Target (New!)
        jnp.zeros((STAGES, 1)),                     # Fwd Mask
        jnp.zeros((STAGES, MICRO_BATCH_SIZE, DIM)), # Bwd Grad
        jnp.zeros((STAGES, 1))                      # Bwd Mask
    )

    # 4. Loop
    def scan_body(carry, incoming):
        state, pipe_io = carry
        new_data, new_tgt, new_mask = incoming
        p_fd, p_ft, p_fm, p_bd, p_bm = pipe_io

        # Injection Logic:
        # We inject Data AND Targets into Stage 0 simultaneously.
        curr_fd = p_fd.at[0].set(new_data[0])
        curr_ft = p_ft.at[0].set(new_tgt[0]) # Inject Target!
        curr_fm = p_fm.at[0].set(new_mask[0])

        new_state, new_pipe_io = step_1f1b(
            state, 
            (curr_fd, curr_ft, curr_fm, p_bd, p_bm), 
            (weights, jnp.arange(STAGES))
        )
        return (new_state, new_pipe_io), None

    (final_state, _), _ = lax.scan(
        scan_body, 
        (init_state, init_carry_io), 
        (stream_in, stream_tgt, stream_mask)
    )
    return final_state[0]

# --- Verification ---
print("--- Production-Grade 1F1B (With Target Carrier) ---")
key = jax.random.PRNGKey(42)

w_global = jax.random.normal(key, (STAGES, DIM, DIM))
x_data = jax.random.normal(key, (MICRO_BATCHES, MICRO_BATCH_SIZE, DIM))
y_data = jax.random.normal(key, (MICRO_BATCHES, MICRO_BATCH_SIZE, DIM))

# Format for Injection (Just place in Stage 0 slot)
x_pad = jnp.zeros((MICRO_BATCHES, STAGES, MICRO_BATCH_SIZE, DIM)).at[:, 0].set(x_data)
# Targets also start at Stage 0! They will travel to Stage 3 automatically.
y_pad = jnp.zeros((MICRO_BATCHES, STAGES, MICRO_BATCH_SIZE, DIM)).at[:, 0].set(y_data)

print("Running Pipeline...")
grads_pipe = jax.jit(train_1f1b)(x_pad, y_pad, w_global)

print("Running Reference...")
def loss_ref(all_x, w, all_y):
    def model(x):
        for i in range(STAGES): x = dense_layer(x, w[i])
        return x
    preds = jax.vmap(model)(all_x)
    return jnp.sum(0.5 * (preds - all_y) ** 2) / MICRO_BATCHES

grads_ref = jax.jit(jax.grad(loss_ref, argnums=1))(x_data, w_global, y_data)

diff = jnp.max(jnp.abs(grads_pipe - grads_ref))
print(f"Diff: {diff:.6f}")

if diff < 1e-5:
    print("✅ Success: Pipeline matches Reference.")
else:
    print("❌ Failure: Gradients mismatch.")