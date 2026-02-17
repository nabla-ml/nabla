import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax

jax.config.update("jax_enable_x64", True)  # Enable float64

import jax.numpy as jnp
from jax import lax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

# --- Configuration ---
STAGES = 4
MICRO_BATCHES = 32  # Large number of batches
MICRO_BATCH_SIZE = 4
DIM = 16
STASH_SIZE = 8  # Fixed small buffer (2 * STAGES)


# --- Model ---
def dense_layer(x, w):
    return jax.nn.relu(x @ w)


# --- 1F1B Pipeline Step ---
def step_1f1b_fn(state, inputs, constants):
    """
    Executes one step of the 1F1B schedule on a single stage.
    """
    # Unpack
    grads, stash, w_idx, r_idx = state
    fwd_in, fwd_mask, bwd_in, bwd_mask = inputs
    weights, targets, stage_idx = constants

    is_last_stage = stage_idx == STAGES - 1

    # --- A. Forward Pass ---
    # Compute output and capture VJP function (for backward later)
    fwd_out_raw, vjp_fn = jax.vjp(dense_layer, fwd_in, weights)

    # Last stage logic: Compute Loss Gradient (MSE) instead of passing activation
    # Loss Grad = (Output - Target) / N
    loss_grad = (fwd_out_raw - targets) / MICRO_BATCHES
    fwd_out = jnp.where(is_last_stage, loss_grad, fwd_out_raw)

    # Mask bubbles (if valid data didn't arrive, output zeros)
    fwd_out = jnp.where(fwd_mask, fwd_out, jnp.zeros_like(fwd_out))

    # Stash input 'x' for the future backward pass
    # We take the first element of the residuals (which is 'x')
    x_input = (fwd_in,)
    stash = jax.lax.dynamic_update_slice(
        stash, x_input[0][None, ...], (0, w_idx[0], 0, 0)
    )
    w_idx = (w_idx + 1) % STASH_SIZE

    # --- B. Backward Pass ---
    # Determine input gradient:
    # - Last Stage: Uses the 'loss_grad' (fwd_out) we just computed
    # - Others: Uses 'bwd_in' coming from the right
    grad_in = jnp.where(is_last_stage, fwd_out, bwd_in)
    do_bwd = jnp.where(is_last_stage, fwd_mask, bwd_mask)

    # Retrieve stashed input 'x' from the past
    x_saved = jax.lax.dynamic_slice(
        stash, (0, r_idx[0], 0, 0), (1, 1, MICRO_BATCH_SIZE, DIM)
    )[0]
    r_idx = (r_idx + 1) % STASH_SIZE

    # Recompute gradients using saved X and current W
    _, vjp_fn_reconstructed = jax.vjp(dense_layer, x_saved, weights)
    d_x, d_w = vjp_fn_reconstructed(grad_in)

    # Accumulate gradients (Masked)
    grads += jnp.where(do_bwd, d_w, 0)
    d_x = jnp.where(do_bwd, d_x, 0)

    # --- C. Communication (Shift) ---
    # Forward: Move Right (i -> i+1)
    # Backward: Move Left (i -> i-1)
    right_perm = [(i, (i + 1) % STAGES) for i in range(STAGES)]
    left_perm = [(i, (i - 1) % STAGES) for i in range(STAGES)]

    fwd_out_shifted = lax.ppermute(fwd_out, axis_name="stage", perm=right_perm)
    fwd_mask_shifted = lax.ppermute(fwd_mask, axis_name="stage", perm=right_perm)

    bwd_out_shifted = lax.ppermute(d_x, axis_name="stage", perm=left_perm)
    bwd_mask_shifted = lax.ppermute(do_bwd, axis_name="stage", perm=left_perm)

    return (grads, stash, w_idx, r_idx), (
        fwd_out_shifted,
        fwd_mask_shifted,
        bwd_out_shifted,
        bwd_mask_shifted,
    )


# --- Sharding Wrapper ---
mesh = Mesh(mesh_utils.create_device_mesh((STAGES,)), axis_names=("stage",))

step_1f1b = shard_map(
    step_1f1b_fn,
    mesh=mesh,
    in_specs=(
        # State: Grads, Stash, W_ptr, R_ptr
        (P("stage", None, None), P("stage", None, None, None), P("stage"), P("stage")),
        # Inputs: Fwd_D, Fwd_M, Bwd_D, Bwd_M
        (
            P("stage", None, None),
            P("stage", None),
            P("stage", None, None),
            P("stage", None),
        ),
        # Constants: Weights, Targets, Stage_ID
        (P("stage", None, None), P("stage", None, None), P("stage")),
    ),
    out_specs=(
        # State Return
        (P("stage", None, None), P("stage", None, None, None), P("stage"), P("stage")),
        # Output Return
        (
            P("stage", None, None),
            P("stage", None),
            P("stage", None, None),
            P("stage", None),
        ),
    ),
)


# --- Training Loop ---
def train_1f1b(inputs, targets, weights):
    TOTAL_STEPS = MICRO_BATCHES + 2 * STAGES

    # 1. Prepare Inputs (Stream + Padding)
    # We append zeros to allow the pipeline to flush
    pad_len = TOTAL_STEPS - MICRO_BATCHES
    stream_in = jnp.concatenate(
        [inputs, jnp.zeros((pad_len, STAGES, MICRO_BATCH_SIZE, DIM))]
    )

    # Targets must align with data arrival at the LAST stage.
    # Data T @ Stage 0 -> Arrives @ Stage (N-1) at Time T + (N-1)
    stream_tgt = jnp.zeros((TOTAL_STEPS, STAGES, MICRO_BATCH_SIZE, DIM))
    target_times = jnp.arange(MICRO_BATCHES) + (STAGES - 1)
    stream_tgt = stream_tgt.at[target_times].set(targets)

    # Masks (1 for Real Data, 0 for Padding)
    stream_mask = jnp.concatenate(
        [jnp.ones((MICRO_BATCHES, STAGES, 1)), jnp.zeros((pad_len, STAGES, 1))]
    )

    # 2. Initialize State
    # Read pointer starts negative to account for pipeline round-trip delay
    # Delay = 2 * (STAGES - 1 - local_stage_index)
    delays = 2 * (STAGES - 1 - jnp.arange(STAGES))
    init_state = (
        jnp.zeros_like(weights),  # Grads
        jnp.zeros((STAGES, STASH_SIZE, MICRO_BATCH_SIZE, DIM)),  # Stash
        jnp.zeros((STAGES,), dtype=jnp.int64),  # Write Idx
        ((0 - delays) % STASH_SIZE).astype(jnp.int64),  # Read Idx
    )

    # 3. Pipeline Carriers (Tokens moving between stages)
    init_carry_io = (
        jnp.zeros((STAGES, MICRO_BATCH_SIZE, DIM)),  # Fwd Data
        jnp.zeros((STAGES, 1)),  # Fwd Mask
        jnp.zeros((STAGES, MICRO_BATCH_SIZE, DIM)),  # Bwd Grad
        jnp.zeros((STAGES, 1)),  # Bwd Mask
    )

    # 4. Execution Loop
    def scan_body(carry, incoming):
        state, pipe_io = carry
        new_data, new_tgt, new_mask = incoming
        pipe_f_d, pipe_f_m, pipe_b_d, pipe_b_m = pipe_io

        # Inject Fresh Data at Stage 0
        # We overwrite the "incoming" forward token for Stage 0 with fresh data from stream
        curr_fwd_d = pipe_f_d.at[0].set(new_data[0])
        curr_fwd_m = pipe_f_m.at[0].set(new_mask[0])

        new_state, new_pipe_io = step_1f1b(
            state,
            (curr_fwd_d, curr_fwd_m, pipe_b_d, pipe_b_m),
            (weights, new_tgt, jnp.arange(STAGES)),
        )
        return (new_state, new_pipe_io), None

    (final_state, _), _ = lax.scan(
        scan_body, (init_state, init_carry_io), (stream_in, stream_tgt, stream_mask)
    )
    return final_state[0]  # Return accumulated gradients


# --- Verification ---
print("--- 1F1B Minimal Test (Educational) ---")
print(f"Configuration: {MICRO_BATCHES} Microbatches, {STAGES} Stages.")
print(f"Memory Proof: Stash Size is fixed at {STASH_SIZE}.")
print(f"   - GPipe would require Stash Size >= {MICRO_BATCHES} (Linear Memory).")
print(f"   - 1F1B runs with Stash Size {STASH_SIZE} (Constant Memory).")
print("-------------------------------------------")

key = jax.random.PRNGKey(42)

# Random Data
w_global = jax.random.normal(key, (STAGES, DIM, DIM))
x_data = jax.random.normal(key, (MICRO_BATCHES, MICRO_BATCH_SIZE, DIM))
y_data = jax.random.normal(key, (MICRO_BATCHES, MICRO_BATCH_SIZE, DIM))

# Prepare for Pipeline (Shape: [Batch, Stage, ...])
# We inject at Stage 0 and Expect at Stage N-1
x_pad = jnp.zeros((MICRO_BATCHES, STAGES, MICRO_BATCH_SIZE, DIM)).at[:, 0].set(x_data)
y_pad = (
    jnp.zeros((MICRO_BATCHES, STAGES, MICRO_BATCH_SIZE, DIM))
    .at[:, STAGES - 1]
    .set(y_data)
)

print("Running Pipeline...")
grads_pipe = jax.jit(train_1f1b)(x_pad, y_pad, w_global)

print("Running Reference...")


def loss_ref(all_x, w, all_y):
    def model(x):
        for i in range(STAGES):
            x = dense_layer(x, w[i])
        return x

    preds = jax.vmap(model)(all_x)
    return jnp.sum(0.5 * (preds - all_y) ** 2) / MICRO_BATCHES


grads_ref = jax.jit(jax.grad(loss_ref, argnums=1))(x_data, w_global, y_data)

# Compare
diff = jnp.max(jnp.abs(grads_pipe - grads_ref))
print(f"Diff: {diff:.6f}")

if diff < 1e-5:
    print("✅ Success: Pipeline matches Reference.")
else:
    print("❌ Failure: Gradients mismatch.")
