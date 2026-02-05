import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

# --- Configuration ---
DP = 2
STAGES = 4
MICRO_BATCHES = 32
MICRO_BATCH_SIZE = 4
DIM = 16
STASH_SIZE = 8


# --- Model ---
def dense_layer(x, w):
    return jax.nn.relu(x @ w)


# --- 1F1B Step ---
def step_1f1b_fn(state, inputs, constants):
    grads, stash, w_idx, r_idx = state
    fwd_in, fwd_tgt, fwd_msk, bwd_in, bwd_msk = inputs
    weights, stage_idx = constants

    # 1. Squeeze Local Views (Remove Shard Dims (1,1))
    # w_idx: (1, 1) -> Scalar
    # fwd_in: (1, 1, MB, DIM) -> (MB, DIM)
    w_ptr = w_idx[0, 0]
    r_ptr = r_idx[0, 0]
    fwd_in_sq = fwd_in[0, 0]
    fwd_tgt_sq = fwd_tgt[0, 0]
    mask_val = fwd_msk[0, 0, 0]  # Scalar mask
    bwd_in_sq = bwd_in[0, 0]
    bwd_msk_val = bwd_msk[0, 0, 0]
    w_sq = weights[0]  # (DIM, DIM)

    is_last_stage = stage_idx[0] == STAGES - 1

    # --- A. Forward ---
    fwd_out_raw, vjp_fn = jax.vjp(dense_layer, fwd_in_sq, w_sq)

    # Loss Calculation (MSE)
    loss_grad = (fwd_out_raw - fwd_tgt_sq) / MICRO_BATCHES

    # Mux Output
    fwd_out = jnp.where(is_last_stage, loss_grad, fwd_out_raw)
    fwd_out = jnp.where(mask_val, fwd_out, 0)

    # Stash Input
    # Update slice at (0, 0, w_ptr, 0, 0)
    stash = jax.lax.dynamic_update_slice(
        stash, fwd_in_sq[None, None, None, ...], (0, 0, w_ptr, 0, 0)
    )
    w_idx = jnp.full((1, 1), (w_ptr + 1) % STASH_SIZE, dtype=jnp.int64)

    # --- B. Backward ---
    grad_in = jnp.where(is_last_stage, fwd_out, bwd_in_sq)
    do_bwd = jnp.where(is_last_stage, mask_val, bwd_msk_val)

    # Pop Stash
    x_saved = jax.lax.dynamic_slice(
        stash, (0, 0, r_ptr, 0, 0), (1, 1, 1, MICRO_BATCH_SIZE, DIM)
    )[0, 0, 0]

    r_idx = jnp.full((1, 1), (r_ptr + 1) % STASH_SIZE, dtype=jnp.int64)

    # Compute Grads
    _, vjp_fn_reconstructed = jax.vjp(dense_layer, x_saved, w_sq)
    d_x, d_w = vjp_fn_reconstructed(grad_in)

    # Accumulate (Reshape d_w to match grads (1, 1, DIM, DIM))
    grads += jnp.where(do_bwd, d_w, 0)[None, None, ...]
    d_x = jnp.where(do_bwd, d_x, 0)

    # --- C. Communication ---
    # Pack outputs back to (1, 1, ...) for sharding
    outputs_local = (
        fwd_out[None, None, ...],
        fwd_tgt[0, 0][None, None, ...],  # Pass through
        fwd_msk[0, 0][None, None, ...],
        d_x[None, None, ...],
        jnp.where(do_bwd, 1.0, 0.0)[None, None, None],  # bwd_mask
    )

    # Tree-mapped Permutation
    right_perm = [(i, (i + 1) % STAGES) for i in range(STAGES)]
    left_perm = [(i, (i - 1) % STAGES) for i in range(STAGES)]

    def shift(tensor, direction):
        perm = right_perm if direction == "right" else left_perm
        return lax.ppermute(tensor, axis_name="pp", perm=perm)

    # Shift Fwd (Data, Tgt, Mask) Right
    fwd_shifted = [shift(x, "right") for x in outputs_local[:3]]
    # Shift Bwd (Grad, Mask) Left
    bwd_shifted = [shift(x, "left") for x in outputs_local[3:]]

    return (grads, stash, w_idx, r_idx), tuple(fwd_shifted + bwd_shifted)


# --- Sharding Definition ---
mesh = Mesh(mesh_utils.create_device_mesh((DP, STAGES)), axis_names=("dp", "pp"))

step_1f1b = shard_map(
    step_1f1b_fn,
    mesh=mesh,
    in_specs=(
        # State: Grads, Stash, W_idx, R_idx
        (
            P("dp", "pp", None, None),
            P("dp", "pp", None, None, None),
            P("dp", "pp"),
            P("dp", "pp"),
        ),
        # Inputs: Fwd_D, Fwd_T, Fwd_M, Bwd_D, Bwd_M
        (
            P("dp", "pp", None, None),
            P("dp", "pp", None, None),
            P("dp", "pp", None),
            P("dp", "pp", None, None),
            P("dp", "pp", None),
        ),
        # Constants
        (P("pp", None, None), P("pp")),
    ),
    out_specs=(
        (
            P("dp", "pp", None, None),
            P("dp", "pp", None, None, None),
            P("dp", "pp"),
            P("dp", "pp"),
        ),
        (
            P("dp", "pp", None, None),
            P("dp", "pp", None, None),
            P("dp", "pp", None),
            P("dp", "pp", None, None),
            P("dp", "pp", None),
        ),
    ),
)


# --- Training Loop ---
def train_dp_pp(inputs, targets, weights):
    TOTAL_STEPS = inputs.shape[0]

    # Init State
    delays = 2 * (STAGES - 1 - jnp.arange(STAGES))
    init_state = (
        jnp.zeros((DP, STAGES, DIM, DIM)),
        jnp.zeros((DP, STAGES, STASH_SIZE, MICRO_BATCH_SIZE, DIM)),
        jnp.zeros((DP, STAGES), dtype=jnp.int64),
        jnp.tile(((0 - delays) % STASH_SIZE).astype(jnp.int64), (DP, 1)),
    )

    init_carry_io = (
        jnp.zeros((DP, STAGES, MICRO_BATCH_SIZE, DIM)),
        jnp.zeros((DP, STAGES, MICRO_BATCH_SIZE, DIM)),
        jnp.zeros((DP, STAGES, 1)),
        jnp.zeros((DP, STAGES, MICRO_BATCH_SIZE, DIM)),
        jnp.zeros((DP, STAGES, 1)),
    )

    def scan_body(carry, incoming):
        state, pipe_io = carry
        new_data, new_tgt, new_mask = incoming
        p_fd, p_ft, p_fm, p_bd, p_bm = pipe_io

        # Injection (Update Stage 0 slots)
        curr_fd = p_fd.at[:, 0].set(new_data[:, 0])
        curr_ft = p_ft.at[:, 0].set(new_tgt[:, 0])
        curr_fm = p_fm.at[:, 0].set(new_mask[:, 0])

        new_state, new_pipe_io = step_1f1b(
            state,
            (curr_fd, curr_ft, curr_fm, p_bd, p_bm),
            (weights, jnp.arange(STAGES)),
        )
        return (new_state, new_pipe_io), None

    # Mask Logic (Zeros for padding)
    stream_mask = jnp.concatenate(
        [
            jnp.ones((MICRO_BATCHES, DP, STAGES, 1)),
            jnp.zeros((TOTAL_STEPS - MICRO_BATCHES, DP, STAGES, 1)),
        ]
    )

    (final_state, _), _ = lax.scan(
        scan_body, (init_state, init_carry_io), (inputs, targets, stream_mask)
    )

    # Sync and Scale
    grads = final_state[0]
    grads_synced = shard_map(
        lambda g: lax.pmean(g, "dp"),
        mesh,
        P("dp", "pp", None, None),
        P("dp", "pp", None, None),
    )(grads)

    return grads_synced[0] / DP


# --- Verification ---
print("--- Clean 1F1B + DP Test ---")
key = jax.random.PRNGKey(42)

w_global = jax.random.normal(key, (STAGES, DIM, DIM))
x_data = jax.random.normal(key, (DP, MICRO_BATCHES, MICRO_BATCH_SIZE, DIM))
y_data = jax.random.normal(key, (DP, MICRO_BATCHES, MICRO_BATCH_SIZE, DIM))

TOTAL_STEPS = MICRO_BATCHES + 2 * STAGES
pad_len = TOTAL_STEPS - MICRO_BATCHES

x_pad = jnp.concatenate(
    [x_data, jnp.zeros((DP, pad_len, MICRO_BATCH_SIZE, DIM))], axis=1
)
y_pad = jnp.concatenate(
    [y_data, jnp.zeros((DP, pad_len, MICRO_BATCH_SIZE, DIM))], axis=1
)

x_pipe = jnp.zeros((TOTAL_STEPS, DP, STAGES, MICRO_BATCH_SIZE, DIM))
y_pipe = jnp.zeros((TOTAL_STEPS, DP, STAGES, MICRO_BATCH_SIZE, DIM))

x_pipe = x_pipe.at[:, :, 0].set(jnp.transpose(x_pad, (1, 0, 2, 3)))
y_pipe = y_pipe.at[:, :, 0].set(jnp.transpose(y_pad, (1, 0, 2, 3)))

print("Running Pipeline...")
grads_pipe = jax.jit(train_dp_pp)(x_pipe, y_pipe, w_global)

print("Running Reference...")
x_flat = x_data.reshape(-1, DIM)
y_flat = y_data.reshape(-1, DIM)


def loss_ref(all_x, w, all_y):
    def model(x):
        for i in range(STAGES):
            x = dense_layer(x, w[i])
        return x

    preds = jax.vmap(model)(all_x)
    return jnp.sum(0.5 * (preds - all_y) ** 2) / (MICRO_BATCHES * DP)


grads_ref = jax.jit(jax.grad(loss_ref, argnums=1))(x_flat, w_global, y_flat)

diff = jnp.max(jnp.abs(grads_pipe - grads_ref))
print(f"Diff: {diff:.6f}")

if diff < 1e-5:
    print("✅ Success.")
else:
    print("❌ Failure.")
