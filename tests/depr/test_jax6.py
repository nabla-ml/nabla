import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

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


# --- 1F1B Core Logic (Idiomatic: Rank-Independent) ---
# This function sees:
# - stash: (STASH, MB, DIM)
# - fwd_in: (MB, DIM)
# - weights: (DIM, DIM)
# No knowledge of DP or PP dimensions!
def step_core_fn(state, inputs, constants):
    grads, stash, w_idx, r_idx = state
    fwd_in, fwd_tgt, fwd_msk, bwd_in, bwd_msk = inputs
    weights, stage_idx = constants

    # Scalars are automatically scalars here due to vmap
    is_last_stage = stage_idx == STAGES - 1

    # --- A. Forward ---
    fwd_out_raw, vjp_fn = jax.vjp(dense_layer, fwd_in, weights)

    loss_grad = (fwd_out_raw - fwd_tgt) / MICRO_BATCHES

    fwd_out = jnp.where(is_last_stage, loss_grad, fwd_out_raw)
    fwd_out = jnp.where(fwd_msk, fwd_out, 0)

    # Stash
    # Use indices directly (no [0,0]).
    # vmap has stripped the shard dims.
    stash = jax.lax.dynamic_update_slice(
        stash,
        fwd_in[None, ...],  # Add stash dim -> (1, MB, DIM)
        (w_idx, 0, 0),  # (StashIdx, MB, Dim)
    )
    w_idx = (w_idx + 1) % STASH_SIZE

    # --- B. Backward ---
    grad_in = jnp.where(is_last_stage, fwd_out, bwd_in)
    do_bwd = jnp.where(is_last_stage, fwd_msk, bwd_msk)

    # Pop Stash
    x_saved = jax.lax.dynamic_slice(stash, (r_idx, 0, 0), (1, MICRO_BATCH_SIZE, DIM))[0]

    r_idx = (r_idx + 1) % STASH_SIZE

    # Gradients
    _, vjp_fn_reconstructed = jax.vjp(dense_layer, x_saved, weights)
    d_x, d_w = vjp_fn_reconstructed(grad_in)

    grads += jnp.where(do_bwd, d_w, 0)
    d_x = jnp.where(do_bwd, d_x, 0)

    # Return
    return (grads, stash, w_idx, r_idx), (fwd_out, fwd_tgt, fwd_msk, d_x, do_bwd)


# --- Sharded Adapter with VMAP ---
def step_1f1b_adapter(state, inputs, constants):
    # Inputs are Rank 5: (1, 1, ...)
    # Constants (Weights) are Rank 3: (1, DIM, DIM) [Missing DP dim]

    # 1. Inner Vmap (PP Axis)
    # Maps over the local PP shard (Axis 0 of what remains).
    # All inputs have this dimension (size 1).
    vmap_pp = jax.vmap(step_core_fn, in_axes=(0, 0, 0))

    # 2. Outer Vmap (DP Axis)
    # Maps over the local DP shard (Axis 0).
    # State/Inputs have this dimension (size 1).
    # Constants (Weights) DO NOT have this dimension (Replicated).
    # We use in_axes=None to broadcast weights to all DP shards.
    vmap_dp = jax.vmap(vmap_pp, in_axes=(0, 0, None))

    # Execute
    new_state, outputs = vmap_dp(state, inputs, constants)

    # 3. Communication (Tree Map)
    fwd_out, fwd_tgt, fwd_msk, bwd_out, bwd_msk = outputs

    right_perm = [(i, (i + 1) % STAGES) for i in range(STAGES)]
    left_perm = [(i, (i - 1) % STAGES) for i in range(STAGES)]

    def shift(x, direction):
        perm = right_perm if direction == "right" else left_perm
        return lax.ppermute(x, axis_name="pp", perm=perm)

    # Shift Forward
    fwd_shifted = [shift(x, "right") for x in (fwd_out, fwd_tgt, fwd_msk)]
    # Shift Backward
    bwd_shifted = [shift(x, "left") for x in (bwd_out, bwd_msk)]

    return new_state, tuple(fwd_shifted + bwd_shifted)


# --- Sharding Definition ---
mesh = Mesh(mesh_utils.create_device_mesh((DP, STAGES)), axis_names=("dp", "pp"))

step_1f1b = shard_map(
    step_1f1b_adapter,
    mesh=mesh,
    in_specs=(
        # State
        (
            P("dp", "pp", None, None),
            P("dp", "pp", None, None, None),
            P("dp", "pp"),
            P("dp", "pp"),
        ),
        # Inputs
        (
            P("dp", "pp", None, None),
            P("dp", "pp", None, None),
            P("dp", "pp", None),
            P("dp", "pp", None, None),
            P("dp", "pp", None),
        ),
        # Constants: Weights P('pp', ...), Stage P('pp')
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

        curr_fd = p_fd.at[:, 0].set(new_data[:, 0])
        curr_ft = p_ft.at[:, 0].set(new_tgt[:, 0])
        curr_fm = p_fm.at[:, 0].set(new_mask[:, 0])

        new_state, new_pipe_io = step_1f1b(
            state,
            (curr_fd, curr_ft, curr_fm, p_bd, p_bm),
            (weights, jnp.arange(STAGES)),
        )
        return (new_state, new_pipe_io), None

    stream_mask = jnp.concatenate(
        [
            jnp.ones((MICRO_BATCHES, DP, STAGES, 1)),
            jnp.zeros((TOTAL_STEPS - MICRO_BATCHES, DP, STAGES, 1)),
        ]
    )

    (final_state, _), _ = lax.scan(
        scan_body, (init_state, init_carry_io), (inputs, targets, stream_mask)
    )

    # Sync: pmean averages over DP
    grads = final_state[0]
    grads_synced = shard_map(
        lambda g: lax.pmean(g, "dp"),
        mesh,
        P("dp", "pp", None, None),
        P("dp", "pp", None, None),
    )(grads)

    # Divide by DP because Reference divides by (MB*DP), and pmean/accum logic gives 2x factor.
    return grads_synced[0] / DP


# --- Verification ---
print("--- Idiomatic 1F1B + DP (Nested VMAP) ---")
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
