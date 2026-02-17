import os

# We need 8 devices: 2 for Data Parallelism * 4 for Pipeline Parallelism
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
DP = 2  # Data Parallel Replicas
STAGES = 4  # Pipeline Stages
MICRO_BATCHES = 32
MICRO_BATCH_SIZE = 4
DIM = 16
STASH_SIZE = 8


# --- Model ---
def dense_layer(x, w):
    return jax.nn.relu(x @ w)


# --- 1F1B Core Logic (Clean, Rank-4) ---
# This function knows NOTHING about DP or Sharding dimensions.
# It operates on a single local shard.
def step_core_fn(state, inputs, constants):
    # Unpack (Scalars and Rank-3 Tensors)
    grads, stash, w_idx, r_idx = state
    fwd_in, fwd_tgt, fwd_msk, bwd_in, bwd_msk = inputs
    weights, stage_idx = constants

    is_last_stage = stage_idx == STAGES - 1

    # --- A. Forward ---
    fwd_out_raw, vjp_fn = jax.vjp(dense_layer, fwd_in, weights)

    loss_grad = (fwd_out_raw - fwd_tgt) / MICRO_BATCHES

    fwd_out = jnp.where(is_last_stage, loss_grad, fwd_out_raw)
    fwd_out = jnp.where(fwd_msk, fwd_out, 0)

    # Stash (Standard Logic, no extra zeros!)
    x_input = (fwd_in,)
    # w_idx is a Scalar here!
    # Stash is (STASH, MB, DIM) -> Rank 3 (vmap stripped sharded dims)
    stash = jax.lax.dynamic_update_slice(stash, x_input[0][None, ...], (w_idx, 0, 0))
    w_idx = (w_idx + 1) % STASH_SIZE

    # --- B. Backward ---
    grad_in = jnp.where(is_last_stage, fwd_out, bwd_in)
    do_bwd = jnp.where(is_last_stage, fwd_msk, bwd_msk)

    # Pop Stash
    # r_idx is a Scalar here!
    x_saved = jax.lax.dynamic_slice(stash, (r_idx, 0, 0), (1, MICRO_BATCH_SIZE, DIM))[0]
    r_idx = (r_idx + 1) % STASH_SIZE

    _, vjp_fn_reconstructed = jax.vjp(dense_layer, x_saved, weights)
    d_x, d_w = vjp_fn_reconstructed(grad_in)

    grads += jnp.where(do_bwd, d_w, 0)
    d_x = jnp.where(do_bwd, d_x, 0)

    return (grads, stash, w_idx, r_idx), (fwd_out, fwd_tgt, fwd_msk, d_x, do_bwd)


# --- Sharded Adapter ---
def step_1f1b_adapter(state, inputs, constants):
    # Inputs here are Rank-5 (DP, PP, ...) with size 1 for sharded dims
    # e.g., w_idx is (1, 1)

    # 1. Lift Core Logic to handle the sharded dimensions
    # Inner Vmap (PP Axis): Maps over everything (0, 0, 0) - default
    step_inner = jax.vmap(step_core_fn)

    # Outer Vmap (DP Axis): Maps State/Inputs, Broadcasts Constants
    # Constants (Weights/StageID) lack the DP dimension, so we must broadcast (None).
    step_vmapped = jax.vmap(step_inner, in_axes=(0, 0, None))

    # 2. Execute Core Logic
    new_state, outputs = step_vmapped(state, inputs, constants)

    # 3. Handle Communication (Must be done on the Sharded View)
    # Unpack outputs to perform ppermute
    fwd_out, fwd_tgt, fwd_msk, bwd_out, bwd_msk = outputs

    # Communicate along 'pp' axis
    right_perm = [(i, (i + 1) % STAGES) for i in range(STAGES)]
    left_perm = [(i, (i - 1) % STAGES) for i in range(STAGES)]

    fwd_out_shifted = lax.ppermute(fwd_out, axis_name="pp", perm=right_perm)
    fwd_tgt_shifted = lax.ppermute(fwd_tgt, axis_name="pp", perm=right_perm)
    fwd_msk_shifted = lax.ppermute(fwd_msk, axis_name="pp", perm=right_perm)

    bwd_out_shifted = lax.ppermute(bwd_out, axis_name="pp", perm=left_perm)
    bwd_msk_shifted = lax.ppermute(bwd_msk, axis_name="pp", perm=left_perm)

    return new_state, (
        fwd_out_shifted,
        fwd_tgt_shifted,
        fwd_msk_shifted,
        bwd_out_shifted,
        bwd_msk_shifted,
    )


# --- Sharding Definition (2D Mesh) ---
# Axes: ('dp', 'pp')
mesh = Mesh(mesh_utils.create_device_mesh((DP, STAGES)), axis_names=("dp", "pp"))

step_1f1b = shard_map(
    step_1f1b_adapter,
    mesh=mesh,
    in_specs=(
        # State: Grads, Stash, W_ptr, R_ptr
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
        # Constants: Weights(Replicated on DP, Sharded PP), Stage_ID(Sharded PP)
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


# Gradient Sync (All-Reduce across DP)
def sync_grads_fn(grads):
    return lax.pmean(grads, axis_name="dp")


sync_grads = shard_map(
    sync_grads_fn,
    mesh=mesh,
    in_specs=P("dp", "pp", None, None),
    out_specs=P("dp", "pp", None, None),
)


# --- Training Loop ---
def train_dp_pp(inputs, targets, weights):
    TOTAL_STEPS = inputs.shape[0]

    # 1. Init State
    delays = 2 * (STAGES - 1 - jnp.arange(STAGES))
    init_state = (
        jnp.zeros((DP, STAGES, DIM, DIM)),
        jnp.zeros((DP, STAGES, STASH_SIZE, MICRO_BATCH_SIZE, DIM)),
        jnp.zeros((DP, STAGES), dtype=jnp.int64),
        jnp.tile(((0 - delays) % STASH_SIZE).astype(jnp.int64), (DP, 1)),
    )

    # 2. Pipeline Carriers
    init_carry_io = (
        jnp.zeros((DP, STAGES, MICRO_BATCH_SIZE, DIM)),
        jnp.zeros((DP, STAGES, MICRO_BATCH_SIZE, DIM)),
        jnp.zeros((DP, STAGES, 1)),
        jnp.zeros((DP, STAGES, MICRO_BATCH_SIZE, DIM)),
        jnp.zeros((DP, STAGES, 1)),
    )

    # 3. Loop
    def scan_body(carry, incoming):
        state, pipe_io = carry
        new_data, new_tgt, new_mask = incoming
        p_fd, p_ft, p_fm, p_bd, p_bm = pipe_io

        # Injection
        curr_fd = p_fd.at[:, 0].set(new_data[:, 0])
        curr_ft = p_ft.at[:, 0].set(new_tgt[:, 0])
        curr_fm = p_fm.at[:, 0].set(new_mask[:, 0])

        new_state, new_pipe_io = step_1f1b(
            state,
            (curr_fd, curr_ft, curr_fm, p_bd, p_bm),
            (weights, jnp.arange(STAGES)),
        )
        return (new_state, new_pipe_io), None

    # Mask: 1 for Real Data, 0 for Padding
    stream_mask = jnp.concatenate(
        [
            jnp.ones((MICRO_BATCHES, DP, STAGES, 1)),
            jnp.zeros((TOTAL_STEPS - MICRO_BATCHES, DP, STAGES, 1)),
        ]
    )

    (final_state, _), _ = lax.scan(
        scan_body, (init_state, init_carry_io), (inputs, targets, stream_mask)
    )

    # 4. Sync Gradients
    accum_grads = final_state[0]
    synced_grads = sync_grads(accum_grads)
    return synced_grads[0]


# --- Verification ---
print(f"--- 1F1B + Data Parallelism (DP={DP}) PP={STAGES}) ---")
key = jax.random.PRNGKey(42)

w_global = jax.random.normal(key, (STAGES, DIM, DIM))
x_data_global = jax.random.normal(key, (DP, MICRO_BATCHES, MICRO_BATCH_SIZE, DIM))
y_data_global = jax.random.normal(key, (DP, MICRO_BATCHES, MICRO_BATCH_SIZE, DIM))

TOTAL_STEPS = MICRO_BATCHES + 2 * STAGES
pad_len = TOTAL_STEPS - MICRO_BATCHES

x_pad = jnp.concatenate(
    [x_data_global, jnp.zeros((DP, pad_len, MICRO_BATCH_SIZE, DIM))], axis=1
)
y_pad = jnp.concatenate(
    [y_data_global, jnp.zeros((DP, pad_len, MICRO_BATCH_SIZE, DIM))], axis=1
)

x_pipe = jnp.zeros((TOTAL_STEPS, DP, STAGES, MICRO_BATCH_SIZE, DIM))
y_pipe = jnp.zeros((TOTAL_STEPS, DP, STAGES, MICRO_BATCH_SIZE, DIM))

x_pipe = x_pipe.at[:, :, 0].set(jnp.transpose(x_pad, (1, 0, 2, 3)))
y_pipe = y_pipe.at[:, :, 0].set(jnp.transpose(y_pad, (1, 0, 2, 3)))

print("Running DP+PP Pipeline...")
grads_pipe = jax.jit(train_dp_pp)(x_pipe, y_pipe, w_global)

print("Running Reference (Global Batch)...")
x_ref_flat = x_data_global.reshape(-1, DIM)
y_ref_flat = y_data_global.reshape(-1, DIM)


def loss_ref(all_x, w, all_y):
    def model(x):
        for i in range(STAGES):
            x = dense_layer(x, w[i])
        return x

    preds = jax.vmap(model)(all_x)
    return jnp.sum(0.5 * (preds - all_y) ** 2) / (MICRO_BATCHES * DP)


grads_ref = jax.jit(jax.grad(loss_ref, argnums=1))(x_ref_flat, w_global, y_ref_flat)

diff = jnp.max(jnp.abs(grads_pipe - grads_ref))
print(f"Diff: {diff:.6f}")

if diff < 1e-5:
    print("✅ Success: DP+PP Pipeline matches Global Reference.")
else:
    print("❌ Failure: Gradients mismatch.")
