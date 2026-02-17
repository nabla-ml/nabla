import os

# --- 0. Setup Fake Cluster ---
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

# --- Configuration ---
STAGES = 4
MICRO_BATCHES = 8
MICRO_BATCH_SIZE = 4
DIM = 16


# --- Common Math Kernel ---
# Both implementations use exactly the same math logic
def dense_layer(x, w):
    return jax.nn.relu(x @ w)


# ==========================================
# IMPLEMENTATION A: The "Shift Register" Pipeline
# ==========================================

devices = mesh_utils.create_device_mesh((STAGES,))
mesh = Mesh(devices, axis_names=("stage",))

batched_layer = vmap(dense_layer, in_axes=(0, None), out_axes=0)


def pipeline_step_fn(current_activations, w_stage):
    # 1. Compute
    # w_stage comes in as (1, D, D) (local shard), squeeze to (D, D)
    y = batched_layer(current_activations, w_stage[0])
    # 2. Shift Right (Circular)
    return lax.ppermute(
        y, axis_name="stage", perm=[(i, (i + 1) % STAGES) for i in range(STAGES)]
    )


pipeline_step = shard_map(
    pipeline_step_fn,
    mesh=mesh,
    in_specs=(P("stage", None, None), P("stage", None, None)),
    out_specs=P("stage", None, None),
)


def pipeline_loss_fn(all_inputs, weights, targets):
    # 1. Padding: We need (STAGES) extra steps to flush the pipe
    padding = jnp.zeros((STAGES, MICRO_BATCH_SIZE, DIM))
    padded_inputs = jnp.concatenate([all_inputs, padding], axis=0)

    # 2. Init State (Bubbles)
    init_pipe = jnp.zeros((STAGES, MICRO_BATCH_SIZE, DIM))

    def scan_body(pipe_state, fresh_input):
        next_pipe = pipeline_step(pipe_state, weights)
        output_popped = next_pipe[0]  # The data wrapping around from Last -> First
        next_pipe = next_pipe.at[0].set(
            fresh_input
        )  # Overwrite garbage with fresh data
        return next_pipe, output_popped

    # 3. Run Pipeline
    _, stream_outputs = lax.scan(scan_body, init_pipe, padded_inputs)

    # 4. Slicing: The first valid output pops at t=STAGES
    # We slice to get exactly MICRO_BATCHES outputs
    valid_preds = stream_outputs[STAGES : (STAGES + MICRO_BATCHES)]

    return jnp.mean((valid_preds - targets) ** 2)


# ==========================================
# IMPLEMENTATION B: The "Reference" (Sequential)
# ==========================================


def reference_loss_fn(all_inputs, weights, targets):
    # This is how a standard non-parallel implementation works:
    # We just loop over layers (0..STAGES) for every input.

    def apply_full_mlp(x_batch):
        # x_batch: (B, D)
        # weights: (STAGES, D, D)
        activations = x_batch
        for i in range(STAGES):
            activations = dense_layer(activations, weights[i])
        return activations

    # Apply to all microbatches in the stream
    # vmap over the 'microbatch' axis (axis 0)
    all_preds = vmap(apply_full_mlp, in_axes=0)(all_inputs)

    return jnp.mean((all_preds - targets) ** 2)


# ==========================================
# VERIFICATION RUN
# ==========================================

# 1. Setup Data
key = jax.random.PRNGKey(42)
w_global = jax.random.normal(key, (STAGES, DIM, DIM))
x_stream = jax.random.normal(key, (MICRO_BATCHES, MICRO_BATCH_SIZE, DIM))
y_targets = jax.random.normal(key, (MICRO_BATCHES, MICRO_BATCH_SIZE, DIM))

print(f"Simulation: {STAGES} Stages, {MICRO_BATCHES} Microbatches")

# 2. Compute Pipeline Gradients
print("\n--- Running Pipeline (Sharded) ---")
grad_pipe_fn = jax.jit(jax.grad(pipeline_loss_fn, argnums=(0, 1)))
d_x_pipe, d_w_pipe = grad_pipe_fn(x_stream, w_global, y_targets)
print("Pipeline Run Complete.")

# 3. Compute Reference Gradients
print("\n--- Running Reference (Sequential) ---")
grad_ref_fn = jax.jit(jax.grad(reference_loss_fn, argnums=(0, 1)))
d_x_ref, d_w_ref = grad_ref_fn(x_stream, w_global, y_targets)
print("Reference Run Complete.")

# 4. Compare
print("\n--- Comparison Results ---")

# Compare Input Gradients
diff_x = jnp.max(jnp.abs(d_x_pipe - d_x_ref))
print(f"Max Difference in Input Gradients:   {diff_x:.8f}")

# Compare Weight Gradients
diff_w = jnp.max(jnp.abs(d_w_pipe - d_w_ref))
print(f"Max Difference in Weight Gradients:  {diff_w:.8f}")

# Final Verdict
if diff_w < 1e-3 and diff_x < 1e-3:
    print("\n[Image of clean green checkmark]")
    print(
        "SUCCESS: The sharded pipeline produces MATHEMATICALLY IDENTICAL gradients to the sequential model."
    )
else:
    print("\n[Image of red warning sign]")
    print("WARNING: Mismatch detected. Check padding/slicing logic.")
