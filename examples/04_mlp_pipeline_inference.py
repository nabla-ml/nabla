# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Pipeline parallel inference with GPipe.

Demonstrates: 4-stage inference pipeline, explicit graph tracing,
comparison with sequential NumPy reference.
"""

import numpy as np

import nabla as nb
from nabla import ops
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.core.sharding import PartitionSpec as P
from nabla.ops import communication
from nabla.transforms import vmap

STAGES = 4
MICRO_BATCHES = 8
MICRO_BATCH_SIZE = 4
DIM = 16


def stage_compute(x, w):
    return ops.relu(ops.matmul(x, w))


def pipeline_step(current_state, fresh_input, weight_stack, mask_0, step_fn, perm):
    """Single GPipe step: compute -> shift -> extract -> inject."""
    computed = step_fn(current_state, weight_stack)
    shifted = communication.ppermute(computed, perm)
    res_part = ops.where(mask_0, shifted, ops.zeros_like(shifted))
    result = ops.reduce_sum(res_part, axis=0)
    next_state = ops.where(mask_0, fresh_input, shifted)
    return next_state, result


def pipeline_inference_loop(
    padded_inputs, weight_stack, current_state, mask_0, step_fn, perm, total_steps
):
    results = []
    for t in range(total_steps):
        start_idx = (t, 0, 0)
        slice_size = (1, MICRO_BATCH_SIZE, DIM)
        fraction = ops.slice_tensor(padded_inputs, start=start_idx, size=slice_size)
        fresh = ops.squeeze(fraction, axis=0)

        current_state, res = pipeline_step(
            current_state, fresh, weight_stack, mask_0, step_fn, perm
        )
        results.append(res)

    return ops.stack(results, axis=0), current_state


def test_pp_inference_clean():
    mesh = DeviceMesh("pp", (STAGES,), ("stage",))
    print(f"Running GPipe Inference Test on Mesh: {mesh}")

    np.random.seed(42)

    w_np = np.random.randn(STAGES, DIM, DIM).astype(np.float32)
    x_np = np.random.randn(MICRO_BATCHES, MICRO_BATCH_SIZE, DIM).astype(np.float32)

    w_spec = [DimSpec.from_raw(d) for d in P("stage", None, None)]
    w_sharded = ops.shard(nb.Tensor.from_dlpack(w_np), mesh, w_spec).realize()

    padding = np.zeros((STAGES, MICRO_BATCH_SIZE, DIM), dtype=np.float32)
    x_padded_np = np.concatenate([x_np, padding], axis=0)
    x_padded_nb = nb.Tensor.from_dlpack(x_padded_np)

    state_np = np.zeros((STAGES, MICRO_BATCH_SIZE, DIM), dtype=np.float32)
    state_sharded = ops.shard(nb.Tensor.from_dlpack(state_np), mesh, w_spec).realize()

    mask_np = np.eye(STAGES, 1).reshape(STAGES, 1, 1).astype(bool)
    mask_0_sharded = ops.shard(nb.Tensor.from_dlpack(mask_np), mesh, w_spec).realize()

    idx = mesh.axis_names.index("stage")
    size = mesh.shape[idx]
    perm = [(i, (i + 1) % size) for i in range(size)]

    step_fn = vmap(
        stage_compute, in_axes=(0, 0), out_axes=0, spmd_axis_name="stage", mesh=mesh
    )

    def trace_wrapper(inputs, weights, state, mask):
        total_steps = MICRO_BATCHES + STAGES
        return pipeline_inference_loop(
            inputs, weights, state, mask, step_fn, perm, total_steps
        )

    traced = nb.core.graph.tracing.trace(
        trace_wrapper, x_padded_nb, w_sharded, state_sharded, mask_0_sharded
    )

    results_np = nb.core.tree_map(lambda x: x.to_numpy(), traced.outputs)
    preds_all = results_np[0]
    vals = preds_all[STAGES : STAGES + MICRO_BATCHES]

    print("Running Reference...")
    outs = []
    for i in range(MICRO_BATCHES):
        act = x_np[i]
        for s in range(STAGES):
            act = np.maximum(act @ w_np[s], 0)
        outs.append(act)
    ref = np.stack(outs)

    diff = np.max(np.abs(vals - ref))
    print(f"Max Diff: {diff:.6f}")

    if diff < 1e-4:
        print("✅ SUCCESS")
    else:
        print("❌ FAILURE")


if __name__ == "__main__":
    test_pp_inference_clean()
