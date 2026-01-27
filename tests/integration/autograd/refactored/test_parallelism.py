# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
import nabla as nb
from nabla import ops
from nabla.core.sharding import DeviceMesh, PartitionSpec as P, DimSpec
from nabla.transforms import vmap
from nabla.ops import communication
from max.dtype import DType

STAGES = 4
MICRO_BATCHES = 8
MICRO_BATCH_SIZE = 4
DIM = 16

def get_pp_permutation(mesh):
    idx = mesh.axis_names.index("stage")
    size = mesh.shape[idx]
    perm = []
    for src in range(len(mesh.devices)):
        coords = list(mesh.get_coordinate(src, ax) for ax in mesh.axis_names)
        coords[idx] = (coords[idx] + 1) % size
        dst = next(d for d in range(len(mesh.devices)) if list(mesh.get_coordinate(d, ax) for ax in mesh.axis_names) == coords)
        perm.append((src, dst))
    return perm

def test_pipeline_parallelism():
    mesh = DeviceMesh("pp", (STAGES,), ("stage",))
    np.random.seed(42)
    
    # 1. Setup Data & Sharded Inputs
    w_np = np.random.randn(STAGES, DIM, DIM).astype(np.float32)
    x_np = np.random.randn(MICRO_BATCHES, MICRO_BATCH_SIZE, DIM).astype(np.float32)
    y_np = np.random.randn(MICRO_BATCHES, MICRO_BATCH_SIZE, DIM).astype(np.float32)
    
    # We must REALIZE sharded constants before tracing so they act as graph inputs (leaves)
    w_spec = [DimSpec.from_raw(d) for d in P("stage", None, None)]
    w_sharded = ops.shard(nb.Tensor.from_dlpack(w_np), mesh, w_spec).realize()
    
    stage_ids_np = np.arange(STAGES).reshape(STAGES, 1, 1).astype(np.int32)
    is_stage_0_mask_np = (stage_ids_np == 0).repeat(MICRO_BATCH_SIZE, axis=1).repeat(DIM, axis=2).astype(bool)
    mask_sharded = ops.shard(nb.Tensor.from_dlpack(is_stage_0_mask_np), mesh, w_spec).realize()

    x_nb = nb.Tensor.from_dlpack(x_np) # Already realized
    y_nb = nb.Tensor.from_dlpack(y_np) # Already realized
    
    # Initialize state as float32 and sharded
    init_state_np = np.zeros((STAGES, MICRO_BATCH_SIZE, DIM), dtype=np.float32)
    init_state_sharded = ops.shard(nb.Tensor.from_dlpack(init_state_np), mesh, w_spec).realize()

    def pipeline_forward_fn(all_inputs, weight_stack, stage_mask, state):
        perm = get_pp_permutation(mesh)
        
        def stage_compute(x, w):
            return ops.relu(ops.matmul(x, w))

        # Vectorized computation across all stages
        pipeline_step_fn = vmap(stage_compute, in_axes=(0, 0), out_axes=0, spmd_axis_name="stage", mesh=mesh)
        
        valid_preds = []

        for t in range(MICRO_BATCHES + STAGES):
            # 1. COMPUTE: All stages process their current micro-batch in parallel
            activations = pipeline_step_fn(state, weight_stack)
            
            # 2. SHIFT: Move data i -> i+1. (N-1) wraps to 0.
            shifted = communication.ppermute(activations, perm)
            
            # 3. EXTRACT: If data at Stage 0 is a finished result from Stage N-1
            # In a pipeline of length N, it takes N steps of computation + N shifts
            # for the first batch to reach extraction point at Stage 0.
            if t >= STAGES:
                # Extract result from Stage 0 slot
                # We use where + reduce_sum as a robust sharded-to-replicated gather
                res_shard = ops.where(stage_mask, shifted, ops.zeros_like(shifted))
                pred = ops.reduce_sum(res_shard, axis=0)
                valid_preds.append(pred)
            
            # 4. INJECT: Put new data at Stage 0 slot
            if t < MICRO_BATCHES:
                fresh_in = ops.gather(all_inputs, ops.constant([t], dtype=DType.int64), axis=0)
                # Overwrite Stage 0 with fresh input, keep shifted data for other stages
                state = ops.where(stage_mask, fresh_in, shifted)
            else:
                # Just move shifted data, Stage 0 gets zeros (flushing)
                state = ops.where(stage_mask, ops.zeros_like(state), shifted)

        return ops.stack(valid_preds, axis=0)

    # --- Nabla Run ---
    traced = nb.core.graph.tracing.trace(pipeline_forward_fn, x_nb, w_sharded, mask_sharded, init_state_sharded)
    print("\n=== TRACED GRAPH ===")
    print(traced)
    print("====================")
    
    preds_nb = traced.outputs.to_numpy()
    
    # --- JAX Reference Forward Pass ---
    import jax
    import jax.numpy as jnp
    
    def ref_forward(x_stream, w_stack):
        def apply_mlp(x_batch):
            act = x_batch
            for i in range(STAGES):
                act = jax.nn.relu(jnp.dot(act, w_stack[i]))
            return act
        return jax.vmap(apply_mlp)(x_stream)

    preds_ref = jax.jit(ref_forward)(x_np, w_np)

    # --- Verification ---
    print(f"Predictions Shape: NB={preds_nb.shape}, JAX={preds_ref.shape}")
    diff = np.max(np.abs(preds_nb - preds_ref))
    print(f"Max Diff: {diff:.6f}")
    
    np.testing.assert_allclose(preds_nb, preds_ref, atol=2e-3, rtol=2e-3)
    print("✓ SUCCESS: Nabla Forward Pipeline Parallelism matches JAX")

if __name__ == "__main__":
    test_pipeline_parallelism()
