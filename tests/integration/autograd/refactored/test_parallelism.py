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

    def pipeline_loss_fn(all_inputs, weight_stack, targets, stage_mask, state):
        perm = get_pp_permutation(mesh)
        
        def stage_compute(x, w):
            return ops.relu(ops.matmul(x, w))

        pipeline_step = vmap(stage_compute, in_axes=(0, 0), out_axes=0, spmd_axis_name="stage", mesh=mesh)
        
        valid_preds = []

        for t in range(MICRO_BATCHES + STAGES):
            activations = pipeline_step(state, weight_stack)
            shifted = communication.ppermute(activations, perm)
            
            if t >= STAGES:
                # Capture output wrapping to stage 0
                # Use masked sum to robustly extract from sharded tensor
                masked_out = ops.where(stage_mask, shifted, ops.zeros_like(shifted))
                pred = ops.reduce_sum(masked_out, axis=0)
                valid_preds.append(pred)
            
            if t < MICRO_BATCHES:
                fresh_in = ops.gather(all_inputs, ops.constant([t], dtype=DType.int64), axis=0)
                state = ops.where(stage_mask, fresh_in, shifted)
            else:
                # Use a properly typed and sharded zero tensor for "bubble" filling
                state = ops.where(stage_mask, ops.zeros_like(state), shifted)

        preds = ops.stack(valid_preds, axis=0)
        diff = preds - targets
        return ops.mean(diff * diff)

    # --- Nabla Run ---
    traced = nb.core.graph.tracing.trace(pipeline_loss_fn, x_nb, w_sharded, y_nb, mask_sharded, init_state_sharded)
    print("\n=== TRACED GRAPH ===")
    print(traced)
    print("====================")
    
    loss_nb = traced.outputs.to_numpy()
    
    cot = nb.Tensor.from_dlpack(np.array(1.0, dtype=np.float32))
    grads = nb.core.autograd.backward_on_trace(traced, cot)
    
    gx_nb = grads[x_nb].to_numpy()
    gw_nb = grads[w_sharded].to_numpy()

    # --- JAX Reference ---
    import jax
    import jax.numpy as jnp
    
    def ref_model(x_stream, w_stack, y_targets):
        def apply_mlp(x_batch):
            act = x_batch
            for i in range(STAGES):
                act = jax.nn.relu(jnp.dot(act, w_stack[i]))
            return act
        preds = jax.vmap(apply_mlp)(x_stream)
        return jnp.mean((preds - y_targets)**2)

    grad_fn = jax.jit(jax.value_and_grad(ref_model, argnums=(0, 1)))
    loss_ref, (gx_ref, gw_ref) = grad_fn(x_np, w_np, y_np)

    # --- Verification ---
    print(f"Loss: NB={loss_nb:.4f}, JAX={loss_ref:.4f}")
    np.testing.assert_allclose(loss_nb, loss_ref, atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(gx_nb, gx_ref, atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(gw_nb, gw_ref, atol=2e-3, rtol=2e-3)
    print("✓ SUCCESS: Nabla Pipeline Parallelism matches JAX (Forward & Backward)")

if __name__ == "__main__":
    test_pipeline_parallelism()
