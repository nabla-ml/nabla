# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import unittest
import numpy as np
import jax
import jax.numpy as jnp
import nabla as nb
from nabla import ops
from nabla.core import trace
from nabla.core.autograd import backward_on_trace
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.ops import communication
from nabla.transforms.vmap import vmap

def mlp_layer(x, weights):
    # Simple MLP: relu(x @ W + b)
    h = ops.relu(x @ weights["W1"] + weights["b1"])
    y = h @ weights["W2"] + weights["b2"]
    return y

def get_pp_permutation(mesh):
    stage_idx = mesh.axis_names.index("stage")
    stage_size = mesh.shape[stage_idx]
    total = len(mesh.devices)

    perm = []
    for src in range(total):
        coords = list(mesh.get_coordinate(src, ax) for ax in mesh.axis_names)
        coords[stage_idx] = (coords[stage_idx] + 1) % stage_size
        dst = next(
            d for d in range(total)
            if list(mesh.get_coordinate(d, ax) for ax in mesh.axis_names) == coords
        )
        perm.append((src, dst))
    return perm

class TestPPMLPTraining(unittest.TestCase):
    def test_pp_grad_accumulation(self):
        # 2D Mesh: data=2 (DP), stage=2 (PP)
        mesh = DeviceMesh("pp_dp", (2, 2), ("data", "stage"))
        STAGES, BATCH, SEQ, D = 2, 4, 8, 16
        
        np.random.seed(42)
        x_init_np = np.random.randn(STAGES, BATCH, SEQ, D).astype(np.float32)
        
        def create_weights_np():
            return {
                "W1": np.random.randn(STAGES, D, D).astype(np.float32) * 0.1,
                "b1": np.random.randn(STAGES, D).astype(np.float32) * 0.1,
                "W2": np.random.randn(STAGES, D, D).astype(np.float32) * 0.1,
                "b2": np.random.randn(STAGES, D).astype(np.float32) * 0.1,
            }
        
        weights_np = create_weights_np()
        
        # 1. Nabla Implementation
        x_nb = nb.Tensor.from_dlpack(x_init_np.copy())
        weights_nb = {
            k: nb.Tensor.from_dlpack(v.copy()) for k, v in weights_np.items()
        }
        
        # Shard inputs
        # x: [stage, batch, seq, d] -> shard stage on mesh.stage, batch on mesh.data
        x_specs = [DimSpec(["stage"]), DimSpec(["data"]), DimSpec([]), DimSpec([])]
        # weights: [stage, d1, d2] -> shard stage on mesh.stage
        w_specs = [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
        b_specs = [DimSpec(["stage"]), DimSpec([])]
        
        x_nb = ops.shard(x_nb, mesh, x_specs)
        weights_nb = {
            k: ops.shard(v, mesh, w_specs if "W" in k else b_specs) 
            for k, v in weights_nb.items()
        }
        
        # Setup VMaps
        # Inner vmap: over batch dimension (axis 1 of x, None for weights as weights are shared per batch)
        # Wait, if we use in_axes=(0, None) and pass the slice from outer vmap...
        layer_batch_mapped = vmap(
            mlp_layer, in_axes=(0, None), out_axes=0,
            spmd_axis_name="data", mesh=mesh
        )
        # Outer vmap: over stage dimension (axis 0 of x, axis 0 of weights)
        layer_stage_batch_mapped = vmap(
            layer_batch_mapped, in_axes=(0, 0), out_axes=0,
            spmd_axis_name="stage", mesh=mesh
        )
        
        forward_perm = get_pp_permutation(mesh)

        def pp_train_step(x, weights):
            curr_x = x
            accum_loss = 0.0
            
            # Run only 1 stage for debugging
            for t in range(1):
                # Compute
                y = layer_stage_batch_mapped(curr_x, weights)
                
                # Handover (Commented for debugging)
                # curr_x = communication.ppermute(y, forward_perm)
                curr_x = y
                
                # Loss accumulation
                accum_loss = accum_loss + ops.reduce_sum(y, axis=list(range(len(y.shape))))
                
            return accum_loss

        print("\nTracing PP MLPTraining Step...")
        traced = trace(pp_train_step, x_nb, weights_nb)
        print(traced)
        
        print("Computing Gradients...")
        cotangent = nb.Tensor.from_dlpack(np.array(1.0, dtype=np.float32))
        grads_nb = backward_on_trace(traced, cotangent)
        print("Gradients computed.")
        
        # 2. Reference Implementation (Single-device, no sharding)
        def ref_mlp_layer(x, w):
            h = jax.nn.relu(jnp.matmul(x, w["W1"]) + w["b1"])
            y = jnp.matmul(h, w["W2"]) + w["b2"]
            return y
            
        def ref_loop(x, w):
            curr_x = x
            loss = 0.0
            for t in range(1):
                vmapped_layer = jax.vmap(ref_mlp_layer, in_axes=(0, 0))
                y = vmapped_layer(curr_x, w)
                # curr_x = jnp.roll(y, shift=1, axis=0)
                curr_x = y
                loss += jnp.sum(y)
            return loss

        grad_fn_jax = jax.grad(ref_loop, argnums=(0, 1))
        gx_jax, gw_jax = grad_fn_jax(x_init_np, weights_np)
        
        # 3. Compare Results
        gx_nb = grads_nb[x_nb].to_numpy()
        np.testing.assert_allclose(gx_nb, gx_jax, rtol=1e-4, atol=1e-5, err_msg="Input gradient mismatch")
        print("  \u2713 Input gradient matched!")
        
        for k in weights_np.keys():
            gnb = grads_nb[weights_nb[k]].to_numpy()
            gjax = gw_jax[k]
            np.testing.assert_allclose(gnb, gjax, rtol=1e-4, atol=1e-5, err_msg=f"Weight grad mismatch for {k}")
            print(f"  \u2713 {k} gradient matched!")

if __name__ == "__main__":
    unittest.main()
