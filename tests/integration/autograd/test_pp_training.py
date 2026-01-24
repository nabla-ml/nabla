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

def transformer_layer(x, weights):
    # Simple transformer-like layer
    Q = x @ weights["Wq"]
    K = x @ weights["Wk"]
    V = x @ weights["Wv"]

    Kt = ops.swap_axes(K, -1, -2)
    scores = (Q @ Kt) / (int(Q.shape[-1]) ** 0.5)
    attn = ops.softmax(scores, axis=-1)

    context = attn @ V
    attn_out = context @ weights["Wo"]
    x = x + attn_out

    h = ops.relu(x @ weights["W1"])
    mlp_out = h @ weights["W2"]

    return x + mlp_out

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

class TestPPActivationAccumulation(unittest.TestCase):
    def _create_weights_np(self, stages, d_model):
        d_ff = 4 * d_model
        return {
            k: np.random.randn(stages, d_model, d_model if k != "W1" else d_ff).astype(np.float32) * 0.01
            for k in ["Wq", "Wk", "Wv", "Wo"]
        } | {
            "W1": np.random.randn(stages, d_model, d_ff).astype(np.float32) * 0.01,
            "W2": np.random.randn(stages, d_ff, d_model).astype(np.float32) * 0.01,
        }

    def test_pp_grad_accumulation(self):
        # 2D Mesh: data=2 (DP), stage=2 (PP)
        mesh = DeviceMesh("pp_dp", (2, 2), ("data", "stage"))
        STAGES, BATCH, SEQ, D = 2, 4, 16, 32
        # Number of microbatches = BATCH (each microbatch size 1 if we vmap)
        # Actually in test_pp_transformer_clean, BATCH=4 means each device gets batch 2.
        
        np.random.seed(42)
        x_init_np = np.random.randn(STAGES, BATCH, SEQ, D).astype(np.float32)
        weights_np = self._create_weights_np(STAGES, D)
        
        # 1. Nabla Implementation
        x_nb = nb.Tensor.from_dlpack(x_init_np.copy())
        weights_nb = {
            k: nb.Tensor.from_dlpack(v.copy()) for k, v in weights_np.items()
        }
        
        # Shard inputs
        x_specs = [DimSpec(["stage"]), DimSpec(["data"]), DimSpec([]), DimSpec([])]
        w_specs = [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
        
        x_nb = ops.shard(x_nb, mesh, x_specs)
        weights_nb = {
            k: ops.shard(v, mesh, w_specs) for k, v in weights_nb.items()
        }
        
        # Setup VMaps
        layer_batch_mapped = vmap(
            transformer_layer, in_axes=(0, None), out_axes=0,
            spmd_axis_name="data", mesh=mesh
        )
        layer_stage_batch_mapped = vmap(
            layer_batch_mapped, in_axes=(0, 0), out_axes=0,
            spmd_axis_name="stage", mesh=mesh
        )
        
        forward_perm = get_pp_permutation(mesh)

        def pp_train_step(x, weights):
            curr_x = x
            accum_loss = 0.0
            
            # Unroll loop for S + B - 1 steps.
            # Simplified: just run S steps for now to see gradients move.
            for t in range(STAGES):
                # Compute
                y = layer_stage_batch_mapped(curr_x, weights)
                
                # Handover
                curr_x = communication.ppermute(y, forward_perm)
                
                # Accumulate some dummy "loss" to check gradients
                # Sum everything to a scalar
                accum_loss = accum_loss + ops.reduce_sum(y, axis=list(range(len(y.shape))))
                
            return accum_loss

        print("\nTracing PP Training Step...")
        traced = trace(pp_train_step, x_nb, weights_nb)
        print(f"Trace captured with {len(traced.nodes)} nodes.")
        
        print("Computing Gradients...")
        # Target loss is 1.0 cotangent
        cotangent = nb.Tensor.from_dlpack(np.array(1.0, dtype=np.float32))
        grads_nb = backward_on_trace(traced, cotangent)
        print("Gradients computed.")
        
        # 2. Reference Implementation (Single-device, no sharding)
        def ref_transformer_layer(x, w):
            q = jnp.matmul(x, w["Wq"])
            k = jnp.matmul(x, w["Wk"])
            v = jnp.matmul(x, w["Wv"])
            scores = jnp.matmul(q, k.transpose(0, 2, 1)) / (D**0.5)
            # JAX softmax
            exp_scores = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
            attn = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)
            context = jnp.matmul(attn, v)
            attn_out = jnp.matmul(context, w["Wo"])
            x = x + attn_out
            h = jax.nn.relu(jnp.matmul(x, w["W1"]))
            mlp_out = jnp.matmul(h, w["W2"])
            return x + mlp_out
            
        def ref_loop(x, w):
            curr_x = x
            loss = 0.0
            for t in range(STAGES):
                # Stage-wise vmap
                vmapped_layer = jax.vmap(ref_transformer_layer, in_axes=(0, 0))
                y = vmapped_layer(curr_x, w)
                # PP shift
                curr_x = jnp.roll(y, shift=1, axis=0)
                loss += jnp.sum(y)
            return loss

        grad_fn_jax = jax.grad(ref_loop, argnums=(0, 1))
        # weights_np is Dict[str, [S, D, D]]
        gx_jax, gw_jax = grad_fn_jax(x_init_np, weights_np)
        
        # 3. Compare Results
        # Check input grad
        gx_nb = grads_nb[x_nb].to_numpy()
        np.testing.assert_allclose(gx_nb, gx_jax, rtol=1e-4, atol=1e-5, err_msg="Input gradient mismatch")
        print("  ✓ Input gradient matched!")
        
        # Check weight grads
        for k in weights_np.keys():
            gnb = grads_nb[weights_nb[k]].to_numpy()
            gjax = gw_jax[k]
            np.testing.assert_allclose(gnb, gjax, rtol=1e-4, atol=1e-5, err_msg=f"Weight grad mismatch for {k}")
            print(f"  ✓ {k} gradient matched!")

if __name__ == "__main__":
    unittest.main()
