import numpy as np
import unittest
from nabla import ops
from nabla.sharding import DeviceMesh, DimSpec
from nabla.core import trace
from nabla.ops import communication
from nabla.transforms.vmap import vmap

def transformer_layer(x, weights):
    Q = x @ weights['Wq']
    K = x @ weights['Wk']
    V = x @ weights['Wv']
    
    Kt = ops.swap_axes(K, -1, -2)
    scores = (Q @ Kt) / (int(Q.shape[-1]) ** 0.5)
    attn = ops.softmax(scores, axis=-1)
    
    context = attn @ V
    attn_out = context @ weights['Wo']
    x = x + attn_out
    
    h = ops.relu(x @ weights['W1'])
    mlp_out = h @ weights['W2']
    
    return x + mlp_out

class TestTransformerPPClean(unittest.TestCase):
    def _create_weights(self, stages, d_model):
        d_ff = 4 * d_model
        return {
            k: np.random.randn(stages, d_model, d_model if k != 'W1' else d_ff).astype(np.float32)
            for k in ['Wq', 'Wk', 'Wv', 'Wo']
        } | {
            'W1': np.random.randn(stages, d_model, d_ff).astype(np.float32),
            'W2': np.random.randn(stages, d_ff, d_model).astype(np.float32)
        }

    def test_clean_trace(self):
        mesh = DeviceMesh("pp_dp", (2, 2), ("data", "stage"))
        STAGES, BATCH, SEQ, D = 2, 4, 16, 32
        
        x_data = np.random.randn(STAGES, BATCH, SEQ, D).astype(np.float32)
        weights_data = self._create_weights(STAGES, D)
        
        x_specs = [DimSpec(["stage"]), DimSpec(["data"]), DimSpec([]), DimSpec([])]
        w_specs = [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
        
        x = ops.shard(ops.constant(x_data), mesh, x_specs)
        weights = {k: ops.shard(ops.constant(v), mesh, w_specs) for k, v in weights_data.items()}
        
        layer_batch_mapped = vmap(transformer_layer, in_axes=(0, None), out_axes=0, spmd_axis_name="data", mesh=mesh)
        layer_stage_batch_mapped = vmap(layer_batch_mapped, in_axes=(0, 0), out_axes=0, spmd_axis_name="stage", mesh=mesh)

        def full_pipeline_step(x, weights):
            y = layer_stage_batch_mapped(x, weights)
            
            stage_idx = mesh.axis_names.index("stage")
            stage_size = mesh.shape[stage_idx]
            total = len(mesh.devices)
            
            perm = []
            for src in range(total):
                coords = list(mesh.get_coordinate(src, ax) for ax in mesh.axis_names)
                coords[stage_idx] = (coords[stage_idx] + 1) % stage_size
                dst = next(d for d in range(total) if list(mesh.get_coordinate(d, ax) for ax in mesh.axis_names) == coords)
                perm.append((src, dst))
            return communication.ppermute(y, perm)

        print("\n--- CLEAN TRACE ---")
        t = trace(full_pipeline_step, x, weights)
        print(t)
        
        self.assertIn("matmul", str(t))
        self.assertIn("ppermute", str(t))
        
        out = full_pipeline_step(x, weights).to_numpy()
        self.assertEqual(out.shape, x_data.shape)

if __name__ == "__main__":
    unittest.main()
