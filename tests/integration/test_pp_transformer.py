# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import numpy as np
import unittest
from nabla import ops
from nabla.core.sharding import DeviceMesh, DimSpec
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
    return x + (h @ weights['W2'])

def get_pipeline_perm(mesh):
    if "stage" not in mesh.axis_names:
        return None
    stage_idx = mesh.axis_names.index("stage")
    stage_size = mesh.shape[stage_idx]
    total = len(mesh.devices)
    perm = []
    for src in range(total):
        coords = list(mesh.get_coordinate(src, ax) for ax in mesh.axis_names)
        coords[stage_idx] = (coords[stage_idx] + 1) % stage_size
        dst = next(d for d in range(total) if list(mesh.get_coordinate(d, ax) for ax in mesh.axis_names) == coords)
        perm.append((src, dst))
    return perm

def full_pipeline_step(x, weights):
    y = vmap(transformer_layer, in_axes=(0, 0), out_axes=0)(x, weights)
    mesh = y._impl.sharding.mesh if hasattr(y, '_impl') and y._impl.sharding else None
    if mesh:
        perm = get_pipeline_perm(mesh)
        if perm:
            return communication.ppermute(y, perm)
    stages = int(x.shape[0])
    return communication.ppermute(y, [(i, (i + 1) % stages) for i in range(stages)])

class TestTransformerPP(unittest.TestCase):
    def _create_weights(self, stages, d_model):
        d_ff = 4 * d_model
        return {k: np.random.randn(stages, d_model, d_model if k not in ['W1', 'W2'] else (d_ff if k == 'W1' else d_model)).astype(np.float32) for k in ['Wq', 'Wk', 'Wv', 'Wo']} | \
               {'W1': np.random.randn(stages, d_model, d_ff).astype(np.float32), 'W2': np.random.randn(stages, d_ff, d_model).astype(np.float32)}

    def _numpy_layer(self, x, weights, d_model):
        Q, K, V = x @ weights['Wq'], x @ weights['Wk'], x @ weights['Wv']
        scores = (Q @ np.swapaxes(K, -1, -2)) / (d_model ** 0.5)
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        x = x + (attn @ V @ weights['Wo'])
        return x + (np.maximum(x @ weights['W1'], 0) @ weights['W2'])

    def _verify(self, mesh, x_shape, dp_axis=None):
        STAGES, D = x_shape[0], x_shape[-1]
        x_data = np.random.randn(*x_shape).astype(np.float32)
        weights_data = self._create_weights(STAGES, D)
        
        x_specs = [DimSpec(["stage"])] + ([DimSpec([dp_axis])] if dp_axis else [])
        x_specs += [DimSpec([])] * (len(x_shape) - len(x_specs))
        w_specs = [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
        
        x = ops.shard(ops.constant(x_data), mesh, x_specs)
        weights = {k: ops.shard(ops.constant(v), mesh, w_specs) for k, v in weights_data.items()}
        
        out_nabla = full_pipeline_step(x, weights).to_numpy()
        
        out_numpy = np.zeros_like(x_data)
        for i in range(STAGES):
            out_numpy[(i + 1) % STAGES] = self._numpy_layer(x_data[i], {k: v[i] for k, v in weights_data.items()}, D)
            
        np.testing.assert_allclose(out_nabla, out_numpy, atol=1e-2, rtol=1e-3)

    def test_pp_trace(self):
        mesh = DeviceMesh("pp", (2,), ("stage",))
        x = ops.shard(ops.constant(np.zeros((2, 16, 32), np.float32)), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])])
        weights = {k: ops.shard(ops.constant(v), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]) for k, v in self._create_weights(2, 32).items()}
        t = trace(full_pipeline_step, x, weights)
        self.assertIn("ppermute", str(t))

    def test_pp_values(self):
        self._verify(DeviceMesh("pp", (2,), ("stage",)), (2, 16, 32))
        self._verify(DeviceMesh("pp", (2,), ("stage",)), (2, 4, 16, 32))

    def test_pp_dp_values(self):
        mesh = DeviceMesh("pp_dp", (2, 2), ("data", "stage"))
        self._verify(mesh, (2, 4, 16, 32), dp_axis="data")

if __name__ == "__main__":
    unittest.main()
