
import unittest
import numpy as np
from nabla.core.tensor import Tensor
from nabla import ops
from nabla.sharding import DeviceMesh, DimSpec, ShardingSpec
from nabla.ops import control_flow

class TestScanSharding(unittest.TestCase):
    def test_scan_produces_sharded_output(self):
        """Verify that scan propagates sharding from inner fn to output buffer."""
        mesh = DeviceMesh("mesh", (2,), axis_names=("x",))
        
        # Simple functionality: y = x + 1
        # x is sharded on 'x'. y will be sharded on 'x'.
        # We want the scan output buffer to be sharded on 'x' as well.
        
        # Input: (Length=4, Partitioned=4)
        # We shard dimension 1. Dim 0 is the scan axis.
        
        # Logical shape: (4, 4)
        input_np = np.random.randn(4, 4).astype(np.float32)
        x = Tensor.from_dlpack(input_np)
        
        # Shard x on axis 1
        spec = ShardingSpec(mesh, [DimSpec([]), DimSpec(["x"])])
        x = ops.shard(x, mesh, spec.dim_specs)
        
        init = Tensor.from_dlpack(np.zeros((), dtype=np.float32)) # Scalar carry
        
        def scan_fn(carry, x_i):
            # x_i shape (4,) sharded on 'x'
            # y_i = x_i + 1
            y_i = x_i + 1.0
            return carry, y_i
            
        final_carry, stacked_ys = control_flow.scan(scan_fn, init, x)
        
        # Check stacked_ys sharding
        # Should be (4, 4) sharded on dim 1
        # Current implementation likely returns Replicated because buffer was Replicated
        
        self.assertIsNotNone(stacked_ys._impl.sharding, "Output should have sharding spec")
        self.assertFalse(stacked_ys._impl.sharding.is_fully_replicated(), "Output should not be fully replicated")
        
        # Check dim specs
        # Dim 0: Replicated (Scan axis)
        # Dim 1: "x" (Data axis)
        specs = stacked_ys._impl.sharding.dim_specs
        self.assertEqual(specs[1].axes, ['x'], "Dim 1 should be sharded on x (Found: {})".format(specs[1].axes))

    def test_stack_sharding(self):
        """Verify that stack preserves sharding on non-stacked axes."""
        mesh = DeviceMesh("mesh", (2,), axis_names=("x",))
        # Input (2, 4) sharded on dim 1
        x = Tensor.from_dlpack(np.random.randn(2, 4).astype(np.float32))
        spec = ShardingSpec(mesh, [DimSpec([]), DimSpec(["x"])])
        x = ops.shard(x, mesh, spec.dim_specs)
        
        # Stack two of them
        # Result (2, 2, 4) if axis=0
        y = ops.view.stack([x, x], axis=0)
        
        self.assertIsNotNone(y._impl.sharding)
        specs = y._impl.sharding.dim_specs
        # Dim 0: Replicated (New axis)
        # Dim 1: Replicated (Old dim 0)
        # Dim 2: Sharded "x" (Old dim 1)
        self.assertEqual(len(specs), 3)
        self.assertEqual(specs[2].axes, ['x'])


class TestPipelineMLP(unittest.TestCase):
    def test_spmd_pipeline_mlp(self):
        """Test true SPMD Pipeline Parallel MLP execution.
        
        - h: (6, 4) sharded on stage -> (2, 4) per stage [batch dim sharded]
        - W: (12, 4) sharded on stage -> (4, 4) per stage [different W per stage]
        - b: (4,) REPLICATED - same bias for all stages, broadcasts with (2,4) output
        """
        from nabla.ops import communication
        
        BATCH_PER_STAGE = 2
        HIDDEN_DIM = 4
        NUM_STAGES = 3
        
        def stage_fn_spmd(h, W, b):
            """Each stage computes: relu(h_local @ W_local + b)"""
            out = h @ W  # SPMD matmul: (2,4) @ (4,4) = (2,4) per stage
            out = out + b  # Add: (6,4) + (4,) = (6,4)
            return ops.relu(out)

        def pipeline_loop_spmd(h, W, b, num_iters):
            perm = [(i, (i + 1) % NUM_STAGES) for i in range(NUM_STAGES)]
            for _ in range(num_iters):
                h = stage_fn_spmd(h, W, b)
                h = communication.ppermute(h, perm)
            return h

        mesh = DeviceMesh("pp", (NUM_STAGES,), ("stage",))
        
        # Setup Data
        scale = 0.5
        W_concat_np = np.concatenate([
            np.eye(HIDDEN_DIM, dtype=np.float32) * scale,
            np.eye(HIDDEN_DIM, dtype=np.float32) * scale,
            np.eye(HIDDEN_DIM, dtype=np.float32) * scale,
        ], axis=0)
        
        b_np = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        
        x_np = np.array([[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        h_concat_np = np.concatenate([x_np, np.zeros_like(x_np), np.zeros_like(x_np)], axis=0)
        
        # Create Tensors
        W_input = Tensor.from_dlpack(W_concat_np)
        b_input = Tensor.from_dlpack(b_np)
        h_input = Tensor.from_dlpack(h_concat_np)
        
        # Shard
        h = ops.shard(h_input, mesh, [DimSpec(["stage"]), DimSpec([])])
        W = ops.shard(W_input, mesh, [DimSpec(["stage"]), DimSpec([])])
        # b is replicated
        
        # Run pipeline
        result = pipeline_loop_spmd(h, W, b_input, num_iters=NUM_STAGES)
        
        # Verify
        result_np = result.to_numpy()
        
        self.assertEqual(tuple(int(d) for d in result.shape), (6, 4))
        
        # Check first row values (from successful run in trace)
        # [[0.3, 0.6, 0.9, 1.2]]
        expected_row0 = np.array([0.3, 0.6, 0.9, 1.2], dtype=np.float32)
        np.testing.assert_allclose(result_np[0], expected_row0, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
