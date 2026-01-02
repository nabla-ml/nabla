
"""Test 3D Parallelism on an MLP.

This test validates that a Multi-Layer Perceptron (MLP) can be correctly
executed on a 3D DeviceMesh, combining:
1.  Data Parallelism (DP)
2.  Tensor Parallelism (TP)
3.  Sequence/Pipeline Parallelism (SP) or another axis.

It verifies both metadata propagation and numerical correctness.
"""

import asyncio
import numpy as np
import pytest
from nabla import Tensor, DeviceMesh, DimSpec, ops
from nabla.sharding.spec import ShardingSpec

class MLPBlock:
    """Megatron-style Block: Linear(Col) -> ReLU -> Linear(Row).
    
    This block structure allows for efficient Tensor Parallelism:
    1. Linear 1 (Col Parallel): Splits execution, output is sharded on TP axis.
    2. ReLU: Elementwise, preserves sharding.
    3. Linear 2 (Row Parallel): Takes sharded input, produces partial output (needs AllReduce).
    
    In Nabla's auto-prop system, we define the weights sharding, and the system
    should automatically deduce the activations sharding and insert comms.
    """
    def __init__(self, d_model: int, hidden: int, mesh: DeviceMesh):
        self.d_model = d_model
        self.hidden = hidden
        
        # W1: (d_model, hidden) -> Split hidden on TP (dim 1) 
        # (Column Parallel)
        self.np_w1 = np.random.normal(0, 0.02, (d_model, hidden)).astype(np.float32)
        self.w1 = Tensor.from_dlpack(self.np_w1)
        # Shard: (Replicated, TP)
        self.w1 = self.w1.shard(mesh, [DimSpec([]), DimSpec(["tp"])])

        # W2: (hidden, d_model) -> Split hidden on TP (dim 0)
        # (Row Parallel)
        self.np_w2 = np.random.normal(0, 0.02, (hidden, d_model)).astype(np.float32)
        self.w2 = Tensor.from_dlpack(self.np_w2)
        # Shard: (TP, Replicated)
        self.w2 = self.w2.shard(mesh, [DimSpec(["tp"]), DimSpec([])])
        
    def __call__(self, x: Tensor) -> Tensor:
        h = x @ self.w1
        h = ops.relu(h)
        out = h @ self.w2
        return out

class DeepMLP:
    def __init__(self, d_model, hidden, num_blocks, mesh):
        self.blocks = [MLPBlock(d_model, hidden, mesh) for _ in range(num_blocks)]
        
    def __call__(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x

def test_mlp_3d_parallelism():
    """Run Deep MLP with DP, TP, and SP on a 2x2x2 Mesh."""
    import asyncio
    
    async def run():
        # 3D Mesh: (dp=2, tp=2, sp=2) - 8 Devices
        mesh = DeviceMesh("cluster", (2, 2, 2), ("dp", "tp", "sp"))
        
        batch = 4
        seq = 4
        d_model = 8
        hidden = 16
        num_blocks = 4  # Deeper network
        
        # Input: (batch, seq, d_model)
        # Sharding: (dp, sp, None) 
        np_x = np.random.randn(batch, seq, d_model).astype(np.float32)
        x = Tensor.from_dlpack(np_x)
        x = x.shard(mesh, [DimSpec(["dp"]), DimSpec(["sp"]), DimSpec([])])
        
        print(f"\n--- Initial Input Sharding: {x._impl.sharding} ---")
        
        model = DeepMLP(d_model, hidden, num_blocks, mesh)
        
        print(f"\n--- Running Deep MLP ({num_blocks} blocks) ---")
        
        # Forward pass
        output = model(x)
        
        print("\n--- Propagation Results (Pre-Execution) ---")
        out_spec = output._impl.sharding
        print(f"Final Output Sharding Spec: {out_spec}")
        
        assert out_spec is not None
        assert out_spec.dim_specs[0].axes == ["dp"], "Batch dim lost 'dp' sharding"
        assert out_spec.dim_specs[1].axes == ["sp"], "Seq dim lost 'sp' sharding"
        assert not out_spec.dim_specs[2].axes, "Hidden dim should be replicated (no 'tp')"
        
        print("✓ Propagation verification PASSED: Output matches (DP, SP, R) layout.")
        
        # Trigger execution
        print("\n--- Triggering Execution (Realize) ---")
        await output.realize
        
        print("Execution complete.")
        print(f"Output shape: {output.shape}")
        
        print("✓ Sharding propagation verified correctly for Deep 3D MLP.")
        
        # --- Advanced Capability 1: Reshape on Sharded Dimensions ---
        print("\n--- Advanced: Reshaping Sharded Dimensions ---")
        
        from nabla.ops.view import reshape
        flattened = reshape(output, (batch, seq * d_model))
        
        await flattened.realize
        
        flat_spec = flattened._impl.sharding
        print(f"Flattened shape: {flattened.shape}")
        print(f"Flattened sharding: {flat_spec}")
        
        # Verify 'sp' is preserved in the second dimension
        assert flat_spec.dim_specs[0].axes == ["dp"]
        assert flat_spec.dim_specs[1].axes == ["sp"]
        
        print("✓ Factor-based propagation correctly tracked sharding through reshape.")

        # --- Advanced Capability 2: Explicit Resharding (Gather) ---
        print("\n--- Advanced: Explicit Resharding (Gather) ---")
        
        replicated_spec = [DimSpec([]), DimSpec([])]
        gathered = flattened.shard(mesh, replicated_spec)
        
        await gathered.realize
        
        final_spec = gathered._impl.sharding
        print(f"Gathered sharding: {final_spec}")
        
        assert final_spec.is_fully_replicated()
        
        print("✓ Explicit resharding successfully triggered gather/replication.")
        
        # --- Advanced Capability 3: Numerical Verification vs NumPy ---
        print("\n--- Advanced: Numerical Correctness Check ---")
        
        # 1. Compute Expected NumPy Result
        print("Computing expected result with NumPy...")
        expected = np_x
        for block in model.blocks:
            w1 = block.np_w1
            w2 = block.np_w2
            
            # Linear 1
            h = expected @ w1
            # ReLU
            h = np.maximum(0, h)
            # Linear 2
            expected = h @ w2
            
        # Reshape expected to match 'gathered' tensor (batch, seq*d_model)
        expected_flat = expected.reshape(batch, seq * d_model)
        
        # 2. Extract Actual Result
        assert gathered._impl.is_realized
        assert len(gathered._impl._storages) > 0
        actual_flat = gathered._impl._storages[0].to_numpy()
        
        # 3. Compare with Tolerance
        tolerance = 1e-4 
        print(f"Comparing shapes: Actual {actual_flat.shape} vs Expected {expected_flat.shape}")
        
        np.testing.assert_allclose(actual_flat, expected_flat, rtol=tolerance, atol=tolerance)
        print("✓ Numerical verification PASSED: Nabla 3D execution matches NumPy exact inference.")

    asyncio.run(run())

if __name__ == "__main__":
    test_mlp_3d_parallelism()
