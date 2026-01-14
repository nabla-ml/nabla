"""Test xpr visualization of 3D parallelism sharding for MLP.

Based on tests/integration/with_sharding/test_mlp_3d.py
This traces a Megatron-style MLPBlock with tensor parallelism.
"""

import numpy as np
import nabla
from nabla import Tensor, DeviceMesh, DimSpec, ops
from nabla.utils.debug import capture_trace

# =============================================================================
# MLPBlock: Megatron-style Tensor Parallel Layer
# =============================================================================

def mlp_block_forward(x, w1, w2):
    """Megatron-style MLPBlock forward pass.
    
    Pattern:
    1. Linear 1 (Col Parallel): x @ w1 -> sharded on TP axis
    2. ReLU: Elementwise, preserves sharding
    3. Linear 2 (Row Parallel): h @ w2 -> needs AllReduce
    
    Args:
        x: Input (batch, seq, d_model) - sharded on DP, SP
        w1: First weight (d_model, hidden) - sharded on TP (column parallel)
        w2: Second weight (hidden, d_model) - sharded on TP (row parallel)
    """
    # Column parallel matmul: input is replicated on dim 2, w1 splits output
    h = x @ w1
    # Elementwise ReLU preserves sharding
    h = ops.relu(h)
    # Row parallel matmul: sharded input, produces partial output
    out = h @ w2
    return out


# =============================================================================
# Test: 2D Tensor Parallelism (simpler case)
# =============================================================================

def test_2d_tensor_parallel():
    """Trace MLPBlock with 2D mesh (DP x TP)."""
    print("=" * 70)
    print("TRACE: 2D Tensor Parallel MLP (DP x TP)")
    print("=" * 70)
    print()
    
    # 2D Mesh: (dp=2, tp=2) - 4 devices
    mesh = DeviceMesh("mesh", (2, 2), ("dp", "tp"))
    print(f"Mesh: {mesh.name} shape={mesh.shape} axes={mesh.axis_names}")
    
    batch, seq, d_model, hidden = 4, 8, 16, 32
    
    # Input: (batch, seq, d_model) - sharded on DP
    x = Tensor.ones((batch, seq, d_model))
    x = x.shard(mesh, [DimSpec(["dp"]), DimSpec([]), DimSpec([])])
    
    # W1: (d_model, hidden) - Column Parallel: replicated on dim 0, sharded on TP for dim 1
    w1 = Tensor.ones((d_model, hidden))
    w1 = w1.shard(mesh, [DimSpec([]), DimSpec(["tp"])])
    
    # W2: (hidden, d_model) - Row Parallel: sharded on TP for dim 0, replicated on dim 1
    w2 = Tensor.ones((hidden, d_model))
    w2 = w2.shard(mesh, [DimSpec(["tp"]), DimSpec([])])
    
    print(f"\nInput shapes:")
    print(f"  x: {tuple(x.shape)} @ {x._impl.sharding}")
    print(f"  w1: {tuple(w1.shape)} @ {w1._impl.sharding}")
    print(f"  w2: {tuple(w2.shape)} @ {w2._impl.sharding}")
    print()
    
    # Trace the forward pass
    print("-" * 70)
    print("TRACE OUTPUT:")
    print("-" * 70)
    def wrapped_forward(x, w1, w2):
        out = mlp_block_forward(x, w1, w2)
        return out + 1

    trace = capture_trace(wrapped_forward, x, w1, w2)
    print(trace)
    print()
    
    # Numerical verification
    print("-" * 70)
    print("NUMERICAL VERIFICATION:")
    print("-" * 70)
    
    # Run the actual computation
    output = mlp_block_forward(x, w1, w2)
    
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output sharding: {output._impl.sharding}")
    
    # Get numpy result
    try:
        result_np = output.to_numpy()
        print(f"Output numpy shape: {result_np.shape}")
        print(f"Output values (first 3): {result_np.flat[:3]}")
        
        # Expected numpy computation
        # x is ones(4, 8, 16), w1 is ones(16, 32), w2 is ones(32, 16)
        # h = x @ w1 = ones(4, 8, 16) @ ones(16, 32) = 16 * ones(4, 8, 32)
        # h_relu = relu(h) = 16 * ones(4, 8, 32)
        # out = h @ w2 = 16 * ones(4, 8, 32) @ ones(32, 16) = 16 * 32 * ones(4, 8, 16)
        expected_val = 16 * 32  # = 512
        print(f"Expected value (all elements): {expected_val}")
        
        if np.allclose(result_np, expected_val, rtol=1e-3):
            print("✓ Numerical verification PASSED!")
        else:
            print(f"✗ MISMATCH! Got {result_np.flat[0]}, expected {expected_val}")
            print(f"  With TP=2, if ALL_REDUCE missing, expect {expected_val / 2} = 256")
    except Exception as e:
        print(f"Error getting numpy: {e}")
    
    return trace


# =============================================================================
# Test: 3D Tensor Parallelism (full case)
# =============================================================================

def test_3d_tensor_parallel():
    """Trace MLPBlock with 3D mesh (DP x TP x SP)."""
    print("=" * 70)
    print("TRACE: 3D Tensor Parallel MLP (DP x TP x SP)")
    print("=" * 70)
    print()
    
    # 3D Mesh: (dp=2, tp=2, sp=2) - 8 devices
    mesh = DeviceMesh("cluster", (2, 2, 2), ("dp", "tp", "sp"))
    print(f"Mesh: {mesh.name} shape={mesh.shape} axes={mesh.axis_names}")
    
    batch, seq, d_model, hidden = 4, 8, 16, 32
    
    # Input: (batch, seq, d_model) - sharded on DP and SP
    x = Tensor.ones((batch, seq, d_model))
    x = x.shard(mesh, [DimSpec(["dp"]), DimSpec(["sp"]), DimSpec([])])
    
    # W1: (d_model, hidden) - Column Parallel: replicated on dim 0, sharded on TP for dim 1
    w1 = Tensor.ones((d_model, hidden))
    w1 = w1.shard(mesh, [DimSpec([]), DimSpec(["tp"])])
    
    # W2: (hidden, d_model) - Row Parallel: sharded on TP for dim 0, replicated on dim 1
    w2 = Tensor.ones((hidden, d_model))
    w2 = w2.shard(mesh, [DimSpec(["tp"]), DimSpec([])])
    
    print(f"\nInput shapes:")
    print(f"  x: {tuple(x.shape)} @ {x._impl.sharding}")
    print(f"  w1: {tuple(w1.shape)} @ {w1._impl.sharding}")
    print(f"  w2: {tuple(w2.shape)} @ {w2._impl.sharding}")
    print()
    
    # Trace the forward pass
    print("-" * 70)
    print("TRACE OUTPUT:")
    print("-" * 70)
    trace = capture_trace(mlp_block_forward, x, w1, w2)
    print(trace)
    print()
    
    return trace


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n")
    test_2d_tensor_parallel()
    print("\n\n")
    test_3d_tensor_parallel()
