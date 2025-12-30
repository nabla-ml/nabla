#!/usr/bin/env python3
"""
Megatron-LM Style Sharding Demo
===============================

This demo implements a Transformer MLP block using a 2D Device Mesh to demonstrate
hybrid parallelism (Data Parallelism + Tensor Parallelism).

Architecture:
    Input (Batch Sharded) -> 
    Linear 1 (Column Parallel, Weights Sharded on Output) -> 
    GeLU -> 
    Linear 2 (Row Parallel, Weights Sharded on Input) -> 
    Output (Batch Sharded)

Key Concepts:
- **Data Parallelism (DP)**: All activations are sharded on the batch dimension.
- **Tensor Parallelism (TP)**: Weights are split. 
    - Weights [H, 4H] splits columns -> Output is [B, S, 4H] (sharded on last dim)
    - Weights [4H, H] splits rows -> Input is [B, S, 4H] (sharded on last dim)
- **AllReduce**: The second linear layer performs a reduction over the sharded contracting dimension, resulting in a partial sum that must be AllReduced to recover the correct values.

Mesh Layout:
    (2, 2) mesh named "hybrid" with axes ("data", "model")
"""

import asyncio
import numpy as np
from nabla import Tensor, ops
from nabla.sharding import DeviceMesh, DimSpec

def print_sharding(name: str, tensor: Tensor) -> None:
    """Pretty print sharding info for a tensor."""
    spec = tensor._impl.sharding
    if spec is None:
        sharding_str = "None"
    elif spec.is_fully_replicated():
        sharding_str = "Replicated"
    else:
        dims = []
        for i, d in enumerate(spec.dim_specs):
            if d.axes:
                dims.append(f"d{i}:{d.axes}")
        sharding_str = ", ".join(dims) if dims else "Replicated"
    
    shape_str = str(tuple(int(d) for d in tensor.shape))
    print(f"  {name:25s} {shape_str:<15s} [{sharding_str}]")

def run_megatron_demo():
    print("\n" + "="*80)
    print("  MEGATRON-STYLE SHARDING DEMO")
    print("  Hybrid Parallelism: Data (Batch) x Model (Tensor)")
    print("="*80)

    # 1. Define Device Mesh
    # 4 Devices organized as 2x2 grid
    mesh = DeviceMesh("hybrid", shape=(2, 2), axis_names=("data", "model"))
    print(f"\n[1] Device Mesh: {mesh}")

    # 2. Dimensions
    B, S, H, FF = 8, 16, 64, 256
    print(f"[2] Model Config: Batch={B}, Seq={S}, Hidden={H}, FFN={FF}")

    # 3. Create Inputs & Weights
    
    # Input X: [B, S, H] -> Sharded on Batch ("data")
    np_x = np.random.randn(B, S, H).astype(np.float32)
    x = Tensor.from_dlpack(np_x).trace()
    x = x.shard(mesh, [DimSpec(["data"]), DimSpec([]), DimSpec([])])
    
    # Weight 1 (Column Parallel): [H, FF] -> Sharded on Output ("model")
    # This means each device holds a subset of the output features (columns).
    np_w1 = np.random.randn(H, FF).astype(np.float32) * 0.1
    w1 = Tensor.from_dlpack(np_w1).trace()
    w1 = w1.shard(mesh, [DimSpec([]), DimSpec(["model"])])

    # Weight 2 (Row Parallel): [FF, H] -> Sharded on Input ("model")
    # This means each device holds a subset of the input features (rows).
    np_w2 = np.random.randn(FF, H).astype(np.float32) * 0.1
    w2 = Tensor.from_dlpack(np_w2).trace()
    w2 = w2.shard(mesh, [DimSpec(["model"]), DimSpec([])])

    print("\n[3] Initial Sharding:")
    print_sharding("Input X", x)
    print_sharding("Weight W1 (Col)", w1)
    print_sharding("Weight W2 (Row)", w2)

    # 4. Forward Pass
    print("\n[4] Forward Pass Operations:")

    # Layer 1: Linear (Column Parallel)
    # X [B, S, H] @ W1 [H, FF] -> H1 [B, S, FF]
    # Sharding: 
    #   X is sharded on B ("data")
    #   W1 is sharded on FF ("model")
    #   Result H1 inherits BOTH: Sharded on B ("data") and FF ("model")
    #   This is a "fully sharded" intermediate state.
    h1 = x @ w1
    print_sharding("H1 = X @ W1", h1)

    # Activation (Elementwise)
    # Preserves sharding
    h1 = ops.unary.relu(h1)
    print_sharding("H1 (Activated)", h1)

    # Layer 2: Linear (Row Parallel)
    # H1 [B, S, FF] @ W2 [FF, H] -> Out [B, S, H]
    # Sharding:
    #   H1 sharded on FF ("model") [Contracting Dim!]
    #   W2 sharded on FF ("model") [Contracting Dim!]
    #   Since the contracting dimension is sharded on "model", this triggers an AllReduce on "model".
    #   The output should presumably still be sharded on B ("data") because that axis didn't participate in reduction.
    out = h1 @ w2
    print_sharding("Out = H1 @ W2", out)

    # Residual
    result = x + out
    print_sharding("Result = X + Out", result)

    # 5. Correctness Check
    print("\n[5] Verifying Numerical Correctness...")
    asyncio.run(result.realize)
    
    # Reference (Numpy)
    ref_h1 = np.maximum(np_x @ np_w1, 0)
    ref_out = ref_h1 @ np_w2
    ref_result = np_x + ref_out
    
    actual = result.to_numpy()
    
    if np.allclose(actual, ref_result, atol=1e-5):
        print("✓ SUCCESS: Output matches numpy reference.")
    else:
        print("✗ FAILURE: Output mismatch.")
        print(f"  Max Diff: {np.abs(actual - ref_result).max()}")

if __name__ == "__main__":
    run_megatron_demo()
