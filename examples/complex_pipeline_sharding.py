#!/usr/bin/env python3
"""
Complex Pipeline Sharding Demo
==============================

This demo simulates a long chain of operations with mixed sharding constraints
to stress-test the propagation algorithm's stability over many steps.

Pipeline:
1. Input (Replicated) -> Broadcast -> Sharded A
2. Sharded A -> Elementwise -> Sharded B
3. Sharded B @ Sharded C -> Output D (Reduction)
4. Output D -> Reshape -> E
5. E + Sharded F -> G
6. Reduction(G) -> Final Loss

We deliberately mix replicated, sharded, and different axis constraints.
"""

import asyncio
import numpy as np
from nabla import Tensor, ops
from nabla.sharding import DeviceMesh, DimSpec

def print_sharding(name: str, tensor: Tensor) -> None:
    """Pretty print sharding info."""
    spec = tensor._impl.sharding
    if spec is None:
        s = "Replicated (Implicit)"
    elif spec.is_fully_replicated():
        s = "Replicated (Explicit)"
    else:
        dims = [f"d{i}:{d.axes}" for i, d in enumerate(spec.dim_specs) if d.axes]
        s = ", ".join(dims) if dims else "Replicated"
    print(f"  {name:20s} {str(tuple(int(d) for d in tensor.shape)):<15s} [{s}]")

def run_pipeline_demo():
    print("\n" + "="*80)
    print("  COMPLEX PIPELINE SHARDING DEMO")
    print("="*80)

    # Mesh: 2x2 devices
    mesh = DeviceMesh("pipe_mesh", (2, 2), ("x", "y"))
    print(f"\n[1] Mesh: {mesh}")

    # 1. Inputs
    # A: start Replicated (scalar)
    a = Tensor.from_dlpack(np.array(2.0, dtype=np.float32)).trace()
    
    # B: Sharded on dim 0 ("x")
    b_np = np.random.randn(16, 64).astype(np.float32)
    b = Tensor.from_dlpack(b_np).trace()
    b.shard(mesh, [DimSpec(["x"]), DimSpec([])])
    
    print("\n[2] Initial Inputs:")
    print_sharding("A (Scalar)", a)
    print_sharding("B (Sharded)", b)

    # 2. Pipeline
    print("\n[3] Pipeline Execution:")
    
    # Step 1: Scalar Broadcast + Add -> Sharded
    # A (Scalar) + B (Sharded "x") -> C (Sharded "x")
    c = a + b
    print_sharding("C = A + B", c)
    
    # Step 2: Matmul with Sharded Weights
    # C [16, 64] @ W [64, 32] -> D [16, 32]
    # W sharded on dim 1 ("y")
    w_np = np.random.randn(64, 32).astype(np.float32)
    w = Tensor.from_dlpack(w_np).trace()
    w.shard(mesh, [DimSpec([]), DimSpec(["y"])])
    
    # C contributes "x" to dim 0. W contributes "y" to dim 1.
    # Result D should be [16(x), 32(y)]. VALID on 2D mesh.
    print_sharding("W (Sharded d1)", w)
    
    d = c @ w
    print_sharding("D = C @ W", d)

    # Step 3: Reduction
    # Reduce sum over dim 0 ("x").
    # D [16(x), 32(y)] -> sum(dim0) -> E [32(y)]
    e = ops.reduce_sum(d, axis=0)
    print_sharding("E = sum(D, 0)", e)
    
    # Step 4: Reshape
    # E [32(y)] -> [8, 4]
    # 32 elements. If 32 is sharded on "y" (size 2). 16 per device.
    # Reshape to [8, 4]. 32 elements.
    # Propagating sharding through reshape is tricky.
    # If "y" splits the 32 elements into [0..16] and [16..32].
    # [8, 4] flattens to 32.
    # [0..16] corresponds to rows 0..4.
    # [16..32] corresponds to rows 4..8.
    # So dim 0 of output should technically be sharded on "y".
    # Let's see if View propagation handles this!
    f = ops.reshape(e, (8, 4))
    print_sharding("F = reshape(E)", f)
    
    # Step 5: Verify
    print("\n[4] Verification:")
    
    with open("debug_out.txt", "w") as f_out:
        f_out.write(f"DEBUG SPEC C: {c._impl.sharding}\n")
        f_out.write(f"DEBUG SPEC W: {w._impl.sharding}\n")
        try:
            f_out.write(f"DEBUG SPEC D: {d._impl.sharding}\n")
        except NameError:
             pass
    
    # Reference
    # C is sharded on d0. W is sharded on d1.
    ref_c = 2.0 + b_np
    ref_d = ref_c @ w_np
    ref_e = ref_d.sum(axis=0)
    ref_f = ref_e.reshape(8, 4)

    # Verification
    # Note: We evaluate F directly. Evaluating E separately would reset the graph,
    # invalidating F's symbolic dependency on the pre-evaluated E.
    asyncio.run(f.realize)
    actual = f.to_numpy()
    
    if np.allclose(actual, ref_f, atol=1e-5):
         print("✓ SUCCESS: Robust against conflicts (End-to-End).")
    else:
         print("✗ FAILURE: F mismatch.")
         print(f" Max diff: {np.abs(actual - ref_f).max()}")
    
    # Reference
    # Need to simulate the conflict resolution logic for reference?
    # C is sharded on d0. W is sharded on d1.
    # Op is MatMul.
    # If system picks "Output sharded on d0", W must be all-gathered.
    # If system picks "Output sharded on d1", C must be all-gathered.
    # Nabla (and GSPMD) usually prioritized "output" sharding or "input" sharding based on cost.
    # Our simple propagation might just pick one.
    # Let's calculate reference purely mathematically.
    ref_c = 2.0 + b_np
    ref_d = ref_c @ w_np
    ref_e = ref_d.sum(axis=0)
    ref_f = ref_e.reshape(8, 4)
    
    if np.allclose(actual, ref_f, atol=1e-5):
         print("✓ SUCCESS: Robust against conflicts.")
    else:
         print("✗ FAILURE: Numeric mismatch.")
         print(f" Max diff: {np.abs(actual - ref_f).max()}")

if __name__ == "__main__":
    run_pipeline_demo()
