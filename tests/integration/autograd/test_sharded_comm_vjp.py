# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace
from nabla.core.sharding import DeviceMesh, DimSpec

def test_shard_vjp():
    """Test VJP for shard operation."""
    mesh = DeviceMesh("test_mesh", (2,), ("tp",))
    
    # 1. Replicated input
    x_data = np.random.randn(8, 4).astype(np.float32)
    x = nb.Tensor.from_dlpack(x_data)
    
    def shard_fn(a):
        # Shard on axis 0
        return nb.ops.shard(a, mesh, [DimSpec(["tp"]), DimSpec([])])
    
    print("\n--- Tracing Shard ---")
    traced = trace(shard_fn, x)
    print(traced)
    
    # Cotangent should be sharded matching the output of shard_fn
    # Since shard_fn output is sharded (4, 4) on each of 2 devices
    # Total global shape (8, 4)
    cot_data = np.ones((8, 4), dtype=np.float32)
    # We need to manually shard the cotangent or use a sharded ones_like
    y = shard_fn(x)
    from nabla.ops.creation import full_like
    cotangent = full_like(y, 1.0)
    
    print("\n--- Running Backward ---")
    try:
        grads = backward_on_trace(traced, cotangent)
        grad_x = grads[x]
        print(f"Success! grad_x sharding: {grad_x.sharding}")
        print(f"grad_x shape: {grad_x.shape}")
        
        # Verify correctness: grad_x should be replicated ones
        np.testing.assert_allclose(grad_x.to_numpy(), np.ones((8, 4)), rtol=1e-5)
        print("✓ Verified grad_x values")
        
    except Exception as e:
        print(f"Failed: {e}")
        # Expected to fail if vjp_rule is missing
        raise e

def test_all_gather_vjp():
    """Test VJP for all_gather operation."""
    mesh = DeviceMesh("test_mesh", (2,), ("tp",))
    
    # 1. Sharded input
    x_data = np.random.randn(8, 4).astype(np.float32)
    x_full = nb.Tensor.from_dlpack(x_data)
    x = nb.ops.shard(x_full, mesh, [DimSpec(["tp"]), DimSpec([])])
    
    def gather_fn(a):
        return nb.ops.all_gather(a, axis=0)
    
    print("\n--- Tracing AllGather ---")
    traced = trace(gather_fn, x)
    print(traced)
    
    # Output of AllGather is replicated (8, 4)
    y = gather_fn(x)
    from nabla.ops.creation import full_like
    cotangent = full_like(y, 1.0)
    
    print("\n--- Running Backward ---")
    try:
        grads = backward_on_trace(traced, cotangent)
        grad_x = grads[x]
        print(f"Success! grad_x sharding: {grad_x.sharding}")
        
        # Verify correctness: grad_x should be sharded ones
        # On device 0: (4, 4) ones. On device 1: (4, 4) ones.
        np.testing.assert_allclose(grad_x.to_numpy(), np.ones((8, 4)), rtol=1e-5)
        print("✓ Verified grad_x values")
        
    except Exception as e:
        print(f"Failed: {e}")
        raise e

if __name__ == "__main__":
    test_shard_vjp()
    test_all_gather_vjp()
