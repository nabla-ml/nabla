# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import numpy as np
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace
from nabla.core.sharding import DeviceMesh, DimSpec

def test_full_trace_fb():
    """Trace both forward and backward pass to see the full graph."""
    mesh = DeviceMesh("test_mesh", (2,), ("tp",))
    
    x1_data = np.random.randn(8, 4).astype(np.float32)
    x2_data = np.random.randn(4, 8).astype(np.float32)
    
    x1 = nb.Tensor.from_dlpack(x1_data)
    x2 = nb.Tensor.from_dlpack(x2_data)
    
    # Pre-shard inputs
    x1 = nb.ops.shard(x1, mesh, [DimSpec(["tp"]), DimSpec([])])
    # x2 remains replicated
    
    def forward_fn(a, b):
        res = nb.matmul(a, b)
        return nb.reduce_sum(res, axis=0)

    def full_pipeline(a, b):
        # 1. Capture forward trace (this will actually capture into the OUTER trace if we are already tracing)
        # But wait, trace(forward_fn, a, b) will return a Trace object we can use for backward.
        f_trace = trace(forward_fn, a, b)
        
        # 2. Get output cotangent
        from nabla.ops.creation import full_like
        cot = full_like(f_trace.outputs, 1.0)
        
        # 3. Compute backward
        grads = backward_on_trace(f_trace, cot)
        
        # Return grads to be captured in the outer trace
        return grads[a], grads[b]

    print("\n--- Tracing Full Step (Forward + Backward) ---")
    t_full = trace(full_pipeline, x1, x2)
    print(t_full)
    
    # The output should show the forward ops followed by the backward ops (communications, etc)
    # Forward: shard, shard, matmul, reduce_sum
    # Backward: VJP of reduce_sum -> AllGather (of cotangent?) 
    #           VJP of matmul -> two matmuls
    #           VJP of shard -> AllGather (to get back to replicated grad)

if __name__ == "__main__":
    test_full_trace_fb()
