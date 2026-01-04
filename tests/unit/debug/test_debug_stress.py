"""
Factor Propagation Stress Test

Demonstrates Nabla's factor-based sharding propagation through:
1. Conflicting input shardings (triggering implicit reshard)
2. Matmul with partial computation
3. AllReduce for aggregating partial results
4. Reshape that merges/splits sharded dimensions
5. AllGather to replicate results
"""

import pytest
from nabla import Tensor, DeviceMesh, DimSpec
import nabla.ops as ops
from nabla.sharding import ShardingSpec
from nabla.utils import debug


def test_full_sharding_showcase():
    """
    Comprehensive test of sharding propagation and communication.
    
    Input shardings:
    - A: [4, 8] <dp, tp> (fully sharded on 2x2 mesh)
    - B: [4, 8] <tp, dp> (CONFLICT - swapped axes)
    
    The elementwise add(A, B) must trigger all_gather to align the shardings.
    Then matmul with replicated W, followed by reshape factor tracking.
    """
    m = DeviceMesh("mesh", shape=(2, 2), axis_names=("dp", "tp"))
    
    # === Inputs with CONFLICTING sharding ===
    # A: [4, 8] sharded <dp, tp> -> local [2, 4]
    A = Tensor.zeros((4, 8)).with_sharding(m, [DimSpec(["dp"]), DimSpec(["tp"])])
    
    # B: [4, 8] sharded <tp, dp> -> local [2, 4] (same shape, different layout!)
    # Elementwise op on A+B requires alignment
    B = Tensor.zeros((4, 8)).with_sharding(m, [DimSpec(["tp"]), DimSpec(["dp"])])
    
    # W: [8, 4] replicated (so matmul output inherits from A/B sharding)
    W = Tensor.zeros((8, 4)).with_sharding(m, [DimSpec([]), DimSpec([])])
    
    def forward(a, b, w):
        # Step 1: Add A + B -> CONFLICT! Must gather both to replicated first
        # Current logic: detect conflict, gather to replicated
        # Expected trace: all_gather(a), all_gather(b), add
        h = ops.add(a, b)
        
        # Step 2: Matmul [4, 8]<*, *> @ [8, 4]<*, *> -> [4, 4]<*, *>
        # Since h is replicated and W is replicated, output is replicated
        h = ops.matmul(h, w)
        
        # Step 3: Shard the result explicitly to <dp, tp>
        # [4, 4] -> local [2, 2]
        h = h.with_sharding(m, [DimSpec(["dp"]), DimSpec(["tp"])])
        
        # Step 4: Reshape [4, 4]<dp, tp> -> [16]<dp, tp>
        # Factors dp, tp merge into single dimension
        out = ops.reshape(h, (16,))
        
        # Step 5: AllGather to replicate
        out = ops.all_gather(out, axis=0)
        
        return out

    trace = debug.capture_trace(forward, A, B, W)
    print("\n" + "=" * 70)
    print("TRACE: Full Sharding Showcase")
    print("=" * 70)
    print(debug.GraphPrinter(trace).to_string())


if __name__ == "__main__":
    test_full_sharding_showcase()
