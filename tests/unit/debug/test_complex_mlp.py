"""
Complex Distributed MLP Sharding Test

This test simulates a distributed MLP training step with Hybrid Sharding (DP + TP).
It stresses the sharding propagation system with:
1.  Column Parallel Linear (DP input -> Hybrid output)
2.  Row Parallel Linear (Hybrid input -> DP output, requires AllReduce)
3.  Reshapes that merge sharded dimensions
4.  Implicit conflict resolution in residual connections
"""

import pytest
from nabla import Tensor, DeviceMesh, DimSpec
import nabla.ops as ops
from nabla.sharding import ShardingSpec
from nabla.utils import debug


def test_hybrid_mlp_sharding_trace():
    """
    Simulates a Hybrid Sharded MLP (DP=2, TP=4) on a 2x4 mesh.
    
    Architecture:
    - Input: [Batch, Seq, Hidden] sharded (DP, *, *)
    - W1: [Hidden, Inter] sharded (*, TP) -> ColParallel
    - W2: [Inter, Hidden] sharded (TP, *) -> RowParallel
    """
    # 1. Setup Mesh (2x4 = 8 devices)
    m = DeviceMesh("cluster", shape=(2, 4), axis_names=("dp", "tp"))
    
    # 2. Define Inputs
    B, S, H, I = 4, 8, 16, 32
    
    # X: [Batch, Seq, Hidden] -> Sharded on 'dp' only (Data Parallel)
    # Local shape on each DP shard: [B/2, S, H]
    X = Tensor.zeros((B, S, H)).with_sharding(m, [DimSpec(["dp"]), DimSpec([]), DimSpec([])])
    
    # W1: [Hidden, Inter] -> Sharded on 'tp' (Column Parallel)
    # Local shape on each TP shard: [H, I/4]
    W1 = Tensor.zeros((H, I)).with_sharding(m, [DimSpec([]), DimSpec(["tp"])])
    
    # W2: [Inter, Hidden] -> Sharded on 'tp' (Row Parallel)
    # Local shape on each TP shard: [I/4, H]
    W2 = Tensor.zeros((I, H)).with_sharding(m, [DimSpec(["tp"]), DimSpec([])])
    
    def forward_step(x, w1, w2):
        # --- Layer 1: Column Parallel ---
        # x: [B, S, H] <dp, *, *>
        # w1: [H, I] <*, tp>
        # matmul -> [B, S, I] <dp, *, tp>
        # (Batch is split by DP, Intermediate dim is split by TP)
        h1 = ops.matmul(x, w1)
        
        # --- Reshape (Flatten) ---
        # [B, S, I] -> [B*S, I]
        # Sharding should merge: <dp, *, tp> -> <dp, tp>
        # (Assuming 'dp' is tracked through the merge of B and S)
        h_flat = ops.reshape(h1, (B*S, I))
        
        # --- Layer 2: Row Parallel ---
        # h_flat: [B*S, I] <dp, tp>
        # w2: [I, H] <tp, *>
        # Contracting dim 'I' is sharded on 'tp'.
        # This implies the result is a PARTIAL SUM.
        # Output: [B*S, H] <dp, *> (but heavily sharded internal state)
        h2 = ops.matmul(h_flat, w2)
        
        # --- AllReduce (Implicit) ---
        # The contracting dim 'I' is sharded on 'tp'.
        # Nabla's matmul op detects this and automatically applies 'all_reduce' 
        # over the 'tp' axis within the op. 
        # So 'h2' is already the correct partial sum reduced result.
        
        # --- Residual Connection with Conflict ---
        # Residual: flattened X -> [B*S, H]
        res = ops.reshape(x, (B*S, H)) # inherits <dp, *>
        
        # FORCE CONFLICT: Make 'res' sharded on <dp, tp> (split hidden dim)
        # This conflicts with h2 which is <dp, *> (replicated hidden)
        res_conflict = res.with_sharding(m, [DimSpec(["dp"]), DimSpec(["tp"])])
        
        # Add: <dp, *> + <dp, tp>
        # Should trigger implicit reshard (likely gather res_conflict to *)
        out = ops.add(h2, res_conflict)
        
        return out

    # Capture Trace
    trace = debug.capture_trace(forward_step, X, W1, W2)
    
    print("\n" + "=" * 80)
    print("TRACE: Hybrid Sharding MLP (DP+TP)")
    print("=" * 80)
    printer = debug.GraphPrinter(trace)
    print(printer.to_string())
    
    # --- Programmatic Verification ---
    # Get the final output node from trace
    out_node = [n for n in trace.nodes if n.op_name == "add"][0]
    out_sharding = out_node.sharding
    
    print(f"\nFinal Output Sharding: {out_sharding}")
    
    # Verify Mesh
    assert out_sharding.mesh.name == "cluster"
    
    # Verify Dimensions: [B*S, H]
    # Dim 0 should be sharded on 'dp' (inherited from X)
    assert "dp" in out_sharding.dim_specs[0].axes
    
    # Dim 1 should be REPLICATED or sharded on 'tp' depending on resolution strategy
    # If we added <dp, *> + <dp, tp>, BASIC strategy typically gathers to common prefix
    # which is <dp>. So dim 1 should be empty (replicated).
    # If AGGRESSIVE strategy was used, it might shard both on tp.
    # Currently default is BASIC.
    # Actually, <dp, *> vs <dp, tp>. Common prefix is... tricky.
    # If dim1 is empty vs ["tp"], common prefix is empty.
    # So expected result is <dp, *>.
    assert not out_sharding.dim_specs[1].axes, "Expected dim 1 to be replicated after conflict resolution"


if __name__ == "__main__":
    test_hybrid_mlp_sharding_trace()
