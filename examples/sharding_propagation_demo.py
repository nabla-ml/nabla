#!/usr/bin/env python3
"""
Sharding Propagation Demo: 2D Hybrid Parallelism
=================================================

This demo shows **hybrid parallelism** using a 2D device mesh that combines:
- **Data Parallelism (DP)**: Shard batches across "data" axis
- **Tensor Parallelism (TP)**: Shard model weights across "model" axis

This is the pattern used by Megatron-LM, FSDP+TP, and other large-scale training systems.

Device Mesh Layout (2x2 = 4 devices):
    
         model axis
         ──────────→
        ┌───────────┬───────────┐
  data  │ device 0  │ device 1  │  ← batch shard 0
  axis  ├───────────┼───────────┤
    │   │ device 2  │ device 3  │  ← batch shard 1
    ↓   └───────────┴───────────┘
            ↑           ↑
         weight      weight
         shard 0     shard 1

Run: python examples/sharding_propagation_demo.py
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
        dims = [f"d{i}:{d.axes}" for i, d in enumerate(spec.dim_specs) if d.axes]
        sharding_str = ", ".join(dims) if dims else "Replicated"
    shape_str = str(tuple(int(d) for d in tensor.shape))
    print(f"  {name:25s} {shape_str:<12s} [{sharding_str}]")


def hybrid_parallelism_demo():
    """
    Demo: Transformer-style MLP with 2D Hybrid Parallelism
    
    Combines data parallelism (batch sharding) with tensor parallelism 
    (column/row sharding of weight matrices) - the Megatron-LM pattern.
    """
    
    print("\n" + "="*75)
    print("  HYBRID PARALLELISM DEMO: Transformer MLP Block")
    print("  2D Mesh = Data Parallelism (batch) × Tensor Parallelism (model)")
    print("="*75)
    
    # =========================================================================
    # 2D Device Mesh: 2 data-parallel × 2 tensor-parallel = 4 devices
    # =========================================================================
    
    mesh = DeviceMesh("hybrid", shape=(2, 2), axis_names=("data", "model"))
    
    print(f"""
┌──────────────────────────────────────────────────────────────────────────┐
│ Device Mesh: {mesh}                              
│                                                                          │
│     "model" axis (tensor parallelism)                                    │
│         ─────────────────→                                               │
│        ┌─────────┬─────────┐                                             │
│  "data"│ dev 0   │ dev 1   │  ← batch shard 0, weight shards 0/1         │
│  axis  ├─────────┼─────────┤                                             │
│    │   │ dev 2   │ dev 3   │  ← batch shard 1, weight shards 0/1         │
│    ↓   └─────────┴─────────┘                                             │
└──────────────────────────────────────────────────────────────────────────┘
""")
    
    # =========================================================================
    # Setup: Transformer MLP dimensions
    # batch=8, seq_len=16, hidden=64, ffn_hidden=128
    # =========================================================================
    
    batch, seq_len, hidden, ffn_hidden = 8, 16, 64, 128
    
    # Input activations: (batch, seq_len, hidden)
    # Sharded on batch dimension (data parallelism)
    np_x = np.random.randn(batch, seq_len, hidden).astype(np.float32)
    x = Tensor.from_dlpack(np_x).trace()
    x.shard(mesh, [
        DimSpec(["data"]),  # batch dim: sharded across data axis
        DimSpec([]),        # seq_len: replicated
        DimSpec([]),        # hidden: replicated
    ])
    
    # FC1 weights: (hidden, ffn_hidden) - Column parallel (shard output dim)
    np_w1 = np.random.randn(hidden, ffn_hidden).astype(np.float32) * 0.1
    w1 = Tensor.from_dlpack(np_w1).trace()
    w1.shard(mesh, [
        DimSpec([]),         # hidden (input): replicated
        DimSpec(["model"]),  # ffn_hidden (output): sharded across model axis
    ])
    
    # FC2 weights: (ffn_hidden, hidden) - Row parallel (shard input dim)
    np_w2 = np.random.randn(ffn_hidden, hidden).astype(np.float32) * 0.1
    w2 = Tensor.from_dlpack(np_w2).trace()
    w2.shard(mesh, [
        DimSpec(["model"]),  # ffn_hidden (input): sharded across model axis
        DimSpec([]),         # hidden (output): replicated
    ])
    
    print("─── INITIAL TENSORS ───")
    print("  Name                      Shape        Sharding")
    print("  " + "─"*60)
    print_sharding("x (activations)", x)
    print_sharding("w1 (FC1, col-parallel)", w1)
    print_sharding("w2 (FC2, row-parallel)", w2)
    
    # =========================================================================
    # Forward Pass: Transformer MLP Block
    # Similar to: FFN(x) = GeLU(x @ W1) @ W2
    # =========================================================================
    
    print("\n─── FORWARD PASS (7 ops) ───")
    print("  Name                      Shape        Sharding")
    print("  " + "─"*60)
    
    # Op 1: First linear (column-parallel matmul)
    # x: (8, 16, 64) @ w1: (64, 128) -> h1: (8, 16, 128)
    # Sharding: data on batch, model on ffn_hidden
    h1 = x @ w1
    print_sharding("h1 = x @ w1", h1)
    
    # Op 2: Activation (GeLU approximation with ReLU for simplicity)
    h1_act = ops.unary.relu(h1)
    print_sharding("h1_act = relu(h1)", h1_act)
    
    # Op 3: Dropout-style scaling
    scale = Tensor.from_dlpack(np.array(0.9, dtype=np.float32)).trace()
    h1_scaled = h1_act * scale
    print_sharding("h1_scaled = h1_act * 0.9", h1_scaled)
    
    # Op 4: Second linear (row-parallel matmul) 
    # This has CONTRACTING dim on "model" axis -> needs AllReduce!
    # h1: (8, 16, 128) @ w2: (128, 64) -> h2: (8, 16, 64)
    h2 = h1_scaled @ w2
    print_sharding("h2 = h1_scaled @ w2", h2)
    
    # Op 5: Residual connection (x + h2)
    out = x + h2
    print_sharding("out = x + h2 (residual)", out)
    
    # Op 6: Layer norm approximation (mean over hidden dim)
    out_mean = ops.reduce_sum(out, axis=-1)  # (8, 16)
    print_sharding("out_mean = sum(out, -1)", out_mean)
    
    # Op 7: Final reduce for loss (sum over all)
    loss = ops.reduce_sum(out_mean, axis=0)  # (16,) then 
    print_sharding("loss = sum(out_mean, 0)", loss)
    
    # =========================================================================
    # Explanation
    # =========================================================================
    
    print(f"""
─── SHARDING ANALYSIS ───

  ✓ Input 'x' sharded on batch (data axis) → data parallelism
  ✓ Weight 'w1' sharded on output dim (model axis) → column TP
  ✓ Weight 'w2' sharded on input dim (model axis) → row TP  
  ✓ h1 gets BOTH: data on batch, model on ffn_hidden (2D sharding!)
  ✓ h2 has AllReduce on model axis (contracting dim was sharded)
  ✓ Reductions on sharded dims trigger AllReduce automatically

  This is the Megatron-LM pattern:
    Column-parallel FC1 → Activation → Row-parallel FC2 (+ AllReduce)
""")
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    print("─── EVALUATION ───")
    asyncio.run(loss.realize)
    
    # Verify against numpy reference
    np_h1 = np.maximum(np_x @ np_w1, 0) * 0.9
    np_h2 = np_h1 @ np_w2
    np_out = np_x + np_h2
    np_loss = np_out.sum(axis=-1).sum(axis=0)
    
    actual = loss.to_numpy()
    max_diff = np.abs(np_loss - actual).max()
    
    print(f"  Expected (numpy): {np_loss[:4]}...")
    print(f"  Actual (sharded): {actual[:4]}...")
    print(f"  Max diff: {max_diff:.2e}")
    
    if np.allclose(np_loss, actual, rtol=1e-4, atol=1e-5):
        print("\n  ✓ Hybrid parallelism works correctly!")
    else:
        print("\n  ✗ Results differ!")
    
    print("\n" + "="*75)


if __name__ == "__main__":
    hybrid_parallelism_demo()
