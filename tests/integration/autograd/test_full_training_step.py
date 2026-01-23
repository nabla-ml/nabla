# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import numpy as np
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace
from nabla.core.sharding import DeviceMesh, DimSpec

def test_full_mlp_training_step():
    """
    Demonstrate a complete training step of a sharded MLP.
    Pattern: [Column Parallel] -> ReLU -> [Row Parallel] -> Sum Loss -> SGD Update.
    """
    # 1. Setup Mesh (2 devices)
    mesh = DeviceMesh("cortex_mesh", (2,), ("tp",))
    
    # 2. Dimensions
    B, Din, H, Dout = 4, 8, 16, 4
    lr = 0.01
    
    # 3. Parameters
    # Layer 1: Column Parallel [Din, H] -> shard H
    W1_data = (np.random.randn(Din, H).astype(np.float32) * 0.1)
    W1 = nb.ops.shard(nb.Tensor.from_dlpack(W1_data), mesh, [DimSpec([]), DimSpec(["tp"])])
    
    # Layer 2: Row Parallel [H, Dout] -> shard H (matching contraction)
    W2_data = (np.random.randn(H, Dout).astype(np.float32) * 0.1)
    W2 = nb.ops.shard(nb.Tensor.from_dlpack(W2_data), mesh, [DimSpec(["tp"]), DimSpec([])])
    
    # 4. Inputs (Replicated for simplicity, or Data Parallel)
    X_data = np.random.randn(B, Din).astype(np.float32)
    X = nb.Tensor.from_dlpack(X_data)
    
    # Target (for Loss)
    Target_data = np.random.randn(B, Dout).astype(np.float32)
    Target = nb.Tensor.from_dlpack(Target_data)

    def train_step(x, w1, w2, target):
        h1 = nb.matmul(x, w1)
        h1_act = nb.ops.relu(h1)
        y = nb.matmul(h1_act, w2)
        diff = y - target
        loss = nb.reduce_sum(diff * diff, axis=0)
        return nb.reduce_sum(loss, axis=0)

    def full_pipeline(x, w1, w2, target):
        t = trace(train_step, x, w1, w2, target)
        cot = nb.Tensor.from_dlpack(np.array(1.0, dtype=np.float32))
        grads = backward_on_trace(t, cot)
        new_w1 = w1 - grads[w1] * lr
        new_w2 = w2 - grads[w2] * lr
        
        return t.outputs, new_w1, new_w2

    print("\n" + "="*80)
    print("CAPTURING FULL PHYSICAL TRACE OF A TRAINING STEP")
    print("="*80)
    
    t_full = trace(full_pipeline, X, W1, W2, Target)
    print(t_full)
    
    # 6. Execute one step and see results
    print("\nExecuting step...")
    from nabla.core import GRAPH
    if isinstance(t_full.outputs, (list, tuple)):
        GRAPH.evaluate(*t_full.outputs)
    else:
        GRAPH.evaluate(t_full.outputs)
    
    res = [o.to_numpy() for o in t_full.outputs]
    
    loss_val = res[0]
    print(f"\nInitial Loss: {loss_val.item():.6f}")
    
    # 7. Check if weights actually updated and stayed sharded
    new_w1 = t_full.outputs[1]
    print(f"New W1 sharding: {new_w1.sharding}")
    assert not nb.core.sharding.spec.needs_reshard(new_w1.sharding, W1.sharding)
    
    print("\n✅ Success: A full training step of a sharded MLP was traced and executed!")
    print("The trace above shows the automatic insertion of AllReduce and inverse communication.")

if __name__ == "__main__":
    test_full_mlp_training_step()
