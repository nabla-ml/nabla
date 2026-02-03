# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
import numpy as np
import nabla as nb
from nabla.transforms import compile
from nabla.core.sharding import DeviceMesh, PartitionSpec as P
from nabla.ops import shard
import time

def test_compile_full_training_step():
    """Test full training step compilation with value_and_grad."""
    print("\n[Test] Full Training Step Compilation")
    
    # 1. Setup Model & Optim
    input_dim = 16
    hidden_dim = 32
    output_dim = 1
    
    w1_np = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
    b1_np = np.zeros(hidden_dim, dtype=np.float32)
    w2_np = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.1
    b2_np = np.zeros(output_dim, dtype=np.float32)
    
    w1 = nb.Tensor.constant(w1_np)
    b1 = nb.Tensor.constant(b1_np)
    w2 = nb.Tensor.constant(w2_np)
    b2 = nb.Tensor.constant(b2_np)
    
    params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    
    def model(x, params):
        h = nb.relu(x @ params["w1"] + params["b1"])
        return h @ params["w2"] + params["b2"]
    
    def loss_fn(params, x, y):
        pred = model(x, params)
        diff = pred - y
        return nb.mean(diff * diff)
    
    # 2. Compile Training Step
    # We compile the entire value_and_grad call!
    @compile
    def train_step(params, x, y, lr=0.01):
        # We MUST set realize=False when calling autograd inside a compiled function
        # to avoid triggering GRAPH.evaluate (which the compiler sees as a side effect).
        loss_val, grads = nb.value_and_grad(loss_fn, realize=False)(params, x, y)
        
        # Simple SGD update within the compiled graph
        new_params = {}
        for k, p in params.items():
            new_params[k] = p - grads[k] * lr
            
        return loss_val, new_params

    # 3. Create Dataset
    x_np = np.random.randn(8, input_dim).astype(np.float32)
    y_np = np.random.randn(8, output_dim).astype(np.float32)
    
    x = nb.Tensor.from_dlpack(x_np)
    y = nb.Tensor.from_dlpack(y_np)
    
    # 4. First Run (Trace + Compile)
    t0 = time.time()
    loss1, params1 = train_step(params, x, y)
    loss1.realize() # trigger execution
    t1 = time.time()
    
    print(f"Run 1 (Compile) Time: {(t1 - t0)*1000:.2f} ms")
    assert train_step.stats.misses == 1
    
    # Verify correctness against eager
    eager_loss, eager_grads = nb.value_and_grad(loss_fn)(params, x, y)
    np.testing.assert_allclose(loss1.to_numpy(), eager_loss.to_numpy(), atol=1e-5)
    
    # 5. Second Run (Cache Hit)
    t2 = time.time()
    loss2, params2 = train_step(params1, x, y) # Pass new params
    loss2.realize()
    t3 = time.time()
    
    print(f"Run 2 (Cache Hit) Time: {(t3 - t2)*1000:.2f} ms")
    assert train_step.stats.hits == 1
    
    print("Full training step compiled successfully!")

def test_sharded_training_step():
    """Test compiled training step with sharded tensors (Static Dims)."""
    try:
        mesh = DeviceMesh("test_mesh", (2,), ("x",))
    except Exception as e:
        pytest.skip(f"Skipping sharded test: {e}")
        return

    print("\n[Test] Sharded Training Step Compilation")

    # 1. Setup Sharded Inputs
    input_dim = 16
    hidden_dim = 32
    
    # Shard weights along hidden dim? Let's shard Data (Batch) for X
    # and weights replicated for now, or shard weights column-wise.
    # Let's do Data Parallel: X is sharded on dim 0.
    
    x_np = np.random.randn(8, input_dim).astype(np.float32)
    w_np = np.random.randn(input_dim, hidden_dim).astype(np.float32)
    
    x = shard(nb.Tensor.from_dlpack(x_np), mesh, P("x", None))
    w = nb.Tensor.from_dlpack(w_np) # Replicated
    
    def forward(x, w):
        return x @ w 
    
    @compile
    def sharded_step(x, w):
        out = forward(x, w)
        return nb.sum(out) # Reduction
    
    # Run 1
    loss = sharded_step(x, w)
    np.testing.assert_allclose(loss.to_numpy(), np.sum(x_np @ w_np), atol=1e-4)
    assert sharded_step.stats.misses == 1
    
    # Run 2
    loss2 = sharded_step(x, w)
    assert sharded_step.stats.hits == 1
    
    print("Sharded training step compiled successfully!")

def test_dynamic_dims_sharding_constraint():
    """Verify that we REJECT sharding + dynamic dims."""
    try:
        mesh = DeviceMesh("test_mesh_fail", (2,), ("x",))
    except Exception:
        pytest.skip("No mesh available")
        
    @compile(dynamic_dims={0: {0: "batch"}})
    def f(x):
        return x * 2.0
    
    x_np = np.random.randn(8, 4).astype(np.float32)
    x = shard(nb.Tensor.from_dlpack(x_np), mesh, P("x", None))
    
    print("\n[Test] Dynamic Dims + Sharding Constraint")
    with pytest.raises(NotImplementedError, match="not yet supported"):
        f(x)
    print("Correctly rejected sharding + dynamic dims")

if __name__ == "__main__":
    test_compile_full_training_step()
    test_sharded_training_step()
    test_dynamic_dims_sharding_constraint()
