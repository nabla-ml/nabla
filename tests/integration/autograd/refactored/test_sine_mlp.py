import os
# Set DEBUG_LAZY_EVAL to False BEFORE any other imports
os.environ["NABLA_DEBUG"] = "0"

import nabla as nb
from nabla import ops
from max.dtype import DType
import numpy as np
import time

def mlp(x, params):
    # params is list: [w1, b1, w2, b2, ...]
    for i in range(0, len(params) - 2, 2):
        w = params[i]
        b = params[i+1]
        x = ops.relu(ops.matmul(x, w) + b)
    x = ops.matmul(x, params[-2]) + params[-1]
    return x

def main():
    print("Setting up Sine Curve MLP training...")
    
    # 1. Generate data: Sine curve from 0 to 1
    # x in [0, 1], y = (sin(4 * pi * x) + 1) / 2
    num_samples = 5
    x_np = np.linspace(0, 1, num_samples).reshape(-1, 1).astype(np.float32)
    y_np = (np.sin(4 * np.pi * x_np) + 1) / 2.0
    
    x = nb.Tensor.from_dlpack(x_np)
    y = nb.Tensor.from_dlpack(y_np)
    
    # 2. Initialize MLP parameters
    # Input: 1, Hidden: 32, 32, Output: 1
    layers = [1, 64, 64, 1]
    params = []
    np.random.seed(42)
    for i in range(len(layers) - 1):
        in_dim = layers[i]
        out_dim = layers[i+1]
        # Glorot initialization
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        w_np = np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
        b_np = np.zeros((1, out_dim)).astype(np.float32)
        
        # Realize them immediately
        w = nb.Tensor.from_dlpack(w_np).realize()
        b = nb.Tensor.from_dlpack(b_np).realize()
        params.extend([w, b])
    
    # Enable tracing for parameters
    for p in params:
        p.is_traced = True
        
    lr = 0.01
    epochs = 30
    
    from nabla.core.autograd import value_and_grad
    
    def loss_fn(params, x, y):
        preds = mlp(x, params)
        diff = preds - y
        return ops.mean(diff * diff)
    
    vg_fn = value_and_grad(loss_fn, argnums=0)
    
    print(f"Starting training for {epochs} epochs (lr={lr})...")
    for epoch in range(epochs):
        t_start = time.perf_counter()
        
        loss, grads = vg_fn(params, x, y)
        
        # Compute updated parameters (as nodes)
        new_params = [p - g * lr for p, g in zip(params, grads)]
        
        # Realize everything at once to avoid multiple compilations
        nb.GRAPH.evaluate(loss, new_params)
        
        params = new_params
        
        if epoch % 1 == 0:
            duration = time.perf_counter() - t_start
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}, Time: {duration:.4f}s")
            # Print grad norms to check learning
            grad_norm = sum([np.linalg.norm(g.to_numpy()) for g in grads])
            print(f"  Grad norm: {grad_norm:.6f}")
            
    print("Training complete.")
    
    # Final params should be realized
    for p in params:
        p.realize()
        
    # Check predictions
    final_preds = mlp(x, params).to_numpy()
    mse = np.mean((final_preds - y_np)**2)
    print(f"Final MSE: {mse:.6f}")
    
    if mse < 0.05:
        print("✅ SUCCESS: Model learned the sine curve!")
    else:
        print("❌ FAILURE: Model did not learn the sine curve (MSE too high).")

if __name__ == "__main__":
    main()
