import os
# Comment out to allow NABLA_DEBUG from environment
# os.environ["NABLA_DEBUG"] = "0"

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
    num_samples = 5
    x_np = np.linspace(0, 1, num_samples).reshape(-1, 1).astype(np.float32)
    y_np = (np.sin(4 * np.pi * x_np) + 1) / 2.0
    
    x = nb.Tensor.from_dlpack(x_np)
    y = nb.Tensor.from_dlpack(y_np)
    
    # 2. Initialize MLP parameters
    layers = [1, 4,8,8,8,8,8,8,8,8,4, 1]
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
        w = nb.Tensor.from_dlpack(w_np)
        b = nb.Tensor.from_dlpack(b_np)
        params.extend([w, b])
    
    # Enable tracing for parameters
    for p in params:
        p.is_traced = True
        
    lr = 0.01
    epochs = 100
    
    from nabla.core.autograd import value_and_grad
    
    def loss_fn(params, x, y):
        preds = mlp(x, params)
        diff = preds - y
        return ops.mean(diff * diff)  # Don't realize here - let batching handle it
    
    # Use realize=False to defer compilation until we batch all operations
    vg_fn = value_and_grad(loss_fn, argnums=0, realize=False)
    
    print(f"Starting training for {epochs} epochs (lr={lr})...")
    for epoch in range(epochs):
        t_start = time.perf_counter()
        
        loss, grads = vg_fn(params, x, y)
        
        # Compute updated parameters (as lazy nodes)
        new_params = [p - g * lr for p, g in zip(params, grads)]
        
        # Realize everything at once in a SINGLE compilation/execution
        nb.realize_all(loss, *new_params)
        
        params = new_params
        
        if epoch % 1 == 0:
            duration = time.perf_counter() - t_start
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}, Time: {duration:.4f}s")
            
    print("Training complete.")
    
    # Final params should be realized
    for p in params:
        p
        
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
