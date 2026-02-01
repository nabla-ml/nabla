"""Benchmark comparison: Nabla vs JAX vs PyTorch for simple MLP training."""

import numpy as np
import time

# ============================================================================
# Common setup
# ============================================================================
num_samples = 5
x_np = np.linspace(0, 1, num_samples).reshape(-1, 1).astype(np.float32)
y_np = (np.sin(4 * np.pi * x_np) + 1) / 2.0

layers = [1, 64, 64, 1]
np.random.seed(42)

# Initialize weights
init_params = []
for i in range(len(layers) - 1):
    in_dim = layers[i]
    out_dim = layers[i+1]
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    w_np = np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
    b_np = np.zeros((1, out_dim)).astype(np.float32)
    init_params.append((w_np, b_np))

lr = 0.01
epochs = 100
warmup = 5

# ============================================================================
# JAX Implementation
# ============================================================================
def benchmark_jax():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", False)
    
    def mlp(params, x):
        for i, (w, b) in enumerate(params[:-1]):
            x = jax.nn.relu(x @ w + b)
        w, b = params[-1]
        return x @ w + b
    
    def loss_fn(params, x, y):
        preds = mlp(params, x)
        return jnp.mean((preds - y) ** 2)
    
    @jax.jit
    def train_step(params, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        new_params = [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]
        return new_params, loss
    
    # Initialize
    params = [(jnp.array(w), jnp.array(b)) for w, b in init_params]
    x = jnp.array(x_np)
    y = jnp.array(y_np)
    
    # Warmup
    for _ in range(warmup):
        params_tmp, _ = train_step(params, x, y)
    params_tmp[0][0].block_until_ready()
    
    # Reset params
    params = [(jnp.array(w), jnp.array(b)) for w, b in init_params]
    
    # Benchmark
    times = []
    for epoch in range(epochs):
        t0 = time.perf_counter()
        params, loss = train_step(params, x, y)
        loss.block_until_ready()
        times.append(time.perf_counter() - t0)
    
    print(f"JAX:     First: {times[0]*1000:.3f}ms, Mean (after warmup): {np.mean(times[warmup:])*1000:.3f}ms, Final loss: {float(loss):.6f}")
    return times

# ============================================================================
# PyTorch Implementation
# ============================================================================
def benchmark_pytorch():
    import torch
    
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList()
            for i in range(len(layers) - 1):
                self.layers.append(torch.nn.Linear(layers[i], layers[i+1]))
        
        def forward(self, x):
            for i, layer in enumerate(self.layers[:-1]):
                x = torch.relu(layer(x))
            return self.layers[-1](x)
    
    # Initialize with same weights
    model = MLP()
    with torch.no_grad():
        for i, (w, b) in enumerate(init_params):
            model.layers[i].weight.copy_(torch.from_numpy(w.T))
            model.layers[i].bias.copy_(torch.from_numpy(b.flatten()))
    
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        optimizer.step()
    
    # Reset
    with torch.no_grad():
        for i, (w, b) in enumerate(init_params):
            model.layers[i].weight.copy_(torch.from_numpy(w.T))
            model.layers[i].bias.copy_(torch.from_numpy(b.flatten()))
    
    # Benchmark
    times = []
    for epoch in range(epochs):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        optimizer.step()
        times.append(time.perf_counter() - t0)
    
    print(f"PyTorch: First: {times[0]*1000:.3f}ms, Mean (after warmup): {np.mean(times[warmup:])*1000:.3f}ms, Final loss: {float(loss):.6f}")
    return times

# ============================================================================
# Nabla Implementation
# ============================================================================
def benchmark_nabla():
    import nabla as nb
    from nabla import ops
    from nabla.core.autograd import value_and_grad
    
    def mlp(x, params):
        for i in range(0, len(params) - 2, 2):
            w = params[i]
            b = params[i+1]
            x = ops.relu(ops.matmul(x, w) + b)
        x = ops.matmul(x, params[-2]) + params[-1]
        return x
    
    def loss_fn(params, x, y):
        preds = mlp(x, params)
        diff = preds - y
        return ops.mean(diff * diff)
    
    vg_fn = value_and_grad(loss_fn, argnums=0, realize=False)
    
    # Initialize
    x = nb.Tensor.from_dlpack(x_np)
    y = nb.Tensor.from_dlpack(y_np)
    
    params = []
    for w, b in init_params:
        params.extend([nb.Tensor.from_dlpack(w), nb.Tensor.from_dlpack(b.reshape(1, -1))])
    for p in params:
        p.is_traced = True
    
    # Warmup  
    for _ in range(warmup):
        loss, grads = vg_fn(params, x, y)
        new_params = [p - g * lr for p, g in zip(params, grads)]
        nb.realize_all(loss, *new_params)
        params = new_params
        for p in params:
            p.is_traced = True
    
    # Reset params
    params = []
    for w, b in init_params:
        params.extend([nb.Tensor.from_dlpack(w), nb.Tensor.from_dlpack(b.reshape(1, -1))])
    for p in params:
        p.is_traced = True
    
    # Benchmark
    times = []
    for epoch in range(epochs):
        t0 = time.perf_counter()
        loss, grads = vg_fn(params, x, y)
        new_params = [p - g * lr for p, g in zip(params, grads)]
        nb.realize_all(loss, *new_params)
        times.append(time.perf_counter() - t0)
        params = new_params
        for p in params:
            p.is_traced = True
    
    print(f"Nabla:   First: {times[0]*1000:.3f}ms, Mean (after warmup): {np.mean(times[warmup:])*1000:.3f}ms, Final loss: {loss.item():.6f}")
    return times

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print(f"Benchmarking MLP training: {layers}, {num_samples} samples, {epochs} epochs")
    print("=" * 70)
    
    jax_times = benchmark_jax()
    pytorch_times = benchmark_pytorch()
    nabla_times = benchmark_nabla()
    
    print("=" * 70)
    jax_mean = np.mean(jax_times[warmup:]) * 1000
    pytorch_mean = np.mean(pytorch_times[warmup:]) * 1000
    nabla_mean = np.mean(nabla_times[warmup:]) * 1000
    
    print(f"\nNabla is {nabla_mean/jax_mean:.1f}x slower than JAX")
    print(f"Nabla is {nabla_mean/pytorch_mean:.1f}x slower than PyTorch")
