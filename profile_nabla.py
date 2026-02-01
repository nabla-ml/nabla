"""Profile Nabla to find the bottleneck."""

import numpy as np
import time
import cProfile
import pstats
from io import StringIO

import nabla as nb
from nabla import ops
from nabla.core.autograd import value_and_grad

num_samples = 5
x_np = np.linspace(0, 1, num_samples).reshape(-1, 1).astype(np.float32)
y_np = (np.sin(4 * np.pi * x_np) + 1) / 2.0

layers = [1, 64, 64, 1]
np.random.seed(42)

init_params = []
for i in range(len(layers) - 1):
    in_dim = layers[i]
    out_dim = layers[i+1]
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    w_np = np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
    b_np = np.zeros((1, out_dim)).astype(np.float32)
    init_params.append((w_np, b_np))

lr = 0.01

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

x = nb.Tensor.from_dlpack(x_np)
y = nb.Tensor.from_dlpack(y_np)

params = []
for w, b in init_params:
    params.extend([nb.Tensor.from_dlpack(w), nb.Tensor.from_dlpack(b.reshape(1, -1))])
for p in params:
    p.is_traced = True

# Warmup
for _ in range(3):
    loss, grads = vg_fn(params, x, y)
    new_params = [p - g * lr for p, g in zip(params, grads)]
    nb.realize_all(loss, *new_params)
    params = new_params

# Reset
params = []
for w, b in init_params:
    params.extend([nb.Tensor.from_dlpack(w), nb.Tensor.from_dlpack(b.reshape(1, -1))])
for p in params:
    p.is_traced = True

def run_one_step():
    global params
    loss, grads = vg_fn(params, x, y)
    new_params = [p - g * lr for p, g in zip(params, grads)]
    nb.realize_all(loss, *new_params)
    params = new_params
    return loss

# Profile one step
print("Profiling one training step...")
profiler = cProfile.Profile()
profiler.enable()

for _ in range(10):
    run_one_step()

profiler.disable()

# Print top 30 time consumers
s = StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(40)
print(s.getvalue())
