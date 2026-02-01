"""Profile benchmark to understand time breakdown."""
import time
import numpy as np

# Patch Operation.__call__ to count calls
op_call_count = 0

import nabla as nb
from nabla.ops import base as ops_base

original_call = ops_base.Operation.__call__

def counting_call(self, *args, **kwargs):
    global op_call_count
    op_call_count += 1
    return original_call(self, *args, **kwargs)

ops_base.Operation.__call__ = counting_call

from nabla.core.autograd import value_and_grad

# Setup from benchmark
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
    from nabla import ops
    for i in range(0, len(params) - 2, 2):
        w = params[i]
        b = params[i+1]
        x = ops.relu(ops.matmul(x, w) + b)
    x = ops.matmul(x, params[-2]) + params[-1]
    return x

def loss_fn(params, x, y):
    from nabla import ops
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

# Warmup (5 steps)
for _ in range(5):
    loss, grads = vg_fn(params, x, y)
    new_params = [p - g * lr for p, g in zip(params, grads)]
    nb.realize_all(loss, *new_params)
    params = new_params

# Reset counter
op_call_count = 0

# Single step timing breakdown
print("=== Single training step breakdown ===")

t0 = time.perf_counter()
loss, grads = vg_fn(params, x, y)
t1 = time.perf_counter()
print(f"  value_and_grad: {(t1-t0)*1000:.2f}ms, Op calls: {op_call_count}")

count_after_vg = op_call_count
t2 = time.perf_counter()
new_params = [p - g * lr for p, g in zip(params, grads)]
t3 = time.perf_counter()
print(f"  SGD update:     {(t3-t2)*1000:.2f}ms, Op calls: {op_call_count - count_after_vg}")

count_after_sgd = op_call_count
t4 = time.perf_counter()
nb.realize_all(loss, *new_params)
t5 = time.perf_counter()
print(f"  realize_all:    {(t5-t4)*1000:.2f}ms, Op calls: {op_call_count - count_after_sgd}")

print(f"\n  TOTAL: {(t5-t0)*1000:.2f}ms, Total Op calls: {op_call_count}")
