"""Profile to understand cache hit vs cache miss paths."""

import time
import numpy as np

import nabla
from nabla.core.autograd import value_and_grad
from nabla.core.graph import engine

# Track execute_on_shards calls
from nabla.core.sharding import spmd
original_execute_on_shards = spmd.execute_on_shards
execute_on_shards_calls = []

def patched_execute_on_shards(*args, **kwargs):
    import traceback
    t0 = time.perf_counter()
    result = original_execute_on_shards(*args, **kwargs)
    execute_on_shards_calls.append((time.perf_counter() - t0, ''.join(traceback.format_stack()[-5:-1])))
    return result

spmd.execute_on_shards = patched_execute_on_shards

# Run simple test
def train_step(params, x, y):
    def loss_fn(p):
        W1, b1, W2, b2 = p
        h = nabla.tanh(x @ W1 + b1)
        pred = h @ W2 + b2
        diff = pred - y
        return nabla.mean(diff * diff)
    
    loss, grads = value_and_grad(loss_fn)(params)
    return loss, grads

# Initialize
np.random.seed(42)
W1 = nabla.Tensor.from_dlpack(np.random.randn(1, 64).astype(np.float32) * 0.1)
b1 = nabla.Tensor.from_dlpack(np.zeros((1, 64), dtype=np.float32))
W2 = nabla.Tensor.from_dlpack(np.random.randn(64, 1).astype(np.float32) * 0.1)
b2 = nabla.Tensor.from_dlpack(np.zeros((1, 1), dtype=np.float32))
params = (W1, b1, W2, b2)

x = nabla.Tensor.from_dlpack(np.random.randn(5, 1).astype(np.float32))
y = nabla.Tensor.from_dlpack(np.sin(x.to_numpy()))

print("=" * 70)
print("Analyzing cache behavior")
print("=" * 70)

# First call - should compile
print("\n--- First training step (cache miss expected) ---")
execute_on_shards_calls.clear()
t0 = time.perf_counter()
loss, grads = train_step(params, x, y)
loss_val = loss.to_numpy()  # Force realization
t1 = time.perf_counter()
print(f"Total time: {(t1-t0)*1000:.2f}ms")
print(f"execute_on_shards calls: {len(execute_on_shards_calls)}")
if execute_on_shards_calls:
    print(f"execute_on_shards total: {sum(t for t, _ in execute_on_shards_calls)*1000:.2f}ms")

# Update params (simple SGD)
lr = 0.01
params = tuple(
    nabla.Tensor.from_dlpack((p.to_numpy() - lr * g.to_numpy()).astype(np.float32))
    for p, g in zip(params, grads)
)

# Second call - should be cache hit
print("\n--- Second training step (cache hit expected) ---")
execute_on_shards_calls.clear()
t0 = time.perf_counter()
loss, grads = train_step(params, x, y)
loss_val = loss.to_numpy()
t1 = time.perf_counter()
print(f"Total time: {(t1-t0)*1000:.2f}ms")
print(f"execute_on_shards calls: {len(execute_on_shards_calls)}")
if execute_on_shards_calls:
    print(f"execute_on_shards total: {sum(t for t, _ in execute_on_shards_calls)*1000:.2f}ms")
    print(f"\nFirst execute_on_shards call stack:")
    print(execute_on_shards_calls[0][1])

# Run a few more to get average
print("\n--- Average of next 10 calls ---")
times = []
exec_calls = []
for _ in range(10):
    params = tuple(
        nabla.Tensor.from_dlpack((p.to_numpy() - lr * g.to_numpy()).astype(np.float32))
        for p, g in zip(params, grads)
    )
    execute_on_shards_calls.clear()
    t0 = time.perf_counter()
    loss, grads = train_step(params, x, y)
    loss_val = loss.to_numpy()
    t1 = time.perf_counter()
    times.append((t1-t0)*1000)
    exec_calls.append(len(execute_on_shards_calls))

print(f"Average time: {np.mean(times):.2f}ms")
print(f"execute_on_shards calls per step: {exec_calls[0]} (should be 0 for cache hits)")
