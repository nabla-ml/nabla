"""Test performance after the output_refs fix."""
import nabla as nb
from nabla.core.autograd import value_and_grad
import numpy as np
import time

# Simple MLP
np.random.seed(42)
N, D_in, H, D_out = 64, 784, 128, 10
x = nb.Tensor.from_dlpack(np.random.randn(N, D_in).astype(np.float32))
y_true = nb.Tensor.from_dlpack(np.random.randn(N, D_out).astype(np.float32))
W1 = nb.Tensor.from_dlpack(np.random.randn(D_in, H).astype(np.float32) * 0.01)
b1 = nb.Tensor.from_dlpack(np.zeros((1, H)).astype(np.float32))
W2 = nb.Tensor.from_dlpack(np.random.randn(H, D_out).astype(np.float32) * 0.01)
b2 = nb.Tensor.from_dlpack(np.zeros((1, D_out)).astype(np.float32))

def loss_fn(params):
    W1, b1, W2, b2 = params
    h = nb.relu(x @ W1 + b1)
    y_pred = h @ W2 + b2
    diff = y_pred - y_true
    return nb.mean(diff * diff)

vg_fn = value_and_grad(loss_fn, argnums=0, realize=False)

# Warmup (first run compiles)
params = [W1, b1, W2, b2]
loss, grads = vg_fn(params)
loss.realize()
for g in grads:
    g.realize()
print('Warmup done')

# Measure 100 epochs
times = []
for epoch in range(100):
    start = time.perf_counter()
    params = [W1, b1, W2, b2]
    loss, grads = vg_fn(params)
    loss.realize()
    for g in grads:
        g.realize()
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    if epoch % 20 == 0 or epoch == 99:
        print(f'Epoch {epoch}: {elapsed:.2f}ms')

print(f'Mean: {np.mean(times):.2f}ms, Std: {np.std(times):.2f}ms')
print(f'First 10: {np.mean(times[:10]):.2f}ms')
print(f'Last 10: {np.mean(times[-10:]):.2f}ms')
