"""Quick check of cache behavior."""
import nabla as nb
from nabla.core.autograd import value_and_grad
import numpy as np
import os

os.environ["NABLA_DEBUG"] = "1"
# Re-import to pick up debug flag
from nabla.core.graph import engine
engine.DEBUG_LAZY_EVAL = True

def train_step(params, x, y):
    def loss_fn(p):
        W1, b1 = p
        h = nb.tanh(x @ W1 + b1)
        return nb.mean(h * h)
    return value_and_grad(loss_fn, realize=False)(params)

W1 = nb.Tensor.from_dlpack(np.random.randn(1, 4).astype(np.float32))
b1 = nb.Tensor.from_dlpack(np.zeros((1, 4), dtype=np.float32))
x = nb.Tensor.from_dlpack(np.random.randn(2, 1).astype(np.float32))
y = nb.Tensor.from_dlpack(np.zeros((2, 4), dtype=np.float32))
params = (W1, b1)

print('=== First call ===')
loss, grads = train_step(params, x, y)
nb.realize_all(loss, *grads)

# Update params manually with realized values
W1_new = nb.Tensor.from_dlpack((W1.to_numpy() - 0.01 * grads[0].to_numpy()).astype(np.float32))
b1_new = nb.Tensor.from_dlpack((b1.to_numpy() - 0.01 * grads[1].to_numpy()).astype(np.float32))
params = (W1_new, b1_new)

print('\n=== Second call ===')
loss, grads = train_step(params, x, y)
nb.realize_all(loss, *grads)

print('\n=== Third call ===')
W1_new = nb.Tensor.from_dlpack((params[0].to_numpy() - 0.01 * grads[0].to_numpy()).astype(np.float32))
b1_new = nb.Tensor.from_dlpack((params[1].to_numpy() - 0.01 * grads[1].to_numpy()).astype(np.float32))
params = (W1_new, b1_new)
loss, grads = train_step(params, x, y)
nb.realize_all(loss, *grads)
