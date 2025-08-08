
import nabla as nb
import numpy as np

LAYERS = [1, 1024, 2048, 4096, 2048, 1024, 1]

def loss_fn(params, x, y):
    for i in range(0, len(params) - 2, 2):
        x = nb.relu(nb.matmul(x, params[i]) + params[i + 1])
    predictions = nb.matmul(x, params[-2]) + params[-1]
    return nb.mean((predictions - y) ** 2)

@nb.jit(auto_device=True)
def train_step(params, x, y, lr):
    loss, grads = nb.value_and_grad(loss_fn)(params, x, y)
    return loss, [p - g * lr for p, g in zip(params, grads)]

params = [p for i in range(len(LAYERS) - 1) for p in (
        nb.glorot_uniform((LAYERS[i], LAYERS[i + 1])),
        nb.zeros((1, LAYERS[i + 1])),)]

for i in range(1001):
    # Generate data on the fly
    x_np = np.random.rand(256, 1).astype(np.float32)
    y_np = np.sin(2 * np.pi * x_np).astype(np.float32)
    x, y = nb.Array.from_numpy(x_np), nb.Array.from_numpy(y_np)

    loss, params = train_step(params, x, y, 0.01)

    if i % 100 == 0: print(f"Iter {i:4d} | Loss: {loss.to_numpy():.4f}")
