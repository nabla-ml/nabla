import nabla as nb

# Defines MLP forward pass and loss.
def loss_fn(params, x, y):
    for i in range(0, len(params) - 2, 2):
        x = nb.relu(x @ params[i] + params[i + 1])
    predictions = x @ params[-2] + params[-1]
    return nb.mean((predictions - y) ** 2)

# JIT-compiled training step via SGD
@nb.jit(auto_device=True)
def train_step(params, x, y, lr):
    loss, grads = nb.value_and_grad(loss_fn)(params, x, y)
    return loss, [p - g * lr for p, g in zip(params, grads)]

# Setup network (hyper)parameters.
LAYERS = [1, 32, 64, 32, 1]
params = [p for i in range(len(LAYERS) - 1) for p in (nb.glorot_uniform((LAYERS[i], LAYERS[i + 1])), nb.zeros((1, LAYERS[i + 1])),)]

# Run training loop.
x, y = nb.rand((256, 1)), nb.rand((256, 1))
for i in range(1001):
    loss, params = train_step(params, x, y, 0.01)
    if i % 100 == 0: print(i, loss.to_numpy())