"""Profile/benchmark the eager (deferred) execution path."""
import nabla as nb
import numpy as np
import time

np.random.seed(42)
n_samples = 500
X_np = np.linspace(0, 1, n_samples).reshape(-1, 1).astype(np.float32)
y_np = (np.sin(8 * np.pi * X_np) + 1) / 2.0
X = nb.Tensor.from_dlpack(X_np)
y = nb.Tensor.from_dlpack(y_np)

layers = [1, 16, 32, 64, 64, 64, 64, 32, 16, 1]
params = {}
for i in range(len(layers) - 1):
    in_dim, out_dim = layers[i], layers[i + 1]
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    w_np = np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
    b_np = np.zeros((out_dim,), dtype=np.float32)
    params[f'layer{i+1}'] = {
        'w': nb.Tensor.from_dlpack(w_np),
        'b': nb.Tensor.from_dlpack(b_np)
    }

def mlp_forward(params, x):
    h = x
    for i in range(1, len(layers)):
        h = h @ params[f'layer{i}']['w'] + params[f'layer{i}']['b']
        if i < len(layers) - 1:
            h = nb.relu(h)
    return h

def loss_fn(params, x, y):
    pred = mlp_forward(params, x)
    diff = pred - y
    return nb.mean(diff * diff)

def train_step_eager(params, x, y):
    loss, grads = nb.value_and_grad(loss_fn, realize=False)(params, x, y)
    lr = 0.01
    new_params = {}
    for layer_name in params.keys():
        new_params[layer_name] = {
            'w': params[layer_name]['w'] - grads[layer_name]['w'] * lr,
            'b': params[layer_name]['b'] - grads[layer_name]['b'] * lr,
        }
    all_outputs = [loss]
    for layer_params in new_params.values():
        all_outputs.extend(layer_params.values())
    nb.realize_all(*all_outputs)
    return loss, new_params

# Warmup
loss, params = train_step_eager(params, X, y)

# Quick benchmark: 3 runs of 200 steps
import sys
n_steps = 200
for run in range(3):
    p = params
    start = time.perf_counter()
    for _ in range(n_steps):
        loss, p = train_step_eager(p, X, y)
    elapsed = time.perf_counter() - start
    print(f'Run {run+1}: {n_steps/elapsed:.1f} steps/sec ({elapsed:.3f}s)')

# Profile if requested
if '--profile' in sys.argv:
    import cProfile, pstats
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(50):
        loss, params = train_step_eager(params, X, y)
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('tottime')
    print('\n=== TOP 40 BY TOTAL TIME ===')
    stats.print_stats(40)
