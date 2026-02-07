"""Profile nabla eager execution to find Python overhead hotspots."""
import cProfile
import pstats
import io
import nabla as nb
import numpy as np

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
loss, params_eager = train_step_eager(params, X, y)

# Profile 50 eager steps
n_steps = 50
pr = cProfile.Profile()
pr.enable()
for i in range(n_steps):
    loss, params_eager = train_step_eager(params_eager, X, y)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s)
ps.sort_stats('cumulative')
print("=== TOP 50 BY CUMULATIVE TIME ===")
ps.print_stats(50)
print(s.getvalue())

s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2)
ps2.sort_stats('tottime')
print("\n=== TOP 50 BY TOTAL TIME (self time) ===")
ps2.print_stats(50)
print(s2.getvalue())

# Also count function calls
s3 = io.StringIO()
ps3 = pstats.Stats(pr, stream=s3)
ps3.sort_stats('calls')
print("\n=== TOP 30 BY CALL COUNT ===")
ps3.print_stats(30)
print(s3.getvalue())
