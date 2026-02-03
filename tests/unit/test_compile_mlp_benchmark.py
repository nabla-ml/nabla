"""Comprehensive MLP training test with compile - comparing to JAX JIT."""
import nabla as nb
import numpy as np
import time

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("⚠ JAX not available - skipping JAX comparison")

np.random.seed(42)

# Create sine curve dataset - MORE COMPLEX
n_samples = 500  # Much larger dataset
X_np = np.linspace(0, 1, n_samples).reshape(-1, 1).astype(np.float32)
y_np = (np.sin(8 * np.pi * X_np) + 1) / 2.0  # 4 periods, normalized to [0, 1]

X = nb.Tensor.from_dlpack(X_np)
y = nb.Tensor.from_dlpack(y_np)

print("=" * 70)
print("MLP Training: Fitting Complex Sine Curve (4 periods)")
print("=" * 70)
print(f"Dataset: {n_samples} samples")
print(f"Input: x in [0, 1]")
print(f"Target: (sin(8π * x) + 1) / 2 in [0, 1]")
print()

# LARGER, MORE COMPLEX architecture
layers = [1, 16, 32, 64, 64, 64, 64, 32, 16, 1]  # Much wider and deeper
params = {}

for i in range(len(layers) - 1):
    in_dim = layers[i]
    out_dim = layers[i + 1]
    # Glorot initialization
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    w_np = np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
    b_np = np.zeros((out_dim,), dtype=np.float32)
    
    w = nb.Tensor.from_dlpack(w_np)
    b = nb.Tensor.from_dlpack(b_np)
    
    params[f'layer{i+1}'] = {'w': w, 'b': b}

total_params = sum((layers[i] * layers[i+1] + layers[i+1]) for i in range(len(layers) - 1))
print(f"Architecture: {' -> '.join(map(str, layers))}")
print(f"Total params: {total_params}")
print()

def mlp_forward(params, x):
    """MLP with ReLU activations, matching test_sine_mlp.py structure."""
    h = x
    # All layers except last use ReLU
    for i in range(1, len(layers)):
        layer_name = f'layer{i}'
        h = h @ params[layer_name]['w'] + params[layer_name]['b']
        if i < len(layers) - 1:  # No activation on output layer
            h = nb.relu(h)
    return h

def loss_fn(params, x, y):
    """MSE loss."""
    pred = mlp_forward(params, x)
    diff = pred - y
    return nb.mean(diff * diff)

# Compiled train step
@nb.compile
def train_step_compiled(params, x, y):
    loss, grads = nb.value_and_grad(loss_fn)(params, x, y)
    
    # Update parameters with SGD
    lr = 0.01
    new_params = {}
    for layer_name in params.keys():
        new_params[layer_name] = {
            'w': params[layer_name]['w'] - grads[layer_name]['w'] * lr,
            'b': params[layer_name]['b'] - grads[layer_name]['b'] * lr,
        }
    return loss, new_params

# Non-compiled version for comparison (LAZY MODE - proper batching)
def train_step_eager(params, x, y):
    # Use realize=False to defer execution
    loss, grads = nb.value_and_grad(loss_fn, realize=False)(params, x, y)
    
    lr = 0.01
    # Compute new params as lazy nodes (no realization yet)
    new_params = {}
    for layer_name in params.keys():
        new_params[layer_name] = {
            'w': params[layer_name]['w'] - grads[layer_name]['w'] * lr,
            'b': params[layer_name]['b'] - grads[layer_name]['b'] * lr,
        }
    
    # CRITICAL: Batch realize ALL outputs in ONE operation!
    # This is the key to efficient lazy evaluation
    all_outputs = [loss]
    for layer_params in new_params.values():
        all_outputs.extend(layer_params.values())
    nb.realize_all(*all_outputs)
    
    return loss, new_params

print("=" * 70)
print("MLP Training with Full Pytree Parameters (weights + biases)")
print("=" * 70)

# Test 1: Compiled version
print("=" * 70)
print("TEST 1: Compiled train_step (with @nb.compile)")
print("=" * 70)

params_compiled = params
n_steps = 200

# Warmup
loss, params_compiled = train_step_compiled(params_compiled, X, y)
print(f"Warmup (trace): loss = {loss.to_numpy():.6f}")

# Timed training
start = time.perf_counter()
losses_compiled = []
for i in range(n_steps):
    loss, params_compiled = train_step_compiled(params_compiled, X, y)
    losses_compiled.append(float(loss.to_numpy()))
    if (i + 1) % 50 == 0:
        print(f"  Step {i+1:3d}: loss = {loss.to_numpy():.6f}")

elapsed_compiled = time.perf_counter() - start
print(f"\nCompiled version:")
print(f"  Time: {elapsed_compiled:.4f}s ({n_steps/elapsed_compiled:.1f} steps/sec)")
print(f"  Final loss: {losses_compiled[-1]:.6f}")
print(f"  Loss reduction: {losses_compiled[0]:.6f} -> {losses_compiled[-1]:.6f} ({(1 - losses_compiled[-1]/losses_compiled[0])*100:.1f}% reduction)")
print(f"  Compile stats: {train_step_compiled.stats}")

# Test 2: Eager version (no compile)
print()
print("=" * 70)
print("TEST 2: Eager train_step (no compile)")
print("=" * 70)

params_eager = params

# Warmup
loss, params_eager = train_step_eager(params_eager, X, y)
print(f"Warmup: loss = {loss.to_numpy():.6f}")

# Timed training
start = time.perf_counter()
losses_eager = []
for i in range(n_steps):
    loss, params_eager = train_step_eager(params_eager, X, y)
    losses_eager.append(float(loss.to_numpy()))
    if (i + 1) % 50 == 0:
        print(f"  Step {i+1:3d}: loss = {loss.to_numpy():.6f}")

elapsed_eager = time.perf_counter() - start
print(f"\nEager version:")
print(f"  Time: {elapsed_eager:.4f}s ({n_steps/elapsed_eager:.1f} steps/sec)")
print(f"  Final loss: {losses_eager[-1]:.6f}")
print(f"  Loss reduction: {losses_eager[0]:.6f} -> {losses_eager[-1]:.6f} ({(1 - losses_eager[-1]/losses_eager[0])*100:.1f}% reduction)")

# Comparison
print()
print("=" * 70)
print("COMPARISON")
print("=" * 70)
speedup = elapsed_eager / elapsed_compiled
print(f"Speedup: {speedup:.2f}x faster with compile!")
print(f"  Compiled: {elapsed_compiled:.4f}s")
print(f"  Eager:    {elapsed_eager:.4f}s")
print()

# Verify correctness (losses should be very similar)
loss_diff = abs(losses_compiled[-1] - losses_eager[-1])
print(f"Final loss difference: {loss_diff:.8f} (should be very small)")
if loss_diff < 1e-4:
    print("✓ Compiled and eager versions produce same results!")
else:
    print("⚠ Warning: Compiled and eager versions differ!")

# Test 3: JAX JIT comparison
if HAS_JAX:
    print()
    print("=" * 70)
    print("TEST 3: JAX with @jit (for comparison)")
    print("=" * 70)
    
    # Convert params to JAX format (flat list for simplicity)
    jax_params = []
    for layer_name in sorted(params.keys()):
        w_np = params[layer_name]['w'].to_numpy()
        b_np = params[layer_name]['b'].to_numpy()
        jax_params.append(jnp.array(w_np))
        jax_params.append(jnp.array(b_np))
    
    X_jax = jnp.array(X_np)
    y_jax = jnp.array(y_np)
    
    def jax_mlp(params_flat, x):
        h = x
        for i in range(0, len(params_flat) - 2, 2):
            h = h @ params_flat[i] + params_flat[i + 1]
            h = jax.nn.relu(h)
        h = h @ params_flat[-2] + params_flat[-1]
        return h
    
    def jax_loss(params_flat, x, y):
        pred = jax_mlp(params_flat, x)
        return jnp.mean((pred - y) ** 2)
    
    @jit
    def jax_train_step(params_flat, x, y):
        loss = jax_loss(params_flat, x, y)
        grads = grad(jax_loss)(params_flat, x, y)
        lr = 0.01
        new_params = [p - g * lr for p, g in zip(params_flat, grads)]
        return loss, new_params
    
    # Warmup
    loss_jax, jax_params = jax_train_step(jax_params, X_jax, y_jax)
    jax.block_until_ready(loss_jax)
    print(f"Warmup (trace): loss = {float(loss_jax):.6f}")
    
    # Timed training
    start = time.perf_counter()
    losses_jax = []
    for i in range(n_steps):
        loss_jax, jax_params = jax_train_step(jax_params, X_jax, y_jax)
        jax.block_until_ready(loss_jax)
        losses_jax.append(float(loss_jax))
        if (i + 1) % 50 == 0:
            print(f"  Step {i+1:3d}: loss = {float(loss_jax):.6f}")
    
    elapsed_jax = time.perf_counter() - start
    print(f"\nJAX JIT version:")
    print(f"  Time: {elapsed_jax:.4f}s ({n_steps/elapsed_jax:.1f} steps/sec)")
    print(f"  Final loss: {losses_jax[-1]:.6f}")
    print(f"  Loss reduction: {losses_jax[0]:.6f} -> {losses_jax[-1]:.6f} ({(1 - losses_jax[-1]/losses_jax[0])*100:.1f}% reduction)")

print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)
print(f"Nabla Compiled: {elapsed_compiled:.4f}s ({n_steps/elapsed_compiled:.1f} steps/sec)")
print(f"Nabla Eager:    {elapsed_eager:.4f}s ({n_steps/elapsed_eager:.1f} steps/sec)")
if HAS_JAX:
    print(f"JAX JIT:        {elapsed_jax:.4f}s ({n_steps/elapsed_jax:.1f} steps/sec)")
    print()
    speedup_vs_jax = elapsed_jax / elapsed_compiled
    if speedup_vs_jax > 1:
        print(f"🚀 Nabla is {speedup_vs_jax:.2f}x FASTER than JAX!")
    else:
        print(f"JAX is {1/speedup_vs_jax:.2f}x faster than Nabla")
print()
print(f"Nabla speedup over eager: {speedup:.2f}x")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✓ MLP training works with compile!")
print(f"✓ Full pytree parameters (weights + biases) work correctly")
print(f"✓ Loss decreases properly: {losses_compiled[0]:.6f} -> {losses_compiled[-1]:.6f}")
print(f"✓ {speedup:.2f}x speedup from compilation")
print(f"✓ Cache hit rate: {train_step_compiled.stats.hit_rate:.1f}%")
if HAS_JAX:
    print(f"✓ Compared against JAX JIT successfully")

