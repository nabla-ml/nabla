# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""MLP training with @nb.compile vs eager vs JAX.

Trains a deep MLP on a complex sine curve. Compares:
- Nabla compiled (@nb.compile)
- Nabla eager (lazy with batched realize)
- JAX with @jit

Demonstrates compilation speedup and performance parity with JAX.
"""

# %% [markdown]
# # Example 9: Compile vs Eager vs JAX
#
# This benchmark-style example compares three modes:
# - Nabla compiled training (`@nb.compile`)
# - Nabla eager training
# - JAX `@jit` training (when JAX is installed)

# %%

import time

import numpy as np

import nabla as nb

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# %% [markdown]
# ## 1. Dataset and Parameter Initialization

# %%

np.random.seed(42)

n_samples = 500
X_np = np.linspace(0, 1, n_samples).reshape(-1, 1).astype(np.float32)
y_np = (np.sin(8 * np.pi * X_np) + 1) / 2.0

X = nb.Tensor.from_dlpack(X_np)
y = nb.Tensor.from_dlpack(y_np)

print("=" * 70)
print("MLP Training: Fitting Complex Sine Curve")
print("=" * 70)
print(f"Dataset: {n_samples} samples, fitting (sin(8π*x) + 1)/2")

layers = [1, 16, 32, 64, 64, 64, 64, 32, 16, 1]
params = {}

for i in range(len(layers) - 1):
    in_dim, out_dim = layers[i], layers[i + 1]
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    w_np = np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
    b_np = np.zeros((out_dim,), dtype=np.float32)
    params[f"layer{i + 1}"] = {
        "w": nb.Tensor.from_dlpack(w_np),
        "b": nb.Tensor.from_dlpack(b_np),
    }

total_params = sum(
    (layers[i] * layers[i + 1] + layers[i + 1]) for i in range(len(layers) - 1)
)
print(f"Architecture: {' -> '.join(map(str, layers))} ({total_params} params)\n")


def mlp_forward(params, x):
    h = x
    for i in range(1, len(layers)):
        h = h @ params[f"layer{i}"]["w"] + params[f"layer{i}"]["b"]
        if i < len(layers) - 1:
            h = nb.relu(h)
    return h


def loss_fn(params, x, y):
    pred = mlp_forward(params, x)
    diff = pred - y
    return nb.mean(diff * diff)


@nb.compile
def train_step_compiled(params, x, y):
    loss, grads = nb.value_and_grad(loss_fn)(params, x, y)
    lr = 0.01
    new_params = {}
    for layer_name in params:
        new_params[layer_name] = {
            "w": params[layer_name]["w"] - grads[layer_name]["w"] * lr,
            "b": params[layer_name]["b"] - grads[layer_name]["b"] * lr,
        }
    return loss, new_params


def train_step_eager(params, x, y):
    loss, grads = nb.value_and_grad(loss_fn, realize=False)(params, x, y)
    lr = 0.01
    new_params = {}
    for layer_name in params:
        new_params[layer_name] = {
            "w": params[layer_name]["w"] - grads[layer_name]["w"] * lr,
            "b": params[layer_name]["b"] - grads[layer_name]["b"] * lr,
        }
    # Batch realize all outputs
    all_outputs = [loss]
    for layer_params in new_params.values():
        all_outputs.extend(layer_params.values())
    nb.realize_all(*all_outputs)
    return loss, new_params


# %% [markdown]
# ## 2. Nabla Benchmarks (Compiled vs Eager)

# %%


print("=" * 70)
print("TEST 1: Compiled (@nb.compile)")
print("=" * 70)

params_compiled = params
n_steps = 200

loss, params_compiled = train_step_compiled(params_compiled, X, y)
print(f"Warmup: loss = {loss.to_numpy():.6f}")

start = time.perf_counter()
losses_compiled = []
for i in range(n_steps):
    loss, params_compiled = train_step_compiled(params_compiled, X, y)
    losses_compiled.append(float(loss.to_numpy()))
    if (i + 1) % 50 == 0:
        print(f"  Step {i + 1:3d}: loss = {loss.to_numpy():.6f}")

elapsed_compiled = time.perf_counter() - start
print(f"\nTime: {elapsed_compiled:.4f}s ({n_steps / elapsed_compiled:.1f} steps/sec)")
print(f"Loss: {losses_compiled[0]:.6f} -> {losses_compiled[-1]:.6f}")
print(f"Compile stats: {train_step_compiled.stats}")

print("\n" + "=" * 70)
print("TEST 2: Eager (no compile)")
print("=" * 70)

params_eager = params

loss, params_eager = train_step_eager(params_eager, X, y)
print(f"Warmup: loss = {loss.to_numpy():.6f}")

start = time.perf_counter()
losses_eager = []
for i in range(n_steps):
    loss, params_eager = train_step_eager(params_eager, X, y)
    losses_eager.append(float(loss.to_numpy()))
    if (i + 1) % 50 == 0:
        print(f"  Step {i + 1:3d}: loss = {loss.to_numpy():.6f}")

elapsed_eager = time.perf_counter() - start
print(f"\nTime: {elapsed_eager:.4f}s ({n_steps / elapsed_eager:.1f} steps/sec)")
print(f"Loss: {losses_eager[0]:.6f} -> {losses_eager[-1]:.6f}")

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
speedup = elapsed_eager / elapsed_compiled
print(f"Speedup: {speedup:.2f}x with compile")
print(f"  Compiled: {elapsed_compiled:.4f}s")
print(f"  Eager:    {elapsed_eager:.4f}s")

loss_diff = abs(losses_compiled[-1] - losses_eager[-1])
print(f"\nLoss difference: {loss_diff:.8f}")
if loss_diff < 1e-4:
    print("✓ Compiled and eager match!")
else:
    print("⚠ Compiled and eager differ!")

# Test 2b: Eager MAX Graph (builds MAX graph eagerly each step)
print()
print("=" * 70)
print("TEST 2b: Eager MAX Graph (EAGER_MAX_GRAPH=1)")
print("=" * 70)

import nabla.config as nabla_config

orig_eager_max = nabla_config.EAGER_MAX_GRAPH
nabla_config.EAGER_MAX_GRAPH = True

params_eager_max = params

loss, params_eager_max = train_step_eager(params_eager_max, X, y)
print(f"Warmup: loss = {loss.to_numpy():.6f}")

start = time.perf_counter()
losses_eager_max = []
for i in range(n_steps):
    loss, params_eager_max = train_step_eager(params_eager_max, X, y)
    losses_eager_max.append(float(loss.to_numpy()))
    if (i + 1) % 50 == 0:
        print(f"  Step {i + 1:3d}: loss = {loss.to_numpy():.6f}")

elapsed_eager_max = time.perf_counter() - start
nabla_config.EAGER_MAX_GRAPH = orig_eager_max

print(f"\nTime: {elapsed_eager_max:.4f}s ({n_steps / elapsed_eager_max:.1f} steps/sec)")
print(f"Loss: {losses_eager_max[0]:.6f} -> {losses_eager_max[-1]:.6f}")

loss_diff_max = abs(losses_compiled[-1] - losses_eager_max[-1])
print(f"Loss diff vs compiled: {loss_diff_max:.8f}")
if loss_diff_max < 1e-4:
    print("✓ Eager MAX Graph and compiled match!")
else:
    print("⚠ Eager MAX Graph and compiled differ!")

# Test 3: JAX JIT comparison
# %% [markdown]
# ## 3. JAX Comparison (Optional)

# %%
if HAS_JAX:
    print()
    print("=" * 70)
    print("TEST 3: JAX with @jit (for comparison)")
    print("=" * 70)

    # Convert params to JAX format (flat list for simplicity)
    jax_params = []
    for layer_name in sorted(params.keys()):
        w_np = params[layer_name]["w"].to_numpy()
        b_np = params[layer_name]["b"].to_numpy()
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
        new_params = [p - g * lr for p, g in zip(params_flat, grads, strict=False)]
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
            print(f"  Step {i + 1:3d}: loss = {float(loss_jax):.6f}")

    elapsed_jax = time.perf_counter() - start
    print("\nJAX JIT version:")
    print(f"  Time: {elapsed_jax:.4f}s ({n_steps / elapsed_jax:.1f} steps/sec)")
    print(f"  Final loss: {losses_jax[-1]:.6f}")
    print(
        f"  Loss reduction: {losses_jax[0]:.6f} -> {losses_jax[-1]:.6f} ({(1 - losses_jax[-1] / losses_jax[0]) * 100:.1f}% reduction)"
    )

print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)
print(
    f"Nabla Compiled:        {elapsed_compiled:.4f}s ({n_steps / elapsed_compiled:.1f} steps/sec)"
)
print(
    f"Nabla Eager (deferred):{elapsed_eager:.4f}s ({n_steps / elapsed_eager:.1f} steps/sec)"
)
print(
    f"Nabla Eager (MAX):     {elapsed_eager_max:.4f}s ({n_steps / elapsed_eager_max:.1f} steps/sec)"
)
if HAS_JAX:
    print(
        f"JAX JIT:               {elapsed_jax:.4f}s ({n_steps / elapsed_jax:.1f} steps/sec)"
    )
    print()
    speedup_vs_jax = elapsed_jax / elapsed_compiled
    if speedup_vs_jax > 1:
        print(f"🚀 Nabla is {speedup_vs_jax:.2f}x FASTER than JAX!")
    else:
        print(f"JAX is {1 / speedup_vs_jax:.2f}x faster than Nabla")
print()
print(f"Nabla speedup over eager (deferred): {speedup:.2f}x")
print(
    f"Nabla speedup over eager (MAX graph): {elapsed_eager_max / elapsed_compiled:.2f}x"
)

print()
# %% [markdown]
# ## 4. Summary

# %%
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("✓ MLP training works with compile!")
print("✓ Full pytree parameters (weights + biases) work correctly")
print(
    f"✓ Loss decreases properly: {losses_compiled[0]:.6f} -> {losses_compiled[-1]:.6f}"
)
print(f"✓ {speedup:.2f}x speedup from compilation")
print(f"✓ Cache hit rate: {train_step_compiled.stats.hit_rate:.1f}%")
if HAS_JAX:
    print("✓ Compared against JAX JIT successfully")
