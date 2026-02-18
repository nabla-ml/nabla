# ===----------------------------------------------------------------------=== #
# Nabla Tutorials - 04: Transforms and Compile
# ===----------------------------------------------------------------------=== #
"""Advanced Transforms and @nb.compile.

This tutorial covers Nabla's powerful transform system:
- vmap — automatic vectorization / batching
- jacrev and jacfwd — full Jacobian computation
- Composing transforms (vmap of grad, Hessians)
- @nb.compile — graph compilation for speed
- Compiled training loops with nn.Module + AdamW
"""

# %% [markdown]
# # Tutorial 4: Transforms and `@nb.compile`
#
# Nabla's transforms are **higher-order functions** that take a function and
# return a new function with modified behavior. They are fully composable
# and work with any Nabla operation, including nn.Modules.
#
# | Transform | What it does |
# |-----------|-------------|
# | `vmap` | Auto-vectorize over a batch dimension |
# | `grad` | Compute gradients (reverse-mode) |
# | `jacrev` | Full Jacobian via reverse-mode |
# | `jacfwd` | Full Jacobian via forward-mode |
# | `compile` | Compile computation graph to MAX graph |

# %%
import numpy as np

import nabla as nb

print("Nabla Transforms & Compile Tutorial")

# %% [markdown]
# ## 1. `vmap` — Automatic Vectorization
#
# `vmap` transforms a function that operates on a single example into one
# that operates on a batch — without writing any batching logic yourself.

# %%
def single_dot(x, y):
    """Dot product of two vectors (no batch dimension)."""
    return nb.reduce_sum(x * y)

# Without vmap: manual loop
x_batch = nb.uniform((5, 3))
y_batch = nb.uniform((5, 3))

# With vmap: automatic vectorization!
batched_dot = nb.vmap(single_dot, in_axes=(0, 0))
result = batched_dot(x_batch, y_batch)
print(f"Batched dot products (5 pairs of 3D vectors):")
print(result)
print(f"Shape: {result.shape}")

# %% [markdown]
# ### `in_axes` and `out_axes`
#
# `in_axes` controls which axis of each argument is the batch axis.
# `out_axes` controls where to place the batch axis in the output.
# Use `None` for arguments that should be broadcast (not batched).

# %%
def weighted_sum(x, w):
    """Weighted sum: w * x, summed."""
    return nb.reduce_sum(w * x)

# x is batched (axis 0), w is shared across the batch
batch_fn = nb.vmap(weighted_sum, in_axes=(0, None))

x_batch = nb.uniform((4, 3))
w = nb.Tensor.from_dlpack(np.array([1.0, 2.0, 3.0], dtype=np.float32))

result = batch_fn(x_batch, w)
print(f"Batched weighted sum (shared weights):")
print(result)
print(f"Shape: {result.shape}")

# %% [markdown]
# ## 2. `vmap` of `grad` — Per-Example Gradients
#
# Composing `vmap` with `grad` gives per-example gradients — something that's
# difficult to do efficiently in most frameworks.

# %%
def per_sample_loss(x, w):
    """Loss for a single sample: (w @ x)^2."""
    return nb.reduce_sum(w * x) ** 2

# grad of the loss w.r.t. w for a single sample
grad_single = nb.grad(per_sample_loss, argnums=1)

# vmap over samples — per-example gradients!
per_example_grad = nb.vmap(grad_single, in_axes=(0, None))

x_batch = nb.Tensor.from_dlpack(
    np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
)
w = nb.Tensor.from_dlpack(np.array([2.0, 3.0], dtype=np.float32))

grads = per_example_grad(x_batch, w)
print("Per-example gradients (3 samples, 2 weights):")
print(grads)
print(f"Shape: {grads.shape}")

# %% [markdown]
# ## 3. `jacrev` and `jacfwd` — Full Jacobians
#
# Recall from Tutorial 2: `jacrev` and `jacfwd` compute full Jacobian matrices.
# Here we show them applied to a more interesting function.

# %%
def neural_layer(x):
    """A simple neural network layer: tanh(Wx + b)."""
    W = nb.Tensor.from_dlpack(
        np.array([[1.0, -0.5], [0.3, 0.8], [-0.2, 0.6]], dtype=np.float32)
    )
    b = nb.Tensor.from_dlpack(np.array([0.1, -0.1, 0.2], dtype=np.float32))
    return nb.tanh(x @ W + b)

x = nb.Tensor.from_dlpack(np.array([1.0, 0.5], dtype=np.float32))

J_rev = nb.jacrev(neural_layer)(x)
J_fwd = nb.jacfwd(neural_layer)(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {neural_layer(x).shape}")
print(f"\nJacobian via jacrev (shape {J_rev.shape}):")
print(J_rev)
print(f"\nJacobian via jacfwd (shape {J_fwd.shape}):")
print(J_fwd)

# %% [markdown]
# ## 4. Composing Jacobians — Hessians
#
# Since transforms compose, we can compute Hessians by nesting:

# %%
def energy(x):
    """Energy function: E(x) = 0.5 * x^T A x where A = [[2, 1], [1, 3]]."""
    A = nb.Tensor.from_dlpack(
        np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    )
    return 0.5 * nb.reduce_sum(x * (A @ x))

x = nb.Tensor.from_dlpack(np.array([1.0, 2.0], dtype=np.float32))
print(f"E(x) = 0.5 * x^T @ A @ x, where A = [[2,1],[1,3]]")
print(f"E([1,2]) = {energy(x)}")
print(f"Gradient: {nb.grad(energy)(x)}")
print(f"  (should be Ax = [4, 7])")

H = nb.jacfwd(nb.grad(energy))(x)
print(f"\nHessian (should be A = [[2,1],[1,3]]):")
print(H)

# %% [markdown]
# ## 5. `@nb.compile` — Graph Compilation
#
# `@nb.compile` traces a function, captures its computation graph, and
# compiles it into an optimized MAX graph. Subsequent calls with the same
# tensor shapes/dtypes hit a cache — dramatically speeding up execution.

# %%
import time

def slow_fn(x, y):
    """A function with many operations."""
    for _ in range(5):
        x = nb.relu(x @ y + x)
    return nb.reduce_sum(x)

@nb.compile
def fast_fn(x, y):
    """Same function, but compiled."""
    for _ in range(5):
        x = nb.relu(x @ y + x)
    return nb.reduce_sum(x)

x = nb.uniform((32, 32))
y = nb.uniform((32, 32))

# Warmup compiled version (first call traces and compiles)
_ = fast_fn(x, y)

# Benchmark eager
start = time.perf_counter()
for _ in range(20):
    _ = slow_fn(x, y)
eager_time = time.perf_counter() - start

# Benchmark compiled
start = time.perf_counter()
for _ in range(20):
    _ = fast_fn(x, y)
compiled_time = time.perf_counter() - start

print(f"Eager:    {eager_time:.4f}s")
print(f"Compiled: {compiled_time:.4f}s")
print(f"Speedup:  {eager_time / max(compiled_time, 1e-9):.1f}x")

# %% [markdown]
# ## 6. Compiled Training Loop
#
# The real power of `@nb.compile` is compiling entire training steps.
# When used with `value_and_grad` and `adamw_update`, the forward pass,
# backward pass, and optimizer step are all fused into a single compiled graph.

# %%
class TinyMLP(nb.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nb.nn.Linear(4, 16)
        self.fc2 = nb.nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(nb.relu(self.fc1(x)))


def my_loss_fn(model, x, y):
    return nb.nn.functional.mse_loss(model(x), y)


@nb.compile
def train_step(model, opt_state, x, y):
    """Compiled training step: forward + backward + optimizer update."""
    loss, grads = nb.value_and_grad(my_loss_fn, argnums=0)(model, x, y)
    model, opt_state = nb.nn.optim.adamw_update(
        model, grads, opt_state, lr=1e-2
    )
    return model, opt_state, loss


# Setup
np.random.seed(0)
X = nb.Tensor.from_dlpack(np.random.randn(100, 4).astype(np.float32))
y = nb.Tensor.from_dlpack(np.random.randn(100, 1).astype(np.float32))

model = TinyMLP()
opt_state = nb.nn.optim.adamw_init(model)

print(f"\nCompiled training loop:")
print(f"{'Step':<8} {'Loss':<12}")
print("-" * 22)

for step in range(50):
    model, opt_state, loss = train_step(model, opt_state, X, y)

    if (step + 1) % 10 == 0:
        print(f"{step + 1:<8} {loss.item():<12.6f}")

# %% [markdown]
# ## 7. Compiled Training with JAX-Style Params
#
# `@nb.compile` works equally well with dict-based parameters.

# %%
from nabla.nn.functional import xavier_normal


def init_params():
    params = {
        "w1": xavier_normal((4, 16)),
        "b1": nb.zeros((1, 16)),
        "w2": xavier_normal((16, 1)),
        "b2": nb.zeros((1, 1)),
    }
    for p in params.values():
        p.requires_grad = True
    return params


def forward(params, x):
    h = nb.relu(x @ params["w1"] + params["b1"])
    return h @ params["w2"] + params["b2"]


def jax_loss_fn(params, x, y):
    pred = forward(params, x)
    diff = pred - y
    return nb.mean(diff * diff)


@nb.compile
def jax_train_step(params, opt_state, x, y):
    loss, grads = nb.value_and_grad(jax_loss_fn, argnums=0)(params, x, y)
    params, opt_state = nb.nn.optim.adamw_update(
        params, grads, opt_state, lr=1e-2
    )
    return params, opt_state, loss


params = init_params()
opt_state = nb.nn.optim.adamw_init(params)

print(f"\nCompiled JAX-style training:")
print(f"{'Step':<8} {'Loss':<12}")
print("-" * 22)

for step in range(50):
    params, opt_state, loss = jax_train_step(params, opt_state, X, y)

    if (step + 1) % 10 == 0:
        print(f"{step + 1:<8} {loss.item():<12.6f}")

# %% [markdown]
# ## Summary
#
# | Transform | Usage | Key benefit |
# |-----------|-------|------------|
# | `vmap(f)` | Auto-batch any function | No manual batching |
# | `vmap(grad(f))` | Per-example gradients | Efficient |
# | `jacrev(f)` / `jacfwd(f)` | Full Jacobians | Compose for Hessians |
# | `@nb.compile` | Compile train step | 5–50x speedup |
#
# All transforms compose freely with each other:
# `compile(vmap(grad(f)))`, `jacfwd(jacrev(f))`, etc.
#
# **Next:** [05a_transformer_pytorch.py](05a_transformer_pytorch.py)
# — Building and training a Transformer.

# %%
print("\n✅ Tutorial 04 completed!")
