# ===----------------------------------------------------------------------=== #
# Nabla Examples - 03b: MLP Training (JAX-Style)
# ===----------------------------------------------------------------------=== #
"""MLP Training with Nabla's Functional (JAX-Style) API.

This example shows how to train the same MLP as 03a, but using a purely
functional style — no classes, just functions and pytree parameter dicts:
- Model as a pure function with explicit parameters
- Parameters stored in nested dicts (pytrees)
- Functional optimizer (adamw_init / adamw_update)
- Training loop with value_and_grad
"""

# %% [markdown]
# # Example 3b: MLP Training (JAX-Style / Functional)
#
# In this style, the model is a **pure function** that takes parameters
# explicitly. Parameters are stored in nested dicts (pytrees). This is
# the same approach used by JAX and Flax.
#
# This example trains the same 2-layer MLP from Example 3a, but purely
# functionally.

# %%
import numpy as np

import nabla as nb
from nabla.nn.functional import xavier_normal

print("Nabla MLP Training — JAX-style (functional)")

# %% [markdown]
# ## 1. Initialize Parameters
#
# Instead of a class, we create a nested dict of parameter tensors.
# Each tensor gets `requires_grad=True` so autodiff can track through it.

# %%
def init_mlp_params(in_dim: int, hidden_dim: int, out_dim: int) -> dict:
    """Initialize MLP parameters as a pytree (nested dict)."""
    params = {
        "fc1": {
            "weight": xavier_normal((in_dim, hidden_dim)),
            "bias": nb.zeros((1, hidden_dim)),
        },
        "fc2": {
            "weight": xavier_normal((hidden_dim, out_dim)),
            "bias": nb.zeros((1, out_dim)),
        },
    }
    # Mark all params as differentiable
    for layer in params.values():
        for p in layer.values():
            p.requires_grad = True
    return params


params = init_mlp_params(4, 32, 1)
print("Parameter shapes:")
for name, layer in params.items():
    for pname, p in layer.items():
        print(f"  {name}.{pname}: {p.shape}")

# %% [markdown]
# ## 2. Define the Forward Pass
#
# The model is a pure function: it takes parameters and input, returns output.
# No side effects, no mutation.

# %%
def mlp_forward(params: dict, x):
    """Pure-function MLP forward pass."""
    x = x @ params["fc1"]["weight"] + params["fc1"]["bias"]
    x = nb.relu(x)
    x = x @ params["fc2"]["weight"] + params["fc2"]["bias"]
    return x


# Quick test
x_test = nb.uniform((3, 4))
y_test = mlp_forward(params, x_test)
print(f"Forward pass test: input {x_test.shape} → output {y_test.shape}")

# %% [markdown]
# ## 3. Create Data & Define Loss
#
# Same synthetic dataset as Example 3a: `y = sin(x0) + cos(x1) + 0.5*x2 - x3`.

# %%
np.random.seed(42)
n_samples = 200
X_np = np.random.randn(n_samples, 4).astype(np.float32)
y_np = (
    np.sin(X_np[:, 0])
    + np.cos(X_np[:, 1])
    + 0.5 * X_np[:, 2]
    - X_np[:, 3]
).reshape(-1, 1).astype(np.float32)

X = nb.Tensor.from_dlpack(X_np)
y = nb.Tensor.from_dlpack(y_np)
print(f"Dataset: X {X.shape}, y {y.shape}")


def loss_fn(params, X, y):
    """MSE loss as a pure function of params."""
    predictions = mlp_forward(params, X)
    diff = predictions - y
    return nb.mean(diff * diff)

initial_loss = loss_fn(params, X, y)
print(f"Initial loss: {initial_loss}")

# %% [markdown]
# ## 4. Training Loop
#
# The key insight: `value_and_grad(loss_fn, argnums=0)` differentiates w.r.t.
# the first argument (`params`), which is a dict. It returns gradients with
# the **same pytree structure** as `params`.

# %%
opt_state = nb.nn.optim.adamw_init(params)
lr = 1e-2
num_epochs = 100

print(f"\n{'Epoch':<8} {'Loss':<12}")
print("-" * 22)

for epoch in range(num_epochs):
    loss, grads = nb.value_and_grad(loss_fn, argnums=0)(params, X, y)

    # grads has the same structure as params:
    # grads["fc1"]["weight"], grads["fc1"]["bias"], etc.
    params, opt_state = nb.nn.optim.adamw_update(
        params, grads, opt_state, lr=lr
    )

    if (epoch + 1) % 10 == 0:
        print(f"{epoch + 1:<8} {loss.item():<12.6f}")

# %% [markdown]
# ## 5. Evaluation

# %%
final_loss = loss_fn(params, X, y)
print(f"\nFinal loss: {final_loss}")

predictions = mlp_forward(params, X)
print(f"\nSample predictions vs targets:")
print(f"{'Prediction':<14} {'Target':<14}")
print("-" * 28)
for i in range(5):
    pred_i = nb.gather(predictions, nb.constant(np.array([i], dtype=np.int64)), axis=0)
    true_i = nb.gather(y, nb.constant(np.array([i], dtype=np.int64)), axis=0)
    print(f"{pred_i.item():<14.4f} {true_i.item():<14.4f}")

# %% [markdown]
# ## 6. Manual SGD (No Optimizer)
#
# The functional style makes it trivial to implement gradient descent manually
# using `tree_map`:

# %%
params_sgd = init_mlp_params(4, 32, 1)
sgd_lr = 0.05

print(f"\nManual SGD training:")
print(f"{'Step':<8} {'Loss':<12}")
print("-" * 22)

for step in range(100):
    loss, grads = nb.value_and_grad(loss_fn, argnums=0)(params_sgd, X, y)

    # Manual SGD: params = params - lr * grads
    params_sgd = nb.tree_map(
        lambda p, g: p - sgd_lr * g, params_sgd, grads
    )

    if (step + 1) % 20 == 0:
        print(f"{step + 1:<8} {loss.item():<12.6f}")

# %% [markdown]
# ## PyTorch-Style vs JAX-Style: Comparison
#
# | Aspect | PyTorch-style (03a) | JAX-style (03b) |
# |--------|-------------------|-----------------|
# | Model | `class MLP(nn.Module)` | `def mlp_forward(params, x)` |
# | Params | Auto-tracked by Module | Explicit dict (pytree) |
# | State | Mutable attributes | Immutable, returned from functions |
# | Optimizer | Can be stateful or functional | Typically functional |
# | `@nb.compile` | Works with both | Works with both |
#
# Both styles are fully supported in Nabla. Choose the one that fits your
# mental model!
#
# **Next:** [04_transforms_and_compile](04_transforms_and_compile)
# — Advanced transforms (vmap, jacrev, jacfwd) and `@nb.compile`.

# %%
print("\n✅ Example 03b completed!")
