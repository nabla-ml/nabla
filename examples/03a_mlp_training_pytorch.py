# ===----------------------------------------------------------------------=== #
# Nabla Examples - 03a: MLP Training (PyTorch-Style)
# ===----------------------------------------------------------------------=== #
"""MLP Training with Nabla's PyTorch-Style API.

This example shows how to train a multi-layer perceptron using Nabla's
nn.Module system — familiar to anyone who has used PyTorch:
- Defining models with nb.nn.Module
- Using nb.nn.Linear layers
- Training with value_and_grad + AdamW optimizer
- Evaluating the trained model
"""

# %% [markdown]
# # Example 3a: MLP Training (PyTorch-Style)
#
# Nabla provides a **PyTorch-style** `nn.Module` API for building and training
# neural networks. Models are defined as classes with `forward()` methods,
# parameters are automatically tracked, and training uses functional transforms
# (`value_and_grad`) combined with the AdamW optimizer.
#
# This example trains a 2-layer MLP on a synthetic regression task.

# %%
import numpy as np

import nabla as nb

print("Nabla MLP Training — PyTorch-style")

# %% [markdown]
# ## 1. Define the Model
#
# Subclass `nb.nn.Module` and define layers in `__init__`. The `forward()`
# method specifies the computation. Parameters (from `nb.nn.Linear`, etc.)
# are automatically registered and tracked.

# %%
class MLP(nb.nn.Module):
    """Two-layer MLP with ReLU activation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nb.nn.Linear(in_dim, hidden_dim)
        self.fc2 = nb.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = nb.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = MLP(4, 32, 1)
print(f"Model architecture:")
print(f"  fc1: Linear({model.fc1.weight.shape})")
print(f"  fc2: Linear({model.fc2.weight.shape})")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")

# %% [markdown]
# ## 2. Create Synthetic Data
#
# We'll create a regression dataset: predict `y = sin(x0) + cos(x1) + 0.5*x2 - x3`.

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

# %% [markdown]
# ## 3. Define the Loss Function
#
# The loss function takes the model as the first argument (so we can use
# `argnums=0` to differentiate w.r.t. model parameters).

# %%
def loss_fn(model, X, y):
    """Mean squared error loss."""
    predictions = model(X)
    return nb.nn.functional.mse_loss(predictions, y)

# Test it
initial_loss = loss_fn(model, X, y)
print(f"Initial loss: {initial_loss}")

# %% [markdown]
# ## 4. Initialize the Optimizer
#
# Nabla provides a **functional optimizer API** that works seamlessly with
# `value_and_grad`. The optimizer state is a pytree (dict of tensors), and
# updates return new model + new state — no mutation.

# %%
opt_state = nb.nn.optim.adamw_init(model)
print(f"Optimizer state keys: {list(opt_state.keys())}")

# %% [markdown]
# ## 5. Training Loop
#
# Each step:
# 1. `value_and_grad` computes the loss and gradients w.r.t. the model
# 2. `adamw_update` returns a new model and optimizer state with parameters updated

# %%
learning_rate = 1e-2
num_epochs = 100

print(f"\n{'Epoch':<8} {'Loss':<12}")
print("-" * 22)

for epoch in range(num_epochs):
    # Compute loss and gradients
    loss, grads = nb.value_and_grad(loss_fn, argnums=0)(model, X, y)

    # Update model parameters
    model, opt_state = nb.nn.optim.adamw_update(
        model, grads, opt_state, lr=learning_rate
    )

    if (epoch + 1) % 10 == 0:
        print(f"{epoch + 1:<8} {loss.item():<12.6f}")

# %% [markdown]
# ## 6. Evaluation
#
# Let's see how well the model fits the data.

# %%
final_loss = loss_fn(model, X, y)
print(f"\nFinal loss: {final_loss}")

# Compare predictions vs targets on a few samples
predictions = model(X)
print(f"\nSample predictions vs targets:")
print(f"{'Prediction':<14} {'Target':<14}")
print("-" * 28)
for i in range(5):
    pred_i = nb.gather(predictions, nb.constant(np.array([i], dtype=np.int64)), axis=0)
    true_i = nb.gather(y, nb.constant(np.array([i], dtype=np.int64)), axis=0)
    print(f"{pred_i.item():<14.4f} {true_i.item():<14.4f}")

# %% [markdown]
# ## 7. Using the Stateful Optimizer (Alternative)
#
# Nabla also supports a stateful optimizer API closer to PyTorch's style.

# %%
# Reset model
model2 = MLP(4, 32, 1)
optimizer = nb.nn.optim.AdamW(model2, lr=1e-2)

print(f"\nStateful AdamW training:")
print(f"{'Step':<8} {'Loss':<12}")
print("-" * 22)

for step in range(50):
    loss, grads = nb.value_and_grad(loss_fn, argnums=0)(model2, X, y)

    # Stateful update — mutates the optimizer's internal state
    model2 = optimizer.step(grads)

    if (step + 1) % 10 == 0:
        print(f"{step + 1:<8} {loss.item():<12.6f}")

# %% [markdown]
# ## Summary
#
# | Concept | API |
# |---------|-----|
# | Define models | `class MyModel(nb.nn.Module)` |
# | Linear layer | `nb.nn.Linear(in_dim, out_dim)` |
# | Loss functions | `nb.nn.functional.mse_loss`, `cross_entropy_loss` |
# | Gradients | `nb.value_and_grad(loss_fn, argnums=0)(model, ...)` |
# | Optimizer init | `opt_state = nb.nn.optim.adamw_init(model)` |
# | Optimizer step | `model, opt_state = nb.nn.optim.adamw_update(...)` |
#
# **Next:** [03b_mlp_training_jax](03b_mlp_training_jax) — the same
# MLP trained in a purely functional (JAX-like) style.

# %%
print("\n✅ Example 03a completed!")
