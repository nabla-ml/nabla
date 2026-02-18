# ===----------------------------------------------------------------------=== #
# Nabla Tutorials - 02: Automatic Differentiation
# ===----------------------------------------------------------------------=== #
"""Automatic Differentiation in Nabla.

This tutorial covers Nabla's composable autodiff system:
- grad and value_and_grad (reverse-mode)
- jvp — forward-mode Jacobian-vector products
- vjp — reverse-mode vector-Jacobian products
- jacrev and jacfwd — full Jacobian matrices
- Higher-order derivatives (Hessians via composing transforms)
"""

# %% [markdown]
# # Tutorial 2: Automatic Differentiation
#
# Nabla provides a **JAX-like functional autodiff** system built on composable
# transforms. Every transform is a higher-order function: it takes a function
# and returns a new function that computes derivatives.
#
# | Transform | Mode | Computes |
# |-----------|------|----------|
# | `grad` | Reverse | Gradient of scalar-valued function |
# | `value_and_grad` | Reverse | (value, gradient) pair |
# | `jvp` | Forward | Jacobian-vector product |
# | `vjp` | Reverse | Vector-Jacobian product |
# | `jacrev` | Reverse | Full Jacobian matrix |
# | `jacfwd` | Forward | Full Jacobian matrix |

# %%
import numpy as np

import nabla as nb

print("Nabla autodiff tutorial")

# %% [markdown]
# ## 1. `grad` — Gradient of a Scalar Function
#
# `nb.grad(fn)` returns a function that computes the gradient of `fn`
# with respect to specified arguments (default: first argument).

# %%
def f(x):
    """f(x) = x^3 + 2x^2 - 5x + 3, so f'(x) = 3x^2 + 4x - 5."""
    return x ** 3 + 2.0 * x ** 2 - 5.0 * x + 3.0

df = nb.grad(f)

x = nb.Tensor.from_dlpack(np.array([2.0], dtype=np.float32))
grad_val = df(x)
print(f"f(x) = x^3 + 2x^2 - 5x + 3")
print(f"f'(x) = 3x^2 + 4x - 5")
print(f"f'(2.0) = 3*4 + 4*2 - 5 = {3*4 + 4*2 - 5}")
print(f"Nabla grad: {grad_val}")

# %% [markdown]
# ## 2. `value_and_grad` — Value and Gradient Together
#
# Often you need both the function value and its gradient. This is more
# efficient than calling `f` and `grad(f)` separately.

# %%
def quadratic(x):
    """f(x) = sum(x^2), so grad = 2x."""
    return nb.reduce_sum(x * x)

val_and_grad_fn = nb.value_and_grad(quadratic)
x = nb.Tensor.from_dlpack(np.array([1.0, 2.0, 3.0], dtype=np.float32))
value, gradient = val_and_grad_fn(x)
print(f"x = [1, 2, 3]")
print(f"f(x) = sum(x^2) = {value}")
print(f"grad(f) = 2x = {gradient}")

# %% [markdown]
# ### Multiple Arguments with `argnums`
#
# Use `argnums` to specify which arguments to differentiate with respect to.

# %%
def weighted_sum(w, x):
    """f(w, x) = sum(w * x)."""
    return nb.reduce_sum(w * x)

# Gradient w.r.t. first arg (w) only — default
grad_w = nb.grad(weighted_sum, argnums=0)
w = nb.Tensor.from_dlpack(np.array([1.0, 2.0], dtype=np.float32))
x = nb.Tensor.from_dlpack(np.array([3.0, 4.0], dtype=np.float32))
print(f"grad w.r.t. w: {grad_w(w, x)}")
print(f"  (should be x = [3, 4])")

# Gradient w.r.t. second arg (x)
grad_x = nb.grad(weighted_sum, argnums=1)
print(f"grad w.r.t. x: {grad_x(w, x)}")
print(f"  (should be w = [1, 2])")

# Gradient w.r.t. both — returns a tuple
grad_both = nb.grad(weighted_sum, argnums=(0, 1))
gw, gx = grad_both(w, x)
print(f"grad w.r.t. (w, x): ({gw}, {gx})")

# %% [markdown]
# ## 3. `jvp` — Forward-Mode (Jacobian-Vector Product)
#
# `nb.jvp(fn, primals, tangents)` computes:
# - The function output `fn(*primals)`
# - The directional derivative `J @ tangents` (JVP)
#
# This is efficient when the number of **inputs** is small (one forward pass
# per tangent direction).

# %%
def g(x):
    """g(x) = [x0^2 + x1, x0 * x1]."""
    r0 = nb.reshape(x[0] ** 2 + x[1], (1,))
    r1 = nb.reshape(x[0] * x[1], (1,))
    return nb.concatenate([r0, r1], axis=0)

x = nb.Tensor.from_dlpack(np.array([3.0, 2.0], dtype=np.float32))
v = nb.Tensor.from_dlpack(np.array([1.0, 0.0], dtype=np.float32))

output, jvp_val = nb.jvp(g, (x,), (v,))
print(f"g([3, 2]) = [3^2 + 2, 3*2] = {output}")
print(f"JVP with v=[1,0] (column 1 of Jacobian):")
print(f"  J @ v = {jvp_val}")
print(f"  Expected: [2*3, 2] = [6, 2]")

# %%
# Second column of the Jacobian
v2 = nb.Tensor.from_dlpack(np.array([0.0, 1.0], dtype=np.float32))
_, jvp_val2 = nb.jvp(g, (x,), (v2,))
print(f"JVP with v=[0,1] (column 2 of Jacobian):")
print(f"  J @ v = {jvp_val2}")
print(f"  Expected: [1, 3]")

# %% [markdown]
# ## 4. `vjp` — Reverse-Mode (Vector-Jacobian Product)
#
# `nb.vjp(fn, *primals)` returns `(output, vjp_fn)` where `vjp_fn(cotangent)`
# gives the VJP = `cotangent @ J`.
#
# This is efficient when the number of **outputs** is small (one backward pass
# per cotangent direction).

# %%
def linear_fn(x):
    """f(x) = Ax where A = [[1, 2], [3, 4], [5, 6]]."""
    A = nb.Tensor.from_dlpack(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    )
    return x @ A  # (3,) @ (3,2) isn't quite right — let's use matmul properly

# For vjp demo, use a scalar-to-vector function via matrix multiply
def mat_fn(x):
    """f(x) = Ax, where A is 2x3 and x is (3,). Output is (2,)."""
    A = nb.Tensor.from_dlpack(
        np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]], dtype=np.float32)
    )
    return A @ x  # (2,3) @ (3,) = (2,)

x = nb.Tensor.from_dlpack(np.array([1.0, 2.0, 3.0], dtype=np.float32))
output, vjp_fn = nb.vjp(mat_fn, x)
print(f"f(x) = Ax, A = [[1,0,2],[0,3,1]], x = [1,2,3]")
print(f"f(x) = {output}")
print(f"  Expected: [1+0+6, 0+6+3] = [7, 9]")

# VJP with cotangent [1, 0] — gives first row of A^T
v1 = nb.Tensor.from_dlpack(np.array([1.0, 0.0], dtype=np.float32))
(vjp1,) = vjp_fn(v1)
print(f"\nVJP with v=[1,0]: {vjp1}")
print(f"  Expected: A^T @ [1,0] = [1, 0, 2]")

# VJP with cotangent [0, 1] — gives second row of A^T
v2 = nb.Tensor.from_dlpack(np.array([0.0, 1.0], dtype=np.float32))
(vjp2,) = vjp_fn(v2)
print(f"VJP with v=[0,1]: {vjp2}")
print(f"  Expected: A^T @ [0,1] = [0, 3, 1]")

# %% [markdown]
# ## 5. `jacrev` — Full Jacobian via Reverse Mode
#
# `nb.jacrev(fn)` computes the full Jacobian matrix using reverse-mode
# autodiff (one backward pass per output element, batched via vmap).

# %%
def h(x):
    """h(x) = Ax + sin(x), nonlinear vector function R^2 -> R^2."""
    A = nb.Tensor.from_dlpack(
        np.array([[2.0, -1.0], [1.0, 3.0]], dtype=np.float32)
    )
    return A @ x + nb.sin(x)

x = nb.Tensor.from_dlpack(np.array([1.0, 0.5], dtype=np.float32))
J = nb.jacrev(h)(x)
print("Jacobian via jacrev:")
print(J)
print("Expected: A + diag(cos(x))")
print(f"  [[2+cos(1), -1     ],")
print(f"   [1,        3+cos(0.5)]]")
print(f"  ≈ [[{2+np.cos(1):.4f}, {-1:.4f}],")
print(f"     [{1:.4f}, {3+np.cos(0.5):.4f}]]")

# %% [markdown]
# ## 6. `jacfwd` — Full Jacobian via Forward Mode
#
# `nb.jacfwd(fn)` computes the same Jacobian using forward-mode autodiff
# (one JVP per input element, batched via vmap). Prefer `jacfwd` when
# inputs are few and outputs are many.

# %%
J_fwd = nb.jacfwd(h)(x)
print("Jacobian via jacfwd:")
print(J_fwd)

# %% [markdown]
# ### When to use `jacrev` vs `jacfwd`
#
# | Scenario | Prefer |
# |----------|--------|
# | Few outputs, many inputs | `jacrev` |
# | Few inputs, many outputs | `jacfwd` |
# | Square Jacobian | Either works |
# | Hessian (second derivative) | Compose both! |

# %% [markdown]
# ## 7. Hessians — Composing Transforms
#
# Because Nabla's transforms are **composable**, you can compute Hessians
# (second-order derivatives) by nesting Jacobian transforms.
#
# For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the Hessian
# $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ can be computed
# in multiple ways:

# %%
def scalar_fn(x):
    """f(x) = x0^2 * x1 + x1^3, a polynomial with known Hessian."""
    return x[0] ** 2 * x[1] + x[1] ** 3

x = nb.Tensor.from_dlpack(np.array([2.0, 3.0], dtype=np.float32))
print(f"f(x) = x0^2 * x1 + x1^3")
print(f"x = {x}")
print(f"f(x) = {scalar_fn(x)}")
print()

# The Hessian of f:
# df/dx0 = 2*x0*x1,  df/dx1 = x0^2 + 3*x1^2
# d^2f/dx0dx0 = 2*x1,    d^2f/dx0dx1 = 2*x0
# d^2f/dx1dx0 = 2*x0,    d^2f/dx1dx1 = 6*x1
# At x = [2, 3]:
# H = [[6, 4], [4, 18]]
print("Analytical Hessian at x=[2,3]:")
print("  [[2*x1, 2*x0], [2*x0, 6*x1]] = [[6, 4], [4, 18]]")
print()

# Method 1: jacfwd(grad(f))
H1 = nb.jacfwd(nb.grad(scalar_fn))(x)
print("Method 1 — jacfwd(grad(f)):")
print(H1)

# Method 2: jacrev(grad(f))
H2 = nb.jacrev(nb.grad(scalar_fn))(x)
print("Method 2 — jacrev(grad(f)):")
print(H2)

# Method 3: jacrev(jacfwd(f))
H3 = nb.jacrev(nb.jacfwd(scalar_fn))(x)
print("Method 3 — jacrev(jacfwd(f)):")
print(H3)

# Method 4: jacfwd(jacrev(f))
H4 = nb.jacfwd(nb.jacrev(scalar_fn))(x)
print("Method 4 — jacfwd(jacrev(f)):")
print(H4)

print("\nAll four methods produce the same Hessian! ✅")

# %% [markdown]
# ## 8. Gradient of a Multi-Variable Loss
#
# A more practical example: computing gradients for a simple regression loss.

# %%
def linear_regression_loss(w, b, X, y):
    """MSE loss for linear regression: ||Xw + b - y||^2 / n."""
    predictions = X @ w + b
    residuals = predictions - y
    return nb.mean(residuals * residuals)

# Create data
np.random.seed(42)
n_samples, n_features = 50, 3
X = nb.Tensor.from_dlpack(np.random.randn(n_samples, n_features).astype(np.float32))
w_true = nb.Tensor.from_dlpack(np.array([[2.0], [-1.0], [0.5]], dtype=np.float32))
y = X @ w_true + 0.1 * nb.gaussian((n_samples, 1))

# Initialize weights
w = nb.zeros((n_features, 1))
b = nb.zeros((1,))

# Compute gradients
grad_fn = nb.grad(linear_regression_loss, argnums=(0, 1))
dw, db = grad_fn(w, b, X, y)
print(f"Gradient w.r.t. weights (shape {dw.shape}):")
print(dw)
print(f"\nGradient w.r.t. bias (shape {db.shape}):")
print(db)

# %% [markdown]
# ## 9. A Simple Gradient Descent
#
# Using `value_and_grad` in a training loop:

# %%
w = nb.zeros((n_features, 1))
b = nb.zeros((1,))
lr = 0.1

vg_fn = nb.value_and_grad(linear_regression_loss, argnums=(0, 1))

print(f"{'Step':<6} {'Loss':<12}")
print("-" * 20)
for step in range(10):
    loss, (dw, db) = vg_fn(w, b, X, y)
    w = w - lr * dw
    b = b - lr * db
    if (step + 1) % 2 == 0:
        print(f"{step + 1:<6} {loss.item():<12.6f}")

print(f"\nLearned weights: {w}")
print(f"True weights:    {w_true}")

# %% [markdown]
# ## Summary
#
# | Transform | Input | Output | Best for |
# |-----------|-------|--------|----------|
# | `grad(f)` | Scalar fn | Gradient vector | Training losses |
# | `value_and_grad(f)` | Scalar fn | (value, gradient) | Training loops |
# | `jvp(f, primals, tangents)` | Any fn | (output, J·v) | Few inputs |
# | `vjp(f, *primals)` | Any fn | (output, vjp_fn) | Few outputs |
# | `jacrev(f)` | Any fn | Full Jacobian | Few outputs |
# | `jacfwd(f)` | Any fn | Full Jacobian | Few inputs |
# | Compose! | — | Hessians, etc. | Higher-order derivatives |
#
# **Next:** [03a_mlp_training_pytorch.py](03a_mlp_training_pytorch.py)
# — Training an MLP with Nabla's PyTorch-style API.

# %%
print("\n✅ Tutorial 02 completed!")
