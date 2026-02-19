# ===----------------------------------------------------------------------=== #
# Nabla Examples - 01: Tensors and Operations
# ===----------------------------------------------------------------------=== #
"""Tensors and Operations in Nabla.

This example covers the fundamentals of Nabla's tensor system:
- Creating tensors (from NumPy, factory functions, constants)
- Basic arithmetic and element-wise operations
- Matrix operations (matmul, transpose, reshape)
- Reduction operations (sum, mean, max, min)
- Shape manipulation (reshape, squeeze, unsqueeze, permute)
- Indexing and slicing
"""

# %% [markdown]
# # Example 1: Tensors and Operations
#
# Welcome to Nabla! This example introduces the core building block of
# the library: the **Tensor**. Nabla tensors are lazy by default — operations
# build a computation graph that is evaluated only when you request the result
# (e.g., by printing or calling `.realize()`).
#
# Let's start by importing Nabla and NumPy.

# %%
import numpy as np

import nabla as nb

print("Nabla imported successfully!")

# %% [markdown]
# ## 1. Creating Tensors
#
# There are several ways to create tensors in Nabla:
#
# 1. **From NumPy arrays** via `nb.Tensor.from_dlpack()` (works with any DLPack source)
# 2. **Factory functions** like `nb.zeros()`, `nb.ones()`, `nb.arange()`, `nb.uniform()`
# 3. **Constants** via `nb.constant()`

# %%
# From NumPy arrays
np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
x = nb.Tensor.from_dlpack(np_array)
print("From NumPy:")
print(x)
print(f"  Shape: {x.shape}, Dtype: {x.dtype}\n")

# %%
# Factory functions — zeros, ones, full
z = nb.zeros((2, 3))
o = nb.ones((2, 3))
f = nb.full((2, 3), 3.14)
print("Zeros:", z)
print("Ones: ", o)
print("Full: ", f)

# %%
# Ranges and random tensors
r = nb.arange(0, 6, dtype=nb.DType.float32)
u = nb.uniform((2, 3), low=-1.0, high=1.0)
g = nb.gaussian((2, 3), mean=0.0, std=1.0)
print("Arange:  ", r)
print("Uniform: ", u)
print("Gaussian:", g)

# %%
# Constants from python lists (via numpy)
c = nb.constant(np.array([10.0, 20.0, 30.0], dtype=np.float32))
print("Constant:", c)

# %% [markdown]
# ## 2. Tensor Properties
#
# Every tensor carries metadata about its shape, dtype, and device.

# %%
x = nb.uniform((3, 4, 5))
print(f"Shape:  {x.shape}")
print(f"Dtype:  {x.dtype}")
print(f"Device: {x.device}")
print(f"Rank:   {x.ndim}")

# %% [markdown]
# ## 3. Arithmetic Operations
#
# Nabla supports standard arithmetic via Python operators and named functions.
# All operations are lazy — they build a graph that is evaluated on demand.

# %%
a = nb.Tensor.from_dlpack(np.array([1.0, 2.0, 3.0], dtype=np.float32))
b = nb.Tensor.from_dlpack(np.array([4.0, 5.0, 6.0], dtype=np.float32))

print("a:    ", a)
print("b:    ", b)
print("a + b:", a + b)
print("a - b:", a - b)
print("a * b:", a * b)
print("a / b:", a / b)
print("a ** 2:", a ** 2)

# %%
# Named function equivalents
print("nb.add(a, b):", nb.add(a, b))
print("nb.mul(a, b):", nb.mul(a, b))
print("nb.sub(a, b):", nb.sub(a, b))
print("nb.div(a, b):", nb.div(a, b))

# %% [markdown]
# ## 4. Element-wise Unary Operations
#
# Nabla provides a rich set of element-wise functions.

# %%
x = nb.Tensor.from_dlpack(np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32))
print("x:       ", x)
print("exp(x):  ", nb.exp(x))
print("log(x+1):", nb.log(x + 1.0))
print("sqrt(x): ", nb.sqrt(x))
print("sin(x):  ", nb.sin(x))
print("cos(x):  ", nb.cos(x))
print("tanh(x): ", nb.tanh(x))

# %%
# Activation functions
x = nb.Tensor.from_dlpack(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32))
print("x:       ", x)
print("relu(x): ", nb.relu(x))
print("sigmoid:", nb.sigmoid(x))
print("gelu(x): ", nb.gelu(x))
print("silu(x): ", nb.silu(x))

# %% [markdown]
# ## 5. Matrix Operations
#
# Matrix multiplication is a first-class operation in Nabla.

# %%
A = nb.uniform((3, 4))
B = nb.uniform((4, 5))
C = nb.matmul(A, B)  # or A @ B
print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"A @ B shape: {C.shape}")
print("A @ B:\n", C)

# %%
# Batched matmul
batch_A = nb.uniform((2, 3, 4))
batch_B = nb.uniform((2, 4, 5))
batch_C = batch_A @ batch_B
print(f"Batched matmul: {batch_A.shape} @ {batch_B.shape} = {batch_C.shape}")

# %%
# Outer product via broadcasting: v1[:, None] * v2[None, :]
v1 = nb.Tensor.from_dlpack(np.array([1.0, 2.0, 3.0], dtype=np.float32))
v2 = nb.Tensor.from_dlpack(np.array([4.0, 5.0], dtype=np.float32))
outer = nb.unsqueeze(v1, axis=1) * nb.unsqueeze(v2, axis=0)
print(f"Outer product ({v1.shape} x {v2.shape}):")
print(outer)

# %% [markdown]
# ## 6. Reduction Operations
#
# Reduce along one or more axes (or all axes for a scalar result).

# %%
x = nb.Tensor.from_dlpack(
    np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
)
print("x:\n", x)
print()

# Full reductions
print("sum(x):  ", nb.reduce_sum(x))
print("mean(x): ", nb.mean(x))
print("max(x):  ", nb.reduce_max(x))
print("min(x):  ", nb.reduce_min(x))

# %%
# Axis-specific reductions
print("sum(axis=0):", nb.reduce_sum(x, axis=0))  # Sum columns
print("sum(axis=1):", nb.reduce_sum(x, axis=1))  # Sum rows
print("mean(axis=1):", nb.mean(x, axis=1))
print("max(axis=0): ", nb.reduce_max(x, axis=0))

# %%
# keepdims preserves the reduced dimension
print("sum(axis=1, keepdims=True):", nb.reduce_sum(x, axis=1, keepdims=True))
print(f"  Shape: {nb.reduce_sum(x, axis=1, keepdims=True).shape}")

# %%
# Argmax / Argmin
print("argmax(axis=1):", nb.argmax(x, axis=1))
print("argmin(axis=0):", nb.argmin(x, axis=0))

# %% [markdown]
# ## 7. Shape Manipulation
#
# Nabla supports reshaping, transposing, squeezing, and more — all as lazy ops.

# %%
x = nb.arange(0, 12, dtype=nb.DType.float32)
print(f"Original: shape={x.shape}")
print(x)

# Reshape
r = nb.reshape(x, (3, 4))
print(f"\nReshaped to (3, 4):")
print(r)

# Flatten
f = nb.flatten(r)
print(f"\nFlattened back: shape={f.shape}")

# %%
# Transpose and permute
m = nb.uniform((2, 3, 4))
print(f"Original shape:   {m.shape}")
print(f"Swap axes (1,2):  {nb.swap_axes(m, 1, 2).shape}")
print(f"Permute (2,0,1):  {nb.permute(m, (2, 0, 1)).shape}")
print(f"Move axis 2→0:    {nb.moveaxis(m, 2, 0).shape}")

# %%
# Squeeze and unsqueeze
x = nb.ones((1, 3, 1, 4))
print(f"Original:       {x.shape}")
print(f"Squeeze(0):     {nb.squeeze(x, axis=0).shape}")
print(f"Squeeze(2):     {nb.squeeze(x, axis=2).shape}")

y = nb.ones((3, 4))
print(f"Unsqueeze(0):   {nb.unsqueeze(y, axis=0).shape}")
print(f"Unsqueeze(1):   {nb.unsqueeze(y, axis=1).shape}")

# %% [markdown]
# ## 8. Concatenation and Stacking

# %%
a = nb.ones((2, 3))
b = nb.zeros((2, 3))
print("Concatenate (axis=0):", nb.concatenate([a, b], axis=0).shape)
print("Concatenate (axis=1):", nb.concatenate([a, b], axis=1).shape)
print("Stack (axis=0):      ", nb.stack([a, b], axis=0).shape)
print("Stack (axis=1):      ", nb.stack([a, b], axis=1).shape)

# %% [markdown]
# ## 9. Broadcasting
#
# Nabla follows NumPy broadcasting rules.

# %%
x = nb.uniform((3, 1))
y = nb.uniform((1, 4))
z = x + y  # Broadcasts to (3, 4)
print(f"x: {x.shape} + y: {y.shape} = z: {z.shape}")
print(z)

# %%
# Explicit broadcast
v = nb.Tensor.from_dlpack(np.array([1.0, 2.0, 3.0], dtype=np.float32))
b = nb.broadcast_to(v, (4, 3))
print(f"Broadcast {v.shape} → {b.shape}:")
print(b)

# %% [markdown]
# ## 10. Type Casting

# %%
x = nb.ones((3,), dtype=nb.DType.float32)
print(f"Original dtype: {x.dtype}")

x_int = nb.cast(x, nb.DType.int32)
print(f"Cast to int32:  {x_int.dtype}")

x_f64 = nb.cast(x, nb.DType.float64)
print(f"Cast to float64: {x_f64.dtype}")

# %% [markdown]
# ## 11. Comparisons and Logical Operations

# %%
a = nb.Tensor.from_dlpack(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
b = nb.Tensor.from_dlpack(np.array([2.0, 2.0, 4.0, 3.0], dtype=np.float32))

print("a:", a)
print("b:", b)
print("a == b:", nb.equal(a, b))
print("a > b: ", nb.greater(a, b))
print("a < b: ", nb.less(a, b))
print("a >= b:", nb.greater_equal(a, b))

# %%
# Where (conditional select)
mask = nb.greater(a, b)
result = nb.where(mask, a, b)  # Pick a where a > b, else b
print("where(a > b, a, b):", result)

# %% [markdown]
# ## 12. Softmax

# %%
logits = nb.Tensor.from_dlpack(
    np.array([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]], dtype=np.float32)
)
probs = nb.softmax(logits, axis=-1)
print("Logits:\n", logits)
print("Softmax:\n", probs)
print("Row sums:", nb.reduce_sum(probs, axis=-1))

# %% [markdown]
# ## Summary
#
# In this example you learned how to:
# - Create tensors from NumPy arrays, factory functions, and constants
# - Perform arithmetic, element-wise, and matrix operations
# - Reduce tensors along axes (sum, mean, max, min, argmax)
# - Manipulate shapes (reshape, transpose, squeeze, unsqueeze)
# - Use broadcasting, type casting, comparisons, and softmax
#
# All operations are **lazy** — they build a computation graph that's evaluated
# on demand. This enables powerful optimizations when combined with `@nb.compile`.
#
# **Next:** [02_autodiff](02_autodiff) — Automatic differentiation with Nabla.

# %%
print("\n✅ Example 01 completed!")
