# Basic Operations

This tutorial covers the fundamental operations you can perform with Nabla arrays.

## Array Creation

Nabla provides several ways to create arrays:

```python
import nabla as nb
import numpy as np

# From Python lists
x = nb.array([1, 2, 3, 4])
print(f"Array from list: {x}")

# Filled arrays
zeros = nb.zeros((3, 4))
ones = nb.ones((2, 5))
full_array = nb.full((3, 3), 7.5)

# Random arrays
random_uniform = nb.rand((2, 3))  # Uniform [0, 1)
random_normal = nb.randn((2, 3))  # Standard normal

# Ranges and sequences
range_array = nb.arange(10)  # [0, 1, 2, ..., 9]
linspace_array = nb.linspace(0, 1, 11)  # 11 points from 0 to 1
```

## Array Properties

```python
x = nb.randn((3, 4, 5))

print(f"Shape: {x.shape}")
print(f"Number of dimensions: {x.ndim}")
print(f"Total elements: {x.size}")
print(f"Data type: {x.dtype}")
```

## Element-wise Operations

### Arithmetic Operations

```python
a = nb.array([1, 2, 3, 4])
b = nb.array([2, 3, 4, 5])

# Basic arithmetic
addition = a + b
subtraction = a - b
multiplication = a * b
division = a / b
power = a ** 2

print(f"a + b = {addition}")
print(f"a * b = {multiplication}")
print(f"a ** 2 = {power}")
```

### Mathematical Functions

```python
x = nb.linspace(-np.pi, np.pi, 100)

# Trigonometric functions
sin_x = nb.sin(x)
cos_x = nb.cos(x)
tan_x = nb.tan(x)

# Exponential and logarithmic
exp_x = nb.exp(x)
log_x = nb.log(nb.abs(x) + 1)  # Add 1 to avoid log(0)

# Other functions
sqrt_x = nb.sqrt(nb.abs(x))
abs_x = nb.abs(x)
```

## Broadcasting

Nabla supports NumPy-style broadcasting:

```python
# Broadcasting scalar with array
a = nb.array([[1, 2, 3],
              [4, 5, 6]])
scalar = 10
result = a + scalar  # Adds 10 to each element

# Broadcasting arrays with different shapes
b = nb.array([10, 20, 30])  # Shape: (3,)
result = a + b  # b is broadcast to (2, 3)

print(f"Original array:\n{a}")
print(f"Added vector:\n{result}")
```

## Array Manipulation

### Reshaping

```python
x = nb.arange(12)
print(f"Original shape: {x.shape}")

# Reshape to 2D
reshaped = nb.reshape(x, (3, 4))
print(f"Reshaped to (3, 4):\n{reshaped}")

# Reshape to 3D
reshaped_3d = nb.reshape(x, (2, 2, 3))
print(f"Reshaped to (2, 2, 3):\n{reshaped_3d}")
```

### Transposition

```python
matrix = nb.randn((3, 4))
transposed = nb.transpose(matrix)
# Or use the shorthand:
transposed = matrix.T

print(f"Original shape: {matrix.shape}")
print(f"Transposed shape: {transposed.shape}")
```

### Indexing and Slicing

```python
x = nb.arange(20).reshape((4, 5))

# Basic indexing
element = x[1, 2]  # Element at row 1, column 2
row = x[1, :]      # Entire row 1
column = x[:, 2]   # Entire column 2

# Slicing
subarray = x[1:3, 2:4]  # Rows 1-2, columns 2-3

print(f"Original array:\n{x}")
print(f"Subarray x[1:3, 2:4]:\n{subarray}")
```

## Aggregation Operations

```python
data = nb.randn((5, 4))

# Global reductions
total_sum = nb.sum(data)
mean_val = nb.mean(data)
max_val = nb.max(data)
min_val = nb.min(data)

# Axis-specific reductions
row_sums = nb.sum(data, axis=1)      # Sum along columns (result: shape (5,))
col_means = nb.mean(data, axis=0)    # Mean along rows (result: shape (4,))

print(f"Data shape: {data.shape}")
print(f"Row sums shape: {row_sums.shape}")
print(f"Column means shape: {col_means.shape}")
```

## Linear Algebra

```python
# Matrix multiplication
A = nb.randn((3, 4))
B = nb.randn((4, 5))
C = nb.matmul(A, B)  # Shape: (3, 5)

# Vector operations
v1 = nb.array([1, 2, 3])
v2 = nb.array([4, 5, 6])
dot_product = nb.dot(v1, v2)

print(f"Matrix product shape: {C.shape}")
print(f"Dot product: {dot_product}")
```

## Performance Tips

1. **Use vectorized operations**: Prefer array operations over Python loops
2. **Understand broadcasting**: Avoid unnecessary reshaping
3. **Choose appropriate data types**: Use `float32` instead of `float64` when possible
4. **Minimize array copies**: Use in-place operations when appropriate

```python
# Good: Vectorized operation
x = nb.randn((1000,))
result = nb.sin(x) ** 2 + nb.cos(x) ** 2

# Avoid: Python loops (when possible)
# result = nb.array([math.sin(xi)**2 + math.cos(xi)**2 for xi in x])
```

## Next Steps

Now that you understand basic operations, move on to:
- {doc}`automatic_differentiation` to learn about computing derivatives
- {doc}`vectorization_and_jit` for performance optimization
