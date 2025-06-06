# Basic Operations

This tutorial covers the fundamental operations you can perform with Endia arrays.

## Array Creation

Endia provides several ways to create arrays:

```python
import endia as nd
import numpy as np

# From Python lists
x = nd.array([1, 2, 3, 4])
print(f"Array from list: {x}")

# Filled arrays
zeros = nd.zeros((3, 4))
ones = nd.ones((2, 5))
full_array = nd.full((3, 3), 7.5)

# Random arrays
random_uniform = nd.rand((2, 3))  # Uniform [0, 1)
random_normal = nd.randn((2, 3))  # Standard normal

# Ranges and sequences
range_array = nd.arange(10)  # [0, 1, 2, ..., 9]
linspace_array = nd.linspace(0, 1, 11)  # 11 points from 0 to 1
```

## Array Properties

```python
x = nd.randn((3, 4, 5))

print(f"Shape: {x.shape}")
print(f"Number of dimensions: {x.ndim}")
print(f"Total elements: {x.size}")
print(f"Data type: {x.dtype}")
```

## Element-wise Operations

### Arithmetic Operations

```python
a = nd.array([1, 2, 3, 4])
b = nd.array([2, 3, 4, 5])

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
x = nd.linspace(-np.pi, np.pi, 100)

# Trigonometric functions
sin_x = nd.sin(x)
cos_x = nd.cos(x)
tan_x = nd.tan(x)

# Exponential and logarithmic
exp_x = nd.exp(x)
log_x = nd.log(nd.abs(x) + 1)  # Add 1 to avoid log(0)

# Other functions
sqrt_x = nd.sqrt(nd.abs(x))
abs_x = nd.abs(x)
```

## Broadcasting

Endia supports NumPy-style broadcasting:

```python
# Broadcasting scalar with array
a = nd.array([[1, 2, 3],
              [4, 5, 6]])
scalar = 10
result = a + scalar  # Adds 10 to each element

# Broadcasting arrays with different shapes
b = nd.array([10, 20, 30])  # Shape: (3,)
result = a + b  # b is broadcast to (2, 3)

print(f"Original array:\n{a}")
print(f"Added vector:\n{result}")
```

## Array Manipulation

### Reshaping

```python
x = nd.arange(12)
print(f"Original shape: {x.shape}")

# Reshape to 2D
reshaped = nd.reshape(x, (3, 4))
print(f"Reshaped to (3, 4):\n{reshaped}")

# Reshape to 3D
reshaped_3d = nd.reshape(x, (2, 2, 3))
print(f"Reshaped to (2, 2, 3):\n{reshaped_3d}")
```

### Transposition

```python
matrix = nd.randn((3, 4))
transposed = nd.transpose(matrix)
# Or use the shorthand:
transposed = matrix.T

print(f"Original shape: {matrix.shape}")
print(f"Transposed shape: {transposed.shape}")
```

### Indexing and Slicing

```python
x = nd.arange(20).reshape((4, 5))

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
data = nd.randn((5, 4))

# Global reductions
total_sum = nd.sum(data)
mean_val = nd.mean(data)
max_val = nd.max(data)
min_val = nd.min(data)

# Axis-specific reductions
row_sums = nd.sum(data, axis=1)      # Sum along columns (result: shape (5,))
col_means = nd.mean(data, axis=0)    # Mean along rows (result: shape (4,))

print(f"Data shape: {data.shape}")
print(f"Row sums shape: {row_sums.shape}")
print(f"Column means shape: {col_means.shape}")
```

## Linear Algebra

```python
# Matrix multiplication
A = nd.randn((3, 4))
B = nd.randn((4, 5))
C = nd.matmul(A, B)  # Shape: (3, 5)

# Vector operations
v1 = nd.array([1, 2, 3])
v2 = nd.array([4, 5, 6])
dot_product = nd.dot(v1, v2)

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
x = nd.randn((1000,))
result = nd.sin(x) ** 2 + nd.cos(x) ** 2

# Avoid: Python loops (when possible)
# result = nd.array([math.sin(xi)**2 + math.cos(xi)**2 for xi in x])
```

## Next Steps

Now that you understand basic operations, move on to:
- {doc}`automatic_differentiation` to learn about computing derivatives
- {doc}`vectorization_and_jit` for performance optimization
