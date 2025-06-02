# Getting Started

## Installation

**📦 Now available on PyPI!**

```bash
pip install nabla-ml
```

For development installation:

```bash
git clone https://github.com/nabla-ml/nabla.git
cd nabla
pip install -e ".[dev]"
```

## Requirements

- Python 3.12+
- NumPy >= 1.22
- Modular >= 25.0.0

## Basic Usage

### Creating Arrays

```python
import nabla as nb

# Create arrays
x = nb.array([1, 2, 3, 4])
y = nb.ones((3, 4))
z = nb.randn((2, 3))
```

### Basic Operations

```python
# Element-wise operations
result = nb.sin(x) + nb.cos(y)

# Linear algebra
A = nb.randn((10, 5))
B = nb.randn((5, 3))
C = nb.matmul(A, B)

# Reductions
mean_val = nb.mean(C, axis=0)
sum_val = nb.sum(C, axis=1)
```

### Function Transformations

```python
def simple_function(x):
    return nb.sum(x ** 2)

# Automatic differentiation
grad_fn = nb.grad(simple_function)
gradient = grad_fn(nb.array([1.0, 2.0, 3.0]))

# Vectorization
vmap_fn = nb.vmap(simple_function)
results = vmap_fn(nb.randn((5, 3)))

# JIT compilation
jit_fn = nb.jit(simple_function)
fast_result = jit_fn(nb.randn((1000,)))
```

### Neural Network Example

```python
def mlp_layer(x, weights, bias):
    return nb.relu(nb.matmul(x, weights) + bias)

def mlp(x, params):
    for w, b in zip(params[::2], params[1::2]):
        x = mlp_layer(x, w, b)
    return x

# Create parameters
layer_sizes = [10, 64, 32, 1]
params = []
for i in range(len(layer_sizes) - 1):
    w = nb.randn((layer_sizes[i], layer_sizes[i+1])) * 0.1
    b = nb.zeros((layer_sizes[i+1],))
    params.extend([w, b])

# Forward pass
x = nb.randn((32, 10))  # Batch of 32 samples
output = mlp(x, params)

# Compute gradients
loss_fn = lambda p: nb.mean((mlp(x, p) - target) ** 2)
gradients = nb.grad(loss_fn)(params)
```

## What's Next?

- Check out the {doc}`tutorials/index` for detailed examples
- Browse the {doc}`api/index` for complete API reference  
- See {doc}`examples/index` for real-world applications
