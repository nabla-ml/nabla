# Getting Started

## Installation

**📦 Now available on PyPI!**

```bash
pip install endia-ml
```

For development installation:

```bash
git clone https://github.com/endia-ml/endia.git
cd endia
pip install -e ".[dev]"
```

## Requirements

- Python 3.12+
- NumPy >= 1.22
- Modular >= 25.0.0

## Basic Usage

### Creating Arrays

```python
import endia as nb

# Create arrays
x = nd.array([1, 2, 3, 4])
y = nd.ones((3, 4))
z = nd.randn((2, 3))
```

### Basic Operations

```python
# Element-wise operations
result = nd.sin(x) + nd.cos(y)

# Linear algebra
A = nd.randn((10, 5))
B = nd.randn((5, 3))
C = nd.matmul(A, B)

# Reductions
mean_val = nd.mean(C, axis=0)
sum_val = nd.sum(C, axis=1)
```

### Function Transformations

```python
def simple_function(x):
    return nd.sum(x ** 2)

# Automatic differentiation
grad_fn = nd.grad(simple_function)
gradient = grad_fn(nd.array([1.0, 2.0, 3.0]))

# Vectorization
vmap_fn = nd.vmap(simple_function)
results = vmap_fn(nd.randn((5, 3)))

# JIT compilation
jit_fn = nd.jit(simple_function)
fast_result = jit_fn(nd.randn((1000,)))
```

### Neural Network Example

```python
def mlp_layer(x, weights, bias):
    return nd.relu(nd.matmul(x, weights) + bias)

def mlp(x, params):
    for w, b in zip(params[::2], params[1::2]):
        x = mlp_layer(x, w, b)
    return x

# Create parameters
layer_sizes = [10, 64, 32, 1]
params = []
for i in range(len(layer_sizes) - 1):
    w = nd.randn((layer_sizes[i], layer_sizes[i+1])) * 0.1
    b = nd.zeros((layer_sizes[i+1],))
    params.extend([w, b])

# Forward pass
x = nd.randn((32, 10))  # Batch of 32 samples
output = mlp(x, params)

# Compute gradients
loss_fn = lambda p: nd.mean((mlp(x, p) - target) ** 2)
gradients = nd.grad(loss_fn)(params)
```

## What's Next?

- Check out the {doc}`tutorials/index` for detailed examples
- Browse the {doc}`api/index` for complete API reference  
- See {doc}`examples/index` for real-world applications
