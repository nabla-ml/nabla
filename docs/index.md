# NABLA

[![Development Status](https://img.shields.io/badge/status-pre--alpha-red)](https://github.com/nabla-ml/nabla)
[![PyPI version](https://badge.fury.io/py/nabla-ml.svg)](https://badge.fury.io/py/nabla-ml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Nabla is a Python library that provides three key features:

- **Multidimensional Array computation** (like NumPy) with strong GPU acceleration
- **Composable Function Transformations**: `vmap`, `grad`, `jit`, and more  
- **Deep integration** with (custom) Mojo kernels

```{toctree}
:maxdepth: 1

tutorials/index
examples/index
api/index
```

## Installation

**📦 Now available on PyPI!**

```bash
pip install nabla-ml
```

**Note:** Nabla also includes an [experimental pure Mojo API](https://github.com/nabla-ml/nabla/tree/main/experimental) for native Mojo development.

## Quick Start

```python
import nabla as nb

# Example function using Nabla's array operations
def foo(input):
    return nb.sum(input * input, axes=0)

# Vectorize, differentiate, accelerate
foo_grads = nb.jit(nb.vmap(nb.grad(foo)))
gradients = foo_grads(nb.randn((10, 5)))
```

## Key Features

### Function Transformations

```python
import nabla as nb

# Automatic differentiation
grad_fn = nb.grad(my_loss_function)

# Vectorization  
batch_fn = nb.vmap(single_sample_function)

# Just-in-time compilation
fast_fn = nb.jit(expensive_computation)

# Compose transformations
optimized_grad = nb.jit(nb.grad(nb.vmap(loss_fn)))
```

### Array Operations

```python
# Create arrays
x = nb.array([1, 2, 3, 4])
y = nb.ones((3, 4))
z = nb.randn((2, 3))

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

### JAX Compatibility

Nabla follows JAX's API design, making it easy to port existing JAX code:

```python
# JAX-style function transformations
from nabla import grad, jit, vmap, vjp, jvp

# Same API as JAX
def loss_fn(params, x, y):
    predictions = model(params, x)
    return np.mean((predictions - y) ** 2)

# Compute gradients
grad_fn = grad(loss_fn)
gradients = grad_fn(params, x_batch, y_batch)
```

## Development Setup

For contributors and advanced users:

```bash
# Clone and install in development mode
git clone https://github.com/nabla-ml/nabla.git
cd nabla
pip install -e ".[dev]"

# Run tests
pytest

# Format and lint code
black nabla tests
ruff check nabla tests
```

## Documentation

- **{doc}`tutorials/index`** - Learn Nabla step by step
- **{doc}`examples/index`** - Example gallery showing Nabla in action  
- **{doc}`api/index`** - Complete API reference

## What makes Nabla special?

- **Performance**: Deep integration with Mojo kernels for maximum speed
- **Familiar API**: JAX-compatible interface for easy adoption
- **Composable**: Function transformations that work seamlessly together
- **Extensible**: Easy integration with custom Mojo kernels

- `grad`: Automatic differentiation  
- `vmap`: Vectorization over batch dimensions
- `jit`: Just-in-time compilation for performance
- `vjp`/`jvp`: Vector-Jacobian and Jacobian-vector products

### Array Operations

- Comprehensive mathematical operations (unary, binary, linear algebra)
- NumPy-compatible API design
- GPU acceleration through Mojo kernels
- Broadcasting and shape manipulation

### Integration

- Python API for ease of use
- Mojo kernels for performance-critical operations
- JAX-like transformation semantics
- PyTorch-style imperative programming support

## Installation

```bash
pip install nabla-ml
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
