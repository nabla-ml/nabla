[![Development Status](https://img.shields.io/badge/status-pre--alpha-red)](https://github.com/nabla-ml/nabla)
[![PyPI version](https://badge.fury.io/py/nabla-ml.svg)](https://badge.fury.io/py/nabla-ml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# NABLA

Nabla is a Python library that provides three key features:

- Multidimensional Array computation (like NumPy) with strong GPU acceleration
- Composable Function Transformations: `vmap`, `grad`, `jit`, and more
- Deep integration with (custom) Mojo kernels

## Installation

**📦 Now available on PyPI!**

```bash
pip install nabla-ml
```

## Quick Start

```python
import nabla as nb

# Example function using Nabla's array operations
def foo(input):
    return nb.sum(input ** 2, axis=0)

# Vectorize, differentiate, accelerate
foo_grads = nb.jit(nb.grad(nb.vmap(foo)))
gradients = foo_grads([nb.randn((10, 5))])
```

## Roadmap

- ✅ **Function Transformations**: `vmap`, `jit`, `vjp`, `jvp`, `grad`
- ✅ **Mojo Kernel Integration**: CPU/GPU acceleration working
- 👷 **Extended Operations**: Comprehensive math operations
- 💡 **Enhanced Mojo API**: When Mojo/MAX ecosystem stabilizes

## Development Setup

For contributors and advanced users:

```bash
# Clone and install in development mode
git clone https://github.com/nabla-ml/nb.git
cd nabla
pip install -e ".[dev]"

# Run tests
pytest

# Format and lint code
ruff format nabla/
ruff check nabla/ --fix
```

## Repository Structure

```text
nabla/
├── nabla/                     # Core Python library
│   ├── core/                  # Function transformations and array operations
│   ├── ops/                   # Mathematical operations (binary, unary, linalg)
│   ├── kernels/               # Internal CPU/GPU Mojo kernels (not the built-in MAX kernels)
│   ├── nn/                    # Neural network layers and utilities
│   └── utils/                 # Utilities (broadcasting, formatting, types)
├── tests/                     # Comprehensive test suite
├── examples/                  # MLP training and other examples
└── experimental/              # Pure Mojo API
```

## Contributing

Contributions welcome! Discuss significant changes in Issues first. Submit PRs for bugs, docs, and smaller features.

## License

Nabla is licensed under the [Apache-2.0 license](https://github.com/nabla-ml/nabla/blob/main/LICENSE).

---

*Thank you for checking out Nabla!*
