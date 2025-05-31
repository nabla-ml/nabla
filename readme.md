[![Development Status](https://img.shields.io/badge/status-pre--alpha-red)](https://github.com/nabla-ml/nabla)
[![PyPI version](https://badge.fury.io/py/nabla-ml.svg)](https://pypi.org/project/nabla-ml/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)


# NABLA

### Dynamic Neural Networks and Function Transformations in Python + Mojo

*Nabla provides 3 things:*

- **Multidimensional array operations**: Support for binary, unary, and linear algebra operations on multi-dimensional arrays (Tensors) on CPU and GPU.
- **Dynamic function transformations**: Apply JAX-like transformations like `vmap`, `grad`, `jit` to Python functions.
- **Mojo acceleration**: Seamlessly integrate high-performance Mojo kernels for CPU and GPU execution.


## Installation

**📦 Now available on PyPI!**

```bash
pip install nabla-ml
```

**Requirements**: Python 3.10+, NumPy, Modular (Mojo + MAX for JIT/GPU support)


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
ruff format nabla/
ruff check nabla/ --fix
```

## Roadmap

- ✅ **Function Transformations**: `vmap`, `grad`, `jit`, `vjp`, `jvp`
- ✅ **Mojo Kernel Integration**: CPU/GPU acceleration working
- 👷 **Extended Operations**: Comprehensive math operations
- 💡 **Enhanced Mojo API**: When Mojo ecosystem stabilizes

## Repository Structure

```
nabla/
├── nabla/                     # Core Python library
│   ├── core/                  # Function transformations and array operations
│   ├── ops/                   # Mathematical operations (binary, unary, linalg)
│   ├── mojo_kernels/          # Internal CPU/GPU Mojo kernels (not the built-in MAX kernels)
│   ├── nn/                    # Neural network layers and utilities
│   └── utils/                 # Utilities (broadcasting, formatting, types)
├── tests/                     # Comprehensive test suite
└── nabla-mojo/                # Experimental pure Mojo API
```

## Contributing

Contributions welcome! Discuss significant changes in Issues first. Submit PRs for bugs, docs, and smaller features.

## License

Nabla is licensed under the [Apache-2.0 license](https://github.com/nabla-ml/nabla/blob/main/LICENSE).

---

<p align="center" style="margin-top: 3em; margin-bottom: 2em;"><em>Thank you for checking out Nabla!</em></p>

