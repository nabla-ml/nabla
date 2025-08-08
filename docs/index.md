# Nabla

Nabla is a scientific computing library in Python, featuring:

- Multidimensional array computation with strong GPU acceleration
- JAX-like composable function transformations: `grad`, `vmap`, `jit`, and other automatic differentiation tools
- Deep integration with the MAX compiler and custom Mojo 🔥 kernels

```{toctree}
:maxdepth: 1

api/index
tutorials/index
```

## Installation

```bash
pip install nabla-ml
```
 <!--  -->
## Quick Start

```python
import nabla as nb

# Example function using Nabla's array operations
def foo(input):
    return nb.sum(input * input, axes=-1)

# Differentiate, vectorize, accelerate
foo_grads = nb.jit(nb.vmap(nb.grad(foo)))
gradients = foo_grads(nb.randn((10, 5)).to(nb.accelerator()))
```

## For Developers

1. Clone the repository
2. Create a virtual environment (recommended)
3. Install dependencies

```bash
git clone https://github.com/nabla-ml/nabla.git
cd nabla

python3 -m venv venv
source venv/bin/activate

pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

## Repository Structure

<!-- ![alt text](assets/image.png) -->

```text
nabla/
├── nabla/                     # Core Python library
│   ├── core/                  # Array class and MAX compiler integration
│   ├── nn/                    # Neural network modules and models
│   ├── ops/                   # Mathematical operations (binary, unary, linalg, etc.)
│   ├── transforms/            # Function transformations (vmap, grad, jit, etc.)
│   └── utils/                 # Utilities (formatting, types, MAX-interop, etc.)
├── tests/                     # Comprehensive test suite
├── tutorials/                 # Notebooks on Nabla usage for ML tasks
└── examples/                  # Example scripts for common use cases
```

## Contributing

Contributions welcome! Discuss significant changes in Issues first. Submit PRs for bugs, docs, and smaller features.

## License

Nabla is licensed under the [Apache-2.0 license](https://github.com/nabla-ml/nabla/blob/main/LICENSE).

---

*Thank you for checking out Nabla!*

[![Development Status](https://img.shields.io/badge/status-pre--alpha-red)](https://github.com/nabla-ml/nabla)
[![PyPI version](https://badge.fury.io/py/nabla-ml.svg)](https://badge.fury.io/py/nabla-ml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)