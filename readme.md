# NABLA

Nabla is a Python library that provides three key features:

- Multidimensional Array computation (like NumPy) with strong GPU acceleration
- Composable Function Transformations: `vmap`, `grad`, `jit`, and other Automatic Differentiation tools
- Deep integration with MAX and (custom) Mojo kernels

For tutorials and API reference, visit: [nablaml.com](https://www.nablaml.com/index.html)

## Installation

**Now available on PyPI!**

```bash
pip install nabla-ml
```

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

## Development Setup and Reproducibility

This guide is for contributors or for reproducing the validation and benchmark results presented in the thesis.

<p align="center">
    <img src="assets/image.png" alt="alt text" style="max-width: 400px; display: block; margin: 0 auto;" />
</p>

### 1. Initial Setup

First, clone the repository and set up a virtual environment with all necessary dependencies.

```bash
# Clone the repository
git clone https://github.com/nabla-ml/nabla.git
cd nabla

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install all core and development dependencies
pip install -r requirements-dev.txt
```

### 2. Run the Correctness Validation Suite

This runs the full test suite to verify Nabla's correctness against JAX.

```bash
# Navigate to the unit test directory from the project root
cd nabla/tests/unit

# Execute the unified test script
python unified.py all -all-configs
```

### 3. Run the Performance Benchmarks

This script reproduces the performance benchmarks for Nabla, JAX, and PyTorch.

```bash
# Navigate to the benchmark directory
cd nabla/tests/benchmarks

# Run the benchmark script
python benchmark.py
```

## Repository Structure

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
├── examples/                  # Example scripts for common use cases
└── experimental/              # Core (pure) Mojo library (WIP!)
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