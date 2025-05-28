<h1 align="center">NABLA: Differentiable Programming with Python + Mojo</h1>

<p align="center"><em>A Research Preview</em></p>

Nabla provides **JAX-like function transformations** for automatic differentiation in Python 🔥. Apply transformations like `vmap`, `grad`, `jit`, `vjp`, and `jvp` to pure functions for numerical computing and machine learning, with high-performance Mojo kernel integration for CPU/GPU acceleration.

**Key advantage over JAX**: Seamless integration with Mojo for writing custom high-performance kernels while maintaining the familiar JAX-style functional API.

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#features">Features</a> •
  <a href="#roadmap">Roadmap</a> •
  <a href="https://github.com/nabla-ml/nabla/issues">Report Bug</a>
</p>

## Features

- **Function Transformations**: `grad`, `jit`, `vmap`, `vjp`, `jvp` for automatic differentiation
- **High Performance**: Mojo kernel integration for CPU/GPU acceleration  
- **JAX-like API**: Familiar functional programming interface
- **Pure Functions**: Compose transformations on stateless functions
- **Custom Kernels**: Write performance-critical kernels in Mojo

## Installation

**Note**: Nabla will soon be installable via pip. For now, please install from source.

Get started with Nabla using Python package management.
*(Requires [Python 3.10+](https://www.python.org/downloads/).)*

```bash
git clone https://github.com/nabla-ml/nabla.git
cd nabla

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

## Prerequisites

- **Python 3.10+**
- **NumPy** (for array operations)
- **Optional**: MAX Engine (for Mojo kernel acceleration)

## Core Transformations

- `nb.grad()` - Compute gradients (reverse-mode autodiff)
- `nb.jvp()` - Jacobian-vector product (forward-mode autodiff) 
- `nb.vjp()` - Vector-Jacobian product (reverse-mode autodiff)
- `nb.vmap()` - Vectorize functions over batch dimensions
- `nb.jit()` - Just-in-time compilation for performance

## Development Setup

For contributors and advanced users:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format and lint code
ruff format nabla/
ruff check nabla/ --fix
```

## Roadmap

Following the strategic direction of the MAX project, Nabla prioritizes the **Python API first** with JAX-like function transformations.

- ✅ **Function Transformations**: Core JAX-like transformations (vmap, grad, jit)
- ✅ **Mojo Kernel Integration**: High-performance kernel implementation working
- ✅ **GPU Acceleration**: Full GPU support
- 👷 **Extended Operations**: Comprehensive math operations
- 💡 **Enhanced Mojo API**: When Mojo ecosystem stabilizes, we will integrate further

## Repository Structure

This repository contains:

- **Main Directory**: Primary Python-based Nabla implementation (recommended)
- **`nabla-mojo-experimental/`**: Experimental Mojo implementation (research preview)

The Python version is the main focus and provides the most stable experience. The Mojo version showcases future possibilities but is currently on hold pending MAX Mojo API development.

## General Status & Caveats (Research Preview)

- **API Stability**: APIs are subject to change
- **Completeness**: Operator coverage is growing but not exhaustive
- **Performance**: JIT compilation and Mojo kernel integration provide good performance
- **Documentation**: Currently basic; will be expanded significantly
- **Bugs**: Expect to encounter bugs; please report them!

## Contributing

Contributions welcome! Discuss significant changes in Issues first. Submit PRs for bugs, docs, and smaller features.

## License

Nabla is licensed under the [Apache-2.0 license](https://github.com/nabla-ml/nabla/blob/main/LICENSE).

---

<p align="center" style="margin-top: 3em; margin-bottom: 2em;"><em>Thank you for checking out Nabla!</em></p>

