<!-- <h1 align="center">NABLA</h1>

<h2 align="center"><em>Dynamic Neural Networks and Function Transformations in Python + Mojo</em></h2> -->

# NABLA

### Dynamic Neural Networks and Function Transformations in Python + Mojo

Nabla provides 3 things:

- **Multidimensional array operations**: Support for binary, unary, and linear algebra operations on multi-dimensional arrays (Tensors) on CPU and GPU.
- **Dynamic function transformations**: Apply JAX-like transformations like `vmap`, `grad`, `jit` to Python functions.
- **Mojo acceleration 🔥**: Seamlessly integrate high-performance Mojo kernels for CPU and GPU execution.

<!-- 
<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#roadmap">Roadmap</a> •
  <a href="https://github.com/nabla-ml/nabla/issues">Report Bug</a>
</p> -->

## Installation

**Note**: Nabla will soon be installable via pip. For now, please install from source.

**Requirements**: Python 3.10+, NumPy, Modular (Mojo + MAX for JIT/GPU support)

```bash
git clone https://github.com/nabla-ml/nabla.git
cd nabla

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .
```

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
│   ├── mojo_kernels/          # High-performance Mojo kernels
│   └── utils/                 # Utilities (broadcasting, types)
├── tests/                     # Comprehensive test suite
└── nabla-mojo-experimental/   # Experimental pure Mojo implementation
```

## Status (Research Preview)

- **API Stability**: Subject to change
- **Completeness**: Growing operator coverage  
- **Documentation**: Basic; expanding soon
- **Bugs**: Please report issues!

## Contributing

Contributions welcome! Discuss significant changes in Issues first. Submit PRs for bugs, docs, and smaller features.

## License

Nabla is licensed under the [Apache-2.0 license](https://github.com/nabla-ml/nabla/blob/main/LICENSE).

---

<p align="center" style="margin-top: 3em; margin-bottom: 2em;"><em>Thank you for checking out Nabla!</em></p>

