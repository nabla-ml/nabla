# NABLA

> **⚠️ IMPORTANT: Architecture Under Heavy Restructuring**
> 
> **You are viewing the `main` development branch** with a complete rewrite featuring distributed SPMD execution, multi-device training, and a new graph-based execution model.
> 
> The stable release (v25.7, single-device) is available on the [nabla/v25.7 branch](https://github.com/nabla-ml/nabla/tree/nabla/v25.7). See Installation section below.

---

Welcome! Nabla is a Machine Learning library for the emerging Mojo/Python ecosystem, featuring:

- Gradient computation the PyTorch way (imperatively via .backward())
- Purely-functional, JAX-like composable function transformations: `grad`, `vmap`, `jit`, etc.
- Custom differentiable CPU/GPU kernels

For tutorials and API reference, visit: [nablaml.com](https://nablaml.com/index.html)

## Installation & Quick Start

### For Users: Stable Release (v25.7)

Install from PyPI:
```bash
pip install nabla-ml
```

Example - Simple MLP training:
```python
import nabla as nb

# Defines MLP forward pass and loss.
def loss_fn(params, x, y):
    for i in range(0, len(params) - 2, 2):
        x = nb.relu(x @ params[i] + params[i + 1])
    predictions = x @ params[-2] + params[-1]
    return nb.mean((predictions - y) ** 2)

# JIT-compiled training step via SGD
@nb.jit(auto_device=True)
def train_step(params, x, y, lr):
    loss, grads = nb.value_and_grad(loss_fn)(params, x, y)
    return loss, [p - g * lr for p, g in zip(params, grads)]

# Setup network (hyper)parameters.
LAYERS = [1, 32, 64, 32, 1]
params = [p for i in range(len(LAYERS) - 1) for p in (nb.glorot_uniform((LAYERS[i], LAYERS[i + 1])), nb.zeros((1, LAYERS[i + 1])),)]

# Run training loop.
x, y = nb.rand((256, 1)), nb.rand((256, 1))
for i in range(1001):
    loss, params = train_step(params, x, y, 0.01)
    if i % 100 == 0: print(i, loss.to_numpy())
```

### For Contributors: Development Branch (main)

> **Working on the new distributed architecture?** Clone main:

```bash
git clone https://github.com/nabla-ml/nabla.git
cd nabla
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

⚠️ **Note**: The `main` branch is under active development. APIs are unstable and the example above won't work.

## Repository Structure

> **For developers working on `main` branch:**  
> See [nabla/README.md](nabla/README.md) for detailed architecture documentation of the new distributed execution model.

## Contributing

Contributions welcome! Discuss significant changes in Issues first. Submit PRs for bugs, docs, and smaller features.

## License

Nabla is licensed under the [Apache-2.0 license](https://github.com/nabla-ml/nabla/blob/main/LICENSE).

---

[![Development Status](https://img.shields.io/badge/status-pre--alpha-red)](https://github.com/nabla-ml/nabla)
[![PyPI version](https://badge.fury.io/py/nabla-ml.svg)](https://badge.fury.io/py/nabla-ml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
