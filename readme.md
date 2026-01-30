# NABLA

> **⚠️ IMPORTANT: Architecture Under Heavy Restructuring**
> 
> **You are viewing the `main` development branch** which contains a **complete rewrite** of Nabla with:
> - Distributed SPMD execution (factor-based sharding)
> - Multi-device/multi-GPU training
> - New graph-based execution model
> 
> **For the current stable release (single-device, non-distributed):**
> - PyPI: `pip install nabla-ml` (v25.7)
> - Source: [nabla/v25.7 branch](https://github.com/nabla-ml/nabla/tree/nabla/v25.7)
> 
> The repository structure and code shown below reflect the **new architecture in development**.

---

Welcome! Nabla is a Machine Learning library for the emerging Mojo/Python ecosystem, featuring:

- Gradient computation the PyTorch way (imperatively via .backward())
- Purely-functional, JAX-like composable function transformations: `grad`, `vmap`, `jit`, etc.
- Custom differentiable CPU/GPU kernels

For tutorials and API reference, visit: [nablaml.com](https://nablaml.com/index.html)

## Installation

```bash
pip install nabla-ml
```

## Quick Start

> **Note**: The following example works with the **v25.7 release** (`pip install nabla-ml`).  
> The `main` branch code is under active development and APIs may differ.

*The most simple, but fully functional Neural Network training setup:*

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

## Repository Structure (New Architecture)

> **Note**: This structure reflects the **new distributed architecture** under development on `main`.  
> The [v25.7 release](https://github.com/nabla-ml/nabla/tree/nabla/v25.7) has a different, simpler structure (non-distributed).

```text
nabla/
├── nabla/                     # Core Python library (NEW ARCHITECTURE)
│   ├── core/                  # Execution engine (Tensor, Graph, Sharding, Autograd)
│   │   ├── tensor/            # Dual-object model (Tensor wrapper + TensorImpl state)
│   │   ├── graph/             # Trace recording, rehydration, OutputRefs
│   │   ├── sharding/          # Factor-based SPMD propagation, DeviceMesh
│   │   └── autograd/          # Backward pass engine (BackwardEngine, VJP rules)
│   ├── ops/                   # Operations (arithmetic, reductions, communication, views)
│   │   ├── communication/     # Collective ops (AllReduce, AllGather, etc.)
│   │   └── view/              # Metadata-only ops (reshape, squeeze, etc.)
│   └── transforms/            # Function transformations (grad, vmap, shard_map, compile)
├── tests/                     # Comprehensive test suite
├── tutorials/                 # Notebooks on Nabla usage for ML tasks
└── examples/                  # Example scripts for common use cases
```

**Key architectural changes** from v25.7:
- **Distributed execution**: Tensors can be sharded across multiple devices
- **Factor-based sharding**: Operations describe transformations via factor notation (e.g., `"m k, k n -> m n"`)
- **Eager SPMD**: Sharding decisions and communication happen per-operation, not via compilation
- **Graph tracing**: All operations recorded for autograd and JIT compilation
- **Physical execution model**: Operations implement `physical_execute` to work on `TensorValue` shards

See [nabla/README.md](nabla/README.md) for detailed architecture documentation.

## Contributing

Contributions welcome! Discuss significant changes in Issues first. Submit PRs for bugs, docs, and smaller features.

## License

Nabla is licensed under the [Apache-2.0 license](https://github.com/nabla-ml/nabla/blob/main/LICENSE).

---

[![Development Status](https://img.shields.io/badge/status-pre--alpha-red)](https://github.com/nabla-ml/nabla)
[![PyPI version](https://badge.fury.io/py/nabla-ml.svg)](https://badge.fury.io/py/nabla-ml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
