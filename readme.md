# Nabla: Distributed Deep Learning Framework

> **A JAX-inspired autodiff library with factor-based SPMD sharding, built on [Modular MAX](https://www.modular.com/max).**

[![Development Status](https://img.shields.io/badge/status-pre--alpha-red)](https://github.com/nabla-ml/nabla)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> 
> **Active Development**: This is the `main` development branch with distributed SPMD execution and a new lazy, graph-based execution model. For the older single-device release (v25.7), see [`pip install nabla-ml`](#stable-release-v257).

---

## Quick Examples

### Forward Pass & Autodiff

```python
import nabla
from max.driver import Accelerator, CPU

# Run on CPU or GPU
with nabla.default_device(CPU()):  # Use Accelerator() for GPU
    x = nabla.uniform((4, 8))
    w = nabla.uniform((8, 16))

    # Define loss function
    def compute_loss(x, w):
        return nabla.mean(nabla.relu(x @ w))

    # Compute loss
    loss = compute_loss(x, w)
    print("Loss:", loss)  # Implicitly triggers .realize()

    # Compute gradients
    grad_x, grad_w = nabla.grad(compute_loss, argnums=(0, 1))(x, w)
    print("Gradients:", grad_x.shape, grad_w.shape)
```

### SPMD Sharding

```python
import nabla

# Define 2×4 device mesh
mesh = nabla.DeviceMesh("mesh", (2, 4), ("dp", "tp"))

# Create and shard tensors
x = nabla.uniform((32, 128))
w = nabla.uniform((128, 256))

x_sharded = nabla.shard(x, mesh, nabla.P("dp", None))  # data parallel
w_sharded = nabla.shard(w, mesh, nabla.P(None, "tp"))   # tensor parallel

# Define loss function
def compute_loss(x, w):
    return nabla.mean(nabla.relu(x @ w))

# Compute with automatic communication
loss = compute_loss(x_sharded, w_sharded)
print("Loss:", loss)
```

---

## Development Setup

### Prerequisites

- **Python 3.12+**
- **Modular MAX SDK** (automatically installed via `requirements.txt`)

### Clone and Install

```bash
git clone https://github.com/nabla-ml/nabla.git
cd nabla
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

### GPU Setup

**AMD/NVIDIA (Linux)**: Works out of the box with Modular MAX.

**Apple Silicon (macOS)**: Requires Xcode with Metal toolchain:

```bash
# Verify developer directory
xcode-select -p  # Should be /Applications/Xcode.app/Contents/Developer
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer  # if not

# Install Metal toolchain (if missing)
xcrun -sdk macosx metal  # Should output "no input files"
xcodebuild -downloadComponent MetalToolchain  # if "cannot execute tool 'metal'"
```

### Running Tests

```bash
pytest tests/
```

---

## Architecture Overview

| Principle | Description |
|-----------|-------------|
| **Eager Metadata, Deferred Graph** | Shapes/dtypes computed immediately; MAX graph built lazily on `evaluate()` |
| **Trace-Based Autodiff** | Gradients computed by replaying `OpNode` traces backward (not per-tensor tape) |
| **Factor-Based SPMD** | Sharding uses semantic factors (`m`, `k`, `n`) not raw dimension indices |

### Tensor / TensorImpl (Dual Object Model)

```
Tensor (User API)              TensorImpl (Internal State)
├── .shape, .dtype             ├── _graph_values: list[TensorValue]
├── .realize()                 ├── _buffers: list[driver.Buffer]
├── arithmetic ops             ├── graph_values_epoch: int
└── wraps ──────────────────►  ├── sharding: ShardingSpec
                               ├── output_refs: OpNode
                               └── batch_dims: int
```

**Why two objects?** 
1. **Multi-output support**: Operations like `split` produce multiple `Tensor` objects sharing one `OpNode`.
2. **Lifetime Management**: Decouples the user-facing `Tensor` from the underlying `TensorImpl`. The engine manages `TensorImpl` lifetimes via weakrefs, ensuring efficient memory management even in complex cyclic graphs.

→ Deep dive: [nabla/core/tensor/README.md](nabla/core/tensor/README.md)

### Operation Execution (9-Step Pipeline)

Every operation flows through `Operation.__call__()`:

```
┌───────────────────────────────────────────────────────────────────────────┐
│  1. METADATA          Collect batch_dims, tracing state, sharding info    │
│  2. RESHARD           Insert AllGather/AllToAll if inputs need reshaping  │
│  3. HASH              Compute structural hash for compiled model cache    │
│  4. SHAPE INFERENCE   Compute output shapes (ALWAYS runs eagerly)         │
│  5. EXECUTE           Build MAX graph (deferred by default)               │
│  6. PACKAGE           Create output Tensor (promise or realized)          │
│  7. TRACE             Create OpNode linking outputs → inputs              │
│  8. AUTO-REDUCE       Insert AllReduce if contracting dims were sharded   │
│  9. JVP               Propagate tangents for forward-mode autodiff        │
└───────────────────────────────────────────────────────────────────────────┘
```

Steps 1-4 and 6-9 always run. Step 5 depends on execution mode.

**Why Operation-based tracing?** We trace `OpNode`s (operations) rather than `Tensor`s. This cleanly handles operations with multiple outputs (like SVD or Split), which would be messy and complex with tensor-based tracing (tapes).

→ Deep dive: [nabla/ops/README.md](nabla/ops/README.md)

### Execution Modes

| Mode | `NABLA_EAGER_MAX_GRAPH` | Behavior |
|------|-------------------------|----------|
| **Deferred** (default) | `0` | Graph building delayed until `.realize()`. Enables compiled model caching. |
| **Eager** | `1` | MAX graph nodes built immediately. Useful for debugging. |

> **Crucial Distinction**: "Eager" here refers only to **MAX graph building**. Nabla is **fundamentally lazy**—execution always involves compiling the graph to the MAX engine. Eager mode just builds that graph node-by-node, whereas Deferred mode builds it all at once during `evaluate()`.

```bash
export NABLA_EAGER_MAX_GRAPH=1        # Enable eager mode
export NABLA_VERIFY_EAGER_SHAPES=1    # Validate shape inference
```

### Promise Tensors → Graph Evaluation

Operations create **promise tensors** (shape known, graph deferred):

```
y = x @ w           →  y.shape available immediately
                       y._impl.graph_values_epoch = -1 (promise marker)
                       y._impl.output_refs = OpNode(op=MatmulOp, inputs=[x, w])

y.realize()         →  GRAPH.evaluate(y):
                       1. Cache lookup by structural hash
                       2. HIT: run cached model, skip graph building
                       3. MISS: replay OpNode trace → build MAX graph → compile
```

→ Deep dive: [nabla/core/graph/README.md](nabla/core/graph/README.md)

### Gradient Computation (Reverse-Mode Autodiff)

```
nabla.grad(loss_fn)(x)  →  1. Trace forward, capturing OpNode DAG
                           2. Evaluate forward to get loss value
                           3. Backward: reversed(trace), call op.vjp_rule() per node
                           4. Accumulate cotangents (sum for multi-use tensors)
```

→ Deep dive: [nabla/core/autograd/README.md](nabla/core/autograd/README.md)

### Factor-Based Sharding (SPMD)

Nabla uses **semantic factors** for sharding propagation, deeply inspired by **XLA's novel Shardy compiler system**.

**Why Factor-Based?**
- **Flexibility**: More expressive than traditional GSPMD-style sharding.
- **Future-Proof**: Enables global sharding optimization based on constraints (propagation is just solving constraints).

Example:

```
Matmul factors: "m k, k n → m n"
   m = batch/rows      → shard freely
   k = contracting     → sharding requires AllReduce  
   n = output columns  → shard freely
```

Three-phase propagation per operation: **COLLECT** (input dims → factors) → **RESOLVE** (handle conflicts) → **UPDATE** (factors → output dims, insert communication).

→ Deep dive: [nabla/core/sharding/README.md](nabla/core/sharding/README.md)

---

## Additional Features

Nabla offers more than covered above:

| Feature | Description | See |
|---------|-------------|-----|
| **vmap** | Automatic vectorization via `batch_dims` tracking | [transforms/vmap.py](nabla/transforms/vmap.py) |
| **Dynamic dims** | Compile functions with symbolic dimension support | [transforms/compile.py](nabla/transforms/compile.py) |
| **Control flow** | `cond`, `while_loop`, `scan` for differentiable control | [ops/control_flow.py](nabla/ops/control_flow.py) |
| **Pytree utilities** | General python-tree operations | [core/common/pytree.py](nabla/core/common/pytree.py) |

**Current sharding model**: Explicit, user-controlled via `nabla.shard()`. You decide partition specs.

**Coming soon**: `shard_map` transform — define sharding constraints, let nabla handle propagation automatically.

### Interesting Test Cases

- **DP+PP distributed MLP** (WIP): [tests/integration/autograd/refactored/test_pp_grad3.py](tests/integration/autograd/refactored/test_pp_grad3.py)
- **vmap + sharding**: [tests/unit_v2/test_vmap_sharding.py](tests/unit_v2/test_vmap_sharding.py)
- **vmapped gradients**: [tests/integration/autograd/refactored/test_vmapped.py](tests/integration/autograd/refactored/test_vmapped.py)
- **Pipeline parallel transformer**: [tests/integration/test_pp_transformer.py](tests/integration/test_pp_transformer.py)

---

## Module Structure

```
nabla/
├── core/
│   ├── tensor/          # Tensor/TensorImpl dual model
│   ├── graph/           # GRAPH singleton, OpNode, tracing
│   ├── autograd/        # grad(), backward engine
│   └── sharding/        # DeviceMesh, factor propagation
├── ops/
│   ├── base.py          # Operation.__call__() pipeline
│   ├── binary.py        # add, matmul, etc.
│   ├── unary.py         # relu, exp, etc.
│   ├── communication/   # AllReduce, shard, etc.
│   └── control_flow.py  # cond, scan, while_loop
└── transforms/
    ├── vmap.py          # Automatic batching
    ├── shard_map.py     # SPMD distribution
    └── compile.py       # JIT compilation
```

→ Each submodule has its own `README.md`.

---

## Key Files for Contributors

| To understand... | Start here |
|-----------------|------------|
| Op execution | [nabla/ops/base.py](nabla/ops/base.py) → `Operation.__call__()` |
| Autodiff | [nabla/core/autograd/utils.py](nabla/core/autograd/utils.py) → `backward_on_trace()` |
| Trace replay | [nabla/core/graph/tracing.py](nabla/core/graph/tracing.py) → `Trace.refresh_graph_values()` |
| Adding ops | [nabla/ops/README.md](nabla/ops/README.md) |
| Sharding | [nabla/core/sharding/README.md](nabla/core/sharding/README.md) |

---

## Stable Release (v25.7)

<details>
<summary>pip install nabla-ml (single-device, stable API)</summary>

```bash
pip install nabla-ml
```

From [nabla/v25.7 branch](https://github.com/nabla-ml/nabla/tree/nabla/v25.7):

```python
import nabla

def loss_fn(params, x, y):
    for i in range(0, len(params) - 2, 2):
        x = nabla.relu(x @ params[i] + params[i + 1])
    return nabla.mean((x @ params[-2] + params[-1] - y) ** 2)

@nabla.jit(auto_device=True)
def train_step(params, x, y, lr):
    loss, grads = nabla.value_and_grad(loss_fn)(params, x, y)
    return loss, [p - g * lr for p, g in zip(params, grads)]
```

[![PyPI version](https://badge.fury.io/py/nabla-ml.svg)](https://badge.fury.io/py/nabla-ml)

</details>

---

## Contributing

Contributions welcome! For significant changes, please open an Issue first to discuss.

- **Bug fixes / docs**: Submit PR directly
- **New features**: Discuss in Issues before implementing
- **New ops**: Follow the pattern in [nabla/ops/README.md](nabla/ops/README.md)

## License

Apache-2.0 — see [LICENSE](LICENSE)
