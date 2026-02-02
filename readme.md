# Nabla: Distributed Deep Learning Framework

![alt text](./assets/image-1.png)

> **A JAX-inspired autodiff library with factor-based SPMD sharding, built on [Modular MAX](https://www.modular.com/max).**

[![Development Status](https://img.shields.io/badge/status-pre--alpha-red)](https://github.com/nabla-ml/nabla)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> 
> **Active Development**: This is the `main` development branch with distributed SPMD execution and a refined lazy, MAX-native execution model. For the older single-device release (v25.7), see [`pip install nabla-ml`](#stable-release-v257).

---

## Quick Examples

### Forward Pass & Autodiff

```python
import nabla
from max.driver import Accelerator, CPU

# Run on CPU or GPU
with nabla.default_device(Accelerator()):  # Alternatively, use CPU()
    x = nabla.uniform((4, 8))
    w = nabla.uniform((8, 16))

    # Define loss function
    def compute_loss(x, w):
        return nabla.mean(nabla.relu(x @ w))

    # Compute loss
    loss = compute_loss(x, w)
    print("Loss:", loss)  # Implicitly triggers .realize()

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

x_sharded = nabla.shard(x, mesh, nabla.P("dp", None))  # data parallel
w_sharded = nabla.shard(w, mesh, nabla.P(None, "tp"))  # tensor parallel

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

* **Python 3.12+**
* **Modular MAX SDK** (automatically installed via `requirements.txt`)

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
xcode-select -p  # Should be /Applications/Xcode.app/Contents/Developer
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer  # if not

# Install Metal toolchain (if missing)
xcrun -sdk macosx metal  # Should output "no input files"
xcodebuild -downloadComponent MetalToolchain  # if "cannot execute tool 'metal'"

```

---

## Architecture Overview

| Principle | Description |
| --- | --- |
| **Lazy Execution** | Shapes/dtypes are computed immediately, but the MAX-Graph is compiled/executed only on `.realize()`. |
| **Trace-Based Autodiff** | Gradients computed by replaying `OpNode` traces backward (only traced when needed). |
| **Factor-Based SPMD** | Sharding uses semantic factors (`m`, `k`, `n`) not raw dimension indices. |

### Tensor / TensorImpl (Dual Object Model)

```
Tensor (User API)              TensorImpl (Internal State)
├── .shape(s), .device(s)      ├── _graph_values: list[max.TensorValue]
├── .realize()                 ├── _buffers: list[max.Buffer]
├── arithmetic ops             ├── output_refs: OpNode
└── wraps ──────────────────►  ├── sharding: ShardingSpec
                               ├── is_traced: bool
                               └── batch_dims: int

```

**Why two objects for the same thing?** -> **Lifetime Management**: Decouples the user-facing `Tensor` from the underlying `TensorImpl`. The engine manages `TensorImpl` lifetimes via weakrefs, ensuring efficient memory management even in complex cyclic graphs.

→ Deep dive: [nabla/core/tensor/README.md](https://github.com/nabla-ml/nabla/tree/main/nabla/core/tensor/README.md)

### Operation Execution (The Pipeline)

Every operation on Tensor(s) flows through `Operation.__call__()`. Note that **most steps are optional** and only run when necessary (e.g., during training or distributed execution).

```
┌────────────────────────────────────────────────────────────────────────────┐
│  1. METADATA (Always) Collect batch_dims, tracing state, sharding info     │
│  2. RESHARD  (Cond)   Insert comms if input sharding !match op requirement │
│  3. HASH     (Always) Compute structural hash for compiled model cache     │
│  4. BUILD IR (Cond)   Build MAX graph nodes (if in Immediate Mode)         │
│  5. SHAPE    (Cond)   Infer output shapes (if in Deferred Mode)            │
│  6. PACKAGE  (Always) Create output Tensor from shapes/TensorValues        │
│  7. TRACE    (Cond)   Create OpNode (if is_traced OR Deferred Mode)        │
│  8. REDUCE   (Cond)   Insert AllReduce if contracting dims were sharded    │
│  9. JVP      (Cond)   Propagate tangents (if forward-mode AD active)       │
└────────────────────────────────────────────────────────────────────────────┘

```

**When is an `OpNode` created? (Step 7)**
We avoid the overhead of creating graph nodes unless absolutely necessary. An `OpNode` is only instantiated if:

1. The tensor is actively being traced (e.g., inside `nabla.grad`).
2. We are in **Deferred IR** mode (see below), where we need the trace to build the graph later.

In a standard eager inference run (Immediate IR mode), `OpNode` creation is skipped entirely, making dispatch extremely lightweight.

### Graph Construction Modes

Nabla is **fundamentally lazy**—execution always involves compiling a graph for the MAX engine. However, you can control *when* the Intermediate Representation (IR) is constructed.

| Mode | Env Var | Behavior |
| --- | --- | --- |
| **Deferred IR** (default) | `EAGER_MAX_GRAPH=0` | MAX Graph building is delayed until `.realize()`. We trace a lightweight `OpNode` graph first. Enables faster model caching and whole-graph optimizations. |
| **Immediate IR** | `EAGER_MAX_GRAPH=1` | MAX graph nodes (TensorValues) are built immediately during the forward pass. Useful for debugging shapes/dtypes at the exact line of failure. |

```bash
export EAGER_MAX_GRAPH=1     
export VERIFY_EAGER_SHAPES=1   

```

→ Deep dive: [nabla/ops/README.md](https://github.com/nabla-ml/nabla/tree/main/nabla/ops/README.md)

### Factor-Based Sharding (SPMD)

Nabla uses **semantic factors** for sharding propagation (inspired by XLA's Shardy), rather than raw dimension indices. This decouples the *physical mesh* from the *logical operation*.

**The Translation Bridge:**

1. **User**: Shards a tensor axis on mesh dimension `"dp"` (e.g., `x.shard(mesh, P("dp", None))`).
2. **Op Rule**: The operation (e.g., Matmul) maps that input axis to a semantic factor (e.g., `m`).
3. **Solver**: The engine propagates that factor to the output.

**Example: Matrix Multiplication**
Rule: `"m k, k n → m n"`

| Factor | Semantics | User Input | Engine Action |
| --- | --- | --- | --- |
| **`m`** | Batch/Rows | Sharded on `"dp"` | **Propagates**: Output `m` is also on `"dp"`. |
| **`k`** | Contracting | Sharded on `"tp"` | **Solves**: Insert `AllReduce` (sum partials). |
| **`n`** | Columns | Replicated | **Propagates**: Output `n` is replicated. |

**Three-Phase Propagation:**
**COLLECT** (Input Dims → Factors) → **RESOLVE** (Solve Conflicts) → **UPDATE** (Factors → Output Dims).

→ **Deep Dive**: [nabla/core/sharding/README.md](https://github.com/nabla-ml/nabla/tree/main/nabla/core/sharding/README.md)

---

### Example Feature: Gradient Computation

```
nabla.grad(loss_fn)(x)  →  1. Trace forward, capturing OpNode DAG
                           2. Evaluate forward to get loss value
                           3. Backward: reversed(trace), call op.vjp_rule() per node
                           4. Accumulate cotangents (sum for multi-use tensors)

```

→ Deep dive: [nabla/core/autograd/README.md](https://github.com/nabla-ml/nabla/tree/main/nabla/core/autograd/README.md)

## Additional Features

Nabla offers more than covered above:

| Feature | Description | See |
| --- | --- | --- |
| **vmap** | Automatic vectorization via `batch_dims` tracking | [transforms/vmap.py](https://github.com/nabla-ml/nabla/tree/main/nabla/transforms/vmap.py) |
| **Dynamic dims** | Compile functions with symbolic dimension support | [transforms/compile.py](https://github.com/nabla-ml/nabla/tree/main/nabla/transforms/compile.py) |
| **Control flow** | `cond`, `while_loop`, `scan` for differentiable control | [ops/control_flow.py](https://github.com/nabla-ml/nabla/tree/main/nabla/ops/control_flow.py) |
| **Pytree utilities** | General python-tree operations | [core/common/pytree.py](https://github.com/nabla-ml/nabla/tree/main/nabla/core/common/pytree.py) |

**Current sharding model**: Explicit, user-controlled via `nabla.shard()`. The user decides partition specs.

**WIP**: `shard_map` transform — define sharding constraints, let nabla handle propagation automatically. See [nabla/transforms/shard_map.py](https://github.com/nabla-ml/nabla/tree/main/nabla/transforms/shard_map.py).

### Interesting Test Cases

* **DP+PP distributed MLP** (WIP): [tests/integration/autograd/refactored/test_pp_grad3.py](https://github.com/nabla-ml/nabla/tree/main/tests/integration/autograd/refactored/test_pp_grad3.py)
* **vmap + sharding**: [tests/unit/test_vmap_sharding.py](https://github.com/nabla-ml/nabla/tree/main/tests/unit/test_vmap_sharding.py)
* **vmapped gradients**: [tests/integration/autograd/refactored/test_vmapped.py](https://github.com/nabla-ml/nabla/tree/main/tests/integration/autograd/refactored/test_vmapped.py)
* **Pipeline parallel transformer**: [tests/integration/test_pp_transformer.py](https://github.com/nabla-ml/nabla/tree/main/tests/integration/test_pp_transformer.py)

---

## Module Structure

```
nabla/
├── config.py            # Global environment settings
├── core/
│   ├── tensor/          # Tensor/TensorImpl dual model
│   ├── graph/           # GRAPH singleton, OpNode, tracing
│   ├── autograd/        # trace-based backward engine
│   ├── sharding/        # DeviceMesh, factor propagation
│   └── common/          # Pytree, context management
├── ops/
│   ├── base.py          # Operation.__call__() pipeline
│   ├── binary.py        # add, matmul, etc.
│   ├── unary.py         # relu, exp, etc.
│   ├── creation.py      # full, zeros, ones
│   ├── reduction.py     # mean, reduce_sum
│   ├── view/            # reshape, transpose, indexing
│   ├── communication/   # AllReduce, shard, all_gather
│   ├── control_flow.py  # cond, scan, while_loop
│   ├── multi_output.py  # split, unbind
│   └── custom_op.py     # custom MAX-kernel integration
└── transforms/
    ├── vmap.py          # Automatic batching
    ├── shard_map.py     # SPMD distribution
    └── compile.py       # Static JIT compilation
```

→ Each submodule has its own `README.md`.

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

</details>

---

## Contributing

Contributions welcome! For significant changes, please open an Issue first to discuss.

* **Bug fixes / docs**: Submit PR directly
* **New features**: Discuss in Issues before implementing
* **New ops**: Follow the pattern in [nabla/ops/README.md](https://github.com/nabla-ml/nabla/tree/main/nabla/ops/README.md)

## License

Apache-2.0 — see [LICENSE](https://github.com/nabla-ml/nabla/tree/main/LICENSE)