# Nabla - High-Performance Scientific Computing

> Nabla is a high-performance scientific computing and machine learning framework combining imperative and functional APIs. Seamlessly drop custom Mojo kernels into the autodiff engine and automatically shard distributed workloads, built on [Mojo & MAX](https://www.modular.com/max).

> **Active Development**: This is the `main` development branch with distributed SPMD execution and a refined lazy, MAX-native execution model. Read the docs: [https://nablaml.com](https://nablaml.com).

[![Development Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/nabla-ml/nabla)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

---

## Installation

Nabla requires **Modular nightly**.

```bash
python -m venv venv
source venv/bin/activate
pip install --pre --extra-index-url https://whl.modular.com/nightly/simple/ modular nabla-ml
```

**GPU Support**: 
*   **Linux (AMD/NVIDIA)**: Supported natively via Modular MAX.
*   **macOS (Apple Silicon)**: Requires Xcode Metal toolchain (`xcode-select --install`).

---

## Development Setup

Installation of all dependencies (torch/jax for testing, mypy/black for linting, etc.)

```bash
git clone https://github.com/nabla-ml/nabla.git
cd nabla
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

---

## Feature Showcase

<!-- ![alt text](./assets/title_image.png) -->


### 1. Tensors & Autodiff

Define Python functions and compute gradients using trace-based automatic differentiation. [Read more](nabla/core/autograd/README.md)

```python
import nabla

# Use Accelerator (GPU) or CPU for execution
with nabla.default_device(nabla.Accelerator()):
    x = nabla.uniform((4, 8))
    w = nabla.uniform((8, 16))

    # Define loss function
    def compute_loss(x, w):
        return nabla.mean(nabla.relu(x @ w))

    # Compute loss (implicit .realize() on print)
    loss = compute_loss(x, w)
    print("Loss:", loss)

    # Compute gradients via backward replay
    grad_x, grad_w = nabla.grad(compute_loss, argnums=(0, 1))(x, w)
    print("Gradients:", grad_x.shape, grad_w.shape)
```

### 2. SPMD Sharding

Shard tensors on a logical mesh; operations automatically propagate sharding constraints. [Read more](nabla/core/sharding/README.md)

```python
# Define 2×4 device mesh (Logical DP × TP)
mesh = nabla.DeviceMesh("my_mini_pod", (2, 4), ("dp", "tp"))

# Shard x on 'dp' (rows), w on 'tp' (columns)
x = nabla.shard(nabla.uniform((32, 128)), mesh, nabla.P("dp", None))
w = nabla.shard(nabla.uniform((128, 256)), mesh, nabla.P(None, "tp"))

def compute_loss(x, w):
    return nabla.mean(nabla.relu(x @ w))

# Automatic AllReduce is inserted for 'tp' sum
loss = compute_loss(x, w)
print("Loss (Sharded):", loss)
```

### 3. Mojo Integration

Nabla's core strength is its ability to drop down to **Mojo** for high-performance custom kernels, bridging the gap between high-level Python and bare-metal execution. [Read more](nabla/ops/README.md)

**Mojo Kernel (`kernels/custom_kernel.mojo`)**
```mojo
@compiler.register("my_kernel")
struct MyKernel:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ):
        @parameter
        fn add_one[W: Int](idx: IndexList[x.rank]) -> SIMD[x.dtype, W]:
            return x.load[W](idx) + 1

        foreach[add_one, target=target](output, ctx)
```

**Python Usage**
```python
class AddOneOp(nabla.UnaryOperation):
    name = "my_kernel"

    def kernel(self, x, **kwargs):
        # Concise invocation: (func_name, path, inputs, out_types)
        return nabla.call_custom_kernel("my_kernel", "./kernels", x, x.type)

x = nabla.Tensor.constant([1., 2., 3.])
y = AddOneOp()(x)
```

### 4. Distributed Pipeline Parallelism (GPipe)

Define complex distributed schedules like **GPipe** using `vmap` for parallel execution and `ppermute` for explicit data movement. [Read more](nabla/transforms/README.md)

```python
# Parallel execution across 'num_stages'
@nabla.vmap(in_axes=(0, 0), spmd_axis_name="stage")
def stage_compute(x, w): 
    return nabla.relu(x @ w)

def pipeline_step(current_state, fresh_input, weights, mask_0):
    # 1. Compute: Run all stages in parallel
    computed = stage_compute(current_state, weights)

    # 2. Communicate: Shift activations to the next stage (i -> i+1)
    shifted = nabla.ppermute(computed, perm=[(i, (i + 1) % stages) for i in range(stages)])

    # 3. Control: Stage 0 takes fresh input; others take shifted data
    return nabla.where(mask_0, fresh_input, shifted)
```

### 5. Dynamic Shape Compilation

Compile functions once with symbolic dimensions to handle varying input sizes without recompilation.

```python
# Compile once for ANY batch size (dim 0)
@nabla.compile(dynamic_dims={0: {0: "batch"}})
def square(x):
    return x * x

x_small = nabla.uniform((2, 10))
x_large = nabla.uniform((128, 10))

res1 = square(x_small) # Triggers compilation
res2 = square(x_large) # Reuses compiled graph!
```

---

## Architecture Overview

Nabla relies on three core principles:

1.  **Lazy Execution**: Shapes are computed eagerly, but the computation graph is built and compiled only when `.realize()` is called.
    *   [Read more: Operation Pipeline](nabla/README.md)
2.  **Trace-Based Autodiff**: Gradients are computed by tracing the forward pass and replaying operations in reverse.
    *   [Read more: Autograd Engine](nabla/core/autograd/README.md)
3.  **Factor-Based SPMD**: Sharding is propagated using "semantic factors" (e.g., batch, heads) rather than physical mesh axes.
    *   [Read more: Sharding & Solver](nabla/core/sharding/README.md)

---

## Contributing

* **Bugs/Docs**: Submit PR directly.
* **Features**: Open an Issue first.
* **New Ops**: See [nabla/ops/README.md](https://github.com/nabla-ml/nabla/tree/main/nabla/ops/README.md).

## License

Apache-2.0 — see [LICENSE](https://github.com/nabla-ml/nabla/tree/main/LICENSE)