# Nabla: High-Performance Distributed ML

Nabla is a JAX-inspired autodiff library with factor-based SPMD sharding, built on [Mojo & MAX](https://www.modular.com/max).

```{toctree}
:maxdepth: 2
:hidden:

api/index
tutorials/index
```

---

## Installation

Nabla requires **Modular nightly**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --pre --extra-index-url https://whl.modular.com/nightly/simple/ modular nabla-ml
```

---

## Feature Showcase

### 1. Tensors & Autodiff

Define Python functions and compute gradients using trace-based automatic differentiation. [Explore the code.](https://github.com/nabla-ml/nabla/tree/main/nabla/core/autograd/README.md)

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

Shard tensors on a logical mesh; operations automatically propagate sharding constraints. [Explore the code.](https://github.com/nabla-ml/nabla/tree/main/nabla/core/sharding/README.md)

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

Nabla's core strength is its ability to drop down to **Mojo** for high-performance custom kernels, bridging the gap between high-level Python and bare-metal execution. [Explore the code.](https://github.com/nabla-ml/nabla/tree/main/nabla/ops/README.md)

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

Define complex distributed schedules like **GPipe** using `vmap` for parallel execution and `ppermute` for explicit data movement. [Explore the code.](https://github.com/nabla-ml/nabla/tree/main/nabla/transforms/README.md)

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
    *   [Explore the code.: Operation Pipeline](https://github.com/nabla-ml/nabla/tree/main/nabla/README.md)
2.  **Trace-Based Autodiff**: Gradients are computed by tracing the forward pass and replaying operations in reverse.
    *   [Explore the code.: Autograd Engine](https://github.com/nabla-ml/nabla/tree/main/nabla/core/autograd/README.md)
3.  **Factor-Based SPMD**: Sharding is propagated using "semantic factors" (e.g., batch, heads) rather than physical mesh axes.
    *   [Explore the code.: Sharding & Solver](https://github.com/nabla-ml/nabla/tree/main/nabla/core/sharding/README.md)

---

## Development Setup

### Prerequisites
* **Python 3.12+**
* **Modular MAX SDK** (via `requirements.txt`)

### Installation
```bash
git clone https://github.com/nabla-ml/nabla.git
cd nabla
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

**GPU Support**: 
*   **Linux (AMD/NVIDIA)**: Supported natively via Modular MAX.
*   **macOS (Apple Silicon)**: Requires Xcode Metal toolchain (`xcode-select --install`).

---

## Contributing

* **Bugs/Docs**: Submit PR directly.
* **Features**: Open an Issue first.
* **New Ops**: See [nabla/ops/README.md](https://github.com/nabla-ml/nabla/tree/main/nabla/ops/README.md).

## License

Apache-2.0 — see [LICENSE](https://github.com/nabla-ml/nabla/tree/main/LICENSE)