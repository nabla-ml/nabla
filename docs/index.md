# Nabla: High-Performance Distributed ML

Nabla is a JAX-inspired autodiff library with factor-based SPMD sharding, built on [Mojo & MAX](https://www.modular.com/max).

```{toctree}
:maxdepth: 2
:hidden:

api/index
tutorials/index
```

## Core Principles

1.  **Lazy Execution**: Shapes are computed eagerly, but the computation graph is built and compiled only when `.realize()` is called.
2.  **Trace-Based Autodiff**: Gradients are computed by tracing the forward pass and replaying operations in reverse via the `grad` transform.
3.  **Factor-Based SPMD**: Sharding is propagated using "semantic factors" (e.g., batch, heads) rather than physical mesh axes.

---

## Feature Showcase

### 1. Tensors & Autodiff

```python
import nabla

# Use Accelerator (GPU) or CPU for execution
with nabla.default_device(nabla.Accelerator()):
    x = nabla.uniform((4, 8))
    w = nabla.uniform((8, 16))

    def compute_loss(x, w):
        return nabla.mean(nabla.relu(x @ w))

    # Compute gradients via grad transform
    grad_fn = nabla.grad(compute_loss, argnums=(0, 1))
    grad_x, grad_w = grad_fn(x, w)
```

### 2. SPMD Sharding

```python
# Define 2x4 device mesh (Logical DP x TP)
mesh = nabla.DeviceMesh("my_mesh", (2, 4), ("dp", "tp"))

# Shard x on 'dp' (rows), w on 'tp' (columns)
x = nabla.shard(nabla.uniform((32, 128)), mesh, nabla.P("dp", None))
w = nabla.shard(nabla.uniform((128, 256)), mesh, nabla.P(None, "tp"))

def compute_loss(x, w):
    return nabla.mean(nabla.relu(x @ w))

# Automatic AllReduce is inserted for 'tp' sum
loss = compute_loss(x, w)
```

### 3. Mojo Integration

Nabla allows dropping down to **Mojo** for high-performance custom kernels.

```python
class AddOneOp(nabla.UnaryOperation):
    name = "my_kernel"

    def kernel(self, x, **kwargs):
        return nabla.call_custom_kernel("my_kernel", "./kernels", x, x.type)

x = nabla.Tensor.constant([1., 2., 3.])
y = AddOneOp()(x)
```

---

## Installation

```bash
pip install nabla-ml
```

## Contributing

Nabla is open-source and welcomes contributions! See our [GitHub repository](https://github.com/nabla-ml/nabla) for more details.

License: Apache-2.0