# Nabla Architecture & Core Logic

Nabla is a hybrid Machine Learning library designed for the emerging Mojo/Python ecosystem. It bridges the gap between **PyTorch's imperative usability** and **JAX's functional power**, all powered by the high-performance **MAX** compiler backend.

## 1. Core Philosophy

Nabla is built on three pillars:
1.  **Imperative Frontend**: A `Tensor` object that feels like PyTorch (eager execution by default, mutable state).
2.  **Functional Transformations**: Composable transformations like `grad`, `jit`, and `vmap` inspired by JAX.
3.  **Compiled Backend**: Seamless integration with the Modular Accelerated Xecution (MAX) engine for hardware acceleration.

## 2. System Architecture

The library is organized into four main layers:

### 2.1. Core (`nabla.core`)
The heart of the library.
-   **`Tensor`**: The central data structure. It wraps two possible backends:
    -   **Eager Backend**: `numpy.ndarray` for immediate CPU execution (debugging, dynamic control flow).
    -   **Compiled Backend**: `max.driver.Tensor` for accelerated execution on CPU/GPU.
    -   **Autograd State**: Stores `grad`, `requires_grad`, and graph tracing information (`tangent`, `cotangent`, `vjp_rule`).
-   **`GraphTracer`**: Implements a DFS-based tracing mechanism to capture the computation graph from a set of output tensors. It generates a cache key for JIT compilation.
-   **`ModelFactory`**: Converts the traced Nabla graph into a **MAX Graph**. It handles:
    -   Input/Output typing.
    -   Constant folding (converting non-dynamic inputs to constants).
    -   Operation mapping (Nabla Ops -> MAX Ops).

### 2.2. Operations (`nabla.ops`)
Defines the mathematical primitives.
-   **`Operation` Base Class**: Enforces a strict contract for every op:
    -   `forward()`: Computes the result (eagerly or symbolically).
    -   `maxpr()`: Defines how the op is lowered to the MAX graph.
    -   `eagerxpr()`: Defines how the op is executed eagerly via NumPy.
    -   `vjp_rule()`: Defines the Vector-Jacobian Product for reverse-mode autodiff (`grad`).
    -   `jvp_rule()`: Defines the Jacobian-Vector Product for forward-mode autodiff.
-   **Registration**: Ops are stateless singletons (e.g., `_add_op`) invoked by functional wrappers (e.g., `nabla.add`).

### 2.3. Transformations (`nabla.transforms`)
Implements functional transformations via a unified tracing system.
-   **`grad(fn)`**:
    1.  Wraps inputs in `Tensor` with `traced=True`.
    2.  Executes `fn` to build a computation trace.
    3.  Performs a backward pass on the trace using `vjp_rule`s.
    4.  Returns the gradients.
-   **`jit(fn)`**:
    1.  Traces `fn` to build a graph.
    2.  Hashes the graph to check the compilation cache.
    3.  If miss: Compiles the graph to a MAX `Model` using `ModelFactory`.
    4.  Executes the compiled model.
-   **`vmap(fn)`**:
    1.  Uses `batch_dims` tracking on Tensors.
    2.  Ops automatically handle batch broadcasting (e.g., `broadcast_batch_dims`).
    3.  Allows writing code for a single sample and running it on a batch.

### 2.4. Neural Networks (`nabla.nn`)
High-level abstractions built on top of Core and Ops.
-   **`Module`**: A PyTorch-like base class.
    -   Auto-registers `Parameter`s and `Buffer`s.
    -   Supports `state_dict` loading/saving.
    -   `compile()` method: JIT-compiles the module's `forward` pass for maximum performance.

## 3. Deep Dive: The Life of a Tensor

When you create a tensor `x = nb.tensor([1, 2, 3])`:
1.  It starts backed by a NumPy array (Eager).
2.  If you call `y = x * 2`:
    -   `MulOp` is invoked.
    -   It executes `np.multiply` immediately.
    -   Returns a new Tensor `y`.

**When `jit` is involved:**
1.  `@nb.jit` wraps a function.
2.  When called, it passes **Tracer Tensors** (no data, just shape/dtype).
3.  Operations like `x * 2` don't execute NumPy. Instead, they record themselves in the `Tensor`'s history (`creator_op`).
4.  The result is a graph of Tensors.
5.  `GraphTracer` walks this graph.
6.  `ModelFactory` converts it to a MAX Graph.
7.  The MAX Graph is compiled to machine code.
8.  Real data is fed into the compiled model.

**When `grad` is involved:**
1.  `@nb.grad` wraps a function.
2.  It executes the function (tracing the graph).
3.  It triggers `backward()` on the output.
4.  `backward()` walks the graph in reverse (topological sort).
5.  For each node, it calls `node.op.vjp_rule(inputs, cotangent, output)`.
6.  Gradients accumulate in the leaves.

## 4. Implementation Details

-   **Dual-Mode Execution**: Every op *must* implement both `eagerxpr` (NumPy) and `maxpr` (MAX). This ensures you can debug in Python but deploy with MAX.
-   **Lazy Evaluation**: Tensors can be in a "staged" state (`stage_realization=True`), meaning they exist as nodes in a graph but haven't been computed yet. `realize()` forces computation.
-   **Custom Kernels**: The architecture supports plugging in custom Mojo/C++ kernels via `custom_kernel_path` in Ops.

## 5. Directory Structure

```text
nabla/
├── core/
│   ├── tensor.py          # The God Object (Tensor)
│   ├── graph_execution.py # Tracing & Compilation logic
│   └── execution_context.py # Global state management
├── ops/
│   ├── operation.py       # Base class for all Ops
│   ├── binary.py          # Add, Mul, Sub, Div, etc.
│   ├── unary.py           # Sin, Cos, Exp, Log, etc.
│   └── ...
├── transforms/
│   ├── grad.py            # Gradient transformation
│   ├── jit.py             # Just-In-Time compilation
│   ├── vmap.py            # Vectorization
│   └── utils.py           # Autodiff engine (backward pass)
└── nn/
    ├── modules/           # Layers (Linear, Conv, etc.)
    └── functional/        # Functional interface for layers
```
