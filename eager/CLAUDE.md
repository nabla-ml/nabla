# Eager Module Architecture

This document comprehensively explains the architecture of the `eager` module. It is designed to implement a **Lazy Eager** framework, targeting the usability of PyTorch with the optimization capabilities of JAX.

## 1. Project Structure

The module is flat and modular, designed to minimize circular dependencies while keeping related logic distinct.

```text
eager/
├── tensor.py          # User Functionality: Public API (Tensor class)
├── tensor_impl.py     # System Logic: State & Data (TensorImpl class)
├── ops.py             # Dispatcher: Base Operation class & Autodiff logic
├── binary_ops.py      # Implementation: Binary ops (Add, Mul...)
├── creation.py        # Implementation: Creation ops (Zeros, Ones, Random...)
├── multi_output_ops.py # Implementation: Multi-output ops (Split, TopK...)
├── compute_graph.py   # Engine: Graph compilation & execution manager
├── context.py         # Utilities: Thread-local settings (device, dtype)
├── pytree.py          # Utilities: JAX-compatible tree flattening/unflattening
├── sharding.py        # Sharding: Core state definitions (DeviceMesh, ShardingSpec)
└── sharding_propagation.py # Sharding: Propagation logic and templates
```

## 2. Core Philosophy

The `eager` module separates the **API (User View)** from the **Execution (System View)**.

-   **User View**: "I am creating arrays and adding them immediately."
-   **System View**: "I am recording a graph of operations. I will only compile and run them when the user asks for the concrete data."

This **Lazy Eager** approach allows us to:
1.  **Capture the full graph** without `torch.compile` or `jax.jit` decorators.
2.  **Optimize globally** using the MAX compiler stack.
3.  **Retain imperative debugging**: Users can insert `print()` or breakpoints, and graph construction pauses for Python execution.

## 3. Architecture Deep Dive

### 3.1 The Four Pillars

#### 1. `Tensor` (The Facade)
*   **Role**: User-facing object mimicking `torch.Tensor`.
*   **Behavior**: Stateless wrapper referencing `_impl`. Delegates math (`__add__`) to `ops` module.

#### 2. `TensorImpl` (The Brain)
*   **Location**: `eager/tensor_impl.py`
*   **Role**: Internal state container.
*   **State Machine**:
    *   **Unrealized**: Holds `_values` (Symbolic MAX `TensorValue` nodes).
    *   **Realized**: Holds `_storages` (Concrete `driver.Tensor` data).
*   **Tracing Mode (`traced=True`)**:
    *   Essential for Autodiff and Sharding.
    *   Stores `op` (Operation instance) and `op_args` (Inputs that created this tensor).
    *   Stores `tangent` / `cotangent` (TensorImpls for gradients).
*   **Untraced Mode (Default)**:
    *   Optimized for inference.
    *   No history stored (`op_args` is None). Aggressive GC of intermediate tensors.

#### 3. `Operation` (The Dispatcher)
*   **Location**: `eager/ops.py`
*   **Role**: Stateless singleton defining a transformation.
*   **The Interface**:
    1.  `maxpr(*inputs)`: **Lowering**. Emits symbolic nodes to the MAX graph.
    2.  `__call__(*args)`: **Dispatcher**. Handles implementation:
        *   Converts Tensors → TensorValues.
        *   Calls `maxpr`.
        *   **JVP Auto-Detection**: If inputs have `tangent`s, automatically runs `jvp_rule`.
    3.  `jvp_rule(primals, tangents, output)`: **Forward-Mode**. Computes output tangent.
    4.  `vjp_rule(primals, cotangent, output)`: **Reverse-Mode**. Computes input gradients.

#### 4. `ComputeGraph` (The Engine)
*   **Role**: Singleton `GRAPH` managing usage.
*   **Lifecycle**:
    *   Accumulates ops as user runs code (Phase 1).
    *   Triggers `evaluate()` on data access (Phase 2), compiling and executing the graph.

### 3.2 The Pytree System (`eager/pytree.py`)
Essential for handling nested structures (tuples, lists, dicts) of Tensors. This mirrors JAX's tree utils.
*   **Mixed Args**: Ops can accept `(Tensor, int, tuple)` args.
*   **Multi-Output**: Ops like `split` return tuples of Tensors.
*   **Autodiff**: Gradients must match the nested structure of inputs/outputs.
*   **Key Functions**: `tree_map`, `tree_flatten`, `tree_unflatten`.

### 3.3 Logical vs. Physical Shapes
To support auto-vectorization (`vmap`) natively:
*   **Physical Shape**: Actual shape in the MAX graph (e.g., `[Batch, Batch, H, W]`).
*   **Logical Shape**: User-visible shape (e.g., `[H, W]`).
*   **`batch_dims`**: Integer counter. Number of leading axes that are batch dimensions.
    *   `logical_shape = physical_shape[batch_dims:]`
    *   `physical_axis = logical_axis + batch_dims`

## 4. Key Mechanisms

### A. JVP Auto-Detection (Forward-Mode AD)
We don't need a separate `jvp(fn)` tracer. It happens automatically:
1.  User attaches `tangent` to input tensors: `x._impl.tangent = t._impl`.
2.  `Operation.__call__` detects inputs have tangents.
3.  It calls `self.jvp_rule(args, tangents, output)`.
4.  It attaches the result to the output tensor's `tangent`.
5.  Gradients propagate instantly during the forward pass.

### B. Compilation Flow (Normal vs. Sharded)

**Phase 1: Construction**: User runs code → Symbolic graph built. (Cheap)
**Phase 2: Realization**: User prints/accesses data → `GRAPH.evaluate()`.

*   **Path A: Normal (No Sharding)**
    *   Compiles the existing global graph directly.
    *   Fastest path for single-device.

*   **Path B: Sharded**
    *   **Trigger**: Any tensor has `sharding` spec.
    *   **Requirement**: All tensors must be `traced=True`.
    *   **Process**:
        1.  Walk parent history (`op_args`).
        2.  Build **NEW** graph with local shapes + collective ops (all-gather).
        3.  Compile & Execute per-device.

## 5. Comparison

| Feature | PyTorch | JAX | Nabla Eager |
| :--- | :--- | :--- | :--- |
| **Mental Model** | Imperative | Functional | Imperative |
| **Execution** | Eager | Staged (JIT) | **Lazy** (Imperative graph building) |
| **Gradients** | `.backward()` | `grad(fn)` | **JVP** (Implemented), **VJP** (Planned) |
| **Vectorization** | `vmap` (functorch) | `vmap` (core) | **Core**: `batch_dims` is first-class |
| **Sharding** | `DTensor` | Constraints | **Intrinsic**: `Tensor.sharding` |

## 6. Guide to Extending

### Adding an Operation
Inherit from `Operation`, implement `maxpr`. Use `pytree` if handling complex args.

```python
class MyOp(Operation):
    @property
    def name(self): return "my_op"

    def maxpr(self, x, y):
        # Define lowering to MAX
        return ops.add(x, y)

    def jvp_rule(self, primals, tangents, output):
        # Forward differentiation
        return ops.add(tangents[0], tangents[1])

my_op = MyOp() # Singleton instance
```

### Adding Creation Ops
Creation ops (zeros, ones) return constant Tensors.
*   **no inputs**: `maxpr` takes static args.
*   **JVP**: Returns zeros (constants have zero gradient).
*   **VJP**: Returns empty (no inputs to differentiate).

```python
class ZerosOp(Operation):
    def maxpr(self, shape, dtype):
        return ops.zeros(shape, dtype)
        
    def jvp_rule(self, primals, tangents, output):
        return ops.zeros_like(output)
```

### Using Pytrees
Use `tree_map` to handle inputs generically. This allows your op to work with Tensors, tuples of Tensors, or specific static structures.

```python
def __call__(self, *args):
    # Logic is handled by base class, but if you need custom logic:
    # 1. Convert inputs
    # 2. Call maxpr
    # 3. Check for JVP
    return super().__call__(*args)
```
