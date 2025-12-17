# Eager Module Architecture

This document explains the architecture of the `eager` module. It implements a **Lazy Eager** framework, targeting the usability of PyTorch with the optimization capabilities of JAX.

## 1. Project Structure

```text
eager/
├── tensor.py          # Public API: User-facing Tensor class
├── tensor_impl.py     # Internal: TensorImpl state container
├── tracing.py         # Internal: OutputRefs for operation tracing
├── ops.py             # Base Operation class & autodiff dispatch
├── binary_ops.py      # Binary ops (Add, Mul...)
├── creation.py        # Creation ops (Zeros, Ones, Arange...)
├── multi_output_ops.py # Multi-output ops (Split, Unbind...)
├── graph_utils.py     # Graph traversal utilities
├── compute_graph.py   # Graph compilation & execution
├── context.py         # Thread-local settings (device, dtype)
├── pytree.py          # JAX-compatible tree utilities
├── sharding.py        # DeviceMesh, ShardingSpec definitions
└── sharding_propagation.py # Sharding inference logic
```

## 2. Core Philosophy

The module separates the **API (User View)** from the **Execution (System View)**.

-   **User View**: "I am creating arrays and adding them immediately."
-   **System View**: "I am recording a symbolic MAX graph. I compile and run on data access."

This **Lazy Eager** approach allows:
1.  **Capture the full graph** without explicit `jit` decorators.
2.  **Optimize globally** using the MAX compiler.
3.  **Retain imperative debugging**: `print()` or breakpoints pause graph construction.

## 3. Architecture Deep Dive

### 3.1 The Core Components

#### `Tensor` (The Facade)
- **Role**: User-facing object mimicking `torch.Tensor`.
- **Behavior**: Stateless wrapper referencing `_impl`. Delegates math to ops.

#### `TensorImpl` (The Brain)
- **Role**: Internal state container for each tensor.
- **State Machine**:
  - **Unrealized**: Holds `_values` (Symbolic MAX `TensorValue` nodes).
  - **Realized**: Holds `_storages` (Concrete `driver.Tensor` data).
- **Key Fields**:
  - `traced`: Whether this node is part of a traced graph.
  - `output_refs`: Shared `OutputRefs` instance (operation metadata).
  - `output_index`: Position among sibling outputs (for multi-output ops).
  - `tangent` / `cotangent`: For JVP/VJP autodiff.
  - `batch_dims`: Number of vmap batch dimensions.

#### `OutputRefs` (The Trace Node)
- **Location**: `eager/tracing.py`
- **Role**: Single source of truth for operation metadata. Shared among all outputs of the same operation call.
- **Key Design**: Separates "what operation produced this" from "which output am I" to handle multi-output ops cleanly.

```python
@dataclass(frozen=True)
class OutputRefs:
    _refs: tuple[weakref.ref, ...]  # Weak refs to output TensorImpls
    tree_def: PyTreeDef             # Output structure
    op: Operation                   # The operation instance
    op_args: tuple[Any, ...]        # Input TensorImpls + static args
    op_kwargs: dict[str, Any]       # Keyword arguments
```

**Critical**: `op_args` stores `TensorImpl` references, NOT `Tensor` wrappers. This preserves the weak-ref GC strategy.

#### `Operation` (The Dispatcher)
- **Role**: Stateless singleton defining a transformation.
- **Interface**:
  1. `maxpr(*inputs)`: Emit symbolic nodes to MAX graph.
  2. `__call__(*args)`: Execute on Tensors, handle JVP auto-detection.
  3. `jvp_rule()`: Forward-mode autodiff.
  4. `vjp_rule()`: Reverse-mode autodiff.

### 3.2 Tracing Architecture

The tracing system enables autodiff and function transformations:

```
TensorImpl ──holds──► OutputRefs (shared among siblings)
                          │
                          ├── op: Operation (strong ref)
                          ├── op_args: tuple[TensorImpl | static, ...]
                          ├── op_kwargs: dict
                          ├── tree_def: PyTreeDef
                          └── _refs: tuple[weakref] → output TensorImpls
```

**Multi-output operations** (like `split`) create multiple `TensorImpl`s that share the same `OutputRefs`:
- `a._impl.output_refs is b._impl.output_refs` → `True`
- `a._impl.output_index = 0`, `b._impl.output_index = 1`

**Graph traversal** (`graph_utils.py`) deduplicates via `id(output_refs)` so each operation is processed exactly once.

### 3.3 Traced vs Untraced Mode

| Mode | `traced=False` (Default) | `traced=True` |
|------|--------------------------|---------------|
| **Use case** | Inference | Autodiff, sharding |
| **op_args stored?** | No (empty tuple) | Yes (TensorImpls) |
| **GC behavior** | Intermediates can be collected | Full graph retained |
| **Parent access** | `impl.parents = []` | `impl.parents = [TensorImpl, ...]` |

**Memory strategy**: In untraced mode, `OutputRefs.op_args` is empty, so there are no strong references keeping parent tensors alive.

### 3.4 Symbolic Graph via MAX

The MAX graph values ARE the symbolic representation:
- `TensorValue` nodes are symbolic until `GRAPH.evaluate()` materializes them.
- The `Operation` stored in `OutputRefs` can rebuild symbolic expressions.
- No separate "Jaxpr-like" IR needed — MAX provides this natively.

### 3.5 Pytree System (`eager/pytree.py`)

Essential for handling nested structures:
- **Mixed Args**: Ops accept `(Tensor, int, tuple)` args.
- **Multi-Output**: Ops return tuples/lists/dicts of Tensors.
- **Autodiff**: Gradients match input/output structure.

## 4. Key Mechanisms

### A. JVP Auto-Detection (Forward-Mode AD)

1. User attaches `tangent` to input: `x._impl.tangent = t._impl`
2. `Operation.__call__` detects inputs have tangents.
3. Calls `self.jvp_rule(args, tangents, output)`.
4. Attaches result to `output._impl.tangent`.
5. Gradients propagate instantly during the forward pass.

### B. VJP (Reverse-Mode AD) - Ready for Implementation

The structure supports VJP:
1. Mark inputs as `traced=True`.
2. Run forward computation.
3. Traverse backward using `get_operations_topological()`.
4. For each op, call `vjp_rule(op_args, cotangent, outputs)`.
5. Accumulate cotangents on parent `TensorImpl`s.

### C. Logical vs Physical Shapes

For vmap support:
- **Physical Shape**: Actual shape in MAX graph (e.g., `[Batch, H, W]`).
- **Logical Shape**: User-visible shape (e.g., `[H, W]`).
- **`batch_dims`**: Number of leading batch axes.
  - `logical_shape = physical_shape[batch_dims:]`

## 5. Comparison

| Feature | PyTorch | JAX | Nabla Eager |
|:--------|:--------|:----|:------------|
| **Mental Model** | Imperative | Functional | Imperative |
| **Execution** | Eager | Staged (JIT) | **Lazy** |
| **Trace Representation** | Implicit (grad_fn DAG) | Explicit (Jaxpr) | **MAX symbolic graph** |
| **Multi-output Ops** | grad_fn.next_functions | Pytree-native | **Pytree + OutputRefs** |
| **Gradients** | `.backward()` | `grad(fn)` | **JVP** (done), **VJP** (ready) |
| **Vectorization** | `vmap` (functorch) | `vmap` (core) | **Core**: `batch_dims` |
| **Sharding** | `DTensor` | Constraints | **Intrinsic**: `Tensor.sharding` |

## 6. Guide to Extending

### Adding an Operation

```python
class MyOp(Operation):
    @property
    def name(self): return "my_op"

    def maxpr(self, x, y):
        return ops.add(x, y)

    def jvp_rule(self, primals, tangents, output):
        return tangents[0] + tangents[1]

    def vjp_rule(self, primals, cotangent, output):
        # Return cotangent for each input
        return (cotangent, cotangent)

my_op = MyOp()  # Singleton instance
```

### Adding Multi-Output Operations

```python
class SplitOp(Operation):
    @property
    def name(self): return "split"

    def maxpr(self, x, *, num_splits, axis=0):
        # Return tuple/list of TensorValues
        return tuple(ops.split(x, ...))

split = SplitOp()

# Usage: outputs share OutputRefs
a, b = split(x, num_splits=2, axis=0)
assert a._impl.output_refs is b._impl.output_refs
```

### Accessing Operation Metadata in VJP

```python
def vjp_rule(self, primals, cotangent, output):
    # primals contains TensorImpls directly
    input_impl = primals[0]  # Already TensorImpl, not Tensor
    
    # Access kwargs from output's OutputRefs
    axis = output._impl.output_refs.op_kwargs.get('axis', 0)
    
    # Compute gradients...
    return grads
```

---

## 7. Vmap Transform Implementation

The `vmap` transform is fully implemented in `vmap_trafo.py`.

### 7.1 Project Structure (Updated)

```text
eager/
├── tensor.py          # Public API: User-facing Tensor class
├── tensor_impl.py     # Internal: TensorImpl state container
├── tracing.py         # Internal: OutputRefs for operation tracing
├── ops.py             # Base Operation class & autodiff dispatch
├── binary_ops.py      # Binary ops with batch_dims support
├── view_ops.py        # View operations for axis manipulation
├── vmap_trafo.py      # ✅ NEW: vmap transform implementation
├── creation.py        # Creation ops (Zeros, Ones, Arange...)
├── multi_output_ops.py # Multi-output ops (Split, Unbind...)
├── graph_utils.py     # Graph traversal utilities
├── compute_graph.py   # Graph compilation & execution
├── context.py         # Thread-local settings (device, dtype)
├── pytree.py          # JAX-compatible tree utilities
├── sharding.py        # DeviceMesh, ShardingSpec definitions
└── sharding_propagation.py # Sharding inference logic
```

### 7.2 Vmap Usage

```python
from eager import vmap, Tensor

# Basic usage
@vmap
def square(x): return x * x
result = square(Tensor.arange(0, 5))

# Multiple inputs with different axes
vmap(add, in_axes=(0, None))(batched_x, scalar_y)

# Dict axis spec for pytree inputs
vmap(process, in_axes={'w': 0, 'b': None})(params_dict)

# Pure broadcast with explicit size
vmap(fn, in_axes=None, axis_size=10)(scalar)

# Nested vmap (multiple batch dims)
vmap(vmap(fn))(batch_of_batches)
```

### 7.3 Key Design: Prefix Pytree Semantics

Axis specifications follow JAX's "prefix pytree" pattern:
- **Scalar (int/None)**: Broadcasts to all tensor leaves
- **Container**: Must structurally match the input

```python
# Scalar broadcasts to all keys
vmap(fn, in_axes=0)({"a": x, "b": y})  # Both batched on axis 0

# Dict matches pytree structure
vmap(fn, in_axes={"a": 0, "b": None})({"a": x, "b": y})  # a batched, b broadcast
```

### 7.4 Core Components

| Component | Purpose |
|-----------|---------|
| `_map_prefix(fn, tree, prefix)` | Apply fn to tensor leaves with prefix broadcasting |
| `_collect_from_prefix(fn, tree, prefix)` | Collect results for validation |
| `_batch_tensor(tensor, axis, batch_size)` | Move axis to batch_dims |
| `_unbatch_tensor(tensor, axis)` | Restore axis from batch_dims |

### 7.5 How It Works

1. **Batch inputs**: Move specified axes to `batch_dims` via `move_axis_to_batch_dims`
2. **Execute function**: Ops work transparently with `batch_dims`
3. **Unbatch outputs**: Restore axes via `move_axis_from_batch_dims`

The `batch_dims` counter on `TensorImpl` tracks how many leading physical axes are batch axes. User code only sees logical shapes.

### 7.6 Next Steps

1. **Add VJP/JVP rules** to view ops (currently forward-only)
2. **Implement `vjp_trafo.py`**: Reverse-mode autodiff transform
3. **Handle sharding + vmap**: Sharding applies to physical shapes

