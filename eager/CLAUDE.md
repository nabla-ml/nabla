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
├── compute_graph.py   # Graph compilation & execution (with epoch tracking)
├── context.py         # Thread-local settings (device, dtype)
├── pytree.py          # JAX-compatible tree utilities
├── vmap_trafo.py      # vmap transform for vectorization
├── compile_trafo.py   # compile transform for model caching
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
- **Tracking**: `traced`, `batch_dims`, `cached_shape`, `op`, `op_args`, `op_kwargs`.

**Key Invariant**: After `GRAPH.evaluate()` runs:
- `_values` is cleared (freed).
- `cached_shape`, `cached_dtype`, `cached_device` persist the metadata.

#### `OutputRefs` (The Trace Node)
- **Role**: Links a tensor to its producing operation for VJP graph construction.
- **Created When**: `Operation` is called with `traced=True`.
- **Fields**:
  - `op`: Reference to the `Operation` singleton.
  - `op_args`: Positional arguments (as `TensorImpl` references).
  - `op_kwargs`: Keyword arguments (as `TensorImpl` references).
  - `output_idx`: Index if the op returns multiple outputs (pytree).

**Why Weak References?** `op_args` uses `weakref.ref` to prevent circular references and allow garbage collection of unused intermediates.

**Memory Note**: While `TensorImpl` is unrealized (`_values` exists), we use weak references. Once realized (after `evaluate()`), `_values` is cleared and `OutputRefs` can use strong references if needed for VJP, but currently this is not stored persistently.

**Status**: Fully implemented for tracing. VJP uses this to walk the graph backwards.

#### `Operation` (The Dispatcher)
- **Role**: Base class for all ops. Defines computation + autodiff rules.
- **Singleton Pattern**: Each op type (e.g., `AddOp`, `MulOp`) has exactly one instance globally.
- **Key Methods**:
  - `maxpr(...)`: MAX graph lowering.
  - `vjp_rule(...)`: Reverse-mode gradient.
  - `jvp_rule(...)`: Forward-mode tangent.

### 3.2 Tracing Architecture

When an operation is called:
1. `Operation.__call__` converts `Tensor` → `TensorValue`.
2. Calls `maxpr(...)` to get result `TensorValue`.
3. If `traced=True` or JVP mode, creates `OutputRefs` linking result to inputs.
4. Returns result as new `Tensor`.

### 3.3 Traced vs Untraced Mode

| Mode | Purpose | Behavior |
|------|---------|----------|
| **Untraced** (default) | Fast forward pass | No `OutputRefs`. No VJP graph. Memory efficient. |
| **Traced** | Enable VJP/gradients | Creates `OutputRefs` for each op. VJP graph buildable. |

### 3.4 Symbolic Graph via MAX

The symbolic graph is built using MAX's `max.graph.Graph` API. Operations are lazily added, and the graph is compiled on first data access (via `await tensor` or `tensor.numpy()`).

### 3.5 Pytree System (`eager/pytree.py`)

Pytrees allow operations to return arbitrary nested structures (tuples, lists, dicts) of tensors. This is JAX-compatible and used for multi-output ops like `split`, `unbind`.

## 4. Key Mechanisms

### A. JVP Auto-Detection (Forward-Mode AD)

If any input has a `tangent`, the operation automatically enters JVP mode and calls `jvp_rule`:
- Computes both primal and tangent outputs.
- Tangent is stored in `output._impl.tangent`.

### B. VJP (Reverse-Mode AD) - Ready for Implementation

The `OutputRefs` tracing infrastructure is in place. To implement `.backward()`:
1. Walk the `OutputRefs` graph backwards from loss.
2. For each op, call `vjp_rule(primals, cotangent, output)`.
3. Accumulate gradients in `._impl.grad`.

### C. Logical vs Physical Shapes

- **Logical Shape**: What the user sees (e.g., `(batch, features)`).
- **Physical Shape**: Internal representation with batching prefix (e.g., `(vmap_axis, batch, features)`).
- **`batch_dims`**: Number of leading vmap dimensions in physical shape.

## 5. Comparison

| Feature | PyTorch | JAX | Eager Module |
|---------|---------|-----|--------------|
| Execution | Eager | JIT-required | Lazy Eager |
| `.backward()` | ✅ | ❌ (use `vjp`) | 🚧 (ready) |
| `vmap` | ❌ | ✅ | ✅ |
| `jvp` | ❌ | ✅ | ✅ (auto) |
| `jit` | `torch.compile` | `@jax.jit` | `@compile` |

## 6. Guide to Extending

### Adding an Operation

1. Create class inheriting `Operation`:
```python
class SquareOp(Operation):
    @property
    def name(self) -> str:
        return "square"
    
    def maxpr(self, x: TensorValue) -> TensorValue:
        return ops.mul(x, x)
    
    def jvp_rule(self, primals: tuple, tangents: tuple, output: Any) -> Any:
        (x,), (x_dot,) = primals, tangents
        return 2.0 * x * x_dot  # d/dx(x²) = 2x
    
    def vjp_rule(self, primals: tuple, cotangent: Any, output: Any) -> tuple:
        (x,) = primals
        return (2.0 * x * cotangent,)
```

2. Create singleton and public function:
```python
_square_op = SquareOp()

def square(x: Tensor) -> Tensor:
    return _square_op(x)
```

### Adding Multi-Output Operations

Use pytrees for multiple outputs:

```python
class SplitOp(Operation):
    def maxpr(self, x: TensorValue, num_splits: int) -> tuple[TensorValue, ...]:
        return tuple(ops.split(x, num_splits))
    
    # jvp_rule and vjp_rule handle tuples via pytree
```

### Accessing Operation Metadata in VJP

The `op_args` and `op_kwargs` stored in `OutputRefs` are accessible via `output._impl`:

```python
def vjp_rule(self, primals, cotangent, output):
    # Access kwargs from forward pass
    axis = output._impl.op_kwargs.get('axis', 0)
    # Use axis for gradient computation
    ...
```

## 7. Vmap Transform Implementation

### 7.1 Project Structure (Updated)

```text
eager/
└── vmap_trafo.py      # vmap transform + view operations
    ├── vmap()         # Main transform
    ├── MoveAxisToBatchDimsOp
    ├── MoveAxisFromBatchDimsOp
    ├── ExpandBatchDimsOp
    └── CollapseBatchDimsOp
```

**Key Classes:**
- `VmapContext`: Manages batched execution scope.
- View Ops: Manipulate `batch_dims` counter for dimension tracking.

### 7.2 Vmap Usage

```python
# Vectorize over leading dimension
batched_fn = vmap(fn)
result = batched_fn(batched_input)

# Specify input/output axes
batched_fn = vmap(fn, in_axes=(0, 1), out_axes=2)
```

### 7.3 Key Design: Prefix Pytree Semantics

Unlike JAX's arbitrary pytree `in_axes`, this implementation uses **prefix semantics**:
- `in_axes` is a single structure (tuple/dict) of axis specs.
- Only the **prefix** of the input pytree is matched.

### 7.4 Core Components

- **batch_dims tracking**: Each `TensorImpl` has a `batch_dims` counter.
- **Logical vs Physical**: User sees logical shape; operations use physical shape with batch prefix.
- **BinaryOperation**: Automatically handles broadcasting between batched/unbatched tensors.

### 7.5 How It Works

1. Map inputs → add batch dimension via `MoveAxisToBatchDimsOp`.
2. Run function in `VmapContext`.
3. Operations use physical shape internally.
4. Map outputs → move batch dimension to output axis.

### 7.6 Next Steps

- **Nested vmap**: Stack multiple `VmapContext` layers.
- **Advanced in_axes**: Support `-1` for rightmost axis, `None` for broadcast.

## 8. Compile Transform Implementation

### 8.1 Project Structure (Updated)

```text
eager/
└── compile_trafo.py
    ├── compile()           # Main decorator
    ├── CompiledFunction    # Wrapper class
    └── CompilationStats    # Performance metrics
```

### 8.2 Compile Usage

```python
@compile
def compute(x, y):
    return x @ y + x

# First call compiles
result1 = compute(a, b)  # Compilation happens

# Subsequent calls reuse compiled model
result2 = compute(c, d)  # Cache hit
```

Advanced options:

```python
@compile(fullgraph=True, max_cache_size=10)
def strict_compute(x):
    return x * 2
```

### 8.3 Options

- **`fullgraph`**: If `True`, errors on side effects. If `False` (default), allows print/side effects.
- **`max_cache_size`**: LRU cache limit (default: 128).

### 8.4 CompilationStats

Access via `.stats`:

```python
fn = compile(my_function)
fn(x)
print(fn.stats.cache_hits, fn.stats.cache_misses)
```

### 8.5 How It Works

1. Intercept function call.
2. Build cache key from tensor shapes/dtypes/devices.
3. If cache miss: run function, compile graph, store model.
4. Execute cached model with current inputs.

### 8.6 Cache Key Components

Hash of:
- Python function object
- Input shapes, dtypes, devices
- Static argument values

### 8.7 Mixed Outputs

Supports returning both Tensors and non-Tensors:

```python
@compile
def compute_with_metadata(x):
    return x * 2, 42, {"info": "metadata"}

a, count, meta = compute_with_metadata(x)
# count=42, meta={"info": "metadata"} preserved correctly
```

## 9. Dynamic (Symbolic) Dimensions

### 9.1 Overview

The eager module fully supports **symbolic dimensions** via MAX Graph's native `SymbolicDim` and `StaticDim`. This enables batch-flexible models that can be compiled once and executed with varying batch sizes.

### 9.2 Clean API: Strings and Ints

MAX's `Shape` constructor automatically converts:
- `str` → `SymbolicDim`
- `int` → `StaticDim`

This allows a clean, intuitive API:

```python
# Symbolic batch dimension
x = Tensor.ones(("batch", 128))  # Shape: [Dim('batch'), Dim(128)]

# Multiple symbolic dimensions
A = Tensor.zeros(("batch", "seq_len", 768))

# Mixed: symbolic + static
W = Tensor.ones(("hidden", 512))
```

**All creation ops support this:**
- `Tensor.zeros(("batch", 64))`
- `Tensor.ones((128, "hidden"))`
- `Tensor.full(("N", "M"), 5.0)`
- `Tensor.uniform(("batch", 10))`
- `Tensor.gaussian(("batch", "features"))`

### 9.3 Shape Propagation

**Symbolic dimensions are preserved through all operations:**

```python
x = Tensor.ones(("batch", 128))
W = Tensor.ones((128, "hidden"))

# Matrix multiplication
y = x @ W  # Shape: [Dim('batch'), Dim('hidden')]

# Binary operations
z = y + y  # Shape: [Dim('batch'), Dim('hidden')]

# Broadcasting
bias = Tensor.ones(("hidden",))
out = y + bias  # Shape: [Dim('batch'), Dim('hidden')]
```

**Tested operations:**
- Binary ops: `add`, `mul`, `sub`, `div`
- Matrix multiplication: `matmul`
- Broadcasting
- Operation chains

### 9.4 Implementation Details

**`TensorImpl.cached_shape`:**
- Stores the `max.graph.Shape` including `SymbolicDim`s.
- Persists after `_values` is cleared post-evaluation.
- Used by `add_input()` to construct graph inputs with symbolic signatures.

**`ComputeGraph.add_input()` three-tier fallback:**
```python
if tensor._impl._values:
    # Unrealized: use TensorValue's shape
    shape = tensor._impl._values[0].type.shape
elif tensor._impl.cached_shape:
    # Realized: use persisted symbolic shape
    shape = tensor._impl.cached_shape
else:
    # Fallback: concrete storage shape
    shape = tensor.storage.shape
```

**Why this matters:** After `GRAPH.evaluate()`, `_values` is cleared to free memory. `cached_shape` preserves the symbolic signature, allowing realized tensors to be reused as inputs to new compilations with their symbolic shapes intact.

### 9.5 Unique Dimension Names

**Important:** When building a single graph with multiple symbolic dimensions, use **unique names** to avoid cyclic parameter references:

```python
# ❌ BAD - reusing "batch" creates cycles
x1 = Tensor.ones(("batch", 64))
x2 = Tensor.ones(("batch", 128))  # Same "batch" symbol

# ✅ GOOD - unique names
x1 = Tensor.ones(("b1", 64))
x2 = Tensor.ones(("b2", 128))
```

This is only an issue when accumulating operations in a single graph context. For separate compilations or isolated tests, dimension names can be reused.

### 9.6 Testing

Comprehensive tests in `test_ops_symbolic.py` verify:
- Creation ops preserve symbolic shapes
- Binary ops propagate symbolic dims
- Matrix multiplication handles symbolic batch/output dims
- Broadcasting works with symbolic dimensions
- Operation chains maintain symbolic signatures

**Example test pattern:**
```python
x = Tensor.ones(("b1", 128))
W = Tensor.ones((128, 64))
y = x @ W
assert y._value.type.shape == Shape([SymbolicDim("b1"), StaticDim(64)])
```

### 9.7 Migration Note

**Before:** Shapes were tuples of ints: `(32, 128)`.

**Now:** Use `max.graph.Shape` directly:
- Tuples still work for static shapes: `(32, 128)` → `Shape([StaticDim(32), StaticDim(128)])`
- For symbolic: use strings in tuple: `("batch", 128)` → `Shape([SymbolicDim("batch"), StaticDim(128)])`

All internal operations now work with `max.graph.Shape` objects, enabling full symbolic dimension support throughout the eager module.

