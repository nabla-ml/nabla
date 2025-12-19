# Core Module: Lazy Eager Execution Architecture

## Philosophy

The core module implements a **Lazy Eager** execution model that looks imperative but builds symbolic graphs transparently. The user thinks they're doing PyTorch-style eager execution, but we're secretly constructing a MAX graph that compiles on first data access.

**Key Tension**: How do you build a graph lazily while providing an eager API with immediate shape/dtype information?

---

## The Dual-State Model: Unrealized ↔ Realized

### Why Two States?

Every tensor exists in one of two states:

**Unrealized (Symbolic)**:
- `_values`: List of MAX `TensorValue` graph nodes
- Shape is symbolic (can include `SymbolicDim("batch")`)
- No concrete data yet

**Realized (Concrete)**:
- `_storages`: List of `driver.Tensor` with actual data
- `_values` is cleared (freed to save memory)
- `cached_shape` persists symbolic dimensions!

### The Critical Insight: Shape Caching

**Problem**: After `_values` is cleared, how do we reconstruct the symbolic shape for recompilation?

**Solution**: TensorImpl's `get_realized_shape()` method uses a three-tier fallback:
1. `_values[0].type.shape` if still available
2. `cached_shape` if `_values` was cleared (preserves `SymbolicDim("batch")`)
3. `_storages[0].shape` as last resort (concrete only)

This is used when adding realized tensors as graph inputs. It enables **dynamic batch compilation**: Compile once with `SymbolicDim("batch")`, reuse for any batch size.

---

## Memory Management: The Weakref Dance

### The Problem

Naive approach creates circular references:
```
TensorImpl → OutputRefs → op_args → [TensorImpl, ...]
     ↑_______________________________________________|
```
This prevents garbage collection—intermediates leak forever!

### The Solution

**OutputRefs uses `weakref.ref` for `op_args`**:
- Prevents circular references
- Allows GC of intermediate tensors
- Trade-off: VJP must happen before intermediates are GC'd

**Why this works**: VJP construction walks backwards from loss immediately after forward pass, so intermediates are still alive.

### When _values Gets Cleared

After evaluation completes, `_store_results()` populates `_storages` and clears `_values` to save memory. At this point, the circular reference is broken (no more graph nodes). We continue using weak refs in `OutputRefs` for consistency—they work equally well for both realized and unrealized tensors.

---

## Tensor vs TensorImpl: The Facade Pattern

### Why Separate?

**Tensor** is a stateless facade:
- Implements user-facing API (`__add__`, `.shape`, etc.)
- Lightweight wrapper around `_impl`
- Can be copied cheaply

**TensorImpl** is the mutable state container:
- Holds `_values`, `_storages`, metadata
- Shared between multiple `Tensor` wrappers (multi-output ops)
- Never directly exposed to users

### Multi-Output Sharing

When an operation returns multiple outputs:
```
a, b, c = split(x, 3)
```

All three `Tensor` objects wrap **different** `TensorImpl` objects, but those impls share the **same** `OutputRefs` object. Each knows its `output_idx`.

This enables:
- Efficient metadata sharing
- Single autodiff graph node for the split op
- Correct gradient flow to each output

---

## The ComputeGraph Singleton: Global State Done Right

### Why Global?

All tensors in a process share **one** symbolic MAX graph until `evaluate()` is called. This enables:
- Deferred compilation (batch operations)
- Global optimizations (fusion across operations)
- Side-effect detection (epoch tracking)

### Epoch Tracking: Side Effect Detection

**The mechanism**: `epoch` counter increments every time `evaluate()` completes.

**Used by**: `compile` transform's `fullgraph=True` mode:
```
epoch_before = GRAPH.epoch
result = user_function(proxy_inputs)
if GRAPH.epoch != epoch_before:
    # User code called .numpy() or await! Side effect detected.
    raise RuntimeError or fallback to eager
```

This detects when user code forces intermediate evaluations, which breaks graph compilation.

---

## Batch Dims: Physical vs Logical Shape

### The Vmap Problem

When `vmap` transforms a function, it needs to:
1. Move a logical axis to a "batch" prefix position
2. Hide that axis from the function
3. Move it back after computation

**Solution**: `batch_dims` counter tracks how many leading dimensions are "batch" (invisible to user).

**Example**:
- Physical shape: `(5, 3, 4)` with `batch_dims=1`
- Logical shape: `(3, 4)` (what user sees)
- Batch shape: `(5,)` (hidden from user)

### Propagation Rules

Binary operations: `output.batch_dims = max(x.batch_dims, y.batch_dims)`

This enables nested vmap: each vmap level increments `batch_dims`, and operations transparently preserve all batch dimensions.

---

## Pytree System: Why JAX Compatibility?

### The Need

Multi-output operations need to return arbitrary structures:
```
outputs = {"loss": loss_tensor, "metrics": [acc, f1, prec]}
```

### The Solution

**Pytrees** = nested structures of lists/tuples/dicts containing tensors.

**Operations**:
- `tree_flatten`: `structure → (flat_list, structure_descriptor)`
- `tree_unflatten`: `(flat_list, structure_descriptor) → structure`

**Used by**:
- `Operation.__call__`: Flatten inputs, unflatten outputs
- `vmap`: Apply transformation to all tensor leaves
- `compile`: Cache key includes pytree structure

### Sentinel Values

Special markers for metadata:
- `traced(tensor)`: Force tracing even if not in autodiff
- `with_batch_dims(tensor, n)`: Annotate batch dimension count

These propagate through pytree operations, enabling rich metadata on complex structures.

---

## The OutputRefs Graph: Autodiff Preparation

### What It Is

A parallel graph structure tracking operation provenance:
- Each `TensorImpl` has optional `OutputRefs`
- Points to: operation, input tensor impls (weak refs), kwargs, output index

### Why Not Store in the MAX Graph?

MAX's `TensorValue` graph is **immutable** and **low-level**. We need:
- Python-level metadata (kwargs, operation singletons)
- Weak references for memory management
- Multi-output sibling relationships

So we maintain our own graph in Python, aligned with the MAX graph.

### VJP Backward Pass (Future)

Walk the `OutputRefs` graph backwards:
```
1. Start at loss tensor
2. Get loss._impl.output_refs.op
3. Call op.vjp_rule(inputs, cotangent)
4. Recursively walk to input tensors' OutputRefs
5. Accumulate gradients
```

Weak refs are fine here—VJP happens immediately after forward pass while all tensors are still alive.

---

## Graph Compilation: The Three-Stage Pipeline

### Stage 1: Graph Construction (Lazy)

Operations append nodes to MAX graph via `maxpr()` calls. Nothing executes yet.

### Stage 2: Compilation (On-Demand)

When user accesses data (`.numpy()`, `await`, etc.):
1. Topologically sort unrealized tensors
2. Walk operations, add inputs/ops to MAX graph
3. MLIR lowering + optimization
4. MAX compilation to executable

### Stage 3: Execution

Run compiled model with concrete tensor data, store results in `_storages`.

### The Optimization Window

Because we defer compilation until forced, we can:
- Fuse operations
- Eliminate dead code
- Optimize entire subgraphs

This is the payoff for lazy eager execution.

---

## Context Management: Thread-Local Defaults

### Why ContextVar?

Multiple threads might have different default devices/dtypes. `ContextVar` provides thread-local storage.

### The Defaults Cascade

When creating a tensor, if dtype/device not specified:
```
dtype = user_dtype or context_dtype or infer_from_device()
device = user_device or context_device or CPU
```

Enables: `with default_device(cuda): x = Tensor.ones((3, 4))` without passing device everywhere.

---

## Key Architectural Decisions

### 1. Why Separate Tensor and TensorImpl?

**Enables**: Multi-output operations where multiple Tensors share implementation details.

### 2. Why Clear `_values` After Realization?

**Enables**: Memory efficiency—symbolic graph nodes are large and no longer needed once we have concrete data.

### 3. Why Cache Shape Separately?

**Enables**: Recompiling with symbolic dimensions after `_values` is gone.

### 4. Why Weakrefs in OutputRefs?

**Enables**: Garbage collection of intermediates while preserving autodiff graph.

### 5. Why Global GRAPH Singleton?

**Enables**: Deferred compilation, cross-operation optimization, epoch tracking.

---

## Common Misconceptions

**"Lazy eager is just JIT compilation"**: No. We build a graph **eagerly** (no tracing decorators), but compile **lazily** (on data access). Best of both worlds.

**"Weakrefs break autodiff"**: No. They just require VJP to happen promptly. Since we walk backwards from loss immediately, all intermediates are still alive.

**"batch_dims is for sharding"**: No. It's for vmap. Sharding uses `sharding_spec` (separate metadata).

**"TensorImpl is an optimization"**: No. It's fundamental to the architecture—enables multi-output ops and clean state management.

---

## Future Directions

### Full Autodiff

Complete VJP backward pass walking the OutputRefs graph. Current infrastructure is ready—just need to implement the recursive walk and gradient accumulation.

### Graph Optimization Passes

Before compilation:
- Operation fusion (matmul + add → fused_linear)
- Common subexpression elimination
- Dead code elimination

### Async Compilation

Overlap graph building with compilation of previous graphs. Requires careful epoch management.

### Incremental Recompilation

When a small part of the graph changes, only recompile that subgraph. Requires dependency tracking.
