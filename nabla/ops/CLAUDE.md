# Ops Module: The Operation ABC Pattern

## Philosophy

Every operation is a **singleton object** inheriting from `Operation`. This isn't just an abstraction—it's fundamental to how autodiff, tracing, and batch propagation work.

---

## Why Singletons?

### Identity Checking

When walking the autodiff graph, we need: "Which operation created this tensor?" 

With singletons: `tensor._impl.output_refs.op is _add_op` (fast pointer comparison)

Without singletons: Need name-based lookups or class checks (slower, messier)

### Memory Efficiency

Operations are **stateless**. Creating thousands of `AddOp()` instances would waste memory for no benefit.

### Global Rule Registration

Each operation singleton registers its autodiff rules once when the module loads. VJP/JVP dispatch is just a method lookup.

---

## The Operation ABC Hierarchy

### `Operation` (Base)

Defines: `name`, `maxpr()`, optional `vjp_rule()`, `jvp_rule()`, `sharding_rule()`

**The `__call__` method does all the magic**:
- Extracts tensors from pytree inputs
- Converts to `TensorValue` (MAX graph nodes)
- Calls `maxpr()` to build graph
- Creates `TensorImpl` wrappers for outputs
- Propagates metadata (`batch_dims`, `traced`, etc.)
- Creates `OutputRefs` if in tracing mode
- Handles multi-output pytrees

**Result**: Subclasses only implement `maxpr()` for graph construction. All bookkeeping is automatic.

### `BinaryOperation`

Adds **broadcasting logic**:
- Detects when shapes don't match
- Broadcasts **physical shapes** while respecting **logical semantics**
- Critical for vmap: batch dimensions broadcast separately from logical dimensions

### `UnaryOperation`, `ReduceOperation`, `LogicalShapeOperation`

Specialized ABCs with domain-specific behavior:
- `UnaryOperation`: One input, no broadcasting
- `ReduceOperation`: Translates logical axis → physical axis
- `LogicalShapeOperation`: Shape manipulation that respects `batch_dims`

---

## Physical vs Logical Operations

### The Split

**Logical Operations** (`ops/view.py`, `ops/reduction.py`, etc.):
- User-facing API
- Work with logical shapes (what user sees)
- Translate to physical operations internally

**Physical Operations** (`ops/_physical.py`):
- Internal plumbing for vmap
- Manipulate physical shapes and `batch_dims` counter
- **Not exported** (underscore prefix)

### Why This Matters

When a tensor has `batch_dims=2`:
- Physical shape: `(B1, B2, H, W)`
- Logical shape: `(H, W)` (what user sees)

User calls `reduce_sum(x, axis=0)` expecting to reduce over `H`.

Logical operation translates: `axis=0` (logical) → `physical_axis = batch_dims + 0 = 2 + 0 = 2` (skip batch dimensions).

This translation is why we separate logical from physical ops.

---

## Batch Dims Propagation

### The Rules

**Binary operations**: `max(x.batch_dims, y.batch_dims)`
**Unary operations**: Preserve input's `batch_dims`
**Reductions**: Preserve `batch_dims` (reduce over logical dimensions only)
**View operations**: Preserve `batch_dims` (reshape logical portion only)

### Why Automatic?

Vmap wraps functions in physical ops that manipulate `batch_dims`. Inside the vmapped function, all operations must preserve batch dimensions transparently.

If operations didn't auto-propagate, vmap would break—batched computations would lose batch tracking.

---

## Multi-Output Operations

### The Challenge

How to represent operations like `split(x, 3)` that return `(a, b, c)`?

### The Solution

**All outputs share one `OutputRefs` object**:
- Contains: operation, inputs, kwargs
- Each output has unique `output_idx`
- Sibling relationship tracked via pointer equality

**Why?** Autodiff needs to know:
- These outputs came from the same operation call
- Gradients flow back through shared inputs
- One VJP call produces gradients for all outputs' inputs

### Pytree Integration

`Operation.__call__` uses pytrees to:
- Flatten arbitrarily nested outputs
- Create `TensorImpl` for each tensor leaf
- Link all siblings to shared `OutputRefs`
- Unflatten to match original structure

This enables operations to return dicts, lists, tuples—any pytree structure.

---

## Operation Categories

### `binary.py`

Arithmetic + matmul.  
**Key feature**: Broadcasting respects batch dimensions.

### `unary.py`

Activation functions + math operations.  
**Key feature**: JVP rules for automatic differentiation.

### `creation.py`

Tensor creation (no tensor inputs).  
**Key feature**: Support `traced=True` for gradient tracing even though created from scratch.

### `reduction.py`

Sum, mean, etc.  
**Key feature**: Logical → physical axis translation.

### `view.py`

Reshape, transpose, etc. on logical shape.  
**Key feature**: Preserves `batch_dims`, only manipulates logical portion.

### `multi_output.py`

Split, unbind, etc.  
**Key feature**: Pytree handling, sibling OutputRefs sharing.

### `_physical.py`

Low-level batch/shape manipulation.  
**Not user-facing**: Only used by vmap transform.

---

## Automatic JVP Detection

### How It Works

If **any input** has a `tangent` attribute (from outer `jvp` call), `Operation.__call__` automatically:
1. Calls `maxpr()` for primal forward pass
2. Calls `jvp_rule()` to compute tangent forward pass
3. Attaches tangent to output's `_impl.tangent`

### Why Automatic?

Enables nested JVP composition:
```
jvp(jvp(f))  # Second-order derivatives work automatically
```

No need to manually check for tangents in every operation—base class handles it.

---

## Static Arguments via `op_kwargs`

### The Problem

VJP needs to know what `axis` was used in forward pass:
```
forward: reduce_sum(x, axis=2)
backward: Need axis=2 to know how to un-reduce
```

### The Solution

`OutputRefs.op_kwargs` stores keyword arguments from forward pass.

VJP rule accesses: `output._impl.op_kwargs['axis']`

**Why not store in operation?** Operations are singletons (stateless). Metadata must live with the tensor.

---

## Key Architectural Decisions

### 1. Why Singleton Pattern?

**Enables**: Fast identity checks, memory efficiency, global rule registration.

### 2. Why Separate Physical Ops?

**Enables**: Clean abstraction boundary. Users think in logical space, vmap operates in physical space.

### 3. Why Automatic Metadata Propagation?

**Enables**: Vmap. Without auto-propagation of `batch_dims`, vmap would break.

### 4. Why Shared OutputRefs for Multi-Output?

**Enables**: Correct autodiff—all outputs from one operation call share input gradient flow path.

### 5. Why Pytree Integration in Base Class?

**Enables**: Arbitrary return structures (dicts, nested tuples) without per-operation boilerplate.

---

## Common Misconceptions

**"Operations execute computations"**: No. They build MAX graph nodes. Execution happens in `GRAPH.evaluate()`.

**"batch_dims is for the user API"**: No. It's internal metadata for vmap. Users never directly manipulate it.

**"Physical ops can be used in user code"**: No. They're internal plumbing. Use logical ops instead.

**"Each operation call creates a new op instance"**: No. Singleton pattern means one instance per operation type globally.

---

## Extension Points

### Adding a New Operation

Three steps:
1. Subclass appropriate ABC (`UnaryOperation`, `BinaryOperation`, etc.)
2. Implement `name` property and `maxpr()` method
3. Optionally: `jvp_rule()`, `vjp_rule()`, `sharding_rule()`

Base class handles all bookkeeping automatically.

### Defining Custom Autodiff Rules

Implement `vjp_rule()` and/or `jvp_rule()`.

Access forward pass metadata via `output._impl.op_kwargs`.

Return cotangents/tangents matching input structure.

---

## Future Directions

### Operation Fusion

Before lowering to MAX, combine multiple ops:
- `matmul + add → fused_linear`
- `relu + mul → fused_gelu`

Requires graph analysis pass before `maxpr()` calls.

### Automatic Differentiation Rule Synthesis

For simple ops, derive VJP/JVP rules automatically from `maxpr()` definition using symbolic differentiation.

### Dynamic Operation Dispatch

Choose different `maxpr()` implementations based on input shapes/devices:
- Large matmul → tiled implementation
- Small matmul → direct implementation

Enables perf optimization without changing user API.
