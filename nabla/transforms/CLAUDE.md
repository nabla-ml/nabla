# Transforms Module: Function Transformations

## Philosophy

Transforms wrap user functions to add capabilities: vectorization (vmap), caching (compile), autodiff (future grad/vjp).

Unlike JAX which requires explicit `@jit` decoration, our `compile` transform is **optional**—lazy eager execution already provides most benefits. Transforms are for when you want more control.

---

## Vmap: Automatic Vectorization

### The Core Idea

Transform a function that works on individual examples to work on batches:
```
f: (features,) → (outputs,)
vmap(f): (batch, features) → (batch, outputs)
```

**Implementation**: Prefix semantics—batch dimensions are ALWAYS leading axes, never arbitrary positions.

### The Mechanism

**Entry**: `move_axis_to_batch_dims()` + `incr_batch_dims()`
- Moves user-specified axis to front
- Increments `batch_dims` counter

**Inside vmapped function**:
- All operations see logical shape (batch dimension hidden)
- Operations auto-propagate `batch_dims`
- Batch dimension preserved through computation

**Exit**: `move_axis_from_batch_dims()` + `decr_batch_dims()`
- Moves batch dimension to user-specified output position
- Decrements `batch_dims` counter

### Why Prefix Semantics?

**Alternative**: JAX-style arbitrary axis positions (batch at axis 2 of input, axis 1 of output, etc.)

**Problem**: Complex bookkeeping—must track where batch is at each operation.

**Our choice**: Always at front (physical shape prefix). Simpler, faster, equally expressive.

### Nested Vmap

Each vmap level increments `batch_dims`:
```
vmap(vmap(f)): batch_dims starts at 0 → 1 → 2
```

Operations automatically preserve all batch dimensions via `max()` propagation.

**Result**: Nested vmap Just Works™—no special handling needed.

### In_Axes and Out_Axes

**in_axes**: Which axis is batch for each input
- Integer: specific axis
- None: broadcast (not batched)
- Dict: per-key specification for pytree inputs

**out_axes**: Where to place batch in output
- Integer: specific position
- Default 0: leading dimension

**Pytree handling**: Prefix matching—in_axes structure matches input structure prefix.

### The Broadcast Problem

**Challenge**: `vmap(f)(batched_x, scalar_y)` where `y` isn't batched.

**Solution**: 
- `in_axes=(0, None)`
- `y` isn't moved to batch dims
- Binary ops handle mixed batch/unbatched via auto-broadcasting
- Result has batch_dims from `batched_x`

---

## Shard Map: Automatic Distribution

### The Core Idea

Bridge between "Logical" (user view) and "Physical" (distributed execution). Allows writing code as if it runs on a single device, but executes it partitioned across a mesh.

### The Mechanism: Dual Execution

Replaces complex graph patching with a clean **Trace-and-Replay** model:

1. **Trace Logical Graph**: Captures the computation using standard logical tensors.
2. **Replay on Duals**: Re-executes the graph using physical (sharded) tensors.

**The Workflow**:
1. **Realize Inputs**: Ensures a clean starting state.
2. **Trace**: `trace(func, args)` captures the operation history.
3. **Attach Duals**: Maps input arguments to their sharded physical counterparts (duals) based on `in_specs`.
4. **Replay Loop**:
    - Iterates through captured logical nodes.
    - Resolves arguments to their `.dual` (physical) counterparts.
    - Executes operations on duals (triggering SPMD propagation).
    - Updates output duals (`logical.dual = physical_result`).
5. **Finalize**: Returns the duals of the trace outputs, applying `out_specs` if provided.

### Why Dual Execution?

- **Separation of Concerns**: Tracing captures *what* to compute. Replay captures *how* to distribute it.
- **Robustness**: Handles constants (untraced nodes skipped), complex compositions, and mixed updates cleaner than in-place patching.
- **Verification**: The generated trace acts as a "sharding plan" that can be inspected before execution.

### Correct Usage Pattern

> [!IMPORTANT]
> Use `in_specs` and `out_specs` to specify sharding, NOT internal `shard()` calls.

**✅ Correct Pattern**:
```python
def my_func(x, w):
    return x @ w  # Just the computation

# Sharding specified via in_specs
sharded_fn = shard_map(
    my_func, mesh,
    in_specs={
        0: ShardingSpec(mesh, [DimSpec(['dp']), DimSpec([])]),  # x
        1: ShardingSpec(mesh, [DimSpec([]), DimSpec(['tp'])]),  # w
    },
    out_specs=None
)
```

**❌ Problematic Pattern** (causes double-execution):
```python
def my_func(x, w):
    x = x.shard(mesh, [...])  # DON'T do this inside shard_map
    w = w.shard(mesh, [...])
    return x @ w

shard_map(my_func, mesh, in_specs={0: None, 1: None})
```

**Why?** When `shard()` is called inside the function:
1. During trace: The `shard` op executes and is captured
2. During replay: `shard_map` re-executes the traced `shard` op
3. Result: Double sharding/communication, incorrect numerical results


---

## Compile: Computation Caching

### The Philosophy

Unlike JAX's required `@jit`, our lazy eager model already builds graphs—compile is for:
- **Avoiding Python overhead** on repeated calls
- **Caching compiled models** across different batch sizes
- **Strict mode** (fullgraph=True) to catch side effects

### Caching Strategy

**Cache key**:
- Function identity
- Input shapes  (or symbolic shape with dynamic_dims)
- Input dtypes
- Static argument values
- Pytree structure

**Cache value**:
- Compiled MAX model (executable)
- Input order (for dict kwargs)
- Output pytree structure
- Mask of which outputs are tensors vs static values

### Dynamic Dimensions

**The killer feature**: Compile once, run on any batch size.

**Mechanism**: `dynamic_dims={arg_idx: {dim_idx: "name"}}`
- Marks specific dimensions as symbolic
- Cache key uses `SymbolicDim("batch")` not concrete size
- Different batch sizes → cache hit!

**Example**: `{0: {0: "batch"}}` makes first dimension of first argument symbolic named "batch".

### Mixed Outputs

**Challenge**: User functions might return `(tensor, int, str, ...)`.

**Solution**:
- Pytree flatten to separate tensors from non-tensors
- Compile graph for tensor outputs only
- Store non-tensor values in cached model
- Reconstruct full pytree on cache hit

### Side Effect Detection (fullgraph)

**Problem**: User code might call `.numpy()` mid-computation, forcing evaluation.

**Detection**: Epoch tracking
- Record `GRAPH.epoch` before calling user function
- If epoch changed: evaluation happened inside function
- `fullgraph=True` → error, `fullgraph=False` → fallback to eager

### LRU Eviction

**Cache size limit** (default 64 entries).

When cache full, evict oldest entry (FIFO with move-to-end on hit).

**Why limit?** Compiled models are large—unbounded cache would OOM.

---

## Key Architectural Decisions

### 1. Why Vmap Uses Physical Ops?

**Enables**: Clean separation. User code sees logical shapes, vmap machinery operates on physical shapes.

### 2. Why Prefix Batch Semantics?

**Enables**: Simpler implementation, automatic propagation via binary op `max()` rule.

### 3. Why Compile is Optional?

**Enables**: Lazy eager already builds and optimizes graphs on first access. Compile adds: cross-call caching (avoiding Python overhead on repeated calls), dynamic dimension support (one compiled model for multiple batch sizes), and strict side-effect detection.

### 4. Why Dynamic Dims Per-Argument?

**Enables**: Fine-grained control. Batch dimension might be symbolic, but model width (static) for different models.

### 5. Why Epoch Tracking for Side Effects?

**Enables**: Cheap detection without code inspection. Increment counter on evaluate, compare before/after.

---

## Integration Points

### Vmap + Ops

Vmap relies on:
- Operations auto-propagating `batch_dims`
- `_physical` ops for batch manipulation
- Pytree system for nested inputs/outputs

### Compile + Core

Compile relies on:
- `GRAPH.evaluate(return_model=True)` to get compiled model
- Epoch tracking for side effect detection
- Three-tier shape caching for symbolic dimensions

### Future: Vmap + Compile

**Opportunity**: `compile(vmap(f))` or `vmap(compile(f))`

Currently untested, but designed to support both orderings. May require additional testing to ensure compatibility.

---

## Common Pitfalls

**"Vmap is for parallelism"**: No. It's for vectorization. Parallelism is sharding's job.

**"Compile is required for performance"**: No. Lazy eager already compiles on first access.

**"Dynamic dims must match actual dims"**: No. Dynamic dims define symbolic params, actual shape must match at runtime.

**"fullgraph=True prevents all side effects"**: No. Only detects graph evaluations, not print statements or mutations.

---

## Future Directions

### Grad Transform

Reverse-mode autodiff via OutputRefs graph walking. Would compose with vmap/compile.

### JVP/VJP Transforms

Explicit forward/reverse mode differentiation. Currently built into operations, could be exposed as transforms.

### Pmap (Parallel Map)

Like vmap but distributes across devices. Would use sharding infrastructure.

### Scan Transform

For loops with carried state. Requires different graph building strategy.
