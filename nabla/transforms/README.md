# Function Transforms

[← Back to Root](../README.md)

> **Purpose**: Transforms wrap user functions to alter execution semantics. They are the bridge between user code and hardware execution.

## Core Transforms

### `grad` / `value_and_grad` - Automatic Differentiation

Computes gradients via reverse-mode autodiff:

```python
def loss_fn(params, x, y):
    pred = model(params, x)
    return mean((pred - y) ** 2)

# Get gradient function
grad_fn = nabla.grad(loss_fn)
grads = grad_fn(params, x, y)

# Or get both value and gradients
val, grads = nabla.value_and_grad(loss_fn)(params, x, y)
```

**How it works** (see [core/autograd/](../core/autograd/README.md)):
1. **Trace**: Execute `loss_fn` with tracing enabled, capturing OpNode
2. **Rehydrate**: Restore all intermediate `_values` for current epoch
3. **Backward**: Walk OpNode in reverse, calling `vjp_rule` per operation
4. **Accumulate**: Sum cotangents where tensors are used multiple times

### `vmap` - Automatic Vectorization

Auto-batches operations over leading dimension(s):

```python
def single_example(x, w):
    return x @ w

# Vectorize over first axis of x, broadcast w
batched_fn = vmap(single_example, in_axes=(0, None))
y = batched_fn(x_batch, w)  # x_batch: [B, D], w: [D, D]
```

**How it works**:
1. **Increment batch_dims**: Input tensors get `batch_dims += 1`
2. **Execute normally**: Operations see leading dims as batch
3. **Axis translation**: `LogicalAxisOperation.adapt_kwargs` shifts axis indices by batch_dims
4. **Preserve semantics**: Negative axes still work (index from end)

```
vmap(f)                           Normal f
Input: [B, D]                     Input: [D]
        │                                │
        ▼                                ▼
  batch_dims=1                    batch_dims=0
        │                                │
        ▼                                ▼
  reduce(axis=0)  ────────►  reduce(axis=1 in physical)
        │                                │
        ▼                                ▼
Output: [B]                      Output: scalar
```

### `shard_map` - Distributed Execution

Distributes single-device code across a device mesh:

```python
mesh = DeviceMesh((8,), ["dp"])

@shard_map(mesh, in_specs=(P("dp"), P(None)), out_specs=P("dp"))
def data_parallel_forward(x, params):
    return model(params, x)
```

**How it works** - Trace-and-Replay:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        shard_map Execution Flow                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STEP 1: LOGICAL TRACE                                                  │
│  ─────────────────────                                                  │
│  • Execute function with logical tensor inputs                          │
│  • Operations execute eagerly, sharding propagates per-op               │
│  • Result: Computation graph with sharding annotations                  │
│                                                                         │
│  STEP 2: AUTO-SHARDING (Optional)                                       │
│  ────────────────────────────────                                       │
│  • If auto_sharding=True, extract graph + cost model                    │
│  • Run SimpleSolver ILP/heuristic to find optimal shardings             │
│  • Inject sharding constraints into replay                              │
│                                                                         │
│  STEP 3: PHYSICAL TRACE REPLAY                                          │
│  ─────────────────────────────                                          │
│  • Re-execute with tensor.dual (physical shards)                        │
│  • Same Python code, different execution path                           │
│  • Operations detect dual mode, execute per-shard                       │
│  • Result: Distributed computation with communication ops               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: First trace captures "what to do", second trace executes "how to do it distributed".

### `compile` - JIT Compilation

Defers graph compilation, caches for reuse:

```python
@nabla.compile
def fast_fn(x, y):
    return complex_computation(x, y)

# First call: trace, optimize, compile to MAX executable
result = fast_fn(x, y)

# Subsequent calls: run cached compiled code (no Python overhead)
result = fast_fn(x2, y2)
```

**Pipeline**:
1. **Trace**: Capture computation graph
2. **Optimize**: DCE, CSE, constant folding
3. **Lower**: Generate MAX executable
4. **Cache**: Key by (function, input_shapes, input_dtypes)
5. **Execute**: Bypass Python on subsequent calls

---

## Trace Rehydration (Critical Concept)

Rehydration is needed because graph `_values` are epoch-scoped. After `evaluate()`, old values become stale. Before backward pass (or trace replay), rehydration restores them:

```python
def refresh_graph_values(trace):
    # 1. Find all leaves (constants, inputs without output_refs)
    # 2. Ensure leaves are realized
    # 3. Add leaves to current graph epoch
    
    # 4. For each op in topological order:
    for output_refs in trace.nodes:
        op = output_refs.op
        args = wrap_as_tensors(output_refs.op_args)
        kwargs = output_refs.op_kwargs  # ← original kwargs!
        
        # Re-execute to get fresh values
        result = op.execute(args, kwargs)
        
        # Map values back to original TensorImpls
        for ref, new_impl in zip(output_refs._refs, result_impls):
            original_impl = ref()
            if original_impl:
                original_impl._values = new_impl._values
                original_impl.values_epoch = GRAPH.epoch
```

**Why original kwargs**: `execute` adapts internally. This is the only way rehydration can work correctly.

---

## Component Map

| File | Purpose | Key Exports |
|------|---------|-------------|
| [shard_map.py](shard_map.py) | Distributed execution | `shard_map` |
| [vmap.py](vmap.py) | Vectorization | `vmap` |
| [compile.py](compile.py) | JIT compilation | `compile`, `CompiledFunction` |

---

## Maintenance Guide

> **AI Agents - Critical Rules**:
> 1. **vmap + axis ops**: Ensure `adapt_kwargs` correctly handles batch_dims offset
> 2. **shard_map replay**: Must use same kwargs as original trace
> 3. **compile caching**: Cache key must include all shape/dtype info
