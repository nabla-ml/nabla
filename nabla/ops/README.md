# Operations System

[← Back to Root](../README.md)

> **Purpose**: This module defines all mathematical operations and the `Operation.__call__()` lifecycle—the heart of Nabla's execution model.

## The `__call__` Lifecycle

When you call any operation (e.g., `add(x, y)` or `x + y`), `Operation.__call__()` in [base.py](base.py) orchestrates six phases. Here's exactly what happens, mapped to the actual code:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Operation.__call__() - The 6 Phases                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ PHASE 1: METADATA COLLECTION  (lines ~99-113 in base.py)            │    │
│  │                                                                     │    │
│  │   max_batch_dims = 0                                                │    │
│  │   any_traced = False                                                │    │
│  │   any_sharded = False                                               │    │
│  │   pytree.tree_map(collect_metadata, args)  # scan all inputs        │    │
│  │                                                                     │    │
│  │   Purpose: Determine execution context from inputs                  │    │
│  │   • max_batch_dims: for vmap axis translation                       │    │
│  │   • any_traced: whether to record for autodiff                      │    │
│  │   • any_sharded: whether SPMD logic needed                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ PHASE 2: ADAPTATION (Pre-Execution)  (lines ~115-126 in base.py)    │    │
│  │                                                                     │    │
│  │   # 2a. Translate kwargs (logical → physical)                       │    │
│  │   adapted_kwargs = self.adapt_kwargs(args, kwargs, max_batch_dims)  │    │
│  │                                                                     │    │
│  │   # 2b. Infer what shardings are needed via factor propagation      │    │
│  │   predicted_output_spec, input_shardings, reduce_axes = \           │    │
│  │       spmd.infer_output_sharding(self, args, mesh, adapted_kwargs)  │    │
│  │                                                                     │    │
│  │   # 2c. Reshard inputs if current sharding ≠ required sharding      │    │
│  │   resharded_args = spmd.reshard_inputs(args, input_shardings, mesh) │    │
│  │   # ↑ May insert: shard(), reshard_tensor() via AllGather/AllToAll  │    │
│  │                                                                     │    │
│  │   Key outputs:                                                      │    │
│  │   • reduce_axes: mesh axes needing post-op reduction                │    │
│  │   • resharded_args: inputs with correct shardings for execution     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ PHASE 3: PHYSICAL EXECUTION  (lines ~128-130 in base.py)            │    │
│  │                                                                     │    │
│  │   with GRAPH.graph:                                                 │   │
│  │       raw_result = self.physical_execute(resharded_args, kwargs)    │    │
│  │                                                                     │    │
│  │   CRITICAL: physical_execute receives ORIGINAL kwargs, not adapted! │    │
│  │   It calls adapt_kwargs internally. This ensures rehydration works. │    │
│  │                                                                     │    │
│  │   Inside physical_execute (default implementation):                 │    │
│  │   • Adapts kwargs internally                                        │    │
│  │   • Loops over shards: spmd.execute_on_shards(self.maxpr, ...)      │    │
│  │   • Returns: (shard_values, output_sharding, mesh)                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ PHASE 4: PACKAGING  (lines ~132-177 in base.py)                     │    │
│  │                                                                     │    │
│  │   # Unpack raw_result (tuple or PhysicalResult object)              │    │
│  │   shard_values, output_sharding, res_mesh = raw_result              │    │
│  │                                                                     │    │
│  │   # Handle multi-output ops (split, chunk, etc.)                    │    │
│  │   if isinstance(first_shard, (list, tuple)):                        │    │
│  │       # Unzip: [(a0,b0), (a1,b1)] → ([a0,a1], [b0,b1])              │    │
│  │       ...                                                           │    │
│  │                                                                     │    │
│  │   # Wrap raw TensorValues into nabla.Tensor                         │    │
│  │   output = spmd.create_sharded_output(shard_values, ...)            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ PHASE 5: POST-OP COLLECTIVES  (lines ~179-183 in base.py)           │    │
│  │                                                                     │    │
│  │   if reduce_axes and mesh:                                          │    │
│  │       output = apply_auto_reduction(self, output, mesh, reduce_axes)│    │
│  │                                                                     │    │
│  │   What apply_auto_reduction does (execution_utils.py):              │    │
│  │   • Calls all_reduce_op.simulate_grouped_execution() per tensor     │    │
│  │   • Uses op.collective_reduce_type (sum/max/min/prod)               │    │
│  │   • Updates sharding spec to remove reduced axes                    │    │
│  │   • Sets up OutputRefs for the all_reduce operation                 │    │
│  │                                                                     │    │
│  │   When this triggers: contracting factors (like K in matmul)        │    │
│  │   that were sharded produce partial sums needing reduction.         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ PHASE 6: TRACING + JVP  (lines ~185-202 in base.py)                 │    │
│  │                                                                     │    │
│  │   # 6a. Record for autodiff (graph node creation)                   │    │
│  │   self._setup_output_refs(output, resharded_args, kwargs, ...)      │    │
│  │   # Creates OutputRefs with: op, inputs, kwargs for backward pass   │    │
│  │   # Note: stores resharded_args, not original args                  │    │
│  │                                                                     │    │
│  │   # 6b. Forward-mode autodiff (JVP)                                 │    │
│  │   if any_has_tangent:                                               │    │
│  │       apply_jvp(self, args, output)                                 │    │
│  │       # Calls op.jvp_rule(primals, tangents, output)                │    │
│  │       # Attaches tangent to output._impl.tangent                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│                              return output                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why `physical_execute` Receives Original kwargs

This is a **critical design decision** for trace rehydration:

```python
# In __call__:
raw_result = self.physical_execute(resharded_args, kwargs)  # ← original kwargs!

# Inside physical_execute (default implementation):
def physical_execute(self, args, kwargs):
    adapted_kwargs = self.adapt_kwargs(args, kwargs, batch_dims)  # adapts here
    shard_results = spmd.execute_on_shards(self.maxpr, args, adapted_kwargs, mesh)
```

During **rehydration** (replaying the trace for backward pass), we only have access to the original `kwargs` stored in `OutputRefs`. If `__call__` passed pre-adapted kwargs to `physical_execute`, rehydration would fail or produce wrong results.

---

## Operation Class Hierarchy

```
Operation (base class - defines __call__ lifecycle)
│
├── UnaryOperation        # Single input: relu, exp, neg
│   └── Default physical_execute loops over shards
│
├── BinaryOperation       # Two inputs: add, mul, matmul  
│   └── Overrides __call__ to broadcast shapes BEFORE parent lifecycle
│
├── LogicalAxisOperation  # Ops with axis kwargs: reduce, transpose
│   ├── Implements adapt_kwargs() to translate axis by batch_dims
│   ├── ReduceOperation       # reduce_sum, mean
│   └── LogicalShapeOperation # reshape, transpose
│
└── CollectiveOperation   # Communication ops with custom execution
```

### Key Methods Every Operation Can Implement

| Method | Required | Purpose | Code Location |
|--------|----------|---------|---------------|
| `name` | ✅ | String identifier | Property |
| `maxpr(*args, **kwargs)` | ✅ | MAX primitive (per-shard) | Called in shard loop |
| `physical_execute(args, kwargs)` | Has default | Custom execution logic | Override if needed |
| `adapt_kwargs(args, kwargs, batch_dims)` | Has default | Translate logical→physical kwargs | Override for axis ops |
| `vjp_rule(primals, cotangent, output)` | For autodiff | Backward gradient | [core/autograd/](../core/autograd/) |
| `jvp_rule(primals, tangents, output)` | For forward-mode | Forward tangent | execution_utils.py |
| `sharding_rule(input_shapes, output_shapes)` | For SPMD | Factor-based sharding | propagation.py |
| `collective_reduce_type` | Has default="sum" | Reduction op type | For apply_auto_reduction |

---

## Example: How a Matmul Executes

```python
y = matmul(A, B)  # A: [M, K] sharded on K, B: [K, N]
```

**Phase 1 (Metadata)**: Collects batch_dims=0, traced=True, sharded=True

**Phase 2 (Adaptation)**:
- `infer_output_sharding` uses rule `"m k, k n -> m n"`
- Factor `k` is sharded → appears in reduce_axes
- If B's K dim has different sharding → `reshard_inputs` inserts communication

**Phase 3 (Physical Execution)**:
- Loops over shards, calls `matmul.maxpr(A_shard, B_shard)`
- Each shard computes partial `[M, N]` result

**Phase 4 (Packaging)**: Wraps shard results into output Tensor

**Phase 5 (Post-Op Collectives)**:
- `reduce_axes={'k_axis'}` is non-empty
- `apply_auto_reduction` calls grouped all-reduce
- Produces correct global `[M, N]` result

**Phase 6 (Tracing)**: Records OutputRefs for backward pass

---

## Component Map

| File | Purpose | Key Exports |
|------|---------|-------------|
| [base.py](base.py) | **`__call__` lifecycle**, base classes | `Operation`, `BinaryOperation`, `UnaryOperation`, `ReduceOperation`, `LogicalAxisOperation` |
| [execution_utils.py](execution_utils.py) | **Post-op helpers** | `apply_auto_reduction`, `apply_jvp` |
| [binary.py](binary.py) | Binary ops | `add`, `sub`, `mul`, `div`, `matmul`, `pow` |
| [unary.py](unary.py) | Unary ops | `relu`, `sigmoid`, `tanh`, `exp`, `log`, `neg`, `softmax` |
| [reduction.py](reduction.py) | Reductions | `reduce_sum`, `mean`, `reduce_max`, `reduce_min` |
| [creation.py](creation.py) | Tensor factories | `full`, `zeros`, `ones`, `arange`, `uniform`, `gaussian` |
| [view/](view/README.md) | Shape ops | `reshape`, `transpose`, `squeeze`, `broadcast_to`, `gather`, `scatter` |
| [communication/](communication/README.md) | Collectives | `all_reduce`, `all_gather`, `shard`, `reshard`, `reduce_scatter` |
| [comparison.py](comparison.py) | Comparisons | `equal`, `not_equal`, `greater`, `less` |
| [control_flow.py](control_flow.py) | Control flow | `where`, `cond`, `while_loop`, `scan` |
| [multi_output.py](multi_output.py) | Multi-output | `split`, `chunk`, `unbind` |
| [custom_op.py](custom_op.py) | Extensions | `call_custom_kernel` |

---

## Maintenance Guide

> **AI Agents - Critical Rules**:
> 1. **`physical_execute` contract**: MUST receive original kwargs and adapt internally. Breaking this breaks rehydration.
> 2. **`_setup_output_refs`**: Stores `resharded_args`, not original args. Backward pass needs correctly sharded inputs.
> 3. **`adapt_kwargs`**: Only positive axis indices need translation. Negative indices index from end in both logical and physical.
> 4. **Testing**: Every op needs gradient tests via `nabla.grad(fn)(x)`.
