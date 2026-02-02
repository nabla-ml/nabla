# Operations System

[← Back to Root](../README.md)

> **Purpose**: This module defines all mathematical operations and the `Operation.__call__()` pipeline—the heart of Nabla's execution model.

## The `__call__` Pipeline

When you call any operation (e.g., `add(x, y)` or `x + y`), `Operation.__call__()` in [base.py](base.py) orchestrates a **9-step pipeline**. The key architectural insight: **steps 1-4 always run** (metadata computation), while **step 5 is conditional** based on `NABLA_EAGER_MAX_GRAPH`.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Operation.__call__() - The 9-Step Pipeline              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 1: METADATA COLLECTION  (base.py ~line 101)                    │    │
│  │                                                                     │    │
│  │   max_batch_dims, any_traced, any_sharded, any_has_tangent =        │    │
│  │       collect_metadata(args)                                        │    │
│  │                                                                     │    │
│  │   Purpose: Scan inputs to determine execution context               │    │
│  │   • max_batch_dims: for vmap axis translation                       │    │
│  │   • any_traced: whether to record OpNode for autodiff               │    │
│  │   • any_sharded: whether SPMD logic needed                          │    │
│  │   • any_has_tangent: whether JVP propagation needed                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 2: ADAPTATION & RESHARDING  (base.py ~line 104-105)            │    │
│  │                                                                     │    │
│  │   resharded_args, adapted_kwargs, predicted_output_spec, mesh,      │    │
│  │       reduce_axes = adapt_and_reshard(self, args, kwargs, ...)      │    │
│  │                                                                     │    │
│  │   Internally:                                                       │    │
│  │   • adapt_kwargs(): translate logical→physical (axis by batch_dims) │    │
│  │   • infer_output_sharding(): predict output sharding + reductions   │    │
│  │   • reshard_inputs(): insert AllGather/AllToAll if needed           │    │
│  │                                                                     │    │
│  │   Key outputs:                                                      │    │
│  │   • reduce_axes: mesh axes needing post-op reduction (step 8)       │    │
│  │   • resharded_args: inputs with correct shardings for execution     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 3: COMPUTE STRUCTURAL HASH  (base.py ~line 108)                │    │
│  │                                                                     │    │
│  │   op_hash = compute_structural_hash(self.name, resharded_args,      │    │
│  │                                     adapted_kwargs)                 │    │
│  │                                                                     │    │
│  │   Purpose: Create cache key for compiled model lookup               │    │
│  │   • Unrealized tensors: keyed by their own (op_hash, output_index)  │    │
│  │   • Realized tensors: keyed by (dtype, shape, sharding)             │    │
│  │   This enables GRAPH.evaluate() to skip graph building on cache hit │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 4: COMPUTE PHYSICAL SHAPES - ALWAYS RUNS  (base.py ~line 111)  │    │
│  │                                                                     │    │
│  │   if type(self).compute_physical_shape is Operation.compute_...:    │    │
│  │       raise RuntimeError(f"{self.__class__} must implement ...")    │    │
│  │                                                                     │    │
│  │   output_physical_shapes, output_shard_dtypes, output_shard_devices │    │
│  │       = self.compute_physical_shape(resharded_args, adapted_kwargs, │    │
│  │                                     predicted_output_spec)          │    │
│  │                                                                     │    │
│  │   CRITICAL: This must NOT build MAX graph nodes!                    │    │
│  │   It only computes metadata. Why always run?                        │    │
│  │   • Users need .shape immediately for control flow                  │    │
│  │   • Sharding propagation requires shapes                            │    │
│  │   • Broadcasting validation must happen now                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 5: EAGER EXECUTION (CONDITIONAL) ⚡  (base.py ~line 119)        │    │
│  │                                                                     │    │
│  │   execution_results = eager_execute(self, resharded_args, kwargs,   │    │
│  │                                     adapted_kwargs)                 │    │
│  │                                                                     │    │
│  │   if NABLA_EAGER_MAX_GRAPH=0 (default):                             │    │
│  │       → Returns None (graph building DEFERRED)                      │    │
│  │       → No op.execute() called here                                 │    │
│  │                                                                     │    │
│  │   if NABLA_EAGER_MAX_GRAPH=1:                                       │    │
│  │       → Calls op.execute(resharded_args, kwargs)                    │    │
│  │       → Builds MAX graph nodes immediately                          │    │
│  │       → Returns (shard_graph_values, output_sharding, mesh)         │    │
│  │                                                                     │    │
│  │   verify_eager_shapes() validates against step 4 if enabled         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 6: PACKAGING  (base.py ~line 122)                              │    │
│  │                                                                     │    │
│  │   output = package_outputs(self, execution_results,                 │    │
│  │       output_physical_shapes, output_shard_dtypes, ...)             │    │
│  │                                                                     │    │
│  │   Creates Tensor with metadata from step 4:                         │    │
│  │   • _physical_shapes, _shard_dtypes, _shard_devices always set      │    │
│  │                                                                     │    │
│  │   if EAGER_MAX_GRAPH (step 5 ran):                                  │    │
│  │       output._impl._graph_values = [TensorValue, ...]               │    │
│  │       output._impl.graph_values_epoch = GRAPH.epoch                 │    │
│  │                                                                     │    │
│  │   if DEFERRED (step 5 returned None):                               │    │
│  │       output._impl._graph_values = []                               │    │
│  │       output._impl.graph_values_epoch = -1  ← "PROMISE TENSOR"      │    │
│  │       GRAPH.add_unrealized(output._impl)   ← Track for later eval   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 7: SETUP OUTPUT REFS (OpNode Creation)  (base.py ~line 129)   │    │
│  │                                                                     │    │
│  │   self._setup_output_refs(output, resharded_args, kwargs,           │    │
│  │                           op_hash=op_hash)                          │    │
│  │                                                                     │    │
│  │   Creates OpNode with:                                              │    │
│  │   • _refs: tuple of output TensorImpls                              │    │
│  │   • op: the operation instance                                      │    │
│  │   • op_args: input TensorImpls (for backward traversal)             │    │
│  │   • op_kwargs: ORIGINAL kwargs (critical for rehydration!)          │    │
│  │   • _op_hash: structural hash for caching                           │    │
│  │                                                                     │    │
│  │   Key: Stores ORIGINAL kwargs, not adapted. execute() adapts        │    │
│  │   internally, so rehydration can replay correctly.                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 8: AUTO-REDUCTION  (base.py ~line 130)                         │    │
│  │                                                                     │    │
│  │   output = apply_auto_reduction(self, output, mesh, reduce_axes)    │    │
│  │                                                                     │    │
│  │   If contracting factors (like K in matmul) were sharded:           │    │
│  │   → Partial sums exist across devices                               │    │
│  │   → Inserts AllReduce with op.collective_reduce_type (sum/max/...)  │    │
│  │   → Updates sharding spec to remove reduced axes                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 9: JVP PROPAGATION  (base.py ~line 132-133)                    │    │
│  │                                                                     │    │
│  │   if any_has_tangent:                                               │    │
│  │       apply_jvp(self, args, output)                                 │    │
│  │                                                                     │    │
│  │   Forward-mode autodiff: propagate tangents via op.jvp_rule()       │    │
│  │   Attaches tangent to output._impl.tangent                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│                              return output                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why Graph Building is Deferred by Default

MAX graph construction has overhead. The default deferred mode optimizes for **cache hits**:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Why Defer Graph Building?                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SCENARIO: Training loop iteration                                          │
│  ───────────────────────────────────                                        │
│  for batch in dataloader:           # Thousands of iterations               │
│      loss = model(batch)            # Same computation structure            │
│      grads = grad(loss_fn)(params)  # Same structure, different data        │
│                                                                             │
│  WITH EAGER GRAPH BUILDING (NABLA_EAGER_MAX_GRAPH=1):                       │
│  • Every op.execute() builds MAX graph nodes                                │
│  • evaluate() compiles the graph                                            │
│  • Cache stores compiled model                                              │
│  • Next iteration: graph built AGAIN, then cache hit skips compile          │
│  • Wasted work: graph building every iteration                              │
│                                                                             │
│  WITH DEFERRED GRAPH BUILDING (default):                                    │
│  • Operations only compute metadata (shapes, dtypes)                        │
│  • evaluate() checks cache by op_hash                                       │
│  • Cache HIT → Skip graph building entirely, just run cached model          │
│  • Cache MISS → Build graph via _replay_trace_to_build_graph()              │
│  • Hot paths skip ALL graph construction overhead                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**But shapes must still be eager** because:

- User code needs `.shape` for control flow and debugging
- Sharding propagation needs shapes to plan data movement
- Type checking and broadcasting validation must happen immediately

---

### Why `execute` Receives Original kwargs

This is a **critical design decision** for trace rehydration:

```python
# In __call__:
raw_result = self.execute(resharded_args, kwargs)  # ← original kwargs!

# Inside execute (default implementation):
def execute(self, args, kwargs):
    adapted_kwargs = self.adapt_kwargs(args, kwargs, batch_dims)  # adapts here
    shard_results = spmd.execute_on_shards(self.kernel, args, adapted_kwargs, mesh)
```

During **rehydration** (replaying the trace for backward pass), we only have access to the original `kwargs` stored in `OpNode`. If `__call__` passed pre-adapted kwargs to `execute`, rehydration would fail or produce wrong results.

---

## Operation Class Hierarchy

```text
Operation (base class - defines __call__ pipeline)
│
├── UnaryOperation        # Single input: relu, exp, neg
│   └── Default execute loops over shards
│
├── BinaryOperation       # Two inputs: add, mul, matmul
│   └── Overrides __call__ to broadcast shapes BEFORE parent pipeline
│
├── LogicalAxisOperation  # Ops with axis kwargs: reduce, transpose
│   ├── Implements adapt_kwargs() to translate axis by batch_dims
│   ├── ReduceOperation       # reduce_sum, mean
│   └── LogicalShapeOperation # reshape, transpose
│
└── CollectiveOperation   # Communication ops with custom execution
```

### Key Methods Every Operation Must/Can Implement

| Method                                                  | Required          | Purpose                                        | Notes                                         |
| :------------------------------------------------------ | :---------------- | :--------------------------------------------- | :-------------------------------------------- |
| `name`                                                  | ✅                | String identifier                              | Property                                      |
| `kernel(*args, **kwargs)`                               | ✅                | MAX primitive (per-shard)                      | Called by `execute()` in shard loop           |
| `compute_physical_shape(args, kwargs, output_sharding)` | ✅ **NEW**        | Infer output shapes WITHOUT building MAX graph | Must return `(shapes, dtypes, devices)`       |
| `execute(args, kwargs)`                                 | Has default       | Custom execution logic                         | Override if needed. Receives ORIGINAL kwargs! |
| `adapt_kwargs(args, kwargs, batch_dims)`                | Has default       | Translate logical→physical kwargs              | Override for axis ops                         |
| `vjp_rule(primals, cotangent, output)`                  | For autodiff      | Backward gradient                              | See [core/autograd/](../core/autograd/)       |
| `jvp_rule(primals, tangents, output)`                   | For forward-mode  | Forward tangent                                | [utils.py](utils.py)                          |
| `sharding_rule(input_shapes, output_shapes)`            | For SPMD          | Factor-based sharding                          | propagation.py                                |
| `collective_reduce_type`                                | Has default="sum" | Reduction op type                              | For `apply_auto_reduction`                    |

### The `compute_physical_shape` Contract

Every operation **must** implement `compute_physical_shape`. This is how shapes are computed eagerly even when graph building is deferred:

```python
def compute_physical_shape(
    self, args: tuple, kwargs: dict, output_sharding: Any = None
) -> tuple[list[tuple[int, ...]], list[DType], list[Device]]:
    """
    Infer per-shard physical shapes, dtypes, and devices for outputs.

    CRITICAL: Must NOT build MAX graph nodes! Only compute metadata.

    Returns:
        - output_physical_shapes: list of shapes, one per shard
        - output_shard_dtypes: list of dtypes, one per shard
        - output_shard_devices: list of devices, one per shard
    """
```

**Why required?** In default mode, `op.execute()` isn't called during `__call__`. But users still need `.shape`, and sharding propagation needs shapes. This method provides that without the cost of graph building.

---

## Example: How a Matmul Executes (Default Mode)

```python
y = matmul(A, B)  # A: [M, K] sharded on K, B: [K, N]
```

**Step 1 (Metadata)**: Collects batch_dims=0, traced=True, sharded=True

**Step 2 (Adaptation)**:

- `infer_output_sharding` uses rule `"m k, k n -> m n"`
- Factor `k` is sharded → appears in reduce_axes
- If B's K dim has different sharding → `reshard_inputs` inserts communication

**Step 3 (Hash)**: `op_hash = ("matmul", (A_key, B_key), kwargs_key)`

**Step 4 (Physical Shapes)**: `compute_physical_shape` returns `([M, N], [M, N], ...)` per shard

**Step 5 (Eager Execution)**: `NABLA_EAGER_MAX_GRAPH=0` → Returns `None` (no graph building!)

**Step 6 (Packaging)**:

- Creates output tensor with `_physical_shapes=[(M,N), ...]`
- Sets `graph_values_epoch = -1` (promise tensor)
- Calls `GRAPH.add_unrealized(output._impl)`

**Step 7 (OpNode)**: Records `OpNode` with `op_hash` for cache lookup

**Step 8 (Auto-Reduction)**: `reduce_axes={'k_axis'}` → Inserts AllReduce operation

**Step 9 (JVP)**: Not applicable here

**Later, when `y.numpy()` is called:**

- `GRAPH.evaluate(y)` checks cache by `op_hash`
- Cache MISS: `_replay_trace_to_build_graph()` calls `matmul.execute()`
- Compiles and runs, stores to `y._impl._buffers`
- Caches compiled model for next time

---

## Component Map

| File                                      | Purpose                               | Key Exports                                                                                 |
| :---------------------------------------- | :------------------------------------ | :------------------------------------------------------------------------------------------ |
| [base.py](base.py)                        | **`__call__` pipeline**, base classes | `Operation`, `BinaryOperation`, `UnaryOperation`, `ReduceOperation`, `LogicalAxisOperation` |
| [utils.py](utils.py)                      | **Execution helpers**                 | `eager_execute`, `package_outputs`, `collect_metadata`, `apply_auto_reduction`, `apply_jvp` |
| [binary.py](binary.py)                    | Binary ops                            | `add`, `sub`, `mul`, `div`, `matmul`, `pow`                                                 |
| [unary.py](unary.py)                      | Unary ops                             | `relu`, `sigmoid`, `tanh`, `exp`, `log`, `neg`, `softmax`                                   |
| [reduction.py](reduction.py)              | Reductions                            | `reduce_sum`, `mean`, `reduce_max`, `reduce_min`                                            |
| [creation.py](creation.py)                | Tensor factories                      | `full`, `zeros`, `ones`, `arange`, `uniform`, `gaussian`                                    |
| [view/](view/README.md)                   | Shape ops                             | `reshape`, `transpose`, `squeeze`, `broadcast_to`, `gather`, `scatter`                      |
| [communication/](communication/README.md) | Collectives                           | `all_reduce`, `all_gather`, `shard`, `reshard`, `reduce_scatter`                            |
| [comparison.py](comparison.py)            | Comparisons                           | `equal`, `not_equal`, `greater`, `less`                                                     |
| [control_flow.py](control_flow.py)        | Control flow                          | `where`, `cond`, `while_loop`, `scan`                                                       |
| [multi_output.py](multi_output.py)        | Multi-output                          | `split`, `chunk`, `unbind`                                                                  |
| [custom_op.py](custom_op.py)              | Extensions                            | `call_custom_kernel`                                                                        |

---

## Maintenance Guide

> **AI Agents - Critical Rules**:
>
> 1. **`compute_physical_shape` required**: Every op must implement this. It cannot build graph nodes.
> 2. **`execute` contract**: MUST receive original kwargs and adapt internally. Breaking this breaks rehydration.
> 3. **`_setup_output_refs`**: Stores `resharded_args`, not original args. Backward pass needs correctly sharded inputs.
> 4. **OpNode stores original kwargs**: For rehydration correctness.
> 5. **Testing**: Every op needs gradient tests via `nabla.grad(fn)(x)`.
