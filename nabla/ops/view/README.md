# View Operations

[← Back to Ops](../README.md)

> **Purpose**: Manipulate tensor metadata (shape, axes) without copying data.

## How View Ops Differ from Normal Ops

### The Key Architectural Difference

**Normal ops** inherit `Operation.execute` which:

1. Gets mesh from args
2. Calls `spmd.execute_on_shards(self.kernel, args, kwargs, mesh)`
3. Returns `(shard_results, None, mesh)`

**View ops** inherit from specialized base classes that **override `execute`** to:

1. **Adapt kwargs** (translate logical axis/shape → physical)
2. THEN call `spmd.execute_on_shards` with adapted kwargs

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        View Op Execution Flow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LogicalAxisOperation (squeeze, unsqueeze, swap_axes, reduce_sum)           │
│  ─────────────────────────────────────────────────────────────────          │
│                                                                             │
│  execute(args, kwargs):                                                     │
│      # 1. Compute batch_dims from args                                      │
│      max_batch_dims = max(t.batch_dims for t in args)                       │
│                                                                             │
│      # 2. ADAPT: shift axis indices by batch_dims                           │
│      adapted_kwargs = self.adapt_kwargs(args, kwargs, max_batch_dims)       │
│      # Example: axis=0 with batch_dims=2 → axis=2                           │
│                                                                             │
│      # 3. Execute kernel on each shard with ADAPTED kwargs                  │
│      shard_results = spmd.execute_on_shards(                                │
│          self.kernel, args, adapted_kwargs, mesh                            │
│      )                                                                      │
│                                                                             │
│      return (shard_results, None, mesh)                                     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LogicalShapeOperation (reshape, broadcast_to)                              │
│  ────────────────────────────────────────────                               │
│                                                                             │
│  execute(args, kwargs):                                                     │
│      # 1. Compute batch_dims from args                                      │
│      max_batch_dims = max(t.batch_dims for t in args)                       │
│                                                                             │
│      # 2. ADAPT: prepend batch shape to target shape                        │
│      adapted_kwargs = self.adapt_kwargs(args, kwargs, max_batch_dims)       │
│      # Example: shape=(10,5) with batch_shape=(B1,B2) → shape=(B1,B2,10,5)  │
│                                                                             │
│      # 3. Execute kernel on each shard with ADAPTED kwargs                  │
│      shard_results = spmd.execute_on_shards(                                │
│          self.kernel, args, adapted_kwargs, mesh                            │
│      )                                                                      │
│                                                                             │
│      return (shard_results, None, mesh)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Matters

The user writes **logical** code:

```python
x = squeeze(x, axis=0)  # User thinks: "remove first axis"
```

But under vmap with `batch_dims=2`, the physical tensor has shape `(B1, B2, ...)`. The `axis=0` the user specified is actually `axis=2` in physical space:

```text
Logical:   (10, 5)        squeeze axis=0 → (5,)
Physical:  (B1, B2, 10, 5) squeeze axis=2 → (B1, B2, 5)
```

`LogicalAxisOperation.adapt_kwargs` does this translation automatically.

## The Base Class Hierarchy

```text
Operation (base.py)
    ├── LogicalAxisOperation           # Translates axis kwargs
    │       ├── ReduceOperation        # reduce_sum, reduce_mean, etc.
    │       ├── UnsqueezeOp           
    │       ├── SqueezeOp             
    │       ├── SwapAxesOp            
    │       └── ConcatenateOp         
    │
    └── LogicalShapeOperation          # Translates shape kwargs
            ├── ReshapeOp             
            └── BroadcastToOp         
```

## Batch Dimension Internals

The `batch.py` ops are used internally by `vmap` to manage batch dimensions:

| Op | Purpose |
| :--- | :--- |
| `incr_batch_dims` | Mark another leading dim as batch (bump counter) |
| `decr_batch_dims` | Unmark leading batch dim (decrement counter) |
| `move_axis_to_batch_dims` | Move axis to front, mark as batch |
| `move_axis_from_batch_dims` | Move batch axis to specified position |
| `broadcast_batch_dims` | Broadcast tensor to match batch shape |

Batch dimensions are always **leading axes** in prefix order.

## Component Map

| File | Role | Exported Symbols |
| :--- | :--- | :--- |
| [`shape.py`](shape.py) | **Shape Transformation** | **Classes**: `BroadcastToOp`, `ReshapeOp`, `SliceUpdateOp`, `SliceTensorOp`, `ConcatenateOp`, `BroadcastToPhysicalOp`; **Functions**: `broadcast_to`, `reshape`, `slice_tensor`, `slice_update`, `concatenate`, `stack`, `broadcast_to_physical` |
| [`axes.py`](axes.py) | **Axis Manipulation** | **Classes**: `UnsqueezeOp`, `SqueezeOp`, `SwapAxesOp`, `MoveAxisOp`, `UnsqueezePhysicalOp`, `SqueezePhysicalOp`; **Functions**: `unsqueeze`, `squeeze`, `swap_axes`, `moveaxis`, `unsqueeze_physical`, `squeeze_physical` |
| [`batch.py`](batch.py) | **Batch Dim Internals** | **Classes**: `IncrBatchDimsOp`, `DecrBatchDimsOp`, `MoveAxisToBatchDimsOp`, `MoveAxisFromBatchDimsOp`, `BroadcastBatchDimsOp`; **Functions**: `incr_batch_dims`, `decr_batch_dims`, `move_axis_to_batch_dims`, `move_axis_from_batch_dims`, `broadcast_batch_dims` |
| [`indexing.py`](indexing.py) | **Indexing/Slicing** | **Classes**: `GatherOp`, `ScatterOp`; **Functions**: `gather`, `scatter` |

## Maintenance Guide

> **Note to AI Agents**:
>
> 1. **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2. **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
