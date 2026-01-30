# View Operations

[← Back to Ops Hub](../README.md)

## Philosophy
View operations manipulate tensor **metadata** (shape, strides) without copying or moving the underlying data. In a distributed context, they must define how **logical** dimension changes map to **physical** sharding constraints.

## Architecture & Internals

### Logical vs Physical Views

**Logical View**: User-facing shape transformations operating on global tensor semantics.
- `reshape((1024,))` → `(32, 32)`: Changes logical shape
- `unsqueeze(axis=0)`: Adds dimension of size 1
- `transpose()`: Swaps dimensions

**Physical View**: Per-shard adaptations for distributed execution.
- Each shard computes its local shape transformation
- Sharding propagation determines if transformation is valid without communication

### Conservative Reshape

**Problem**: Arbitrary reshapes on sharded tensors can create complex strided layouts unsupported by compute kernels.

**Solution**: When reshape crosses sharded boundaries, gather the tensor to replicated state before reshaping.

Example:
```python
# Tensor sharded on dim 0: shape [1024/8, 128] per shard
x = x.shard(mesh, P("dp"))
# Reshape [1024, 128] -> [32, 32, 128]
y = x.reshape(32, 32, 128)
# First dimension splits 1024 -> (32, 32)
# Since original dim 0 was sharded, automatic AllGather inserted
```

This prioritizes correctness over performance for distributed reshapes.

### Broadcasting in SPMD

Broadcasting naturally handled by factor-based sharding:

**Elementwise with broadcasting**:
```python
x = x.shard(mesh, P("dp"))  # Shape: [batch/dp, features]
y = y.shard(mesh, P())       # Shape: [features] replicated
z = x + y  # Broadcasting: [batch/dp, features] + [features]
# Factor rule handles [1] -> [batch] dimension addition
# Output: [batch/dp, features] - preserves sharding
```

### Batch Dimension Operations

Internal operations for `vmap` transform:

- `incr_batch_dims`: Increment batch dimension count (for nested vmaps)
- `decr_batch_dims`: Decrement after vmap unwrapping
- `move_axis_to_batch_dims`: Convert regular axis to batch axis
- `move_axis_from_batch_dims`: Extract batch axis to regular axis

These maintain the invariant that batch dimensions are always leading dimensions in prefix order.

## Component Map

| File | Role | Exported Symbols |
| :--- | :--- | :--- |
| [`shape.py`](shape.py) | **Shape Transformation** | **Classes**: `ReshapeOp`, `BroadcastToOp`, `ConcatenateOp`, `SliceTensorOp`, `BroadcastToPhysicalOp`<br>**Functions**: `reshape`, `broadcast_to`, `concatenate`, `stack`, `slice_tensor`, `broadcast_to_physical` |
| [`axes.py`](axes.py) | **Axis Manipulation** | **Classes**: `UnsqueezeOp`, `SqueezeOp`, `SwapAxesOp`, `MoveAxisOp`, `UnsqueezePhysicalOp`, `SqueezePhysicalOp`<br>**Functions**: `unsqueeze`, `squeeze`, `swap_axes`, `moveaxis`, `unsqueeze_physical`, `squeeze_physical` |
| [`batch.py`](batch.py) | **Batch Dim Internals** | **Classes**: `IncrBatchDimsOp`, `DecrBatchDimsOp`, `MoveAxisToBatchDimsOp`, `MoveAxisFromBatchDimsOp`, `BroadcastBatchDimsOp`<br>**Functions**: `incr_batch_dims`, `decr_batch_dims`, `move_axis_to_batch_dims`, `move_axis_from_batch_dims`, `broadcast_batch_dims` |
| [`indexing.py`](indexing.py) | **Indexing/Slicing** | **Classes**: `GatherOp`, `ScatterOp`<br>**Functions**: `gather`, `scatter` |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
