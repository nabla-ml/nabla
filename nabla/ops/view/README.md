# View Operations

[← Back to Ops](../README.md)

## Philosophy
View operations manipulate tensor **metadata** (shape, strides) without copying or moving the underlying data. In a distributed context, they must define how **logical** dimension changes map to **physical** sharding constraints.

## Architecture & Internals

### Conservative Reshape

View operations change tensor metadata without copying data. For distributed tensors:

- **Reshape crossing sharded boundaries**: Automatic AllGather to replicated state before reshaping
- **Rationale**: Correctness over performance. Complex strided layouts unsupported by compute kernels.

### Broadcasting

Broadcasting handled naturally by factor-based sharding. Operations automatically handle dimension expansion (e.g., `[features]` → `[batch, features]`) during propagation.

### Batch Dimensions

Internal ops for `vmap` maintain batch dimension invariants:
- `incr_batch_dims`, `decr_batch_dims`: Adjust batch dimension count
- `move_axis_to_batch_dims`, `move_axis_from_batch_dims`: Convert axes
- Batch dimensions always appear as leading axes in prefix order

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
