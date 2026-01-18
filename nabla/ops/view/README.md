# View Operations

[← Back to Ops Hub](../README.md)

## Philosophy
View operations manipulate tensor **metadata** (shape, strides) without copying or moving the underlying data. In a distributed context, they must define how **logical** dimension changes map to **physical** sharding constraints.

## Architecture & Internals

### Logical vs Physical View
*   **Logical**: User sees `reshape(1024) -> (32, 32)`.
*   **Physical**: Each shard might just change its local index math, OR if the reshape crosses sharded boundaries, it might require communication (global gather).

### Key Mechanisms
*   **`LogicalShapeOperation`**: Base class for ops that only change shape (`reshape`, `broadcast_to`).
*   **Conservative Reshape**: If a logical reshape crosses a sharded axis, we conservatively gather the tensor to avoid complex strided-sharding logic.
*   **Broadcast**: Handled via `OpShardingRuleTemplate`.
    *   `1 -> N`: Replicates the data along the new dimension (no communication if existing shards match).

> [!NOTE] Design Decision: Conservative Reshape
> *   **Choice**: `reshape()` gathers sharded tensors if the reshape touches sharded dimensions.
> *   **Why**: Correctness first. Arbitrary reshapes on distributed data can lead to extremely complex "strided sharding" layouts that are hard to support in kernels.
> *   **Trade-off**: Performance hit on distributed reshapes (requires network traffic).

## Component Map

| File | Role | Key Concepts |
| :--- | :--- | :--- |
| [`shape.py`](shape.py) | **Shape Ops**. | `ReshapeOp`, `BroadcastToOp`, `ConcatenateOp` |
| [`axes.py`](axes.py) | **Axis Ops**. | `transpose`, `squeeze`, `permute` |
| [`batch.py`](batch.py) | **Batch Ops**. | `unsqueeze` (add 1s), batch-aware transforms. |
| [`indexing.py`](indexing.py) | **Slicing**. | `__getitem__` logic, `slice`, `gather` (local). |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
