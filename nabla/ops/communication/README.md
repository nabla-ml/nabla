# Communication Operations

[← Back to Ops Hub](../README.md)

## Philosophy
Communication operations are explicit instructions to move data across the mesh. In Nabla, these are treated as regular `Operation`s that take a `Tensor`, perform a collective (via MAX/NCCL), and return a `Tensor` with a new `ShardingSpec`.

## Architecture & Internals

### The Collective Base
All comms inherit from `CollectiveOperation` in [`base.py`](base.py).
1.  **Hydrate**: Ensures inputs are realized (sync point).
2.  **Execute**: Calls the low-level MAX collective primitive.
3.  **Spec Update**: Computes the resulting `ShardingSpec` (e.g., `all_gather` removes a dim from the spec).

### Cost Modeling
These ops implement `estimate_cost(...)`. This is used by the Auto-Sharding solver (in `shard_map`) to weigh the cost of transferring bytes vs recomputing.

> [!NOTE] Design Decision: Explicit Comms
> *   **Choice**: Comms are just Ops.
> *   **Why**: Allows them to be traced, optimized, and inspected in the standard graph. Auto-sharding is just "inserting comm ops" into the graph.
> *   **Trade-off**: Users can shoot themselves in the foot by manually sharding incorrectly if they bypass the auto-spmd engine.

## Component Map

| File | Role | Key Concepts |
| :--- | :--- | :--- |
| [`base.py`](base.py) | **Base Class**. | `CollectiveOperation`, `estimate_cost` |
| [`shard.py`](shard.py) | **Entry**. | `shard(x, mesh, spec)` - The primary way to introduce sharding. |
| [`all_gather.py`](all_gather.py) | **Gather**. | `all_gather` - Collects data from all devices. |
| [`all_reduce.py`](all_reduce.py) | **Reduce**. | `all_reduce` - Sums/Means data across devices. |
| [`all_to_all.py`](all_to_all.py) | **Shuffle**. | `all_to_all` - Scatters and gathers data. |
| [`reduce_scatter.py`](reduce_scatter.py) | **Scatter**. | `reduce_scatter` - Reduces and then scatters. |
| [`reshard.py`](reshard.py) | **Compiler**. | `ReshardOp` - Automatic transition between specs. |
| [`p_permute.py`](p_permute.py) | **Permute**. | `ppermute` - Peer-to-peer permutation. |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
