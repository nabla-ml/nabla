# Communication Operations

[← Back to Ops Hub](../AGENTS.md)

## 🧠 Philosophy
Communication operations are explicit instructions to move data across the mesh. In Nabla, these are treated as regular `Operation`s that take a `Tensor`, perform a collective (via MAX/NCCL), and return a `Tensor` with a new `ShardingSpec`.

## 🏗️ Architecture & Internals

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

## 🗺️ Component Map

| File | Role | Key Concepts |
| :--- | :--- | :--- |
| [`base.py`](base.py) | **Base Class**. | `CollectiveOperation`, `estimate_cost` |
| [`shard.py`](shard.py) | **Entry**. | `shard(x, mesh, spec)` - The primary way to introduce sharding. |
| [`all_gather.py`](all_gather.py) | **Gather**. | `all_gather` - Collects data from all devices. |
| [`all_reduce.py`](all_reduce.py) | **Reduce**. | `all_reduce` - Sums/Means data across devices. |

## 🤖 Maintenance Guide
> **Note to AI Agents**: Update this file if you add new collective intrinsics.
> This file must remain the source of truth for high-level architecture.
