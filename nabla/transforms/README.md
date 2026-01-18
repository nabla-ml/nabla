# Function Transforms

[← Back to Root](../README.md)

## Philosophy
Transforms wrap a user function to alter its execution semantics. They are the bridge between "Logical Code" (what the user writes) and "Physical Execution" (what runs on hardware).

## Architecture & Internals

### 1. `shard_map` (Distribution)
Allows writing single-device code and running it on a mesh.
*   **Trace**: Runs the function with logical tensors to capture the graph.
*   **Solver (Optional)**: If `auto_sharding=True`, extracts the graph to JSON, solves for optimal sharding using `SimpleSolver`, and applies solution constraints.
*   **Replay**: Re-executes the graph using **Physical Tensors** (`Tensor.dual`). This injects the SPMD logic (propagation + resharding) into the graph.

### 2. `vmap` (Vectorization)
Auto-batches operations.
*   **Prefix Semantics**: We forcefully push batch dimensions to the *front* of the physical shape.
*   **Propagation**: When `vmap` sees a binary op, it unifies the batch dimensions of inputs (`max(rank(x), rank(y))`) and propagates them to the output.

> [!NOTE] Design Decision: Physical Trace & Replay
> *   **Choice**: `shard_map` traces logic, then replays physical ops.
> *   **Why**: Cleanest separation. The user writes math; we swap the inputs for "Sharded Tensors" and run the exact same math ops, but the ops themselves are smart enough to emit communication code when they see sharded inputs.
> *   **Trade-off**: Requires every Op to support sharded inputs.

## Component Map

| File | Role | Key Concepts |
| :--- | :--- | :--- |
| [`shard_map.py`](shard_map.py) | **Distribution**. | `shard_map`, `_ShardingGraphExtractor` |
| [`vmap.py`](vmap.py) | **Vectorization**. | `vmap`, `BatchTracer` |
| [`compile.py`](compile.py) | **Optimization**. | `compile`, `JIT` |

## Maintenance Guide
> **Note to AI Agents**: Update this file if you modify the trace/replay machinery.
> This file must remain the source of truth for high-level architecture.
