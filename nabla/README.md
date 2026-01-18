# Nabla Architecture

## Philosophy
Nabla implies a philosophy of **"Lazy-Eager" Execution** on top of a **Unified SPMD Runtime**.
*   **Lazy-Eager**: Code looks eager (like PyTorch), but is lazily traced into a graph and compiled just-in-time when you access data.
*   **Unified SPMD**: You write code for a single device. We use a compiler pass (Sharding Propagation) to automatically parallelize it across a cluster of TPUs/GPUs.

## Architecture & Internals

### The Lifecycle
1.  **Interact**: User manipulates `Tensor` objects (pointers to graph nodes).
2.  **Trace**: Calls like `x + y` record nodes in the global `ComputeGraph`.
3.  **Compile**: Accessing data (e.g., `print(x)`) triggers the compiler.
4.  **Shard**: The compiler runs **Sharding Propagation** to split the graph for the mesh.
5.  **Execute**: The graph runs on the accelerator.

> [!NOTE] Design Decision: The "Dual" System
> *   **Choice**: Every sharded logical tensor is backed by a graph of *physical* shards.
> *   **Why**: Allows us to reuse the same graph engine for both single-device and distributed execution. Scalable.
> *   **Trade-off**: Complexity in the `shard_map` trace-and-replay engine.

## Module Map

| Module | Purpose | Documentation |
| :--- | :--- | :--- |
| **`core`** | **The Engine**. State, Graph, Sharding. | [**Read Docs**](core/README.md) |
| **`ops`** | **The Logic**. Operation definitions. | [**Read Docs**](ops/README.md) |
| **`transforms`** | **The Bridge**. `vmap`, `shard_map`. | [**Read Docs**](transforms/README.md) |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Read Recursively**: Start here, then follow links to understand specific subsystems.
> 2.  **Keep Updated**: If you refactor code, YOU MUST update the corresponding `README.md` file.
> 3.  **Template**: Always use the `Philosophy` -> `Internals` -> `Map` -> `Maintenance` structure.
