# Core Internals

[← Back to Root](../AGENTS.md)

## 🧠 Philosophy
The `core` module contains the engines that drive Nabla: State Management, Graph Compilation, and Distributed Execution. It is organized into semantic submodules to strictly separate concerns and avoid circular dependencies.

## 🏗️ Architecture & Internals
The core follows a layered architecture:
1.  **Bottom (Common)**: Shared utilities (`pytree`, `context`) used by everyone.
2.  **State (Tensor)**: The data containers (`TensorImpl`) that hold values and metadata.
3.  **Logic (Graph)**: The engine (`ComputeGraph`) that records operations on Tensors.
4.  **Distribution (Sharding)**: The compiler pass that annotates the Graph with physical execution info.

> [!NOTE] Design Decision: Layered Core
> *   **Choice**: Strict hierarchy. `sharding` imports `tensor`, `tensor` imports `graph`, `graph` imports `common`.
> *   **Why**: Circular dependencies are the death of large Python projects.
> *   **Trade-off**: Sometimes requires "forward references" or delayed imports (e.g., `Tensor` knowing about `ShardingSpec` but not the `propagate` logic).

## 🗺️ Component Map

| Submodule | Purpose | Key File |
| :--- | :--- | :--- |
| **[`tensor/`](tensor/AGENTS.md)** | **State**. The `Tensor` object and `TensorImpl`. | [`tensor/impl.py`](tensor/impl.py) |
| **[`graph/`](graph/AGENTS.md)** | **Brain**. The `ComputeGraph` and compilation. | [`graph/engine.py`](graph/engine.py) |
| **[`sharding/`](sharding/AGENTS.md)** | **Distribution**. SPMD progagation engine. | [`sharding/spmd.py`](sharding/spmd.py) |
| **[`common/`](common/AGENTS.md)** | **Utils**. Pytree and Context contexts. | [`common/context.py`](common/context.py) |

## 🤖 Maintenance Guide
> **Note to AI Agents**: Update this file if you add new submodules to `core`.
> This file must remain the source of truth for high-level architecture.
