# Core Internals

[← Back to Root](../README.md)

## Philosophy
The `core` module contains the engines that drive Nabla: State Management, Graph Compilation, and Distributed Execution. It is organized into semantic submodules to strictly separate concerns and avoid circular dependencies.

## Architecture & Internals
The core follows a layered architecture:
1.  **Bottom (Common)**: Shared utilities (`pytree`, `context`) used by everyone.
2.  **State (Tensor)**: The data containers (`TensorImpl`) that hold values and metadata.
3.  **Logic (Graph)**: The engine (`ComputeGraph`) that records operations on Tensors.
4.  **Distribution (Sharding)**: The compiler pass that annotates the Graph with physical execution info.

> [!NOTE] Design Decision: Layered Core
> *   **Choice**: Strict hierarchy. `sharding` imports `tensor`, `tensor` imports `graph`, `graph` imports `common`.
> *   **Why**: Circular dependencies are the death of large Python projects.
> *   **Trade-off**: Sometimes requires "forward references" or delayed imports (e.g., `Tensor` knowing about `ShardingSpec` but not the `propagate` logic).

## Component Map

| Submodule | Purpose | Key File |
| :--- | :--- | :--- |
| **[`tensor/`](tensor/README.md)** | **State**. The `Tensor` object and `TensorImpl`. | [`tensor/impl.py`](tensor/impl.py) |
| **[`graph/`](graph/README.md)** | **Brain**. The `ComputeGraph` and compilation. | [`graph/engine.py`](graph/engine.py) |
| **[`sharding/`](sharding/README.md)** | **Distribution**. SPMD progagation engine. | [`sharding/spmd.py`](sharding/spmd.py) |
| **[`common/`](common/README.md)** | **Utils**. Pytree and Context contexts. | [`common/context.py`](common/context.py) |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
