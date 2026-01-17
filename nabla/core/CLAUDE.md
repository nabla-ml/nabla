# Nabla Core Internals

[← Back to Root](../CLAUDE.md)

## Philosophy
The `core` module is organized into semantic submodules to strictly separate concerns and avoid circular dependencies:

*   **`tensor`**: High-level API (`Tensor`) and low-level state (`TensorImpl`).
*   **`graph`**: The symbolic execution engine (`ComputeGraph`, `Trace`).
*   **`common`**: Shared utilities (`Context`, `PyTree`) used by both.

## Key Components

### 1. Tensor vs TensorImpl
We use the **Facade Pattern** to separate API from state.

-   **[`tensor/api.py`](tensor/api.py)** (`Tensor`): Immutable-ish wrapper. Handles API calls (`__add__`, `.shape`).
-   **[`tensor/impl.py`](tensor/impl.py)** (`TensorImpl`): Mutable state container. Holds:
    -   `_values`: Symbolic MAX graph nodes (when unrealized).
    -   `_storages`: Concrete data (when realized).
    -   `output_refs`: Provenance for autodiff.

**Why?** Multi-output operations (like `split`) return multiple `Tensor`s that share provenance in their `TensorImpl`s but manage their own values.

### 2. Dual State Lifecycle
A tensor exists in one of two mutually exclusive states:

| State | Backing Data | Description |
| :--- | :--- | :--- |
| **Unrealized** | `_values` (Symbolic) | A node in the graph. No memory allocated on device. |
| **Realized** | `_storages` (Concrete) | Actual data on device. Graph node is dropped to save memory. |

**Transition**: `graph.engine.ComputeGraph.evaluate([tensors])` compiles the subgraph leading to the requested tensors and fills their `_storages`.

### 3. The Global Graph
The singleton `GRAPH` in [`graph/engine.py`](graph/engine.py) captures all operations.
-   **No Context Managers**: The graph is always active.
-   **Epoch Tracking**: We increment an epoch counter on every evaluation to detect side effects in strict mode.

### 4. Pytree Support
Nabla supports JAX-like Pytrees (nested dicts/lists/tuples) natively via [`common/pytree.py`](common/pytree.py).

## Source Map

### `tensor/`
| File | Purpose |
| :--- | :--- |
| [`api.py`](tensor/api.py) | **The API**. `Tensor` class and operator overloading. |
| [`impl.py`](tensor/impl.py) | **The State**. `TensorImpl` class, internal metadata, and weakrefs. |

### `graph/`
| File | Purpose |
| :--- | :--- |
| [`engine.py`](graph/engine.py) | **The Brain**. `ComputeGraph`, compilation pipeline, and execution loop. |
| [`tracing.py`](graph/tracing.py) | **Provenance**. `Trace` object and `OutputRefs` for graph construction. |
| [`utils.py`](graph/utils.py) | **Traversal**. Topological sort and graph visualization tools. |

### `common/`
| File | Purpose |
| :--- | :--- |
| [`context.py`](common/context.py) | **Thread-local**. Default device and dtype management. |
| [`pytree.py`](common/pytree.py) | **Structure**. Utilities for flattening/unflattening nested containers. |
