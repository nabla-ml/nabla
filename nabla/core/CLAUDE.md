# Nabla Core Internals

[← Back to Root](../CLAUDE.md)

## The Core Loop
Nabla masquerades as an eager library but executes lazily.

1.  **Facade**: User interacts with [`Tensor`](tensor.py).
2.  **Symbolic**: Operations build a MAX graph in the background via [`ComputeGraph`](compute_graph.py).
3.  **Compilation**: Data access triggers JIT compilation and execution.

## Key Components

### 1. Tensor vs TensorImpl
We use the **Facade Pattern** to separate API from state.

-   **[`Tensor`](tensor.py)**: Immutable-ish wrapper. Handles API calls (`__add__`, `.shape`).
-   **[`TensorImpl`](tensor_impl.py)**: Mutable state container. Holds:
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

**Transition**: `compute_graph.evaluate([tensors])` compiles the subgraph leading to the requested tensors and fills their `_storages`.

### 3. The Global Graph
The singleton `GRAPH` in [`compute_graph.py`](compute_graph.py) captures all operations.
-   **No Context Managers**: The graph is always active.
-   **Epoch Tracking**: We increment an epoch counter on every evaluation to detect side effects in strict mode.

### 4. Pytree Support
Nabla supports JAX-like Pytrees (nested dicts/lists/tuples) natively.
-   **[`pytree.py`](pytree.py)**: Utilities for flattening/unflattening structures.
-   **Usage**: `vmap` and `compile` work over arbitrary nested structures, not just tensors.

## Source Map

| File | Purpose |
| :--- | :--- |
| [`compute_graph.py`](compute_graph.py) | **The Brain**. Manages the global MAX graph, compilation pipeline, and execution loop. |
| [`tensor.py`](tensor.py) | **The API**. User-facing properties and operator overloading. |
| [`tensor_impl.py`](tensor_impl.py) | **The State**. Internal metadata, memory management, and weakrefs. |
| [`context.py`](context.py) | **Thread-local**. Default device and dtype management. |
| [`trace.py`](trace.py) | **Provenance**. Infrastructure for autodiff and backwards pass walking. |
