# Tensor System

[← Back to Reference](../../README.md)

## Philosophy
The Tensor system uses the **Facade Pattern** to strictly separate user-facing API from internal state. This allows a single opaque `Tensor` object to represent data that might be:
1.  **Symbolic**: A node in a computation graph.
2.  **Realized**: A concrete buffer on a GPU/TPU.
3.  **Sharded**: Distributed across multiple devices.

## Architecture & Internals

### The Dual-Object Model
*   **`Tensor` (API)**: Immutable-ish wrapper. Implements operator overloading (`__add__`), shape access, and NumPy compatibility. It holds a reference to `TensorImpl`.
*   **`TensorImpl` (State)**: The heavy lifter. It contains:
    *   `_values`: Symbolic graph nodes (for lazy execution).
    *   `_storages`: Concrete data (for eager/realized execution).
    *   `output_refs`: Provenance (what op created this?).
    *   `sharding`: The distributed layout.

### Lazy State Management
Tensors exist in a superposition of states. A `Tensor` usually starts as **Unrealized** (holding `_values`). When `data` is requested (e.g., `.numpy()`), it triggers the graph engine to compile the subgraph and fill `_storages`, becoming **Realized**.

> [!NOTE] Design Decision: Facade Pattern
> *   **Choice**: Separate `Tensor` and `TensorImpl`.
> *   **Why**: Multi-output operations (like `split`) produce multiple `Tensor`s. These need to share back-references to the *same* creating operation for autodiff and graph traversal. `TensorImpl` handles this shared state (`OutputRefs`), while `Tensor` remains a lightweight handle.
> *   **Trade-off**: Double object allocation overhead for every tensor.

## Component Map

| File | Role | Key Concepts |
| :--- | :--- | :--- |
| [`api.py`](api.py) | **The API**. | `Tensor`, `__add__`, `realize()`, `numpy()` |
| [`impl.py`](impl.py) | **The State**. | `TensorImpl`, `_storages`, `_values`, `sharding` |

## Maintenance Guide
> **Note to AI Agents**: Update this file if you modify the state management logic in `impl.py` or the public API in `api.py`.
> This file must remain the source of truth for high-level architecture.
