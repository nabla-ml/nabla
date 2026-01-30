# Tensor System

[← Back to Core](../README.md)

## Philosophy
The Tensor system uses the **Facade Pattern** to strictly separate user-facing API from internal state. This allows a single opaque `Tensor` object to represent data that might be:
1.  **Symbolic**: A node in a computation graph.
2.  **Realized**: A concrete buffer on a GPU/TPU.
3.  **Sharded**: Distributed across multiple devices.

## Architecture & Internals

### The Dual-Object Model
*   **`Tensor` (API)**: Immutable-ish wrapper. Implements operator overloading (`__add__`), shape access, and NumPy compatibility. It holds a reference to `TensorImpl`.
*   **`TensorImpl` (State)**: The heavy lifter. It contains:
    *   `_values`: List of symbolic graph nodes (one per shard, for lazy execution).
    *   `_storages`: List of concrete data blocks (one per shard, for eager/realized execution).
    *   `output_refs`: Provenance (what op created this?).
    *   `sharding`: The distributed layout.

### Lazy State Management
Tensors exist in a superposition of states. A `Tensor` usually starts as **Unrealized** (holding `_values`). When `data` is requested (e.g., `.numpy()`), it triggers the graph engine to compile the subgraph and fill `_storages`, becoming **Realized**.

> [!NOTE] Design Decision: Facade Pattern
> *   **Choice**: Separate `Tensor` and `TensorImpl`.
> *   **Why**: Multi-output operations (like `split`) produce multiple `Tensor`s. These need to share back-references to the *same* creating operation for autodiff and graph traversal. `TensorImpl` handles this shared state (`OutputRefs`), while `Tensor` remains a lightweight handle.
> *   **Trade-off**: Double object allocation overhead for every tensor.

## Component Map

| File | Role | Exported Symbols |
| :--- | :--- | :--- |
| [`api.py`](api.py) | **The API** | **Classes**: `Tensor`<br>**Factory Methods**: `constant`, `full`, `zeros`, `ones`, `arange`, `uniform`, `gaussian`<br>**Key Properties**: `shape`, `dtype`, `device`, `sharded`, `local_shape`, `global_shape`<br>**Key Methods**: `numpy()`, `item()`, `realize()`, `shard()`, `with_sharding()`<br>**Re-exports**: `defaults`, `default_device`, `default_dtype`, `defaults_like` |
| [`impl.py`](impl.py) | **The State** | **Classes**: `TensorImpl`<br>**Internal Properties**: `_values`, `_storages`, `sharding`, `traced`, `dual`, `batch_dims`, `output_refs`<br>**Methods**: `realize()`, `gather()`, `to_numpy()`, `physical_global_shape`, `logical_local_shape` |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
