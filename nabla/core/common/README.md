# Common Core Utilities

[← Back to Reference](../../README.md)

## Philosophy
The `common` module provides the foundational infrastructure used across the entire library. It handles global state (context) and structural data manipulation (pytrees). It is designed to be **stateless logic** (pytree) and **thread-local state** (context) to avoid side effects in the graph engine.

## Architecture & Internals

### Pytree
Nabla uses a **JAX-compatible Pytree system**. A pytree is a tree of Python containers (list, tuple, dict) where the leaves are Tensors (or other data).
- **Flattening**: Converts arbitrary nested structures into a flat list of leaves + a `PyTreeDef` (structure).
- **Fast Traversal**: `tree_map` avoids intermediate object creation by traversing structures in-place.
- **Node Registry**: Supports custom types (like `OutputRefs`) via a registry system.

### Context
We use `contextvars` to manage global defaults (DEVICE, DTYPE) in a thread-safe way.
- **Scopes**: `with default_device(d): ...` sets the context variable for the duration of the block.
- **Inheritance**: New tasks/threads inherit the current context, ensuring consistent tensor creation in async workflows.

> [!NOTE] Design Decision: JAX-style Pytrees
> *   **Choice**: Implementing a native Python `tree_map` (recursively matching structures) instead of relying on `jax.tree_utils`.
> *   **Why**: Zero dependencies. Allows us to define our own "nodes" (like `TensorImpl`) without interfering with JAX if the user mixes them.
> *   **Trade-off**: Maintenance burden of complex structural matching logic.

## Component Map

| File | Role | Key Concepts |
| :--- | :--- | :--- |
| [`context.py`](context.py) | **Global State**. | `_DEFAULT_DEVICE`, `_DEFAULT_DTYPE`, `defaults_like()` |
| [`pytree.py`](pytree.py) | **Structure**. | `tree_map`, `tree_flatten`, `PyTreeDef` |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
