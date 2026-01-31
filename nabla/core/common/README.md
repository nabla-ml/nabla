# Common Core Utilities

[← Back to Core](../README.md)

## Philosophy

The `common` module provides the foundational infrastructure used across the entire library. It handles global state (context) and structural data manipulation (pytrees). It is designed to be **stateless logic** (pytree) and **thread-local state** (context) to avoid side effects in the graph engine.

## Architecture & Internals

### Pytree

Nabla uses a **JAX-compatible Pytree system**. A pytree is a tree of Python containers (list, tuple, dict) where the leaves are Tensors (or other data).

- **Flattening**: Converts arbitrary nested structures into a flat list of leaves + a `PyTreeDef` (structure).
- **Fast Traversal**: `tree_map` avoids intermediate object creation by traversing structures in-place.
- **Node Registry**: Supports custom types (like `OpNode`) via a registry system.
- **Tensor Awareness**: Includes helpers like `tensor_leaves`, `traced`, and `with_batch_dims` specifically for Nabla internal metadata.

### Context

We use `contextvars` to manage global defaults (DEVICE, DTYPE) in a thread-safe way.

- **Scopes**: `with default_device(d): ...` sets the context variable for the duration of the block.
- **Inheritance**: New tasks/threads inherit the current context, ensuring consistent tensor creation in async workflows.
- **Inference Session**: Manages a global `max.engine.api.InferenceSession` singleton (`_session()`) for compiling and running kernels.

> [!NOTE] Design Decision: JAX-style Pytrees
>
> - **Choice**: Implementing a native Python `tree_map` (recursively matching structures) instead of relying on `jax.tree_utils`.
> - **Why**: Zero dependencies. Allows us to define our own "nodes" (like `TensorImpl`) without interfering with JAX if the user mixes them.
> - **Trade-off**: Maintenance burden of complex structural matching logic.

## Component Map

| File | Role | Exported Symbols |
| :--- | :--- | :--- |
| [`context.py`](context.py) | **Global State** & **Defaults** | **Classes**: None; **Functions**: `defaults`, `default_device`, `default_dtype`, `defaults_like`, `contextvar_context`; **Internals**: `_session`, `_default_device`, `_default_dtype` |
| [`pytree.py`](pytree.py) | **Structure Handling** | **Classes**: `PyTreeDef`; **Functions**: `tree_flatten`, `tree_unflatten`, `tree_leaves`, `tree_structure`, `tree_map`; **Tensor Helpers**: `traced`, `untraced`, `with_batch_dims`, `tensor_leaves`, `is_tensor`, `is_tensor_value` |

## Maintenance Guide

> **Note to AI Agents**:
>
> 1. **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2. **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
