# Core Internals

[← Back to Root](../README.md)

## Philosophy
The `core` module contains the engines that drive Nabla: State Management, Graph Compilation, and Distributed Execution. It is organized into semantic submodules to strictly separate concerns and avoid circular dependencies.

## Architecture & Internals

The core follows a layered architecture:

1. **Bottom (Common)**: Shared utilities (`pytree`, `context`) used across all modules.
2. **State (Tensor)**: Data containers (`Tensor`, `TensorImpl`) holding values, metadata, and sharding specs.
3. **Logic (Graph)**: The execution engine (`ComputeGraph`) that records operations as a DAG.
4. **Distribution (Sharding)**: The compiler that annotates graphs with SPMD execution information.
5. **Differentiation (Autograd)**: Gradient computation through reverse-mode autodiff.

### Execution Model: Logical vs Physical

Operations execute in two distinct layers to enable robust tracing and distributed execution:

**Logical Layer** (`op.execute`):
- Entry point for all operations
- Validates inputs (shapes, dtypes), handles broadcasting and type promotion
- Calls `preshard_inputs` to move data before computation
- Delegates to `physical_execute` for actual computation
- Wraps raw results into `nabla.Tensor` objects
- Creates symbolic graph nodes during tracing

**Physical Layer** (`physical_execute`):
- Loops over each device shard index
- Calls `op._transform_shard_kwargs()` to adapt arguments per shard
- Executes `op.maxpr()` on each shard (the MAX Engine primitive)
- Returns `PhysicalResult(symbolic_nodes, computed_values)`
- Must run inside `graph.context()` to access lazy values

This separation ensures:
- Traced graphs can be replayed without triggering recursive propagation
- Physical execution is independent of tracing machinery
- Clear boundary between user-facing API and SPMD implementation

### Strict Layering

Hierarchy: `sharding` imports `tensor`, `tensor` imports `graph`, `graph` imports `common`. Circular dependencies are strictly forbidden to maintain clean architecture as the codebase scales.

## Component Map

| Submodule | Purpose | Exported Symbols (in `nabla.core`) |
| :--- | :--- | :--- |
| **[`tensor/`](tensor/README.md)** | **State** | `Tensor`, `TensorImpl`, `OutputRefs` |
| **[`graph/`](graph/README.md)** | **Brain** | `ComputeGraph`, `GRAPH`, `driver_tensor_type`, `Trace`, `trace`, `get_operations_topological`, `get_all_impls_topological`, `print_trace_graph`, `apply_to_operations` |
| **[`sharding/`](sharding/README.md)** | **Distribution** | (Not directly re-exported via `core`, accessed via `nabla.core.sharding`) |
| **[`autograd/`](autograd/README.md)** | **Differentiation** | (Accessed via `nabla.grad`, `nabla.value_and_grad`, etc.) |
| **[`common/`](common/README.md)** | **Utils** | `defaults`, `default_device`, `default_dtype`, `defaults_like`, `tree_map`, `tree_flatten`, `tree_unflatten`, `tree_leaves`, `tree_structure`, `PyTreeDef`, `tensor_leaves`, `traced`, `untraced`, `with_batch_dims` |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
