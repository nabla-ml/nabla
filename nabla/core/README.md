# Core Internals

[← Back to Root](../README.md)

## Philosophy
The `core` module contains the engines that drive Nabla: State Management, Graph Compilation, and Distributed Execution. It is organized into semantic submodules to strictly separate concerns and avoid circular dependencies.

## Architecture & Internals

Layered architecture with strict import hierarchy:

1. **Common**: Shared utilities (`pytree`, `context`) - no dependencies
2. **Tensor**: Data containers (`Tensor`, `TensorImpl`) - imports graph
3. **Graph**: Execution engine (`ComputeGraph`) - imports common
4. **Sharding**: SPMD execution via factor-based propagation - imports tensor
5. **Autograd**: Reverse-mode autodiff - imports graph, tensor

### Execution Model: Eager Operations with Graph Recording

Operations execute in a single eager pass with integrated sharding:

**Eager Execution**:
1. **Input Validation**: Check shapes, dtypes, broadcasting, type promotion
2. **Sharding Propagation**: Run three-phase factor-based algorithm (COLLECT → RESOLVE → UPDATE)
3. **Reshard**: Execute AllReduce/AllGather immediately if input shardings mismatch required shardings
4. **Per-Shard Execution**: Loop over device mesh shards, call `op.maxpr()` with local shard data
5. **Result Packaging**: Wrap results into new `Tensor` objects with sharding metadata

**Graph Recording** (happens simultaneously):
6. **Create Node**: Add `OutputRefs` to global `ComputeGraph` for tracing/autodiff

**Key Distinction**: 
- **Sharding is eager** - Data movement (AllReduce, AllGather) happens immediately during operation call
- **Graph is lazy** - Compilation to MAX executable deferred until data access (`.numpy()`, `print()`)

### The Dual Tensor System

Enables trace rehydration for `shard_map`:

**Logical Tensor**: User-facing `Tensor` object with global shape, sharding metadata  
**Physical Shards**: List of per-device shard objects (one per mesh device)

**Why Dual System**:
- Captured graphs can be replayed with different tensor implementations (logical vs physical)
- `shard_map` traces once with logical tensors, replays with physical tensors (dual execution paths)
- Physical replay: operations access `tensor.dual` to get per-shard data without triggering recursion
- Enables robust trace rehydration without re-executing Python logic

### Physical Execution Context

Physical execution (shard loop) must run inside `graph.context()` to:
- Access lazy tensor values safely without triggering recursive compilation
- Prevent graph recording during physical shard operations
- Enable clean separation between logical tracing and physical SPMD execution

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
