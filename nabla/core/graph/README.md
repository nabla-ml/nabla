# Graph Engine & Tracing

[← Back to Core](../README.md)

## Philosophy
The Graph Engine is the "Brain" of Nabla. It is responsible for **capturing** user operations into a symbolic graph, **compiling** that graph into a Maxwell model, and **executing** it. It employs a **Lazy-Eager** strategy: the graph is always building (Lazy), but execution is triggered implicitly when data is inspected (Eager feel).

## Architecture & Internals

### The Global Singleton

The singleton `GRAPH` (`ComputeGraph`) captures all operations. No explicit graph contexts for users to manage.

**Weak References**: The graph tracks unrealized tensors via `weakref.WeakValueDictionary`. Discarded Python variables are automatically garbage collected, and the graph engine drops corresponding dead nodes before compilation.

**Epochs**: Internal `_info_epoch` counter tracks staleness for cache invalidation.

**Lazy Evaluation**: Operations are lazily added to the graph. `evaluate()` compiles and runs the subgraph needed to realize specific tensors.

### Tracing

The `Trace` object captures a subgraph between specific input tensors and output tensors.

**`OutputRefs`**: Shared node structure for all sibling outputs of an operation. Holds:
- `op`: The operation instance
- `op_args`: Input tensors
- `op_kwargs`: Configuration
- `_refs`: Weak references to output `TensorImpl` objects

### Execution Loop

The dispatch loop coordinates logical and physical execution:

1. **Input Validation**: Operation receives arguments, validates shapes/dtypes
2. **Pre-sharding**: `preshard_inputs` moves data to required shardings
3. **Physical Execution**: `physical_execute` runs SPMD kernel on each shard
4. **Result Wrapping**: Raw values wrapped into `Tensor` objects with graph nodes
5. **Graph Recording**: `OutputRefs` node added to global graph

**Context Management**: Physical execution must run inside `graph.context()` to safely access lazy tensor values without triggering recursive compilation.

## Component Map

| File | Role | Exported Symbols |
| :--- | :--- | :--- |
| [`engine.py`](engine.py) | **Execution Loop** | **Classes**: `ComputeGraph`<br>**Singleton**: `GRAPH`<br>**Functions**: `seed`, `driver_tensor_type`<br>**Key Methods**: `add_input`, `add_unrealized`, `evaluate` |
| [`tracing.py`](tracing.py) | **Graph Structure** | **Classes**: `Trace`, `OutputRefs`, `GraphPrinter`<br>**Functions**: `trace`<br>**Key Methods**: `Trace.compute`, `Trace.__str__` (visualization) |
| [`utils.py`](utils.py) | **Algorithms** | **Functions**: `get_operations_topological`, `get_all_impls_topological`, `print_trace_graph`, `apply_to_operations` |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
