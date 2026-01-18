# Graph Engine & Tracing

[← Back to Reference](../../README.md)

## Philosophy
The Graph Engine is the "Brain" of Nabla. It is responsible for **capturing** user operations into a symbolic graph, **compiling** that graph into a Maxwell model, and **executing** it. It employs a **Lazy-Eager** strategy: the graph is always building (Lazy), but execution is triggered implicitly when data is inspected (Eager feel).

## Architecture & Internals

### The Global Singleton
We use a singleton `GRAPH` (`ComputeGraph`) to capture operations. There are no "graph contexts" for the user to manage.
*   **Weak References**: The graph tracks unrealized tensors via `weakref.WeakValueDictionary`. If a user discards a tensor variable in Python, it is garbage collected, and the graph engine automatically drops the corresponding dead nodes before compilation.
*   **Epochs**: We track an `_info_epoch` counter. Every time the graph is compiled/executed, the epoch increments. This helps invalidate staleness in strict-mode compilation.

### Tracing
The `Trace` object captures a subgraph between specific input tensors and output tensors.
*   **`OutputRefs`**: A struct shared by all sibling outputs of an operation. It's the graph node. It holds references to:
    *   `op`: The operation instance.
    *   `op_args`: The inputs to the op.
    *   `op_kwargs`: Configuration.

> [!NOTE] Design Decision: Singleton Graph
> *   **Choice**: One global `ComputeGraph` instead of per-thread or explicit graph scopes.
> *   **Why**: Maximizes distinct "PyTorch-like" feel. Users never accidentally define ops "outside" a graph.
> *   **Trade-off**: Harder to support multi-threaded graph construction (requires strictly thread-local value stacks if we go parallel).

## Component Map

| File | Role | Key Concepts |
| :--- | :--- | :--- |
| [`engine.py`](engine.py) | **Execution Loop**. | `ComputeGraph`, `evaluate`, `_compile_and_execute` |
| [`tracing.py`](tracing.py) | **Graph Structure**. | `Trace`, `OutputRefs`, `trace()`, `GraphPrinter` |
| [`utils.py`](utils.py) | **Algorithms**. | Toposort, cycle detection. |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
