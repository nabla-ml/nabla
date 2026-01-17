# Nabla Core Architecture

[← Back to Root](../CLAUDE.md)

## Philosophy
Nabla combines **imperative usability** with **symbolic efficiency** via a Lazy-Eager execution model. You write Python, we build graphs.

## The Lifecycle

```mermaid
graph LR
    User[User Code] -->|API Calls| Tensor[Tensor Wrapper]
    Tensor -->|Trace| Graph[ComputeGraph]
    Graph -->|JIT Compile| Model[MAX Model]
    Model -->|Execute| Result[Concrete Data]
```

1.  **Interact**: User creates/manipulates `Tensor` objects (pointers).
2.  **Trace**: `Operation` calls record nodes in the global `ComputeGraph`.
3.  **Compile**: Accessing data (e.g., `print(x)`) triggers the `ComptueGraph.evaluate()` loop.
4.  **Execute**: The graph is compiled to a MAX model and executed on the device(s).

## Module Map

| Module | Purpose | Key Entry Point |
| :--- | :--- | :--- |
| **[`core/`](core/CLAUDE.md)** | **Runtime Engine**. Manages state, graphs, and compilation. | [`compute_graph.py`](core/compute_graph.py) |
| **[`ops/`](ops/CLAUDE.md)** | **Operation Library**. Defines the `Operation` ABC and dispatch logic. | [`operation.py`](ops/operation.py) |
| **[`sharding/`](sharding/CLAUDE.md)** | **Distributed**. Factor-based propagation and specifications. | [`propagation.py`](sharding/propagation.py) |
| **[`transforms/`](transforms/CLAUDE.md)** | **Function Transforms**. `vmap`, `compile`, and autodiff. | [`vmap.py`](transforms/vmap.py) |

## Key Concepts

### 1. Dual-State Tensors
Every [`Tensor`](core/tensor.py) is in one of two states:
-   **Unrealized (Traced)**: Holds a symbolic `max.graph.Value`.
-   **Realized (Concrete)**: Holds a `max.driver.Tensor`.

### 2. Singleton Operations
All ops (`Add`, `Matmul`, etc.) are stateless singletons in `ops/`.
-   **Dispatch**: Logic for shape inference, sharding propagation, and graph node creation.
-   **Extensibility**: New ops just need to inherit `Operation` and register themselves.

### 3. Unified Distributed System
Sharding is a compilation pass.
-   **Logical Graph**: User sees full tensors.
-   **Physical Execution**: Compiler (Auto-sharding + SPMD) partitions execution across the mesh.
