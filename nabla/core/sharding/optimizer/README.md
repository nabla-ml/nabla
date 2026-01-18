# Auto-Sharding Optimizer

[← Back to Sharding](../README.md)

## Philosophy
The Optimizer uses a **Cost-Based Heuristic** approach to determine the optimal distributed layout for a computation graph. It balances:
1.  **Compute Efficiency**: Maximizing parallelization (e.g., splitting batch for Data Parallelism).
2.  **Communication Overhead**: Minimizing network traffic (e.g., avoiding AllReduce in Model Parallelism if generic DP is cheaper).

## Architecture & Internals

### The Solver Pipeline (`SimpleSolver`)
The solving process involves four phases:
1.  **Parse**: Converts the logical trace into a graph of `ShardingSpec` constraints.
2.  **Seed**: Identifies heavy operations (like `MatMul`) and uses a Cost Model to propose initial constraints (e.g., "Split K axis" vs "Split Batch axis").
3.  **Propagate**: Runs the bidirectional propagation algorithm to flow these seeded constraints to the rest of the graph.
4.  **Export**: Returns a solution dictionary mapping Node IDs to input/output sharding specs.

### Cost Model
The solver estimates costs to decide between strategies:
*   **Data Parallel (DP)**: Split Batch dimension. Cost = `Compute / N`. Comm = 0 (usually).
*   **Model Parallel (MP)**: Split Contracting dimension. Cost = `Compute / N + AllReduce(Bytes)`.
It queries `AllReduceOp.estimate_cost()` to get realistic communication latency estimates.

## Component Map

| File | Role | Exported Symbols |
| :--- | :--- | :--- |
| [`simple_solver.py`](simple_solver.py) | **The Solver** | **Classes**: `SimpleSolver`<br>**Methods**: `solve`, `_seed_matmul`, `_propagate_node` |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
