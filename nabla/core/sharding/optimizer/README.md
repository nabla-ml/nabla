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

Operations implement cost estimation methods to guide sharding decisions:

**Compute Cost**: Estimated FLOPs for the operation.
- Matmul: `2 * M * N * K` FLOPs
- Binary elementwise: `prod(output_shape)` operations
- Reductions: `prod(input_shape)` operations

**Memory Cost**: Estimated bytes transferred.
- Input/output sizes in bytes
- Intermediate activation memory

**Communication Cost**: Network traffic from resharding.
- AllReduce: `2 * (N - 1) / N * bytes` for ring algorithm
- AllGather: `(N - 1) / N * bytes`
- AllToAll: `(N - 1) / N * bytes`

### Example: Matmul Sharding Decision

Consider `C = matmul(A, B)` with shapes `(1024, 512) @ (512, 2048)` on 8 devices:

**Data Parallel (split batch dimension)**:
- Compute: `2 * 1024 * 512 * 2048 / 8 = 268M FLOPs/device`
- Communication: None (no dimension contraction)
- Total cost: Low

**Tensor Parallel (split contracting dimension K)**:
- Compute: `2 * 1024 * 512 * 2048 / 8 = 268M FLOPs/device`
- Communication: AllReduce of `1024 * 2048 * 4 bytes = 8MB`
- Total cost: Higher due to AllReduce

**Model Parallel (split output features)**:
- Compute: `2 * 1024 * 512 * 2048 / 8 = 268M FLOPs/device`
- Communication: None (output stays sharded)
- Total cost: Low

The solver compares these strategies and selects the one minimizing `compute_time + communication_time`.

## Component Map

| File | Role | Exported Symbols |
| :--- | :--- | :--- |
| [`simple_solver.py`](simple_solver.py) | **The Solver** | **Classes**: `SimpleSolver`<br>**Methods**: `solve`, `_seed_matmul`, `_propagate_node` |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
