# Nabla Sharding System

[← Back to Root](../CLAUDE.md)

## Overview
We implement **Unified SPMD Execution** using **Factor-Based Propagation**.
-   **User View**: Logical tensors (`(1024, 1024)`).
-   **System View**: Physical tensors (`(512, 1024)`) on each device.

## Component Map

| Component | File | Description |
| :--- | :--- | :--- |
| **Specs** | **[`spec.py`](spec.py)** | `DeviceMesh`, `ShardingSpec`, `DimSpec`. Defines *how* data is split. |
| **Runtime** | **[`spmd.py`](spmd.py)** | `infer_output_sharding` (forward) and `reshard_inputs` (backward align). |
| **Propagation** | **[`propagation.py`](propagation.py)** | `OpShardingRule`. Solves constraints via factor graphs (`i -> i`). |
| **Optimizer** | **[`optimizer/`](optimizer/simple_solver.py)** | Cost-based solver to pick sharding for undefined ops. |

## The SPMD Pipeline
For every operation `z = f(x, y)`:

1.  **Inference**: `spmd.infer_output_sharding` delegates to the op's `sharding_rule`.
2.  **Propagation**: The rule solves for output factors based on inputs (e.g., "if x is row-sharded, z must be").
3.  **Resharding**: If inputs don't match the rule's requirements, `spmd.reshard_inputs` inserts `AllGather`/`AllReduce`.
4.  **Execution**: The local function is executed on the physical shards.

## Developer Guide: Sharding Rules

Define propagation using `OpShardingRuleTemplate` in your Op's `sharding_rule` method.

### Syntax Reference

| Symbol | Meaning | Example |
| :--- | :--- | :--- |
| **`i`, `j`, `k`** | **Factors**. Named dimensions required to match. | `i k, k j -> i j` (Matmul) |
| **`...`** | **Batch**. Auto-propagated prefix. | `... i -> ... i` (Elementwise) |
| **`1`** | **Replicated**. No factor on this axis. | `i -> 1` (Reduce sum) |
| **`(a b)`** | **Grouped**. Splitting/Merging dims. | `(a b) -> a b` (Reshape) |
| **`?`** | **Open**. Dimension creates a new independent factor. | `1 -> ?` (Constant) |

### Example
```python
# Matrix Multiplication
# Rule: "batch... rows cols, batch... cols hidden -> batch... rows hidden"
rule = OpShardingRuleTemplate.parse("... m k, ... k n -> ... m n")
```
