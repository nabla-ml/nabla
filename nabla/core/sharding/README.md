# Sharding Engine

[← Back to Core Hub](../README.md)

## Philosophy
Nabla uses **Factor-Based Sharding Propagation** (inspired by GSPMD/Shardy). Instead of mapping "Dimension 0 -> Machine Axis `dp`", we map "Factor `batch` -> Machine Axis `dp`". Operations describe how they transform factors (e.g., `batch features`). The engine solves constraints to determine the layout of every tensor.

## Architecture & Internals

### 1. Specs
*   **`DeviceMesh`**: N-dimensional grid of devices (e.g., `(2, 4)` for 8 GPUs).
*   **`ShardingSpec`**: Describes precisely how a tensor is split.
    *   `DimSpec`: List of mesh axes assigned to a tensor dimension.

### 2. The Propagation Loop (`spmd.py`)
For every Operation `Z = f(X, Y)`:
1.  **Infer**: Ask the op for its `OpShardingRule` (e.g., `i k, k j -> i j`).
2.  **Propagate**:
    *   Convert Input Specs (Dim → Axes) to Factor Specs (Factor → Axes).
    *   Resolve conflicts (priority system).
    *   Project Factor Specs to Output Specs.
3.  **Reshard**: If inputs don't match the required sharding, insert `reshard` (comm ops).

> [!NOTE] Design Decision: Factors vs Dimensions
> *   **Choice**: Propagate sharding via named factors (`i`, `j`, `batch`), not positional dimensions.
> *   **Why**: Handles broadcasting (`1 -> N`) and reshaping (`(a b) -> a b`) naturally.
> *   **Trade-off**: Ops must implement `sharding_rule()` instead of just simple dimension mapping.

## Component Map

| File | Role | Key Concepts |
| :--- | :--- | :--- |
| [`spec.py`](spec.py) | **Data Structures**. | `ShardingSpec`, `DeviceMesh`, `DimSpec` |
| [`spmd.py`](spmd.py) | **Pipeline**. | `infer_output_sharding`, `reshard_inputs` |
| [`propagation.py`](propagation.py) | **Algorithm**. | `OpShardingRule`, `FactorSharding`, `propagate_sharding` |

## Maintenance Guide
> **Note to AI Agents**: Update this file if you modify the propagation algorithm or spec structures.
> This file must remain the source of truth for high-level architecture.
