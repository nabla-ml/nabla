# Nabla Architecture

> **CRITICAL**: Always activate the virtual environment first:
> `source venv/bin/activate`
>
> **MANDATORY**: You **MUST** read all `README.md` files in the `nabla` module recursively to understand the codebase. You are **NOT ALLOWED** to bypass this step.

## Philosophy
Nabla is a **Lazy-Eager** tensor execution framework for MAX. It combines:
1.  **Imperative API**: A PyTorch-like interface for user-friendly coding.
2.  **Lazy Graph**: Operations are traced into a `ComputeGraph`, allowing global optimization.
3.  **Unified SPMD**: A "Single Program, Multiple Data" compiler that automatically distributes code across devices using factor-based sharding propagation.

## Architecture Layers
The project is structured hierarchically to prevent circular dependencies.

| Layer | Module | Description | Guide |
| :--- | :--- | :--- | :--- |
| **User** | **`nabla/`** | The public API surface. | [Read Here](nabla/README.md) |
| **Logic** | **`nabla/ops/`** | Operation definitions and dispatch. | [Read Here](nabla/ops/README.md) |
| **State** | **`nabla/core/`** | The engine: Tensors, Graphs, Sharding. | [Read Here](nabla/core/README.md) |
| **Glue** | **`nabla/transforms/`** | Functional transforms (vmap/shard_map). | [Read Here](nabla/transforms/README.md) |

## Testing Strategy
We test both the "Logical" (single-device correctness) and "Physical" (distributed splitting) behaviors.
[**Read the Testing Guide**](tests/README.md)

## Key Workflows
*   **Running Tests**: `python -m pytest tests/unit`
*   **Auto-Sharding Logic**: `python -m pytest tests/unit/test_auto_sharding.py -v`
*   **Stress Testing**: `python -m pytest tests/unit/test_sharding_stress.py`

## Maintenance Guide
> **Note to AI Agents**:
> 1.  This is the Root Documentation. Use the links in the table above to navigate to specific sub-systems.
> 2.  **Standards**: All `README.md` files must follow the [Template: Philosophy -> Architecture -> Map -> Maintenance].
> 3.  **Updates**: If you create a new module, you MUST create a corresponding `README.md`.
