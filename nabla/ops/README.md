# Operations System

[← Back to Root](../README.md)

## Philosophy
In Nabla, every operation (`Add`, `Matmul`, `Reshape`) is a **stateless singleton** class inheriting from `Operation`. The `Operation` class acts as a central dispatcher that handles:
1.  **Logic**: Building the symbolic MAX graph.
2.  **Physics**: Defining sharding propagation rules.
3.  **Cost**: Estimating compute/comm costs for auto-sharding.

## Architecture & Internals

### The Dispatch Loop
When you call `x + y`, `Operation.__call__` executes:
1.  **Infer**: Uses `sharding_rule()` to determine output sharding.
2.  **Reshard**: Calls `reshard_inputs()` (hooks into `core/sharding`) to satisfy input requirements.
3.  **Execute**: Runs `maxpr()` (Maximum Intermediate Representation) to trace the logic into the graph.

> [!NOTE] Design Decision: Stateless Singletons
> *   **Choice**: ops are `const` singletons (e.g., `_add_op = AddOp()`).
> *   **Why**: No per-call overhead for object creation. Easy to register globally.
> *   **Trade-off**: Must pass all state (like `mesh`) as arguments to `__call__`.

## Component Map

| Submodule/File | Role | Key Concepts |
| :--- | :--- | :--- |
| [`base.py`](base.py) | **The Interface**. | `Operation`, `UnaryOperation`, `BinaryOperation` |
| **[`communication/`](communication/README.md)** | **Collectives**. | `all_reduce`, `shard` |
| **[`view/`](view/README.md)** | **Metadata**. | `reshape`, `transpose` |
| `binary.py`, `unary.py` | **Logical Ops**. | `add`, `mul`, `sin`, `exp` |

## Maintenance Guide
> **Note to AI Agents**: Update this file if you change the `Operation` base class or add new op categories.
> This file must remain the source of truth for high-level architecture.
