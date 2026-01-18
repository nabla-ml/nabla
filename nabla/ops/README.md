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

| Submodule/File | Role | Exported Symbols |
| :--- | :--- | :--- |
| [`base.py`](base.py) | **The Interface** | **Classes**: `Operation`, `BinaryOperation`, `LogicalAxisOperation`, `ReduceOperation`, `LogicalShapeOperation`, `UnaryOperation` |
| [`dispatch.py`](dispatch.py) | **Dispatcher** | **Internal**: `execute_operation` (Handles SPMD reasoning) |
| **[`communication/`](communication/README.md)** | **Collectives** | `all_reduce`, `shard`, `reshard`, `all_gather`, `reduce_scatter` |
| **[`view/`](view/README.md)** | **Metadata** | `reshape`, `transpose`, `squeeze`, `unsqueeze`, `broadcast_to`, `gather`, `scatter` |
| [`binary.py`](binary.py) | **Math** | `add`, `mul`, `sub`, `div`, `matmul` |
| [`unary.py`](unary.py) | **Elementwise** | `relu`, `sigmoid`, `tanh`, `exp`, `neg`, `softmax` |
| [`reduction.py`](reduction.py) | **Reductions** | `reduce_sum`, `mean` |
| [`comparison.py`](comparison.py) | **Logic** | `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal` |
| [`creation.py`](creation.py) | **Factories** | `full`, `arange`, `constant`, `zeros`, `ones`, `uniform`, `gaussian`, `normal` |
| [`control_flow.py`](control_flow.py) | **Control Flow** | `where`, `cond`, `while_loop`, `scan` |
| [`multi_output.py`](multi_output.py) | **Multi-Out** | `split`, `chunk`, `unbind`, `minmax` |
| [`custom_op.py`](custom_op.py) | **Extension** | `call_custom_kernel` |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
