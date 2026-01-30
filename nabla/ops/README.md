# Operations System

[← Back to Root](../README.md)

## Philosophy
In Nabla, every operation (`Add`, `Matmul`, `Reshape`) is a **stateless singleton** class inheriting from `Operation`. The `Operation` class acts as a central dispatcher that handles:
1.  **Logic**: Building the symbolic MAX graph.
2.  **Physics**: Defining sharding propagation rules.
3.  **Cost**: Estimating compute/comm costs for auto-sharding.

## Architecture & Internals

### Operation Hierarchy

```
Operation (base class)
├── UnaryOperation          # Single-input ops (ReLU, Exp, Neg)
├── BinaryOperation         # Two-input ops (Add, Mul, Matmul)
├── LogicalAxisOperation    # Ops with axis parameters
│   ├── ReduceOperation     # Reductions (Sum, Mean)
│   └── LogicalShapeOperation  # Shape ops (Reshape, Transpose)
└── CollectiveOperation     # Communication primitives (AllReduce, AllGather)
```

**Key Methods Every Operation Implements**:

- `maxpr(*args, **kwargs)`: MAX Engine primitive execution
- `sharding_rule(*arg_shapes)`: Returns einsum-like factor notation for propagation
- `vjp(cotangents, *primals, **kwargs)`: Gradient computation (VJP)
- `_transform_shard_kwargs(shard_idx, **kwargs)`: Per-shard argument adaptation (optional)

### The Dispatch Loop

Operation execution flows through two layers:

**Logical Layer** (`op.execute`):
1. Validate inputs (shapes, dtypes, broadcasting)
2. Call `preshard_inputs` to satisfy sharding requirements
3. Invoke `physical_execute` for computation
4. Wrap results into `nabla.Tensor` objects
5. Record graph node via `OutputRefs`

**Physical Layer** (`physical_execute`):
1. Loop over each shard index in the device mesh
2. Call `op._transform_shard_kwargs(shard_idx, kwargs)` for per-shard arguments
3. Execute `op.maxpr()` on shard data
4. Return `PhysicalResult(symbolic_nodes, computed_values)`

**Execution Context**: Physical execution must occur inside `graph.context()` to access lazy tensor values without triggering recursive compilation.

### Sharding Propagation

Operations define sharding via einsum-like notation:

**Matmul**: `"m k, k n -> m n"`
- Factor `m`: Maps to A's rows, C's rows
- Factor `k`: Maps to A's cols, B's rows (contracting)
- Factor `n`: Maps to B's cols, C's cols

**Binary elementwise**: `"d0 d1, d0 d1 -> d0 d1"`
- Both inputs must match on all dimensions
- Output inherits sharding

**Reduction on axis 0**: `"d0 d1 -> d1"`
- Factor `d0` contracts away
- If `d0` was sharded, output has partial sums requiring AllReduce

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
