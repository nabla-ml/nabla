# Operations System

[← Back to Root](../README.md)

## Philosophy
In Nabla, every operation (`Add`, `Matmul`, `Reshape`) is a **stateless singleton** class inheriting from `Operation`. The `Operation` class acts as a central dispatcher that handles:
1.  **Logic**: Building the symbolic MAX graph.
2.  **Physics**: Defining sharding propagation rules.
3.  **Cost**: Estimating compute/comm costs for auto-sharding.

## Architecture & Internals

## Architecture & Internals

### Operation Hierarchy

Operations are stateless singletons inheriting from `Operation`:

```
Operation (base class)
├── UnaryOperation          # Single-input (ReLU, Exp, Neg)
├── BinaryOperation         # Two-input (Add, Mul, Matmul)
├── LogicalAxisOperation    # With axis parameters
│   ├── ReduceOperation     # Reductions (Sum, Mean)
│   └── LogicalShapeOperation  # Shape manipulation (Reshape, Transpose)
└── CollectiveOperation     # Communication (AllReduce, AllGather)
```

**Key Methods**:
- `maxpr(*args, **kwargs)`: MAX Engine primitive execution per shard
- `sharding_rule(*arg_shapes)`: Returns einsum-like factor notation for propagation
- `vjp(cotangents, *primals, **kwargs)`: Vector-Jacobian product for autodiff
- `_transform_shard_kwargs(shard_idx, **kwargs)`: Adapt arguments per shard (optional)

### Operation Execution Flow

Eager execution with integrated sharding:

1. **Validate Inputs**: Check shapes, dtypes, handle broadcasting and type promotion
2. **Infer Sharding Rule**: Get factor-based rule (e.g., `"m k, k n -> m n"` for matmul)
3. **Propagate Sharding**: Run three-phase algorithm (COLLECT → RESOLVE → UPDATE)
   - Determine output sharding from input shardings
   - Identify required input shardings (may differ from current)
4. **Reshard Inputs**: Execute communication ops immediately if current ≠ required sharding
   - Partial sums → AllReduce
   - Sharded → Replicated → AllGather
   - Axis redistribution → AllToAll
5. **Execute Per-Shard**: Loop over device mesh shard indices
   - Call `_transform_shard_kwargs(shard_idx, kwargs)` to adapt arguments
   - Execute `maxpr(shard_args, shard_kwargs)` for each shard
6. **Package Results**: Wrap shard results into new `Tensor` with computed sharding
7. **Record Graph Node**: Add `OutputRefs` to global graph for tracing/autodiff

**Execution Context**: Per-shard execution (step 5) runs inside `graph.context()` to access lazy values without recursion.

### Sharding Rules

Operations define factor transformations using einsum-like notation:

**Matmul**: `"m k, k n -> m n"`
- Factors: `m` (rows), `k` (contracting), `n` (cols)
- Factor `k` appears in inputs but not output → contracting factor
- If `k` is sharded, output has partial sums → AllReduce

**Binary Elementwise**: `"d0 d1, d0 d1 -> d0 d1"`
- All factors preserved
- Inputs must have compatible shardings
- Output inherits factor shardings

**Reduction**: `"d0 d1 -> d1"` (reduce over axis 0)
- Factor `d0` contracts
- If `d0` was sharded → partial sums → AllReduce

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
