# Sharding Engine

[← Back to Core](../README.md)

## Philosophy
Nabla uses **Factor-Based Sharding Propagation** (inspired by GSPMD/Shardy). Instead of mapping "Dimension 0 -> Machine Axis `dp`", we map "Factor `batch` -> Machine Axis `dp`". Operations describe how they transform factors (e.g., `batch features`). The engine solves constraints to determine the layout of every tensor.

## Architecture & Internals

### 1. Specs
*   **`DeviceMesh`**: N-dimensional grid of devices (e.g., `(2, 4)` for 8 GPUs).
*   **`ShardingSpec`**: Describes precisely how a tensor is split.
    *   `DimSpec`: List of mesh axes assigned to a tensor dimension.

### 2. The Propagation Algorithm

Runs **eagerly per-operation** during execution (not as a compilation phase).

**Three-Phase Process**:

**COLLECT Phase**: Convert dimension shardings to factor shardings.
- For matmul `C = A @ B` with rule `"m k, k n -> m n"`:
  - Factor `m` collects sharding from A's dimension 0
  - Factor `k` collects from both A's dimension 1 and B's dimension 0
  - Factor `n` collects from B's dimension 1
- Each factor accumulates all shardings from dimensions it maps to

**RESOLVE Phase**: Resolve conflicts using priority system.
- Explicit replication (via `replicated_axes`) overrides all other constraints
- Lower `priority` value = higher priority (0 = strongest constraint)
- At equal priority, higher parallelism wins (more sharded axes preferred)
- Common prefix fallback if no clear winner
- Detect contracting factors: present in inputs but absent in outputs

**UPDATE Phase**: Project resolved factor shardings back to output dimensions.
- Map each output dimension to its corresponding factors
- Inherit sharding from factor assignments
- Mark axes as `partial=True` if they hold unreduced partial sums from contracting factors
- These partial axes trigger automatic AllReduce insertion

**Reshard Phase**: Execute communication ops immediately if needed.
- Input sharding ≠ required sharding → AllGather or AllToAll executes now
- Output has partial sums → AllReduce executes immediately after computation
- No lazy insertion - communication happens during operation execution

### Examples

**Column-parallel matmul** (no communication):
```python
mesh = DeviceMesh((8,), ["tp"])
w = w.shard(mesh, P(None, "tp"))  # [hidden, features/8] per device
y = x @ w  # Output: [batch, features/8] - stays sharded on tp
# Factor n sharded on tp, preserved in output, no contracting factors
```

**Row-parallel matmul** (immediate AllReduce):
```python
x = x.shard(mesh, P("tp"))        # [batch, features/8]
w = w.shard(mesh, P("tp", None))  # [features/8, hidden]
y = x @ w  # Factor k (contracting) sharded on tp
# Output has partial_sum_axes={'tp'} → AllReduce executes before returning y
```

**Data-parallel** (no forward communication):
```python
mesh = DeviceMesh((8,), ["dp"])
x = x.shard(mesh, P("dp"))  # [batch/8, features]
y = relu(x)  # Factor d0 sharded on dp, preserved → no communication
```

## Component Map

| File | Role | Exported Symbols |
| :--- | :--- | :--- |
| [`spec.py`](spec.py) | **Data Structures** | **Classes**: `DeviceMesh`, `DimSpec`, `ShardingSpec`, `PartitionSpec` (alias `P`)<br>**Shape Helpers**: `compute_local_shape`, `compute_global_shape`, `get_num_shards`, `needs_reshard`<br>**Utils**: `parse_sub_axis` |
| [`spmd.py`](spmd.py) | **Pipeline** | **Functions**: `infer_output_sharding`, `reshard_inputs`, `get_shard_args`, `create_sharded_output`, `reshard_tensor`, `ensure_specs`, `get_mesh_from_args`, `create_replicated_spec` |
| [`propagation.py`](propagation.py) | **Algorithm** | **Classes**: `OpShardingRule`, `OpShardingRuleTemplate`, `FactorSharding`, `FactorShardingState`<br>**Enums**: `OpPriority`, `PropagationStrategy`<br>**Functions**: `propagate_sharding`, `run_hierarchical_propagation_pass` |
| **[`optimizer/`](optimizer/)** | **Solver** | `simple_solver.py` - ILP/Heuristic solver (`SimpleSolver`) for auto-sharding. |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
