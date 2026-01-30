# Sharding Engine

[← Back to Core](../README.md)

## Philosophy
Nabla uses **Factor-Based Sharding Propagation** (inspired by GSPMD/Shardy). Instead of mapping "Dimension 0 -> Machine Axis `dp`", we map "Factor `batch` -> Machine Axis `dp`". Operations describe how they transform factors (e.g., `batch features`). The engine solves constraints to determine the layout of every tensor.

## Architecture & Internals

### 1. Specs
*   **`DeviceMesh`**: N-dimensional grid of devices (e.g., `(2, 4)` for 8 GPUs).
*   **`ShardingSpec`**: Describes precisely how a tensor is split.
    *   `DimSpec`: List of mesh axes assigned to a tensor dimension.

### 2. The Propagation Loop (`spmd.py`)

For every Operation `Z = f(X, Y)`:

1. **Infer**: Query operation for `OpShardingRule` (e.g., `"m k, k n -> m n"` for matmul)
2. **Propagate**: Three-phase algorithm:
   - **COLLECT**: Convert dimension specs to factor specs
   - **RESOLVE**: Resolve conflicts via priority system
   - **UPDATE**: Project factor specs back to output dimension specs
3. **Reshard**: Insert communication ops if input shardings don't match requirements

### Three-Phase Propagation Algorithm

**COLLECT Phase**: Convert dimension shardings to factor shardings.

Example: Matmul `C = A @ B` with rule `"m k, k n -> m n"`
- If `A` has `DimSpec(["dp"], ...)` on dimension 0, factor `m` collects `["dp"]`
- If `B` has `DimSpec(["tp"], ...)` on dimension 1, factor `n` collects `["tp"]`
- Factor `k` collects from both `A`'s dimension 1 and `B`'s dimension 0

**RESOLVE Phase**: Resolve conflicts using priority system.

Rules (in order):
1. Explicit replication (via `replicated_axes`) always wins
2. Higher-priority specs override lower-priority specs (lower `priority` value = higher priority)
3. More parallelism wins at equal priority
4. Common prefix fallback if no clear winner

**UPDATE Phase**: Project resolved factor shardings to output dimensions.

- Detect contracting factors (present in inputs, absent in outputs)
- Mark output axes as `partial=True` if they hold unreduced partial sums
- Example: Row-parallel matmul where `k` is sharded produces output with `partial_sum_axes={'tp'}`

### Practical Examples

**Column-Parallel Matmul** (Megatron-LM pattern):
```python
mesh = DeviceMesh((8,), ["tp"])
w = w.shard(mesh, P(None, "tp"))  # Shard output features
y = x @ w  # No AllReduce needed, outputs are sharded on "tp"
```

**Row-Parallel Matmul** (requires reduction):
```python
x = x.shard(mesh, P("tp"))  # Shard input features
w = w.shard(mesh, P("tp", None))  # Shard input features
y = x @ w  # Auto-insert AllReduce on "tp" axis
# Factor k is sharded on "tp", contracts away → partial sums → AllReduce
```

**Data-Parallel Elementwise**:
```python
mesh = DeviceMesh((8,), ["dp"])
x = x.shard(mesh, P("dp"))
y = relu(x)  # Preserves "dp" sharding, no communication
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
