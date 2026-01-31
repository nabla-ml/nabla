# Communication Operations

[← Back to Ops](../README.md)

> **Purpose**: Move data across the device mesh via collective primitives (AllReduce, AllGather, etc.).

## How Communication Ops Differ from Normal Ops

### The Key Architectural Difference

**Normal ops** (matmul, add, relu) follow this flow in `execute`:

```text
execute → spmd.execute_on_shards → self.kernel(shard_value)
                   (loops over shards)       (MAX graph op)
```

**Communication ops COMPLETELY OVERRIDE `execute`** to work directly on the `list[TensorValue]`:

```text
execute → self._reduce_logic(shard_values)
                   (custom logic that COORDINATES across all shards)
```

This is critical because collectives aren't "per-shard" operations—they inherently involve ALL shards talking to each other.

### Inside AllReduceOp.execute

```python
def execute(self, args: tuple, kwargs: dict) -> Any:
    sharded_tensor = args[0]
    
    # 1. Get the list of TensorValues (one per device)
    values = sharded_tensor.values  # list[TensorValue]
    
    # 2. Call _reduce_logic which operates on ALL shards at once
    with GRAPH.graph:
        reduced_values = self._reduce_logic(
            values,           # ALL shard values
            mesh=mesh,
            reduce_op="sum",  # sum/max/min/prod
            reduce_axes=reduce_axes
        )
    
    # 3. Compute new ShardingSpec (partial_sum_axes cleared)
    output_spec = self._compute_output_spec(sharded_tensor, reduced_values, ...)
    
    # 4. Return (values, spec, mesh) tuple - NOT calling spmd.execute_on_shards!
    return (reduced_values, output_spec, mesh)
```

### The _reduce_logic: What Actually Happens

```python
def _reduce_logic(self, shard_values: list[TensorValue], ...):
    if mesh.is_distributed:
        # REAL distributed: use MAX's allgather + local reduce
        gathered = max_allgather(shard_values, signal_buffers, axis=0)
        # Each device now has all data, reduce locally
        for gathered_tensor in gathered:
            reduced = sum(chunks)  # ops.add for each chunk
        return result_values
    else:
        # SIMULATION mode: just sum across the list
        result = shard_values[0]
        for sv in shard_values[1:]:
            result = ops.add(result, sv)  # Direct MAX graph ops
        return [result] * len(shard_values)  # Broadcast to all
```

**Key insight**: Communication ops work on `list[TensorValue]` as a whole, not per-element.

## The CollectiveOperation Base Class

All communication ops inherit from `CollectiveOperation` ([base.py](base.py)):

```text
Operation
    └── CollectiveOperation
            ├── AllReduceOp      # Sum/max/min/prod across shards
            ├── AllGatherOp      # Concatenate shards → replicated
            ├── ReduceScatterOp  # Reduce then scatter
            ├── AllToAllOp       # Axis redistribution
            ├── PPermuteOp       # Cyclic peer-to-peer shift
            └── AxisIndexOp      # Get device index within axis
```

`CollectiveOperation` provides:

- `_derive_mesh()`: Extract mesh from tensor or kwargs
- `_get_reduce_axes()`: Determine which mesh axes to reduce over
- `_get_physical_axis()`: Convert logical axis → physical (accounting for batch_dims)
- `infer_sharding_spec()`: Default adaptation that validates inputs and computes output spec

## Automatic Insertion

Communication ops are inserted **eagerly** by the sharding engine:

| Trigger | Inserted Op | Where |
| :--- | :--- | :--- |
| Contracting dim sharded → partial sums | `AllReduceOp` | `apply_auto_reduction` (Phase 5) |
| Input needs to be replicated | `AllGatherOp` | `reshard_inputs` (Phase 2) |
| Change sharded axis | `AllToAllOp` | `reshard_inputs` (Phase 2) |

```python
# In apply_auto_reduction (execution_utils.py):
if reduce_axes and mesh:
    all_reduce_op = AllReduceOp()
    result = all_reduce_op.simulate_grouped_execution(
        shard_values, mesh, reduce_axes, reduce_op=op.collective_reduce_type
    )
```

## Component Map

| File | Role | Exported Symbols |
| :--- | :--- | :--- |
| [`base.py`](base.py) | **Base Class** | **Classes**: `CollectiveOperation`; **Methods**: `estimate_cost`, `communication_cost` |
| [`shard.py`](shard.py) | **Entry Point** | **Classes**: `ShardOp`; **Functions**: `shard`, `create_replicated_spec` |
| [`reshard.py`](reshard.py) | **Transition** | **Functions**: `reshard`, `reshard_tensor` (Smart transition between any two specs) |
| [`all_reduce.py`](all_reduce.py) | **Reduce** | **Classes**: `AllReduceOp`, `PMeanOp`; **Functions**: `all_reduce`, `pmean` |
| [`all_gather.py`](all_gather.py) | **Gather** | **Classes**: `AllGatherOp`, `GatherAllAxesOp`; **Functions**: `all_gather`, `gather_all_axes` |
| [`reduce_scatter.py`](reduce_scatter.py) | **Scatter** | **Classes**: `ReduceScatterOp`; **Functions**: `reduce_scatter` |
| [`all_to_all.py`](all_to_all.py) | **Shuffle** | **Classes**: `AllToAllOp`; **Functions**: `all_to_all` |
| [`p_permute.py`](p_permute.py) | **Permute** | **Classes**: `PPermuteOp`; **Functions**: `ppermute` (Peer-to-peer cyclic shifts) |
| [`axis_index.py`](axis_index.py) | **Metadata** | **Classes**: `AxisIndexOp`; **Functions**: `axis_index` (Get device index within mesh axis) |

## Maintenance Guide

> **Note to AI Agents**:
>
> 1. **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2. **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
