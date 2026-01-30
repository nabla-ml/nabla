# Communication Operations

[← Back to Ops](../README.md)

## Philosophy
Communication operations are explicit instructions to move data across the mesh. In Nabla, these are treated as regular `Operation`s that take a `Tensor`, perform a collective (via MAX/NCCL), and return a `Tensor` with a new `ShardingSpec`.

## Architecture & Internals

### Execution Model

Communication ops inherit from `CollectiveOperation` and execute eagerly:

1. Force input realization (synchronization point)
2. Execute MAX/NCCL collective primitive immediately
3. Return new tensor with updated `ShardingSpec`

**Critical**: Must implement `physical_execute` to prevent recursion during SPMD execution.

### Automatic Insertion

Inserted eagerly by `reshard_inputs` during operation execution:

- **Partial sums** (contracting dimension sharded) → `AllReduce`
- **Sharded → Replicated** → `AllGather`  
- **Axis redistribution** → `AllToAll`

### Common Patterns

**Column-parallel** (no comm): Shard output features → `w.shard(mesh, P(None, "tp"))`  
**Row-parallel** (AllReduce): Shard contracting dim → automatic AllReduce on tp axis  
**Data-parallel**: Shard batch → no forward comm, AllReduce for weight gradients in backward

## Component Map

| File | Role | Exported Symbols |
| :--- | :--- | :--- |
| [`base.py`](base.py) | **Base Class** | **Classes**: `CollectiveOperation`<br>**Methods**: `estimate_cost`, `communication_cost` |
| [`shard.py`](shard.py) | **Entry Point** | **Classes**: `ShardOp`<br>**Functions**: `shard` (Primary way to introduce sharding) |
| [`reshard.py`](reshard.py) | **Transition** | **Classes**: `ReshardOp`<br>**Functions**: `reshard` (Smart transition between any two specs) |
| [`all_reduce.py`](all_reduce.py) | **Reduce** | **Classes**: `AllReduceOp`, `PMeanOp`<br>**Functions**: `all_reduce`, `pmean` |
| [`all_gather.py`](all_gather.py) | **Gather** | **Classes**: `AllGatherOp`, `GatherAllAxesOp`<br>**Functions**: `all_gather`, `gather_all_axes` |
| [`reduce_scatter.py`](reduce_scatter.py) | **Scatter** | **Classes**: `ReduceScatterOp`<br>**Functions**: `reduce_scatter` |
| [`all_to_all.py`](all_to_all.py) | **Shuffle** | **Classes**: `AllToAllOp`<br>**Functions**: `all_to_all` |
| [`p_permute.py`](p_permute.py) | **Permute** | **Classes**: `PPermuteOp`<br>**Functions**: `ppermute` (Peer-to-peer cyclic shifts) |
| [`axis_index.py`](axis_index.py) | **Metadata** | **Classes**: `AxisIndexOp`<br>**Functions**: `axis_index` (Get specialized index for each shard) |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
