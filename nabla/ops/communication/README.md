# Communication Operations

[← Back to Ops](../README.md)

## Philosophy
Communication operations are explicit instructions to move data across the mesh. In Nabla, these are treated as regular `Operation`s that take a `Tensor`, perform a collective (via MAX/NCCL), and return a `Tensor` with a new `ShardingSpec`.

## Architecture & Internals

### The Collective Base

All communication operations inherit from `CollectiveOperation` in [base.py](base.py):

1. **Hydrate**: Forces input realization (synchronization point)
2. **Execute**: Invokes MAX/NCCL collective primitive
3. **Spec Update**: Computes resulting `ShardingSpec` transformation

**Critical**: Collective operations MUST implement `physical_execute` to prevent recursion during SPMD execution. They operate directly on shard data without triggering sharding propagation.

### Automatic Insertion

Communication operations are automatically inserted by `reshard_inputs` when:

1. **Partial Sum Reduction**: Operation produces `partial_sum_axes` (e.g., matmul with sharded contracting dimension) → Insert `AllReduce`
2. **Sharding Mismatch**: Input sharding doesn't match required sharding:
   - Sharded → Replicated: `AllGather`
   - Replicated → Sharded: Split data (no comm needed)
   - Sharded axis A → Sharded axis B: `AllToAll`

### Sharding Patterns

**Column-Parallel Pattern** (no communication):
```python
# Model Parallel: shard output features
w = w.shard(mesh, P(None, "tp"))  # Weight: [hidden, features/tp]
y = x @ w  # Output: [batch, features/tp]
# No AllReduce needed - output stays sharded
```

**Row-Parallel Pattern** (requires AllReduce):
```python
# Model Parallel: shard input features
x = x.shard(mesh, P(None, "tp"))  # Input: [batch, features/tp]
w = w.shard(mesh, P("tp", None))  # Weight: [features/tp, hidden]
y = x @ w  # Contracting dim sharded → partial sums
# Automatic AllReduce on "tp" axis
```

**Data-Parallel Pattern** (no communication for forward):
```python
# Data Parallel: shard batch
x = x.shard(mesh, P("dp"))  # Input: [batch/dp, features]
w = w.shard(mesh, P())      # Weight: [features, hidden] replicated
y = x @ w  # Output: [batch/dp, hidden]
# No communication in forward pass
# AllReduce needed in backward for weight gradients
```

### Cost Modeling

Communication operations estimate network costs for auto-sharding:

- **AllReduce**: `2 * (N - 1) / N * bytes` (ring algorithm)
- **AllGather**: `(N - 1) / N * bytes`
- **AllToAll**: `(N - 1) / N * bytes`
- **ReduceScatter**: `(N - 1) / N * bytes`

These estimates guide the auto-sharding optimizer when comparing parallelization strategies.

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
