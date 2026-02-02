# Sharding Engine

[← Back to Core](../README.md)

> **Purpose**: Implements SPMD (Single Program Multiple Data) execution via factor-based sharding propagation.

## How Sharding Works

### The Big Picture

When you call an operation on sharded tensors, the sharding engine:

1. **Infers** what sharding the output will have
2. **Determines** what sharding each input MUST have
3. **Reshards** inputs if they don't match (inserts communication)
4. **Detects** contracting axes that need post-op reduction

This happens **eagerly per-operation** in Step 2 of `Operation.__call__()`.

### Sharding Specs

```python
from nabla.core.sharding import DeviceMesh, P

# Create a mesh of 8 devices with one axis named "dp"
mesh = DeviceMesh((8,), ["dp"])

# Or 2D mesh: 2 data-parallel × 4 tensor-parallel
mesh = DeviceMesh((2, 4), ["dp", "tp"])

# Shard a tensor
x = x.shard(mesh, P("dp", None))  # Shard dim 0 on "dp", replicate dim 1
# P is alias for PartitionSpec
```

**ShardingSpec** describes how each tensor dimension maps to mesh axes:

- `P("dp")` → dimension sharded across "dp" axis
- `P(None)` → dimension replicated
- `P("dp", "tp")` → dimension sharded across both (nested sharding)

## Factor-Based Propagation

Unlike dimension-based sharding, Nabla uses **factors** (inspired by GSPMD):

```text
Matmul rule: "m k, k n -> m n"
                │ │    │ │
                │ │    │ └── Factor 'n' = columns
                │ │    └──── Factor 'k' = contracting (DISAPPEARS in output)
                │ └───────── Factor 'k' = contracting  
                └─────────── Factor 'm' = rows
```

**Why factors?** They capture the semantic meaning of dimensions, not just positions. Factor `k` in matmul represents the dimension being summed over—if sharded, we need AllReduce.

### The Three-Phase Algorithm

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    Factor Propagation (per operation)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PHASE 1: COLLECT                                                       │
│  ───────────────                                                        │
│  Map input dimension shardings → factor shardings                       │
│                                                                         │
│  Example: matmul(A, B) with rule "m k, k n -> m n"                      │
│  • A is [M, K] sharded as P("dp", "tp")                                 │
│  • B is [K, N] sharded as P("tp", None)                                 │
│                                                                         │
│  Collected:                                                             │
│  • Factor 'm': {"dp"} (from A dim 0)                                    │
│  • Factor 'k': {"tp"} (from A dim 1) + {"tp"} (from B dim 0)            │
│  • Factor 'n': {} (from B dim 1, replicated)                            │
│                                                                         │
│  PHASE 2: RESOLVE                                                       │
│  ───────────────                                                        │
│  Resolve conflicts when multiple inputs contribute to same factor       │
│                                                                         │
│  Priority rules:                                                        │
│  1. Explicit replication wins (via replicated_axes)                     │
│  2. Lower priority value wins (0 = strongest)                           │
│  3. More parallelism wins (prefer sharded over replicated)              │
│  4. Common prefix fallback                                              │
│                                                                         │
│  Detect contracting factors: in inputs but not outputs (factor 'k')     │
│  → These become reduce_axes for Step 8 of __call__                       │
│                                                                         │
│  PHASE 3: UPDATE                                                        │
│  ─────────────                                                          │
│  Project factor shardings → output dimension shardings                  │
│                                                                         │
│  Output [M, N]:                                                         │
│  • Dim 0 (factor 'm'): sharded on "dp"                                  │
│  • Dim 1 (factor 'n'): replicated                                       │
│                                                                         │
│  Since 'k' was sharded on "tp", output has partial sums on "tp"         │
│  → reduce_axes = {"tp"} returned to __call__                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Communication Insertion

### Pre-Op: reshard_inputs()

If input sharding doesn't match what the operation needs:

```python
# Called in __call__ Step 2
resharded_args = spmd.reshard_inputs(args, input_shardings, mesh)
```

This may insert:

- **shard()**: Replicated → Sharded
- **all_gather()**: Sharded → Replicated
- **reshard_tensor()**: AllToAll for axis redistribution

### Post-Op: apply_auto_reduction()

If contracting factors were sharded (reduce_axes non-empty):

```python
# Called in __call__ Step 8
if reduce_axes and mesh:
    output = apply_auto_reduction(self, output, mesh, reduce_axes)
```

This inserts grouped AllReduce using `op.collective_reduce_type` (default: "sum").

## Examples

### Data Parallel (no communication in forward)

```python
mesh = DeviceMesh((8,), ["dp"])
x = x.shard(mesh, P("dp"))        # [batch/8, features]
w = w.shard(mesh, P(None, None))  # [features, hidden] replicated

y = x @ w  # Rule: "m k, k n -> m n"
# Factor 'm' (batch) sharded on "dp" 
# Factor 'k' (features) replicated → no partial sums
# Factor 'n' (hidden) replicated
# Output: [batch/8, hidden] sharded on "dp"
# No communication!
```

### Tensor Parallel Column (no communication)

```python
mesh = DeviceMesh((8,), ["tp"])
w = w.shard(mesh, P(None, "tp"))  # [features, hidden/8]

y = x @ w  # x is replicated
# Factor 'k' replicated, factor 'n' sharded on "tp"
# Output: [batch, hidden/8] sharded on "tp"
# No communication!
```

### Tensor Parallel Row (AllReduce needed)

```python
w1 = w1.shard(mesh, P(None, "tp"))  # Column parallel
w2 = w2.shard(mesh, P("tp", None))  # Row parallel

h = x @ w1       # [batch, hidden/8], no communication
y = h @ w2       # Factor 'k' (hidden) sharded on "tp"!
# → reduce_axes = {"tp"}
# → apply_auto_reduction inserts AllReduce
# Output: [batch, out_features] replicated
```

## Developing Custom Sharding Rules

If you are implementing a custom `Operation`, you need to tell the solver how your axes map to factors. You do this by implementing `sharding_rule`:

```python
def sharding_rule(self, input_shapes, output_shapes, **kwargs):
    from ..core.sharding.propagation import OpShardingRuleTemplate
    
    # Define factors for inputs and outputs (einsum-style)
    # Example for Batched MatMul:
    #   Input A: [batch(b), m, k]
    #   Input B: [batch(b), k, n]
    #   Output:  [batch(b), m, n]
    
    rule_str = "b m k, b k n -> b m n"
    
    return OpShardingRuleTemplate.from_string(rule_str).instantiate(
        input_shapes, output_shapes
    )
```

**How it works:**
1.  **Parse**: The template parses `"b m k"` and assigns factors to dimensions.
2.  **Constraint**: If User shards Input A's dim 0 (batch) on `"dp"`, then factor `b` is bound to `"dp"`.
3.  **Propagate**: Since Output also has `b` at dim 0, Output dim 0 is automatically assigned `"dp"`.

## Component Map

| File | Purpose | Key Exports |
| :--- | :--- | :--- |
| [spec.py](spec.py) | Data structures | `DeviceMesh`, `ShardingSpec`, `DimSpec`, `PartitionSpec` (alias `P`), `compute_local_shape`, `compute_global_shape`, `needs_reshard` |
| [spmd.py](spmd.py) | SPMD pipeline | `infer_output_sharding`, `reshard_inputs`, `create_sharded_output`, `execute_on_shards`, `get_mesh_from_args` |
| [propagation.py](propagation.py) | Factor algorithm | `propagate_sharding`, `OpShardingRule`, `OpShardingRuleTemplate` |
| [optimizer/](optimizer/) | Auto-sharding solver | `SimpleSolver` |

## Maintenance Guide

> **AI Agents - Critical Rules**:
>
> 1. **reduce_axes**: Returned by `infer_output_sharding`, used by `apply_auto_reduction`. Don't lose this!
> 2. **Factor notation**: Operations define sharding via `"m k, k n -> m n"` style rules
> 3. **Eager execution**: Communication happens immediately, not lazily
> 4. **partial_sum_axes**: Track which mesh axes have unreduced partial sums
