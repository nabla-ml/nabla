# Sharding Module: Distributed Tensor Partitioning

## Overview

This module enables distributing tensors across device meshes with automatic sharding propagation. Unlike many frameworks, nabla supports **eager sharded execution** via SPMD (Single Program Multiple Data).

**Status**: Specification, propagation, and SPMD execution are implemented. Multi-device hardware execution awaits MAX backend support.

---

## Module Structure

| File | Purpose |
|------|---------|
| `spec.py` | DeviceMesh, ShardingSpec, DimSpec |
| `propagation.py` | Factor-based algorithm, op templates |
| `spmd.py` | Execution helpers (slice, reshard, align) |

---

## Core Concepts

### DeviceMesh

Logical n-dimensional grid of devices with named axes:

```
@cluster = <["dp"=2, "tp"=4]>  # 8 devices: 2×4
```

### ShardingSpec

Per-tensor sharding description:
- **dim_specs**: How each dimension is sharded
- **replicated_axes**: Explicitly replicated mesh axes

### DimSpec

Per-dimension specification:
- **axes**: Which mesh axes shard this dim (major→minor)
- **is_open**: Can receive additional sharding during propagation
- **priority**: Lower = stronger (user annotations beat inferred)

---

## Factor-Based Propagation

**Inspired by XLA Shardy/GSPMD**. Uses einsum-like factor mappings:

```
matmul: (m, k), (k, n) → (m, n)
```

### Three-Phase Algorithm

1. **Collect**: Project dimension→factor shardings
2. **Merge**: Resolve conflicts using priority + strategy
3. **Update**: Project factor→dimension shardings

### Conflict Resolution

- **BASIC**: Take longest common prefix (conservative)
- **AGGRESSIVE**: Pick higher parallelism option

### Operations Templates

Templates generate factor mappings from shapes:
- `matmul_template(batch_dims)` — handles batched matmul
- `elementwise_template(rank)` — all dims map 1:1
- `reduce_template(rank, reduce_dims)` — reduced dims are contracting factors
- `transpose_template(rank, perm)` — permuted factor order
- `broadcast_with_shapes_template(in_shape, out_shape)` — handles expansion

---

## SPMD Execution Flow

When `Operation.__call__` detects sharded inputs:

1. **Infer output sharding** via `infer_output_sharding(op, args, mesh)`
2. **Detect conflicts** — reshard to replicated if axes overlap incompatibly
3. **Per-shard execution** — run `maxpr()` on each shard's data
4. **Wrap results** — create output tensor with N shard values

Key functions in `spmd.py`:
- `has_sharded_inputs(args)` — detection
- `get_mesh_from_args(args)` — extract mesh
- `slice_for_shard(tensor, shape, sharding, idx)` — local slicing
- `reshard_tensor(tensor, from_spec, to_spec, mesh)` — resharding

---

## User API

```python
from nabla import Tensor, DeviceMesh, DimSpec

mesh = DeviceMesh("test", (2,), ("x",))
A = Tensor.ones((4, 8)).trace()

# Functional: returns new sharded tensor
A_sharded = A.shard(mesh, [DimSpec(["x"]), DimSpec([])])

# Explicit resharding constraint
B = (A_sharded @ W).with_sharding(mesh, [DimSpec([]), DimSpec([])])
```

---

## Current Limitations

- **Single-machine simulation**: All shards run on same device (awaiting MAX multi-device backend)
- **Limited op coverage**: Templates exist for common ops (`matmul`, `elementwise`, `reduce`, `transpose`, `broadcast`); complex ops need manual rules
- **No cost model**: `AGGRESSIVE` conflict resolution doesn't consider communication cost

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Factor-based propagation | Handles reshapes, complex ops correctly |
| Functional `shard()` | Matches nabla's immutable tensor philosophy |
| Eager SPMD | No separate compilation phase for sharding |
| Priority system | User annotations override inferred sharding |

---

## Testing

```bash
source venv/bin/activate

# Unit tests (propagation, conflict resolution)
python -m pytest tests/unit/sharding/ -v

# Integration tests (SPMD execution, end-to-end)
python -m pytest tests/integration/with_sharding/ -v
```
