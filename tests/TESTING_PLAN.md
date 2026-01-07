# Nabla Comprehensive Testing Plan

## Context Summary

This document provides a complete plan for implementing rigorous unit tests for the nabla ML framework.

### Current Status (Updated January 2026)

**Test Files:**
- `conftest.py` - Shared fixtures and helpers ✅
- `test_physical_ops.py` - Physical operations tests ✅
- `test_logical_ops.py` - Logical operations without batch_dims ✅
- `test_vmap_ops.py` - Basic vmap transform tests ✅
- `test_vmap_sharding_comprehensive.py` - vmap + sharding inside function ✅
- `test_vmap_advanced.py` - Advanced in_axes/out_axes variations ✅ (NEW)
- `test_binary_ops_sharding.py` - Binary ops sharding propagation ✅ (NEW)

### Key Findings

1. **vmap + sharding tests are CORRECT**: Tests in `test_vmap_sharding_comprehensive.py` properly
   apply sharding INSIDE the vmapped function on logical shapes. This is the intended pattern.

2. **Binary ops DO have sharding rules**: The `Operation.sharding_rule()` method returns
   `elementwise_template(rank)` for binary ops, which correctly propagates sharding from
   both inputs to the output.

3. **Two valid testing patterns for vmap + sharding**:
   - Pattern A (test_vmap_sharding_comprehensive.py): Shard INSIDE vmap function on logical shapes
   - Pattern B (test_vmap_ops.py): Pre-shard inputs before vmap (data parallelism pattern)

### Key Learnings:
- `tensor.shape` returns **LOGICAL** shape (excludes batch_dims prefix)
- `tensor._impl.physical_shape` returns **PHYSICAL** shape
- `DeviceMesh` constructor: `DeviceMesh("name", (shape,), ("axis_names",))`
- Tensor realization is async: `asyncio.run(tensor.realize)`
- Use `Tensor.from_dlpack(np_array)` for tensor creation

---

## Test Hierarchy Rationale

The test hierarchy follows a **dependency order** - each level assumes the previous level is correct:

```
Level 1: Physical Ops     → Foundation (no batch_dims awareness)
Level 2: Logical Ops      → Core ops without batch_dims (baseline correctness)
Level 3: vmap Transforms  → Automatic batching (relies on physical ops)
Level 4: Sharding Ops     → Communication primitives
Level 5: Combined         → vmap + sharding together
```

**Why this order?**
- Physical ops are used internally by vmap - they MUST work first
- Logical ops establish numerical correctness against numpy
- vmap tests use `vmap()` directly (not manual batch_dims) so batch handling is automatic
- Sharding tests verify communication ops work independently
- Combined tests verify the full stack

---

## Operations Inventory

### 1. Binary Operations (test ONE representative: `add`)
- `add`, `sub`, `mul`, `div` - all inherit from `BinaryOperation`
- `matmul` - special case (2D contraction)

### 2. Unary Operations (test ONE representative: `relu`)
- `relu`, `sigmoid`, `tanh`, `exp`, `neg`, `abs` - all inherit from `UnaryOperation`

### 3. Reduction Operations (test ALL - different semantics)
- `reduce_sum` - sum along axis
- `mean` - mean along axis

### 4. View Operations (test ALL - unique axis semantics)
- `reshape` - change shape preserving elements
- `squeeze` - remove dim of size 1
- `unsqueeze` - add dim of size 1
- `swap_axes` - swap two axes
- `broadcast_to` - expand dims of size 1
- `moveaxis` - move axis to new position

### 5. Physical Operations (test ALL - foundation)
- `reduce_sum_physical`, `mean_physical`
- `squeeze_physical`, `unsqueeze_physical`
- `broadcast_to_physical`

### 6. Communication Operations (test ALL - sharding primitives)
- `shard` - partition tensor across mesh
- `reshard` - change sharding spec
- `all_gather` - gather shards to replicated
- `all_reduce` - reduce across shards (sum only in MAX)

### 7. Batch Management Operations (metadata only - light testing)
- `incr_batch_dims`, `decr_batch_dims`
- `move_axis_to_batch_dims`, `move_axis_from_batch_dims`

---

## Test File Structure

```
tests/unit/ops_comprehensive/
├── conftest.py                 # Shared fixtures and helpers
├── test_physical_ops.py        # Level 1: Physical ops foundation
├── test_logical_ops.py         # Level 2: Logical ops baseline  
├── test_vmap_ops.py            # Level 3: vmap transforms
├── test_communication_ops.py   # Level 4: Sharding primitives
└── test_combined.py            # Level 5: vmap + sharding
```

---

## Detailed Test Plan

### conftest.py - Shared Fixtures

```python
# CRITICAL FIXES NEEDED:
# 1. DeviceMesh("name", (shape,), ("axis",)) - needs name argument
# 2. asyncio.run(tensor.realize) - realize is async
# 3. Use Tensor.from_dlpack(np_array) for creation

import pytest
import asyncio
import numpy as np
from nabla import Tensor, DeviceMesh
from nabla.sharding.spec import DimSpec

def make_array(*shape, seed=42, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(dtype)

def tensor_from_numpy(arr):
    return Tensor.from_dlpack(arr)

def to_numpy(t):
    asyncio.run(t.realize)
    if t._impl.is_sharded and t._impl._storages:
        shards = [s.to_numpy() for s in t._impl._storages]
        spec = t._impl.sharding
        sharded_dims = [i for i, d in enumerate(spec.dim_specs) if d.axes]
        if not sharded_dims:
            return shards[0]
        return np.concatenate(shards, axis=sharded_dims[0])
    return np.array(t._impl.data)

@pytest.fixture
def mesh_1d():
    return DeviceMesh("mesh_1d", (4,), ("dp",))

@pytest.fixture  
def mesh_2d():
    return DeviceMesh("mesh_2d", (2, 2), ("dp", "tp"))
```

---

### test_physical_ops.py - Level 1 Foundation

**Purpose**: Test physical ops that vmap uses internally. Must pass before anything else.

**Test Classes**:
```
TestReduceSumPhysical
  - test_reduce_axis (parametrize shape × axis)
  - test_keepdims
  - test_sharded_reduce_non_sharded_axis
  - test_sharded_reduce_sharded_axis

TestMeanPhysical
  - test_mean_axis (parametrize shape × axis)
  - test_keepdims
  - test_sharded

TestSqueezePhysical
  - test_squeeze_axis (parametrize shape × axis × expected_shape)
  - test_sharded_squeeze_non_sharded_dim
  
TestUnsqueezePhysical
  - test_unsqueeze_axis (parametrize shape × axis × expected_shape)
  - test_sharded_unsqueeze

TestBroadcastToPhysical
  - test_broadcast (parametrize shape × target_shape)
  - test_broadcast_preserves_sharding
```

---

### test_logical_ops.py - Level 2 Baseline

**Purpose**: Establish numerical correctness without batch_dims. All assertions against numpy.

**Test Classes**:
```
TestBinaryOpsBasic
  - test_add, test_sub, test_mul, test_div (parametrize shapes)
  
TestBinaryOpsBroadcasting
  - test_add_broadcast (parametrize shape_a × shape_b × expected)
  
TestBinaryOpsSharding
  - test_add_both_sharded_same
  - test_add_sharded_replicated
  - test_add_broadcast_sharded

TestUnaryOpsBasic
  - test_relu, test_sigmoid, test_tanh, test_exp, test_neg (parametrize shapes)
  
TestUnaryOpsSharding
  - test_relu_sharded, test_sigmoid_sharded

TestReductionOps
  - test_reduce_sum (parametrize shape × axis)
  - test_mean (parametrize shape × axis)
  - test_reduce_sum_keepdims
  
TestReductionOpsSharding
  - test_reduce_sum_non_sharded_axis
  - test_reduce_sum_sharded_axis

TestReshape
  - test_reshape (parametrize shape × new_shape)
  
TestSqueeze
  - test_squeeze (parametrize shape × axis × expected_shape)
  
TestUnsqueeze
  - test_unsqueeze (parametrize shape × axis × expected_shape)
  
TestSwapAxes
  - test_swap_axes (parametrize shape × axis1 × axis2 × expected_shape)
  
TestMoveaxis
  - test_moveaxis (parametrize shape × source × dest × expected_shape)
  
TestBroadcastTo
  - test_broadcast_to (parametrize shape × target_shape)

TestViewOpsSharding
  - test_reshape_sharded
  - test_swap_axes_sharded
```

---

### test_vmap_ops.py - Level 3 Transforms

**Purpose**: Test vmap using `vmap()` directly. Never manually set batch_dims.

**Pattern**: Create batched input, apply `vmap(fn)`, verify output shape and values.

**Test Classes**:
```
TestVmapBinaryOps
  - test_vmap_add (parametrize batch_size)
  - test_vmap_mul
  - test_vmap_add_broadcast_within_batch

TestVmapBinaryOpsNested
  - test_nested_vmap_add (vmap(vmap(add)))
  - test_nested_vmap_mul

TestVmapUnaryOps
  - test_vmap_relu (parametrize batch_size)
  - test_vmap_sigmoid
  - test_vmap_tanh
  - test_vmap_neg

TestVmapUnaryOpsNested
  - test_nested_vmap_relu
  - test_triple_vmap_relu

TestVmapReductionOps
  - test_vmap_reduce_sum
  - test_vmap_reduce_sum_axis_0
  - test_vmap_mean
  - test_nested_vmap_reduce_sum

TestVmapViewOps
  - test_vmap_reshape
  - test_vmap_squeeze
  - test_vmap_unsqueeze
  - test_vmap_swap_axes
  - test_nested_vmap_reshape

TestVmapMatmul
  - test_vmap_matmul
  - test_nested_vmap_matmul

TestVmapComposite
  - test_vmap_mlp_layer (relu(x @ W + b))
  - test_vmap_normalize ((x - mean) / std)
```

---

### test_communication_ops.py - Level 4 Sharding Primitives

**Purpose**: Test sharding operations independently.

**Test Classes**:
```
TestShardOp
  - test_shard_1d_axis0
  - test_shard_1d_axis1
  - test_shard_2d_mesh
  - test_shard_replicated
  - test_shard_numerical_values (verify shard contents)

TestAllGather
  - test_all_gather_1d
  - test_all_gather_2d_mesh
  - test_all_gather_numerical

TestAllReduce
  - test_all_reduce_sum (no op param - only sum supported)
  - test_all_reduce_numerical

TestReshardOp
  - test_reshard_change_sharding
  - test_reshard_to_replicated
```

---

### test_combined.py - Level 5 Full Stack

**Purpose**: Test vmap + sharding together.

**Test Classes**:
```
TestVmapWithSharding
  - test_vmap_relu_sharded
  - test_vmap_add_sharded
  - test_vmap_matmul_sharded_batch
  - test_nested_vmap_sharded

TestVmapShardingComposite
  - test_mlp_layer_sharded
  - test_attention_head_sharded (if applicable)
```

---

## Implementation Order

Execute tests after creating each file:

```bash
# 1. Fix conftest.py first
pytest tests/unit/ops_comprehensive/conftest.py -v

# 2. Physical ops (foundation)
pytest tests/unit/ops_comprehensive/test_physical_ops.py -v

# 3. Logical ops (baseline)
pytest tests/unit/ops_comprehensive/test_logical_ops.py -v

# 4. vmap ops  
pytest tests/unit/ops_comprehensive/test_vmap_ops.py -v

# 5. Communication ops
pytest tests/unit/ops_comprehensive/test_communication_ops.py -v

# 6. Combined
pytest tests/unit/ops_comprehensive/test_combined.py -v

# Full suite
pytest tests/unit/ops_comprehensive/ -v
```

---

## Key API Signatures to Remember

```python
# Reductions - axis is REQUIRED keyword-only
reduce_sum(x, *, axis: int, keepdims: bool = False)
mean(x, *, axis: int, keepdims: bool = False)

# Physical ops
reduce_sum_physical(x, axis: int, keepdims: bool = False)
mean_physical(x, axis: int, keepdims: bool = False)
squeeze_physical(x, axis: int)
unsqueeze_physical(x, axis: int)
broadcast_to_physical(x, shape: tuple)

# Communication
all_reduce(sharded_tensor)  # NO op parameter, only sum
all_gather(sharded_tensor)
shard(tensor, mesh, dim_specs)
reshard(tensor, mesh, new_specs)

# vmap
vmap(fn)(batched_inputs)  # Automatically handles batch_dims

# DeviceMesh
DeviceMesh("name", (shape_tuple,), ("axis_names",))
```

---

## Critical Insights

1. **tensor.shape = LOGICAL shape** (excludes batch_dims)
2. **tensor._impl.physical_shape = PHYSICAL shape** (full shape)
3. **realize is async**: `asyncio.run(tensor.realize)`
4. **Tensor creation**: `Tensor.from_dlpack(np_array)`
5. **vmap handles batch_dims automatically** - never set manually in tests
6. **all_reduce has NO op parameter** - only sum is supported by MAX

---

## Next Steps

1. Fix `conftest.py` with correct API calls
2. Run `test_physical_ops.py` and fix any failures
3. Run `test_logical_ops.py` and fix any failures  
4. Run `test_vmap_ops.py` and fix any failures
5. Create `test_communication_ops.py`
6. Create `test_combined.py`
7. Ensure full suite passes
