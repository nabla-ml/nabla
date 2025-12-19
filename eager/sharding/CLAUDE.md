# Sharding Module: Distributed Tensor Partitioning

## Philosophy

Sharding enables distributing large tensors and computations across multiple devices. This module provides the **specification** (how to shard) and **propagation** (infer sharding across operations), but not yet the execution (actually running sharded).

**Status**: Infrastructure ready, execution pending MAX multi-device support.

---

## Core Concepts (spec.py)

### DeviceMesh

Logical grid of devices. Example: `DeviceMesh((4, 8))` = 4×8 grid of 32 devices.

**Axis names**: `("data", "model")` for semantic clarity.

**Why mesh?** Expresses multi-dimensional parallelism:
- Axis 0: data parallelism
- Axis 1: model/tensor parallelism

### ShardingSpec

Describes how a tensor is partitioned across a mesh.

**Representation**: Per-dimension specification:
- Which mesh axes this tensor dimension is sharded over
- Whether replicated
- Sub-axis splits for complex patterns

**Example**: Tensor `(batch, features)` sharded on mesh `("data", "model")`:
- `batch` → shard over "data" axis
- `features` → shard over "model" axis

### DimSpec

Per-dimension sharding specification.

**Options**:
- Sharded on specific mesh axes
- Replicated across all devices
- Partially sharded (sub-axis splitting)

---

## Sharding Propagation (propagation.py)

### The Problem

User annotates a few tensors with sharding specs. How to infer sharding for all intermediate tensors?

**Challenge**: Different operations have different sharding semantics:
- Matmul: contracting dimension can change sharding
- Elementwise: preserve sharding
- Reductions: reduce over sharded dimension requires all-reduce

### The Solution: Factor-Based Propagation

**Inspiration**: Shardy (Google's sharding propagation in JAX/GSPMD).

**Core idea**: 
- Einsum-like notation **conceptually** describes operation: `(i,k), (k,j) → (i,j)` for matmul
- Each factor (i, k, j) represents a semantic dimension
- Sharding propagates through factors, not tensor positions
- **Implementation**: Uses dictionaries mapping dimension indices to factor lists, with `to_einsum_notation()` for human-readable display

**Algorithm**:
1. **Collect**: Project tensor shardings onto factors
2. **Resolve**: When factors conflict, apply priority rules
3. **Update**: Project factor shardings back to tensors

### Propagation Strategy

**BASIC**: Conservative—only keep common sharding prefix.

**AGGRESSIVE**: Pick highest-parallelism option when conflict.

**Why configurable?** Trade-off between safety and performance.

### Operation Sharding Rules

Each operation defines how its factors relate:
- Matmul: `OpShardingRule.matmul_like()`
- Elementwise: All dimensions map to same factors
- Reduction: Factor disappears

**Future**: Complete rule library for all operations.

---

## Integration Points

### With Core

- `TensorImpl.sharding_spec` stores partitioning metadata
- `GRAPH.evaluate()` detects sharded tensors, triggers sharded path
- Currently: validation only, no execution

### With Ops

- Operations would need sharding rules registered
- Currently: only a few template rules exist

### Future: Execution

- Partition MAX graph across devices
- Insert collectives (all-reduce, all-gather, reduce-scatter)
- Execute sharded computation
- Relies on MAX multi-device support

---

## Key Architectural Decisions

### 1. Why Factor-Based Propagation?

**Enables**: Correct handling of reshapes and complex operations. Position-based propagation breaks on dimension splitting.

### 2. Why Separate Spec from Execution?

**Enables**: Design the API now, implement execution when MAX multi-device ready.

### 3. Why Einsum Notation?

**Enables**: Concise operation semantics. `(i,k),(k,j)→(i,j)` immediately clear.

### 4. Why Propagation Strategies?

**Enables**: User control over safety vs perf trade-off.

---

## Current Limitations

**No real execution**: Can specify and propagate sharding, but `GRAPH.evaluate()` currently uses a dummy implementation (generates zeros). Actual sharded execution awaits MAX multi-device support.

**Limited op coverage**: Only templates for common patterns.

**No cross-mesh**: Can't propagate between different device meshes.

**No cost model**: AGGRESSIVE mode maximizes parallelism, not performance.

---

## Future Directions

### Full Sharded Execution

- Partition graph across mesh
- Insert communication collectives
- Execute on multiple devices
- Reassemble results

### Cost-Based Optimization

Replace AGGRESSIVE strategy with cost model:
- Consider communication overhead
- Profile operation costs
- Choose minimum-time sharding

### Auto-Sharding

Automatic sharding selection given:
- Model
- Mesh
- Performance objectives

No manual annotations needed.

### Cross-Mesh Support

Transfer tensors between different meshes (CPU ↔ TPU, different topologies).

### Pipeline Parallelism

Combine sharding with pipeline stages for very large models.

---

## Common Misconceptions

**"Sharding is for vmap"**: No. Vmap is single-device vectorization. Sharding is multi-device parallelism.

**"We support distributed training"**: Not yet. Infrastructure exists, execution doesn't.

**"ShardingSpec is like PyTorch DDP"**: No. More expressive—supports arbitrary partitioning, not just data parallelism.

**"Sharding propagation is for optimization"**: Partially. It's also for correctness—ensures sharding is consistent across operations.

---

## Why Shardy?

**Shardy** (sharding + study) is Google's approach in JAX. We adopt the same principles:
- Factor-based propagation
- Einsum operation semantics
- Conflict resolution strategies

**Benefit**: Proven approach, works for massive models (100B+ params).

**Difference**: Our implementation is standalone, not tied to XLA compiler.
