# Design Document: nabla/ops Execution Model Refactoring

**Status**: In Progress / Evolving  
**Target Audience**: Systems Engineers, Core Maintainers

---

## 1. Context & Motivation

The primary driver for this refactoring is to enable **Robust Trace Rehydration**.

### The Problem

Currently, `Trace.rehydrate` relies on a mix of captured `op_args` and stale metadata (like `adapted_kwargs` stored in `TensorImpl`). This is fragile. Specifically, logic that happens "on the fly" during execution—like `auto_reduction` after a distributed matmul—is lost during tracing because it resides outside the captured graph nodes. When we rehydrate a trace (re-execute it to reconstruct the graph), these steps are skipped, leading to incorrect graphs (e.g., missing critical reductions).

### The Solution

We must strictly separate the **Logical Adaptation** (User API, broadcasting, input prep) from the **Physical Execution** (SPMD kernel runner).

We introduce `physical_execute` (formerly `maxpr_new`): a self-contained, robust entry point that takes valid `nabla.Tensors`, derives all necessary physical context (mesh, batch dims, sharding) internally from those tensors, and executes the operation. Both **Eager Mode** and **Rehydration** will verify and call this exact same method, guaranteeing consistency.

---

## 2. Architecture

### The Two Layers

#### 1. Logical Layer: `Operation.__call__`

**Role**: The User Interface & Adapter.

**Responsibility**:
- Input Validation
- **Adaptation**: Transforming inputs (broadcasting, type promotion) and performing pre-op resharding (calling `reshard_inputs`)
- **Delegation**: Calling `physical_execute` with the prepared, valid inputs
- **Packaging**: Wrapping the raw results from `physical_execute` into `nabla.Tensor` objects and attaching `OutputRefs` for tracing

#### 2. Physical Layer: `Operation.physical_execute` (Generic Shard Loop)

**Role**: The SPMD Kernel Runner.

**Signature**: `def physical_execute(self, args: tuple, kwargs: dict) -> list[TensorValue]`

**Contract**:
- **Delegation**: By default, `Operation` implements a generic `physical_execute` that:
  - Derives `output_sharding` and `mesh` internally
  - Loops over `shard_idx`
  - Calls `self._transform_shard_kwargs(...)` to lower logical parameters (shapes, slices) to physical ones
  - Calls `self.maxpr(...)` on the individual shard
- **Context**: Must run inside `with GRAPH.graph:`
- **GSPMD Efficiency**: View ops remain O(1) in terms of communication because they operate on existing shards. Communication is only triggered in the Logical Layer (`__call__`) if `spmd.reshard_inputs` detects a sharding mismatch.

### Shared Primitives (`nabla/ops/base.py`)

- **PhysicalResult**: A standard container `(shard_values, output_sharding, mesh)` to ensure consistent return types across all physical ops.

### The Rehydration Path

`Trace.rehydrate` will no longer try to emulate execution. It will simply:

1. Load the captured input `nabla.Tensors`
2. Call `op.physical_execute(inputs, original_kwargs)`
3. Take the returned raw values and inject them into the existing `TensorImpl` nodes

---

## 3. Implementation Status & Learnings

### Progress So Far

| Component | Status |
|-----------|--------|
| **Infrastructure** | ✅ `nabla/ops/base.py` and `tracing.py` updated |
| **UnaryOperation** | ✅ `physical_execute` implemented |
| **BinaryOperation** | ✅ `physical_execute` implemented |
| **MatmulOp** | ✅ `physical_execute` implemented |
| **ReduceOperation** | 🔄 In Progress |
| **ShardOp** | ✅ Verified by `test_pp_grad2.py` |
| **AllReduceOp** | ✅ Implemented |
| **AllGatherOp** | ✅ Implemented |
| **AllToAllOp** | ✅ Implemented |
| **PPermuteOp** | ✅ Implemented |

### Critical Learnings

During implementation, several key constraints emerged that were not in the original plan:

#### 1. The `hydrate()` Anti-Pattern

**Issue**: We initially placed `x.hydrate()` calls inside `physical_execute` to ensure inputs were realized.  
**Correction**: `physical_execute` **MUST** assume its inputs are already valid `Tensor` objects or TensorValues available in the current graph context. `hydrate()` implies eager realization logic which breaks tracing.

#### 2. The PhysicalResult Wrapper Necessity

**Issue**: `ShardOp` initially returned `list[TensorValue]`. `Trace.rehydrate` failed to recognize this structure, skipping graph updates.  
**Correction**: All `physical_execute` implementations must consistently return a **PhysicalResult** object (tuple of `(shard_values, output_sharding, mesh)`) so callers (`rehydrate`, `__call__`) can reliably unpack output shards and sharding metadata.

#### 3. Graph Context for Lazy Values

**Issue**: Accessing `Tensor.values` (even property access) checks for `_buffer_value`. In MAX Engine, this triggers a check against the current graph context.  
**Correction**: `physical_execute` must wrap its body in `with GRAPH.graph:` to ensure a valid context exists for lazy value access and new node creation.

#### 4. Recursion in Collective Adaptation

**Issue**: Collective operations (AllGather/Reduce) are themselves part of the `shard()` logic. If `Operation.__call__` triggers `reshard_inputs` using default propagation, it can create an infinite loop.  
**Correction**: Collective operations **MUST** implement `infer_sharding_spec`.

#### 5. Integer Hardening (MLIR/nanobind)

**Issue**: MLIR bindings and nanobind are extremely strict about types. Passing `Dim` objects (from `max.graph`) or floats into `ops.reshape` or sharding rules can cause cryptic `SystemError` or `ValueError`.  
**Correction**: **ALWAYS** cast shapes and coordinates to `tuple[int, ...]` before passing them to the physical engine (`ops.*`) or sharding rules.

#### 6. Trace Rehydration Safety

**Issue**: When rehydrating, `op.physical_execute` is called with captured tensors. If it tries to call `spmd.infer_output_sharding`, it might trigger a recursive loop because spmd calls `op.infer_sharding_spec`.  
**Correction**: `physical_execute` should either use a manual `infer_sharding_spec` that explicitly propagates `ShardingSpec` objects or perform Spec propagation internally without re-entering the spmd dispatcher.

---

## 4. Operation-Specific Implementation Patterns

### Pattern A: Simple Elementwise (Unary/Binary)

```python
class UnaryOperation(Operation):
    def physical_execute(self, args: tuple, kwargs: dict) -> Any:
        from ..core import GRAPH
        from ..core.sharding import spmd
        
        mesh = spmd.get_mesh_from_args(args)
        
        with GRAPH.graph:
            shard_results = spmd.execute_on_shards(
                self.maxpr, args, kwargs, mesh, op=self
            )
                
        return (shard_results, None, mesh)
```

### Pattern B: Reduction Operations

Reduction ops need special handling because:
1. They may reduce over sharded axes (triggering cross-shard AllReduce)
2. Output shape differs from input shape
3. kwargs contain axis information that needs per-shard transformation

```python
class ReduceOperation(LogicalAxisOperation):
    def physical_execute(self, args: tuple, kwargs: dict) -> Any:
        from ..core import GRAPH
        from ..core.sharding import spmd
        
        mesh = spmd.get_mesh_from_args(args)
        
        with GRAPH.graph:
            shard_results = spmd.execute_on_shards(
                self.maxpr, args, kwargs, mesh, op=self
            )
        
        return (shard_results, None, mesh)
```

### Pattern C: Communication Operations

Communication ops are special because they coordinate across devices and must implement `infer_sharding_spec` to avoid recursion:

```python
class AllReduceOp(Operation):
    def infer_sharding_spec(self, args, mesh, kwargs):
        # Manual spec propagation without re-entering SPMD dispatcher
        ...
    
    def physical_execute(self, args: tuple, kwargs: dict) -> Any:
        # Custom multi-device coordination
        ...
```

---

## 5. Reduction Operations Deep Dive

### Current State

The `mean` function is implemented differently from other reductions:

```python
def mean(x, *, axis=None, keepdims=False):
    # Implemented as sum / count (NOT using MeanOp.maxpr directly)
    s = reduce_sum(x, axis=axis, keepdims=keepdims)
    count = compute_count(x, axis)
    return s / count
```

This is **intentional** for distributed correctness:
- When reducing over a sharded axis, `ops.mean(local_shard)` would compute the wrong value
- `sum(all_shards) / total_count` gives the correct global mean
- The AllReduce handles the cross-shard sum, then we divide by total count

### The MeanOp.__call__ Override

The `MeanOp` class has an explicit `__call__` override:

```python
class MeanOp(ReduceOperation):
    def __call__(self, x, *, axis: int, keepdims: bool = False):
        return super().__call__(x, axis=axis, keepdims=keepdims)
```

This is benign—it just forwards to the parent. The key insight is that `nb.mean()` function doesn't use `_mean_op(...)` directly for distributed tensors; it uses the `reduce_sum` + division pattern.

---

## 6. Definition of Done

- [ ] `Operation.__call__` handles **ONLY** adaptation and wrapping
- [x] `UnaryOperation.physical_execute` implemented
- [x] `BinaryOperation.physical_execute` implemented
- [x] `MatmulOp.physical_execute` implemented  
- [ ] `ReduceOperation.physical_execute` implemented (current task)
- [ ] Zero legacy `execute` methods remain in `nabla/ops/communication/`
- [x] All `tests/unit_v2/` tests pass

---

## 7. Current Status: All Tests Passing ✅

As of the latest run:
- **297/297 tests pass** in `tests/unit_v2/`
- `debug_mean.py` executes correctly
- The `mean` function correctly computes distributed means using `sum/count` pattern

---

## 8. Next Steps

1. **Add `physical_execute` to `ReduceOperation`** for consistency (even though tests pass)
2. **Audit remaining ops** for legacy `execute` usage
3. **Stress test** with complex vmap + sharding compositions
4. **Document** the sharding rule patterns for each op category
