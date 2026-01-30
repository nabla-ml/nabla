# Nabla Architecture

## Philosophy

Nabla implements **Eager Execution with Lazy Graph Recording** using **Factor-Based SPMD Sharding**.

**Execution Model**: Operations execute immediately (including data movement and sharding), but are recorded in a computation graph. Graph compilation to MAX executables happens lazily when accessing concrete data (`.numpy()`, `print()`).

**Factor-Based Sharding**: Operations define transformations via **named factors** (e.g., `"m k, k n -> m n"` for matmul) rather than positional dimensions. The three-phase propagation algorithm (COLLECT → RESOLVE → UPDATE) runs **eagerly per-operation** to determine output sharding and required input shardings.

**SPMD Execution**: Write code for a single logical device. Each operation automatically handles distributed execution via the dual tensor system: logical tensors contain multiple shard objects, operations loop over shards and execute `maxpr()` (MAX primitives) per shard.

## Architecture & Internals

### Operation Execution Flow

When you call an operation (e.g., `x + y` or `matmul(a, b)`):

1. **Input Validation**: Check shapes, dtypes, handle broadcasting and type promotion
2. **Sharding Inference**: Call operation's `sharding_rule()` to get factor-based propagation rule
3. **Propagate Sharding**: Run three-phase algorithm eagerly:
   - COLLECT: Map input dimension shardings → factor shardings
   - RESOLVE: Resolve conflicts via priority, detect contracting factors
   - UPDATE: Map factor shardings → output dimension shardings
4. **Reshard Inputs**: If input shardings don't match required shardings, execute communication ops immediately:
   - AllReduce for partial sums (contracting dimensions)
   - AllGather for sharded → replicated
   - AllToAll for axis redistribution
5. **Execute**: Loop over device mesh shards, call `op.maxpr()` per shard with local data
6. **Package Results**: Wrap results into new `Tensor` with sharding metadata
7. **Record Graph**: Create `OutputRefs` node for tracing/autodiff

**Key Point**: Steps 1-6 execute eagerly. Step 7 records for potential JIT compilation.

### The Dual Tensor System

Every sharded tensor is represented by:
- **Logical Tensor** (`Tensor`): User-facing object with global shape and sharding metadata
- **Physical Shards**: List of shard objects (one per device), each holding local data

This dual representation enables:
- **Trace Rehydration**: Captured graphs can be replayed with different shard configurations without re-executing Python
- **Unified Engine**: Same graph recording works for single-device and distributed execution
- **shard_map**: Trace with logical tensors, replay with physical shards (different execution paths)

### Sharding Propagation Details

**Factor-Based Rules**: Operations specify how factors transform (einsum notation):
- Matmul: `"m k, k n -> m n"` - factor `k` contracts (appears in inputs, not output)
- Binary elementwise: `"d0 d1, d0 d1 -> d0 d1"` - all factors preserved
- Reduction: `"d0 d1 -> d1"` - factor `d0` contracts

**Conflict Resolution**:
1. Explicit replication (via `replicated_axes`) overrides everything
2. Higher priority (lower `priority` value) wins
3. More parallelism wins at same priority
4. Common prefix fallback

**Automatic Communication**:
- Contracting factors that are sharded → Output gets `partial_sum_axes` → AllReduce executes immediately
- Mismatched input sharding → AllGather/AllToAll executes immediately before computation

## Module Map

| Module | Purpose | Documentation |
| :--- | :--- | :--- |
| **`core/`** | **The Engine**. Contains the `Tensor` state, `ComputeGraph` recorder, and the `Sharding` compiler. | [**Read Docs**](core/README.md) |
| **`ops/`** | **The Logic**. Defines all mathematical operations (`Add`, `Matmul`), their gradients, and sharding rules. | [**Read Docs**](ops/README.md) |
| **`transforms/`** | **The Bridge**. Functional transformations that alter execution: `vmap` (vectorize), `shard_map` (distribute), `compile` (optimize). | [**Read Docs**](transforms/README.md) |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Read Recursively**: Start here, then follow links to understand specific subsystems.
> 2.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 3.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
> 4.  **Template**: Always use the `Philosophy` -> `Internals` -> `Map` -> `Maintenance` structure.
