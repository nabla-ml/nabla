# Nabla Architecture

> **Purpose**: This document explains the core concepts and lifecycles in Nabla. Start here to understand how everything fits together, then follow links for implementation details.

## Core Philosophy

Nabla is a **tensor library with automatic differentiation** built on three pillars:

1. **Eager Execution, Lazy Compilation**: Operations execute immediately (you see results right away), but the underlying MAX graph compilation is deferred until you actually need concrete values (`.numpy()`, `print()`).

2. **Transparent Distributed Execution**: Write single-device code. Nabla automatically handles sharding, communication, and SPMD execution across devices.

3. **Trace-Based Autodiff**: Rather than storing gradient tape per-tensor, Nabla captures operation graphs and computes gradients by replaying traces backward.

---

## Key Lifecycles

Understanding Nabla requires understanding three lifecycles: **Operation Execution**, **Tracing & Rehydration**, and **Gradient Computation**.

### 1. Operation Execution Lifecycle (`__call__`)

Every operation (e.g., `add`, `matmul`, `relu`) goes through the same six-phase lifecycle in `Operation.__call__()`. This is the heart of Nabla's execution model:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Operation.__call__() Lifecycle                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: METADATA COLLECTION                                               │
│  ─────────────────────────────                                              │
│  • Scan all input tensors to gather: max_batch_dims, any_traced, any_sharded│
│  • These flags determine which subsequent phases are needed                 │
│                                                                             │
│  Phase 2: ADAPTATION (Resharding Inputs)                                    │
│  ───────────────────────────────────────                                    │
│  • Call op.adapt_kwargs() to translate logical kwargs → physical kwargs     │
│    (e.g., axis=0 in logical space → axis=batch_dims+0 in physical space)    │
│  • Call spmd.infer_output_sharding() to determine:                          │
│    - What sharding the output will have                                     │
│    - What sharding each input MUST have for correct execution               │
│    - Which axes need collective reduction (contracting dimensions)          │
│  • Call spmd.reshard_inputs() to insert AllGather/AllToAll if inputs don't  │
│    match required shardings                                                 │
│                                                                             │
│  Phase 3: PHYSICAL EXECUTION                                                │
│  ─────────────────────────────                                              │
│  • Call op.execute(resharded_args, kwargs)                         │
│  • This loops over each shard in the mesh and calls op.kernel() per shard    │
│  • Returns raw TensorValues (one per shard) + output sharding info          │
│                                                                             │
│  Phase 4: PACKAGING                                                         │
│  ─────────────────────                                                      │
│  • Wrap raw shard values into nabla.Tensor objects                          │
│  • Handle structured outputs (tuples, lists, dicts from multi-output ops)   │
│  • Attach sharding metadata to output tensors                               │
│                                                                             │
│  Phase 5: POST-OP COLLECTIVES                                               │
│  ────────────────────────────                                               │
│  • If output has partial sums (from contracting sharded dimensions),        │
│    execute AllReduce immediately to produce correct global result           │
│                                                                             │
│  Phase 6: TRACING (Graph Recording)                                         │
│  ───────────────────────────────────                                        │
│  • Call _setup_output_refs() to create OpNode node containing:          │
│    - Weak refs to output TensorImpls                                        │
│    - The operation instance                                                 │
│    - The input arguments (as TensorImpls)                                   │
│    - Original kwargs (for rehydration)                                      │
│  • This enables backward traversal for autodiff                             │
│                                                                             │
│  Phase 7: JVP (Forward-Mode Autodiff)                                       │
│  ─────────────────────────────────────                                      │
│  • If any input has a tangent, call op.jvp_rule() to propagate tangents     │
│  • Attach computed tangent to output tensor                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: Phases 1-5 execute eagerly (real computation happens). Phase 6 records for later replay. Phase 7 handles forward-mode differentiation.

→ **Detailed reference**: [ops/README.md](ops/README.md) explains each phase with code pointers.

---

### 2. Tracing & Rehydration Lifecycle

**Tracing** captures what operations happened. **Rehydration** replays those operations to restore graph values.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Trace Lifecycle                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FORWARD PASS (Tracing)                                                     │
│  ──────────────────────                                                     │
│  1. User calls trace(fn, *args) or implicitly via grad(fn)                  │
│  2. Input tensors marked as traced=True                                     │
│  3. Function executes, each operation records OpNode via Phase 6        │
│  4. Trace.compute() walks backward from outputs, collects nodes in topo     │
│     order via DFS on OpNode.op_args                                     │
│  5. Result: Trace object with .inputs, .outputs, .nodes (list of OpNode)│
│                                                                             │
│  REHYDRATION (Graph Value Restoration)                                      │
│  ──────────────────────────────────────                                     │
│  When: Called before backward pass, or when replaying a trace               │
│                                                                             │
│  Why needed: Graph values (_values) are epoch-scoped. After evaluate(),     │
│  the graph resets. Rehydration rebuilds _values for intermediate tensors.   │
│                                                                             │
│  How it works (Trace.refresh_graph_values()):                                          │
│  1. Find all leaf tensors (no output_refs) → ensure they're realized        │
│  2. Add leaves to current graph epoch via GRAPH.add_input()                 │
│  3. Iterate through nodes in topological order:                             │
│     a. Reconstruct args by wrapping TensorImpls as Tensors                  │
│     b. Call op.adapt_kwargs() for current batch_dims                        │
│     c. Call op.execute(args, kwargs) to recompute values           │
│     d. Map produced values back to original output TensorImpls              │
│  4. Now all intermediate tensors have valid _values in current epoch        │
│                                                                             │
│  Key design: execute receives ORIGINAL kwargs, not pre-adapted.    │
│  It performs adaptation internally. This ensures rehydration correctness.   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

→ **Detailed reference**: [core/graph/README.md](core/graph/README.md) for tracing, [core/autograd/README.md](core/autograd/README.md) for backward pass.

---

### 3. Gradient Computation Lifecycle (Autodiff)

Nabla uses **reverse-mode autodiff** via trace-based VJP (Vector-Jacobian Product):

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Backward Pass Lifecycle                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User calls: grad(fn)(x) or value_and_grad(fn)(x)                           │
│                                                                             │
│  1. TRACE: Execute fn(x), capture Trace object                              │
│                                                                             │
│  2. REHYDRATE: Call trace.refresh_graph_values() to restore all intermediate _values   │
│                                                                             │
│  3. INITIALIZE: Create cotangent_map with output cotangent (usually 1.0)    │
│                                                                             │
│  4. BACKWARD ITERATION: For each node in reversed(trace.nodes):             │
│     a. Skip if op has no vjp_rule or no output has cotangent                │
│     b. Reconstruct primals and outputs as Tensors from TensorImpls          │
│     c. Gather output cotangents from cotangent_map                          │
│     d. Call op.vjp_rule(primals, cotangent, output) → input cotangents      │
│     e. Reduce cotangents to match primal shapes (handle broadcasting)       │
│     f. Accumulate into cotangent_map (sum if tensor used multiple times)    │
│                                                                             │
│  5. FINALIZE: Extract gradients for original inputs from cotangent_map      │
│     - Handle partial sum resolution (AllReduce if needed)                   │
│     - Reshard to match input sharding                                       │
│                                                                             │
│  Result: dict[input_tensor → gradient_tensor]                               │
│                                                                             │
│  Key insight: VJP rules compute d(loss)/d(input) given d(loss)/d(output)    │
│  Each op's vjp_rule is the "chain rule" step for that operation.            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

→ **Detailed reference**: [core/autograd/README.md](core/autograd/README.md)

---

## The Dual Tensor System

Every tensor has two representations that enable powerful features:

| Aspect | Logical View | Physical View |
| :--- | :--- | :--- |
| **Shape** | Global shape (what user sees) | Per-shard local shape |
| **Storage** | `Tensor` with sharding metadata | `TensorImpl._values` (list per shard) |
| **Purpose** | User API, shape reasoning | Actual computation |

**Why this matters**:

- **Trace Rehydration**: Replay captured ops without re-executing Python
- **shard_map**: Same code runs differently (logical trace vs physical execution)
- **Epoch Management**: Values are scoped to graph epochs; rehydration restores them

---

## Sharding (SPMD Execution)

Nabla uses **factor-based sharding propagation** rather than dimension-based:

```python
# Matmul factors: "m k, k n -> m n"
#   m = batch/rows, k = contracting, n = columns
# If k is sharded, output has partial sums → AllReduce inserted automatically
```

**Three-phase propagation** (runs eagerly per-operation):

1. **COLLECT**: Gather shardings from input dimensions to factors
2. **RESOLVE**: Resolve conflicts using priority rules
3. **UPDATE**: Project factor shardings to output dimensions

→ **Detailed reference**: [core/sharding/README.md](core/sharding/README.md)

---

## Module Map

| Module | Purpose | Start Here |
| :--- | :--- | :--- |
| **[core/](core/README.md)** | Tensor state, graph engine, sharding, autodiff | Understanding internals |
| **[ops/](ops/README.md)** | Operation definitions, `__call__` lifecycle, VJP/JVP rules | Adding new ops |
| **[transforms/](transforms/README.md)** | `vmap`, `shard_map`, `compile` | High-level transforms |

### Core Submodules

| Submodule | Purpose | Key Concepts |
| :--- | :--- | :--- |
| [core/tensor/](core/tensor/README.md) | Tensor/TensorImpl facade | Dual object model, lazy realization |
| [core/graph/](core/graph/README.md) | Graph recording, tracing | OpNode, Trace, rehydration |
| [core/autograd/](core/autograd/README.md) | Gradient computation | BackwardEngine, VJP, cotangent accumulation |
| [core/sharding/](core/sharding/README.md) | SPMD distribution | Factor propagation, DeviceMesh |

---

## Quick Reference

**To understand how an operation executes**: Read `Operation.__call__()` in [ops/base.py](ops/base.py)

**To understand gradient computation**: Read `backward_on_trace()` in [core/autograd/utils.py](core/autograd/utils.py)

**To understand trace rehydration**: Read `Trace.refresh_graph_values()` in [core/graph/tracing.py](core/graph/tracing.py)

**To add a new operation**: Implement `kernel()`, `vjp_rule()`, and optionally `sharding_rule()` - see [ops/README.md](ops/README.md)

---

## Maintenance Guide

> **Note to AI Agents**:
>
> 1. **Read Recursively**: Start here, then follow links to understand specific subsystems.
> 2. **Update Requirement**: Update this file when modifying architecture. Keep lifecycle diagrams accurate.
> 3. **Focus on Lifecycles**: This file explains "how things flow". Submodule READMEs explain "what things are".
