# Nabla Module Architecture

> **See [../readme.md](../readme.md)** for getting started, setup instructions, and architecture overview.
>
> This document covers **internal implementation details** for contributors working on the nabla module.

---

## Core Philosophy

Nabla is a **tensor library with automatic differentiation** built on three pillars:

1. **Eager Metadata, Deferred Graph Building**: Shapes, dtypes, and sharding specs are computed immediately. MAX graph construction is **deferred by default**—tensors are created as "promises" that only build graph nodes when `evaluate()` is triggered.

2. **Transparent Distributed Execution**: Write single-device code. Nabla automatically handles sharding, communication, and SPMD execution across devices.

3. **Trace-Based Autodiff**: Rather than storing gradient tape per-tensor, Nabla captures operation graphs and computes gradients by replaying traces backward.

---

## Execution Modes

Nabla supports **two execution modes**, controlled by the `NABLA_EAGER_MAX_GRAPH` environment variable:

### Default Mode: Deferred Graph Building (`NABLA_EAGER_MAX_GRAPH=0`)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Default: Deferred Graph Building                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  y = x + 1              ← Operation.__call__() runs:                        │
│                           • Computes output shapes, dtypes, devices         │
│                           • Does NOT call op.execute() (no MAX graph yet)   │
│                           • Creates "promise tensor" (graph_values_epoch=-1)│
│                           • Registers via GRAPH.add_unrealized(y._impl)     │
│                           • Records OpNode for later replay                 │
│                                                                             │
│  z = y * 2              ← Same: shapes computed, no MAX graph, promise      │
│                                                                             │
│  print(z.numpy())       ← GRAPH.evaluate(z) triggered:                      │
│                           1. Check cache for compiled model (by op_hash)    │
│                           2. Cache HIT → Skip graph building entirely!      │
│                           3. Cache MISS → Build graph via _replay_trace()   │
│                           4. Compile & execute on device                    │
│                           5. Store results, clear trace references          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why defer graph building?** MAX graph construction has overhead. If we hit the compiled model cache (same computation structure), we skip graph building entirely and just replay the cached model with new data. This is a significant performance win for hot paths.

**Why compute shapes eagerly?** Even though graph building is deferred:

- Users need `.shape`, `.dtype`, `.device` immediately for control flow and debugging
- Sharding propagation requires shape information to determine data movement
- Type checking and broadcasting validation must happen at operation time

### Eager Mode: Immediate Graph Building (`NABLA_EAGER_MAX_GRAPH=1`)

```bash
export NABLA_EAGER_MAX_GRAPH=1
export NABLA_VERIFY_EAGER_SHAPES=1  # Optional: validate shape inference
```

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Eager: Immediate Graph Building                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  y = x + 1              ← Operation.__call__() runs:                        │
│                           • Computes output shapes (same as default)        │
│                           • ALSO calls op.execute() to build MAX graph      │
│                           • Stores _graph_values immediately                │
│                           • Sets graph_values_epoch = GRAPH.epoch           │
│                           • Records OpNode                                  │
│                                                                             │
│  print(z.numpy())       ← GRAPH.evaluate(z) triggered:                      │
│                           • Graph already built, just compile & execute     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**When to use eager mode?** Useful for debugging shape mismatches (with `NABLA_VERIFY_EAGER_SHAPES=1`), or when you want to inspect the MAX graph during development.

---

## Key Lifecycles

Understanding Nabla requires understanding three lifecycles: **Operation Execution**, **Graph Evaluation**, and **Gradient Computation**.

### 1. Operation Execution Lifecycle (`__call__`)

Every operation (e.g., `add`, `matmul`, `relu`) goes through a **9-step pipeline** in `Operation.__call__()`. The key insight: **steps 1-4 always run** (metadata), while **step 5 is conditional** on the execution mode.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Operation.__call__() Pipeline                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: METADATA COLLECTION                                                │
│  ───────────────────────────                                                │
│  max_batch_dims, any_traced, any_sharded, any_has_tangent = collect_metadata│
│  Purpose: Determine execution context from inputs (vmap, autodiff, SPMD)    │
│                                                                             │
│  Step 2: ADAPTATION & RESHARDING                                            │
│  ───────────────────────────────                                            │
│  resharded_args, adapted_kwargs, predicted_output_spec, mesh, reduce_axes = │
│      adapt_and_reshard(self, args, kwargs, any_sharded, max_batch_dims)     │
│  Purpose: Translate logical→physical kwargs, predict output sharding,       │
│           insert AllGather/AllToAll if inputs need resharding               │
│                                                                             │
│  Step 3: COMPUTE STRUCTURAL HASH                                            │
│  ───────────────────────────────                                            │
│  op_hash = compute_structural_hash(self.name,resharded_args,adapted_kwargs) │
│  Purpose: Create cache key for compiled model lookup (critical for perf!)   │
│                                                                             │
│  Step 4: COMPUTE PHYSICAL SHAPES - ALWAYS RUNS!                             │
│  ───────────────────────────────                                            │
│  output_physical_shapes, output_shard_dtypes, output_shard_devices =        │
│      self.compute_physical_shape(resharded_args, adapted_kwargs, ...)       │
│  Purpose: Infer output metadata WITHOUT building MAX graph nodes            │
│  Why always? Users need .shape immediately; sharding needs shapes too       │
│                                                                             │
│  Step 5: EAGER EXECUTION (CONDITIONAL) ⚡                                    │
│  ────────────────────────────────────────                                   │
│  execution_results = eager_execute(self, resharded_args, kwargs, ...)       │
│  • If EAGER_MAX_GRAPH=0: Returns None (graph building deferred)             │
│  • If EAGER_MAX_GRAPH=1: Calls op.execute() to build MAX graph nodes        │
│                                                                             │
│  Step 6: PACKAGING (Create Tensor)                                          │
│  ─────────────────────────────────                                          │
│  output = package_outputs(self, execution_results, shapes, dtypes, ...)     │
│  • If EAGER: output gets _graph_values, graph_values_epoch = current epoch  │
│  • If DEFERRED: output is a "promise" with graph_values_epoch = -1          │
│                 GRAPH.add_unrealized(output._impl) registers for later      │
│                                                                             │
│  Step 7: SETUP OUTPUT REFS (OpNode Creation)                                │
│  ───────────────────────────────────────────                                │
│  self._setup_output_refs(output, resharded_args, kwargs, op_hash=op_hash)   │
│  Creates OpNode with: op, inputs, ORIGINAL kwargs, op_hash                  │
│  This enables trace replay and backward traversal for autodiff              │
│                                                                             │
│  Step 8: AUTO-REDUCTION                                                     │
│  ──────────────────────                                                     │
│  output = apply_auto_reduction(self, output, mesh, reduce_axes)             │
│  If contracting dimensions were sharded, insert AllReduce for partial sums  │
│                                                                             │
│  Step 9: JVP PROPAGATION                                                    │
│  ───────────────────────                                                    │
│  if any_has_tangent: apply_jvp(self, args, output)                          │
│  Forward-mode autodiff: propagate tangents through op.jvp_rule()            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: Steps 1-4, 6-9 always execute (metadata + tracing). Step 5 is where the execution mode matters—it determines whether MAX graph nodes are built immediately or deferred.

→ **Detailed reference**: [ops/README.md](ops/README.md) explains each step with code pointers.

---

### 2. Graph Evaluation Lifecycle (`GRAPH.evaluate`)

When you need concrete values (`.numpy()`, `print()`, or explicit `evaluate()`), the graph engine runs:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GRAPH.evaluate() Lifecycle                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. COLLECT TARGETS                                                         │
│  ──────────────────                                                         │
│  Flatten requested tensors, deduplicate by id(impl)                         │
│                                                                             │
│  2. COMPUTE CACHE KEY                                                       │
│  ────────────────────                                                       │
│  cache_key = tuple(get_tensor_key(t) for t in sorted_targets)               │
│  • Unrealized tensors: keyed by (op_hash, output_index)                     │
│  • Realized tensors: keyed by (dtype, shape, sharding)                      │
│                                                                             │
│  3. CACHE LOOKUP ⚡ (THE FAST PATH)                                          │
│  ─────────────────                                                          │
│  if cache_key in _GRAPH_CACHE:                                              │
│      cached_model, kept_indices = _GRAPH_CACHE[cache_key]                   │
│      → Gather input buffers using kept_indices                              │
│      → Run cached_model(*inputs) directly                                   │
│      → Store results to target tensors                                      │
│      → Skip ALL graph building!                                             │
│                                                                             │
│  4. CACHE MISS: BUILD GRAPH                                                 │
│  ──────────────────────────                                                 │
│  GRAPH.epoch += 1  # Bump epoch (old _graph_values now stale)               │
│  self.graph = Graph("main", ...)  # Fresh MAX graph                         │
│  self._replay_trace_to_build_graph(targets)  # Walk OpNode DAG              │
│                                                                             │
│  5. _replay_trace_to_build_graph()                                          │
│  ─────────────────────────────────                                          │
│  • DFS through OpNode.op_args to collect nodes in topological order         │
│  • For each OpNode:                                                         │
│    - Ensure inputs have valid _graph_values (add_input for realized)        │
│    - Call op.execute() to build MAX graph nodes                             │
│    - Store _graph_values to output TensorImpls                              │
│                                                                             │
│  6. COMPILE & EXECUTE                                                       │
│  ────────────────────                                                       │
│  model = session.load(self.graph)  # Compile to executable                  │
│  results = model(*inputs)          # Run on device                          │
│                                                                             │
│  7. STORE RESULTS & CACHE                                                   │
│  ────────────────────────                                                   │
│  • Store buffers to target._impl._buffers                                   │
│  • Cache: _GRAPH_CACHE[cache_key] = (model, kept_indices)                   │
│                                                                             │
│  8. CLEANUP                                                                 │
│  ─────────                                                                  │
│  _finalize_evaluation()  # Bump epoch, reset graph state                    │
│  _cleanup_trace(targets) # Clear output_refs and _graph_values on targets   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why caching matters**: If you run the same computation structure with different data (e.g., training loop iterations), the cache key matches and you skip graph building entirely. This is why deferred graph building + caching is the default—hot paths are fast.

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
│  1. TRACE: Execute fn(x), capture Trace object with OpNodes                 │
│                                                                             │
│  2. COMPUTE FORWARD: trace.compute() evaluates to get concrete outputs      │
│                                                                             │
│  3. REHYDRATE (if EAGER_MAX_GRAPH=1):                                       │
│     trace.refresh_graph_values()                                            │
│     Why? Forward pass built graph → evaluate() bumped epoch and cleared     │
│     _graph_values → VJP ops need primals with valid graph values            │
│     Solution: Replay trace to rebuild _graph_values in current epoch        │
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
└─────────────────────────────────────────────────────────────────────────────┘
```

**Critical: Why `refresh_graph_values()` in EAGER_MAX_GRAPH mode?**

When `EAGER_MAX_GRAPH=1`, operations build MAX graph nodes immediately. But when `evaluate()` runs (to get forward pass results), it:

1. Compiles and executes the graph
2. **Bumps the epoch** (`GRAPH.epoch += 1`)
3. **Clears `_graph_values`** via `_cleanup_trace()`

The backward pass then needs to call VJP rules, which are themselves operations that (in eager mode) immediately try to build graph nodes. These VJP ops need their input tensors (the forward primals) to have valid `_graph_values` in the **current** epoch. But they're stale/cleared!

**Solution**: Before backward, call `trace.refresh_graph_values()` which:

1. Finds all leaf tensors (inputs) and ensures they're realized
2. Adds them to the current graph epoch
3. **Replays all forward operations** to rebuild `_graph_values`

This is why the code in `backward_on_trace()` has:

```python
if EAGER_MAX_GRAPH:
    trace.refresh_graph_values()
```

→ **Detailed reference**: [core/autograd/README.md](core/autograd/README.md)

---

## The Promise Tensor Pattern

A key architectural concept is the **promise tensor**—a tensor that knows its shape but hasn't built its MAX graph nodes yet:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Promise vs Realized Tensors                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROMISE TENSOR (deferred graph building)                                   │
│  ────────────────────────────────────────                                   │
│  y = x + 1  # In default mode                                               │
│                                                                             │
│  y._impl._physical_shapes = [(4, 8), (4, 8)]  # Known from compute_physical │
│  y._impl._shard_dtypes = [float32, float32]   # Known                       │
│  y._impl._shard_devices = [GPU:0, GPU:1]      # Known                       │
│  y._impl._graph_values = []                    # EMPTY - no MAX nodes yet   │
│  y._impl.graph_values_epoch = -1               # Special marker: "promise"  │
│  y._impl.output_refs = OpNode(...)            # OpNode recorded for replay  │
│                                                                             │
│  GRAPH._unrealized_impls contains y._impl     # Tracked for later evaluate  │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  REALIZED TENSOR (after evaluate)                                           │
│  ─────────────────────────────────                                          │
│  y.numpy()  # Triggers GRAPH.evaluate(y)                                    │
│                                                                             │
│  y._impl._buffers = [driver.Buffer, driver.Buffer]  # Actual device memory  │
│  y._impl.graph_values_epoch = GRAPH.epoch           # Current epoch         │
│  y._impl.output_refs = None                         # Cleared after eval    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why this design?**

1. **Cache efficiency**: Promise tensors carry `op_hash` for cache key computation
2. **Lazy evaluation**: Defer work until actually needed
3. **Memory efficiency**: Don't allocate device memory until necessary

---

## The Dual Tensor System

Every tensor has two representations that enable powerful features:

| Aspect | Logical View | Physical View |
| :--- | :--- | :--- |
| **Shape** | Global shape (what user sees) | Per-shard local shape |
| **Buffers** | `Tensor` with sharding metadata | `TensorImpl._values` (list per shard) |
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
