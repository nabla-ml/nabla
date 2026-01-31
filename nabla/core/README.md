# Core Internals

[← Back to Root](../README.md)

> **Purpose**: This module contains the fundamental building blocks that power Nabla: tensor state management, graph recording, automatic differentiation, and distributed execution.

## How the Core Modules Work Together

Understanding Nabla requires seeing how these components interact during operation execution:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Core Module Interaction Flow                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Code                                                                 │
│      │                                                                      │
│      ▼                                                                      │
│   ┌──────────────┐                                                          │
│   │   TENSOR     │  User-facing Tensor wraps TensorImpl (state)             │
│   │   (facade)   │  Provides .shape, .numpy(), arithmetic operators         │
│   └──────┬───────┘                                                          │
│          │ calls operation (e.g., x + y)                                    │
│          ▼                                                                  │
│   ┌──────────────┐                                                          │
│   │   SHARDING   │  Infers output sharding from input shardings             │
│   │   (spmd.py)  │  Determines if inputs need resharding                    │
│   └──────┬───────┘  Inserts AllGather/AllReduce if needed                   │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────┐                                                          │
│   │    GRAPH     │  Executes kernel() per shard inside graph context         │
│   │   (engine)   │  Records OpNode node for tracing                     │
│   └──────┬───────┘                                                          │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────┐                                                          │
│   │  AUTOGRAD    │  (Later) Walks OpNode backward to compute gradients  │
│   │  (backward)  │  Uses trace rehydration to restore intermediate values   │
│   └──────────────┘                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. The Dual Object Model (Tensor/TensorImpl)

**Problem**: Multi-output operations (like `split`) produce multiple tensors that share the same "parent" operation. How do we track this for autodiff?

**Solution**: Separate user API (`Tensor`) from internal state (`TensorImpl`):

```text
Tensor (user-facing)              TensorImpl (internal state)
├── .shape, .dtype, .device       ├── _values: list[TensorValue]  # lazy graph nodes
├── .numpy(), .item()             ├── _storages: list[driver.Tensor]  # realized data
├── arithmetic operators          ├── sharding: ShardingSpec
└── wraps ─────────────────────► ├── output_refs: OpNode  # parent op info
                                  ├── traced: bool
                                  └── batch_dims: int
```

**OpNode**: When an operation produces outputs, all sibling outputs share the SAME `OpNode` object. This contains:

- The operation that created them
- The input arguments (as TensorImpls)
- Weak references to all output TensorImpls

This enables backward traversal: from any tensor, follow `output_refs.op_args` to find its inputs.

### 2. Graph Epochs and Rehydration

**Problem**: After `evaluate()` compiles and runs a graph, the old `_values` become stale. But autodiff needs intermediate values.

**Solution**: Epochs + Rehydration

**Epochs**: Each graph compilation increments `GRAPH.epoch`. TensorImpl stores `values_epoch` to detect staleness.

**Rehydration** (`Trace.refresh_graph_values()`): Before backward pass, replay all operations:

1. Find leaf tensors (constants, inputs) → ensure realized
2. Add leaves to current graph epoch
3. For each operation in topological order:
   - Call `op.execute()` to recompute _values
   - Update output TensorImpls with fresh values

This is why `execute` receives ORIGINAL kwargs and performs adaptation internally—rehydration doesn't have access to pre-computed adapted kwargs.

### 3. Lazy Evaluation Model

```text
                    ┌─────────────┐
                    │   Tensor    │
                    │  created    │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   UNREALIZED         OPERATION          REALIZED
   (has _values)        called          (has _storages)
        │                  │                  │
        │                  │                  │
        │     _values      │                  │
        │◄─────────────────┤                  │
        │                  │                  │
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                     .numpy() or
                     print() called
                           │
                           ▼
                    ┌─────────────┐
                    │  GRAPH      │
                    │  compiles   │
                    │  & executes │
                    └─────────────┘
```

**Key insight**: Most tensors stay "unrealized" (only `_values`, no `_storages`) until you explicitly need data. This enables graph optimization before execution.

### 4. Physical Execution Context

Operations call `kernel()` inside `GRAPH.graph` context. This:

- Provides access to lazy tensor values without triggering recursive compilation
- Allows graph node creation for the current epoch
- Required for any code that manipulates `_values`

## Module Architecture

Strict import hierarchy to avoid circular dependencies:

```text
Level 0: common/     (pytree, context managers - no deps)
         │
Level 1: graph/      (Graph, imports common)
         │
Level 2: tensor/     (Tensor, TensorImpl - imports graph)
         │
Level 3: sharding/   (ShardingSpec, propagation - imports tensor)
         │
Level 4: autograd/   (grad, backward - imports all above)
```

## Component Map

| Submodule | Purpose | Key Concepts | Documentation |
| :--- | :--- | :--- | :--- |
| **[tensor/](tensor/README.md)** | State management | Tensor/TensorImpl facade, lazy realization | Dual object model |
| **[graph/](graph/README.md)** | Execution engine | GRAPH singleton, OpNode, Trace, rehydration | Graph recording, epochs |
| **[sharding/](sharding/README.md)** | SPMD distribution | Factor propagation, DeviceMesh, resharding | Automatic communication |
| **[autograd/](autograd/README.md)** | Differentiation | BackwardEngine, VJP rules, cotangent accumulation | Trace-based gradients |
| **[common/](common/README.md)** | Utilities | pytree operations, context managers | Shared infrastructure |

## Exported Symbols

From `nabla.core`:

```python
# Tensor System
Tensor, TensorImpl, OpNode

# Graph Engine  
Graph, GRAPH, Trace, trace

# Defaults & Context
defaults, default_device, default_dtype, defaults_like
traced, untraced, with_batch_dims

# PyTree Utilities
tree_map, tree_flatten, tree_unflatten, tree_leaves, tree_structure, PyTreeDef
```

## Maintenance Guide

> **Note to AI Agents**:
>
> 1. **Import Hierarchy**: Respect the levels above. Adding imports that go "up" creates cycles.
> 2. **Rehydration**: If changing operation execution, ensure `execute` can work during rehydration (receives original kwargs).
> 3. **OpNode**: Any change to how operations record their outputs affects autodiff. Test gradients.
