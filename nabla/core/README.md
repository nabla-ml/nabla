# Core Internals

[← Back to Root](../README.md)

> **Purpose**: This module contains the fundamental building blocks that power Nabla: tensor state management, graph recording, automatic differentiation, and distributed execution.

## How the Core Modules Work Together

Understanding Nabla requires seeing how these components interact during the deferred execution flow:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│              Core Module Interaction (Default Deferred Mode)                │
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
│   │ OPERATION    │  __call__() computes shapes via compute_physical_shape   │
│   │ (ops/base)   │  Creates "promise tensor" (graph_values_epoch = -1)      │
│   └──────┬───────┘  Records OpNode with op_hash for cache key               │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────┐                                                          │
│   │    GRAPH     │  GRAPH.add_unrealized() tracks promise tensors           │
│   │   (engine)   │  Later: evaluate() checks cache, builds graph if miss    │
│   └──────┬───────┘  _replay_trace_to_build_graph() walks OpNode DAG         │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────┐                                                          │
│   │  AUTOGRAD    │  backward_on_trace() computes gradients                  │
│   │  (backward)  │  if EAGER_MAX_GRAPH: refresh_graph_values() first        │
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
├── .shape, .dtype, .device       ├── _graph_values: list[TensorValue]  # MAX graph nodes
├── .numpy(), .item()             ├── _buffers: list[driver.Buffer]     # realized data
├── arithmetic operators          ├── _physical_shapes: list[tuple]     # per-shard shapes
└── wraps ─────────────────────► ├── graph_values_epoch: int           # -1 = promise!
                                  ├── sharding: ShardingSpec
                                  ├── output_refs: OpNode               # parent op info
                                  ├── traced: bool
                                  └── batch_dims: int
```

**OpNode**: When an operation produces outputs, all sibling outputs share the SAME `OpNode` object. This contains:

- The operation that created them
- The input arguments (as TensorImpls)  
- The **original** kwargs (critical for rehydration!)
- The `_op_hash` for cache key computation

This enables backward traversal: from any tensor, follow `output_refs.op_args` to find its inputs.

### 2. Graph Epochs and Promise Tensors

**Problem**: In default mode, operations don't build MAX graph nodes immediately. How do we track what needs to be built later?

**Solution**: Promise tensors + epoch-based staleness detection

**Promise Tensor Pattern** (default mode, `EAGER_MAX_GRAPH=0`):
```python
y = x + 1  # In package_outputs():

y._impl._physical_shapes = [(4, 8)]     # Known from compute_physical_shape
y._impl._shard_dtypes = [float32]       # Known
y._impl._shard_devices = [GPU:0]        # Known  
y._impl._graph_values = []              # EMPTY - no MAX nodes yet
y._impl.graph_values_epoch = -1         # Special marker: "PROMISE"
y._impl.output_refs = OpNode(...)       # Recorded for trace replay

GRAPH.add_unrealized(y._impl)           # Track in _unrealized_impls set
```

**Epoch-Based Staleness**:
- Each `evaluate()` call increments `GRAPH.epoch`
- TensorImpl stores `graph_values_epoch` when `_graph_values` were set
- If `impl.graph_values_epoch != GRAPH.epoch`, values are stale
- `_get_valid_graph_values()` returns `[]` for stale tensors

**Rehydration** (`Trace.refresh_graph_values()`): Before backward pass (in `EAGER_MAX_GRAPH` mode), replay all operations to restore `_graph_values` for intermediate tensors that need them.

### 3. Lazy Evaluation Model

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Tensor State Transitions                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐                        ┌───────────────┐                 │
│  │   PROMISE     │     GRAPH.evaluate()   │   REALIZED    │                 │
│  │   TENSOR      │  ──────────────────►   │   TENSOR      │                 │
│  ├───────────────┤     (cache check,      ├───────────────┤                 │
│  │ _graph_values │      build graph,      │ _buffers =    │                 │
│  │   = []        │      compile, run)     │   [data...]   │                 │
│  │ epoch = -1    │                        │ epoch = N     │                 │
│  │ output_refs   │                        │ output_refs   │                 │
│  │   = OpNode    │                        │   = None      │                 │
│  └───────────────┘                        └───────────────┘                 │
│                                                                             │
│  Key: evaluate() clears output_refs after execution                         │
│       This prevents memory leaks and marks tensor as "terminal"             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why deferred by default?** Cache efficiency. If the same computation structure runs again, we skip graph building entirely and just replay the cached compiled model.

---

## Module Architecture

Strict import hierarchy to avoid circular dependencies:

```text
Level 0: common/     (pytree, context managers - no deps)
         │
Level 1: graph/      (ComputeGraph, imports common)
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
| **[tensor/](tensor/README.md)** | State management | Tensor/TensorImpl facade, promise tensors | Dual object model |
| **[graph/](graph/README.md)** | Execution engine | GRAPH singleton, OpNode, evaluate(), caching | Graph recording, epochs |
| **[sharding/](sharding/README.md)** | SPMD distribution | Factor propagation, DeviceMesh, resharding | Automatic communication |
| **[autograd/](autograd/README.md)** | Differentiation | BackwardEngine, VJP rules, refresh_graph_values | Trace-based gradients |
| **[common/](common/README.md)** | Utilities | pytree operations, context managers | Shared infrastructure |

## Exported Symbols

From `nabla.core`:

```python
# Tensor System
Tensor, TensorImpl, OpNode

# Graph Engine  
ComputeGraph, GRAPH, Trace, trace

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
> 2. **Promise tensors**: `graph_values_epoch = -1` marks unrealized. Use `GRAPH.add_unrealized()` to track.
> 3. **Rehydration**: If changing operation execution, ensure `execute` receives original kwargs (not adapted).
> 4. **OpNode**: Any change to how operations record outputs affects both caching and autodiff. Test both.
> 5. **refresh_graph_values()**: Critical for EAGER_MAX_GRAPH backward pass. Don't remove without understanding why it exists.
> 3. **OpNode**: Any change to how operations record their outputs affects autodiff. Test gradients.
