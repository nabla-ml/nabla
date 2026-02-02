# Graph Engine & Tracing

[← Back to Core](../README.md)

> **Purpose**: The graph engine manages lazy evaluation, compiled model caching, and trace-based execution. This is where the "deferred graph building" magic happens.

## Key Concepts

### The Global GRAPH Singleton

All operations are recorded in the global `GRAPH` (`ComputeGraph`). No explicit graph contexts for users.

```python
from nabla.core import GRAPH

# All operations automatically tracked
y = x + 1  # Creates promise tensor, registers via GRAPH.add_unrealized()
z = y * 2  # Same: promise tensor tracked

z.numpy()  # GRAPH.evaluate(z) triggered → cache check → compile/run
```

### Promise Tensors and Unrealized Tracking

In **default mode** (`NABLA_EAGER_MAX_GRAPH=0`), operations create **promise tensors**:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        Promise Tensor Creation                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  y = x + 1  # In package_outputs() when EAGER_MAX_GRAPH=False:          │
│                                                                         │
│  y._impl._physical_shapes = [(4, 8)]     # Known from compute_physical  │
│  y._impl._shard_dtypes = [float32]       # Known                        │
│  y._impl._shard_devices = [GPU:0]        # Known                        │
│  y._impl._graph_values = []              # EMPTY - no MAX nodes yet     │
│  y._impl.graph_values_epoch = -1         # Special marker: "PROMISE"    │
│  y._impl.output_refs = OpNode(...)       # Recorded for trace replay    │
│                                                                         │
│  GRAPH.add_unrealized(y._impl)           # Track for later evaluation   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

The `GRAPH._unrealized_impls` set tracks all promise tensors. When `evaluate()` is called, these are the tensors that need their graph nodes built.

### Epochs and Value Staleness

Graph values (`_graph_values`) are scoped to **epochs**:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           Epoch Lifecycle                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Epoch N                                                                │
│  ────────                                                               │
│  y = x + 1      → y._impl.graph_values_epoch = -1 (promise)             │
│  z = y * 2      → z._impl.graph_values_epoch = -1 (promise)             │
│                                                                         │
│  z.numpy()      → GRAPH.evaluate(z)                                     │
│                   1. Cache MISS: _replay_trace_to_build_graph()         │
│                   2. y and z get _graph_values, epoch = N               │
│                   3. Compile & execute                                  │
│                   4. _finalize_evaluation(): GRAPH.epoch → N+1          │
│                   5. _cleanup_trace(): clear output_refs, _graph_values │
│                                                                         │
│  Epoch N+1                                                              │
│  ──────────                                                             │
│  y._impl.graph_values_epoch = N (stale! N < N+1)                        │
│  y._impl._get_valid_graph_values() returns []                           │
│                                                                         │
│  To use y again for new computation, must rehydrate via                 │
│  Trace.refresh_graph_values() or start fresh trace                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### OpNode: The Graph Node

Every operation creates an `OpNode` linking outputs to their creation context:

```python
@dataclass
class OpNode:
    _refs: tuple[TensorImpl, ...]   # Output TensorImpls (direct refs now, not weakrefs)
    tree_def: PyTreeDef             # Structure of outputs (for multi-output ops)
    op: Operation                   # The operation that created these outputs
    op_args: tuple[Any, ...]        # Input TensorImpls (not Tensors!)
    op_kwargs: dict[str, Any]       # ORIGINAL kwargs (critical for rehydration!)
    _op_hash: tuple[Any, ...]       # Structural hash for cache key
```

**Key design**: `op_kwargs` stores **original** kwargs, not adapted ones. `_op_hash` enables cache lookup.

---

## GRAPH.evaluate() - The Cache-First Execution Model

This is the heart of nabla's performance story. When you need concrete values:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    GRAPH.evaluate() Flow (engine.py)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. COLLECT TARGETS                                                     │
│  ──────────────────                                                     │
│  Flatten requested tensors, deduplicate by id(impl)                     │
│  Skip if all targets are leaves (no output_refs)                        │
│                                                                         │
│  2. COMPUTE CACHE KEY ⚡                                                 │
│  ────────────────────                                                   │
│  for each target:                                                       │
│      if unrealized: key = (op_hash, output_index)                       │
│      if realized:   key = (dtype, shape, sharding)                      │
│  cache_key = tuple(sorted(keys))                                        │
│                                                                         │
│  3. CACHE LOOKUP (THE FAST PATH)                                        │
│  ───────────────────────────────                                        │
│  if cache_key in _GRAPH_CACHE:                                          │
│      cached_model, kept_indices = _GRAPH_CACHE[cache_key]               │
│      inputs = gather_buffers_by_kept_indices()                          │
│      results = cached_model(*inputs)  # Skip ALL graph building!        │
│      store_results_to_targets()                                         │
│      return  # Done!                                                    │
│                                                                         │
│  4. CACHE MISS: BUILD GRAPH                                             │
│  ──────────────────────────                                             │
│  GRAPH.epoch += 1                     # Bump epoch                      │
│  self.graph = Graph("main", ...)      # Fresh MAX graph                 │
│  self._replay_trace_to_build_graph(targets)                             │
│                                                                         │
│  5. _replay_trace_to_build_graph()                                      │
│  ─────────────────────────────────                                      │
│  • DFS through OpNode.op_args → topological sort                        │
│  • For each OpNode in order:                                            │
│    - Ensure inputs have valid _graph_values                             │
│    - Call op.execute(args, original_kwargs) to build MAX nodes          │
│    - Store _graph_values to output TensorImpls                          │
│                                                                         │
│  6. COMPILE & EXECUTE                                                   │
│  ────────────────────                                                   │
│  model = session.load(self.graph)                                       │
│  results = model(*inputs)                                               │
│                                                                         │
│  7. STORE RESULTS & CACHE                                               │
│  ────────────────────────                                               │
│  Store buffers to target._impl._buffers                                 │
│  _GRAPH_CACHE[cache_key] = (model, kept_indices)                        │
│                                                                         │
│  8. CLEANUP                                                             │
│  ─────────                                                              │
│  _finalize_evaluation(): bump epoch, reset graph state                  │
│  _cleanup_trace(targets): clear output_refs and _graph_values           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why this matters**: In a training loop, the same computation structure repeats. After the first iteration builds and caches the model, subsequent iterations hit the cache and skip ALL graph building overhead.

---

## Trace and Rehydration

### Trace: A Captured Subgraph

`trace(fn, *args)` captures the computation graph:

```python
t = trace(lambda x: x * 2 + 1, input_tensor)
# t.inputs = (input_tensor,)
# t.outputs = result_tensor
# t.nodes = [OpNode(mul), OpNode(add)]  # topological order
```

**Trace.compute()**: Walks backward from outputs via `output_refs.op_args`, collecting nodes in topological order via DFS.

### Rehydration: Restoring Graph Values

**Why needed**: Before backward pass (when `EAGER_MAX_GRAPH=1`), intermediate tensors may have stale `_graph_values` because `evaluate()` bumped the epoch and cleared them. Rehydration replays operations to restore them.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    Trace.refresh_graph_values()                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  WHEN CALLED: In backward_on_trace() when EAGER_MAX_GRAPH=1             │
│                                                                         │
│  WHY NEEDED:                                                            │
│  • Forward pass built graph nodes (eager mode)                          │
│  • evaluate() ran to get forward results                                │
│  • evaluate() bumped epoch and cleared _graph_values                    │
│  • VJP operations (in eager mode) need primals with valid graph values  │
│  • Solution: Replay trace to rebuild _graph_values in current epoch     │
│                                                                         │
│  HOW IT WORKS:                                                          │
│  1. Find leaf tensors (no output_refs) → ensure realized                │
│  2. Add leaves to current graph epoch via GRAPH.add_input()             │
│  3. For each OpNode in topological order:                               │
│     a. Wrap TensorImpls as Tensors                                      │
│     b. Call op.execute(args, ORIGINAL_kwargs)                           │
│     c. Map fresh _graph_values back to original TensorImpls             │
│  4. All intermediate tensors now have valid _graph_values               │
│                                                                         │
│  CODE (core/autograd/utils.py):                                         │
│  if EAGER_MAX_GRAPH:                                                    │
│      trace.refresh_graph_values()  # Critical for VJP to work!          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Critical design decision**: Rehydration calls `op.execute(args, kwargs)` with **original kwargs**. This is why `execute` must accept original kwargs and adapt internally—it's the only kwargs stored in OpNode!

---

## Lazy Evaluation Triggers

Tensors stay as "promises" until data is explicitly needed:

```python
y = x + 1      # Promise tensor: has shapes but no graph values (default mode)
z = y * 2      # Another promise

# Triggers evaluate():
print(z.numpy())  # → GRAPH.evaluate(z)
z.item()          # → GRAPH.evaluate(z)  
float(z)          # → GRAPH.evaluate(z)
```

---

## Component Map

| File | Purpose | Key Exports |
| :--- | :--- | :--- |
| [engine.py](engine.py) | Global graph singleton, evaluate(), caching | `ComputeGraph`, `GRAPH`, `_GRAPH_CACHE` |
| [tracing.py](tracing.py) | Trace capture, OpNode, rehydration | `Trace`, `OpNode`, `trace`, `GraphPrinter` |
| [utils.py](utils.py) | Graph traversal algorithms | `get_operations_topological`, `get_all_impls_topological` |

---

## Maintenance Guide

> **AI Agents - Critical Rules**:
>
> 1. **OpNode.op_kwargs**: Must store ORIGINAL kwargs, not adapted. Rehydration depends on this.
> 2. **OpNode._op_hash**: Used for cache key computation. Don't change hash semantics without updating cache.
> 3. **Epochs**: After `evaluate()`, epoch increments. All old `_graph_values` become stale.
> 4. **Promise tensors**: `graph_values_epoch = -1` marks a tensor as unrealized/promise.
> 5. **GRAPH.add_unrealized()**: Must be called for deferred tensors so they're tracked for evaluation.
> 6. **refresh_graph_values()**: Critical for EAGER_MAX_GRAPH mode backward pass. Don't remove!
