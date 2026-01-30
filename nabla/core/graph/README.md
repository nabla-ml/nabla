# Graph Engine & Tracing

[← Back to Core](../README.md)

> **Purpose**: The graph engine captures operations, compiles them to MAX executables, and manages the trace-based execution model.

## Key Concepts

### The Global GRAPH Singleton

All operations are recorded in the global `GRAPH` (`ComputeGraph`). No explicit graph contexts for users.

```python
from nabla.core import GRAPH

# All operations automatically recorded
y = x + 1  # Adds node to GRAPH
z = y * 2  # Adds another node
```

### Epochs and Value Staleness

Graph values (`_values`) are scoped to **epochs**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Epoch Lifecycle                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Epoch N                                                                │
│  ────────                                                               │
│  y = x + 1      → y._impl._values = [TensorValue...]                    │
│                   y._impl.values_epoch = N                              │
│                                                                         │
│  z = y * 2      → z._impl._values = [TensorValue...]                    │
│                   z._impl.values_epoch = N                              │
│                                                                         │
│  z.numpy()      → GRAPH.evaluate(z)                                     │
│                   Compiles graph, executes                              │
│                   GRAPH.epoch += 1  (now N+1)                           │
                                                                          │
│  Epoch N+1                                                              │
│  ──────────                                                             │
│  y._impl.values_epoch = N  (stale! N < N+1)                             │
│  y._impl._get_valid_values() returns []                                 │
│                                                                         │
│  To use y again, must either:                                           │
│  • Create new operations (y becomes leaf, gets fresh values)            │
│  • Rehydrate a trace containing y                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### OutputRefs: The Graph Node

Every operation creates an `OutputRefs` node linking outputs to their creation context:

```python
@dataclass
class OutputRefs:
    _refs: tuple[weakref.ref, ...]  # Weak refs to output TensorImpls
    tree_def: PyTreeDef             # Structure of outputs (for multi-output ops)
    op: Operation                   # The operation that created these outputs
    op_args: tuple[Any, ...]        # Input TensorImpls (not Tensors!)
    op_kwargs: dict[str, Any]       # Original kwargs (for rehydration)
```

**Key design**: `op_kwargs` stores **original** kwargs, not adapted ones. This is essential for rehydration—see why below.

### Trace: A Captured Subgraph

`trace(fn, *args)` captures the computation graph:

```python
t = trace(lambda x: x * 2 + 1, input_tensor)
# t.inputs = (input_tensor,)
# t.outputs = result_tensor
# t.nodes = [OutputRefs(mul), OutputRefs(add)]  # topological order
```

**Trace.compute()**: Walks backward from outputs via `output_refs.op_args`, collecting nodes in topological order via DFS.

## Rehydration: Restoring Graph Values

**Why needed**: Before backward pass (or trace replay), intermediate tensors may have stale `_values`. Rehydration replays operations to restore them.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Trace.rehydrate()                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  def rehydrate(self):                                                   │
│      # 1. Find leaf tensors (no output_refs = constants/inputs)         │
│      leaf_impls = set()                                                 │
│      for ref in self.nodes:                                             │
│          for arg in tree_leaves(ref.op_args):                           │
│              if isinstance(arg, TensorImpl) and not arg.output_refs:    │
│                  leaf_impls.add(arg)                                    │
│                                                                         │
│      # 2. Realize all leaves together                                   │
│      leaf_tensors = [Tensor(impl=impl) for impl in leaf_impls]          │
│      GRAPH.evaluate(*leaf_tensors)                                      │
│                                                                         │
│      # 3. Add leaves to current epoch                                   │
│      for t in leaf_tensors:                                             │
│          GRAPH.add_input(t)                                             │
│          with GRAPH.graph:                                              │
│              t._impl._values = [v[...] for v in t._values]              │
│              t._impl.values_epoch = GRAPH.epoch                         │
│                                                                         │
│      # 4. Replay operations in topological order                        │
│      for output_refs in self.nodes:                                     │
│          op = output_refs.op                                            │
│          args = wrap_as_tensors(output_refs.op_args)                    │
│          kwargs = output_refs.op_kwargs  # ← ORIGINAL kwargs!           │
│                                                                         │
│          # physical_execute adapts kwargs internally                    │
│          with GRAPH.graph:                                              │
│              result = op.physical_execute(args, kwargs)                 │
│                                                                         │
│          # Map fresh values back to original TensorImpls                │
│          for ref, new_impl in zip(output_refs._refs, result_impls):     │
│              old_impl = ref()                                           │
│              if old_impl:                                               │
│                  old_impl._values = new_impl._values                    │
│                  old_impl.values_epoch = GRAPH.epoch                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Critical design decision**: Rehydration calls `op.physical_execute(args, kwargs)` with **original kwargs**. This is why `physical_execute` must accept original kwargs and adapt internally—it's the only kwargs we have stored!

## Lazy Evaluation

Tensors don't execute until data is needed:

```python
y = x + 1      # Just records: y has _values (symbolic), no _storages
z = y * 2      # Records again: z has _values, no _storages

print(z.numpy())  # NOW compilation + execution happens
                  # z._storages gets populated
                  # y._values become stale (epoch incremented)
```

**Evaluation trigger points**: `.numpy()`, `.item()`, `print()`, or explicit `GRAPH.evaluate()`.

## Component Map

| File | Purpose | Key Exports |
|------|---------|-------------|
| [engine.py](engine.py) | Global graph singleton, evaluation | `ComputeGraph`, `GRAPH`, `driver_tensor_type` |
| [tracing.py](tracing.py) | Trace capture, OutputRefs, rehydration | `Trace`, `OutputRefs`, `trace`, `GraphPrinter` |
| [utils.py](utils.py) | Graph traversal algorithms | `get_operations_topological`, `get_all_impls_topological` |

## Maintenance Guide

> **AI Agents - Critical Rules**:
> 1. **OutputRefs.op_kwargs**: Must store ORIGINAL kwargs, not adapted. Rehydration depends on this.
> 2. **Epochs**: After `evaluate()`, epoch increments. All old `_values` become stale.
> 3. **Weak refs**: OutputRefs uses weakrefs to outputs. GC'd tensors return `None` from `ref()`.
> 4. **Graph context**: Physical execution must run inside `GRAPH.graph` context.
