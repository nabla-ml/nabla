# Tensor System

[← Back to Core](../README.md)

> **Purpose**: The dual-object Tensor/TensorImpl model separates user API from internal state, enabling multi-output operations and trace-based autodiff.

## The Dual-Object Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        User Code                                        │
│                            │                                            │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                         Tensor (API)                            │    │
│  │  ─────────────────────────────────────────────────────────────  │    │
│  │  • User-facing, lightweight wrapper                             │    │
│  │  • Implements __add__, __mul__, etc. (operator overloading)     │    │
│  │  • Properties: .shape, .dtype, .device                          │    │
│  │  • Methods: .numpy(), .item(), .shard()                         │    │
│  │  • Holds reference to TensorImpl                                │    │
│  └──────────────────────────┬──────────────────────────────────────┘    │
│                             │                                           │
│                             │ wraps                                     │
│                             ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                       TensorImpl (State)                        │    │
│  │  ─────────────────────────────────────────────────────────────  │    │
│  │  _values: list[TensorValue]    # Lazy graph nodes (per shard)   │    │
│  │  _storages: list[driver.Tensor] # Realized data (per shard)     │    │
│  │  values_epoch: int              # For staleness detection       │    │
│  │                                                                 │    │
│  │  sharding: ShardingSpec         # How tensor is distributed     │    │
│  │  batch_dims: int                # Leading batch dims (vmap)     │    │
│  │  traced: bool                   # Record in computation graph?  │    │
│  │                                                                 │    │
│  │  output_refs: OpNode        # What operation created this?  │    │
│  │  output_index: int              # Which output of that op?      │    │
│  │                                                                 │    │
│  │  tangent: TensorImpl            # For JVP (forward-mode AD)     │    │
│  │  cotangent: TensorImpl          # For VJP (backward-mode AD)    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Two Objects?

**Problem**: Multi-output operations (like `split`) produce multiple tensors that share the same parent operation. How do we track this for autodiff?

```python
a, b, c = split(x, 3)  # 3 Tensors from 1 operation
# All three need to reference the same OpNode for backward pass
```

**Solution**: 
- `TensorImpl` holds the shared `output_refs` 
- All sibling outputs point to the SAME `OpNode` object
- Each has different `output_index` (0, 1, 2)

```python
a._impl.output_refs is b._impl.output_refs is c._impl.output_refs  # True
a._impl.output_index  # 0
b._impl.output_index  # 1  
c._impl.output_index  # 2
```

## Lazy vs Realized State

```
                 UNREALIZED                              REALIZED
                 ──────────                              ────────
  _values:       [TensorValue, ...]       →              [TensorValue, ...]
  _storages:     None                     →              [driver.Tensor, ...]
  
  Trigger: .numpy(), .item(), print(), GRAPH.evaluate()
```

**Key insight**: Most tensors stay unrealized until data is explicitly needed. This enables graph optimization before execution.

### Epoch-Based Staleness

```python
y = x + 1                    # y._impl.values_epoch = current_epoch
z = y * 2                    # z._impl.values_epoch = current_epoch
print(z.numpy())             # Evaluates, epoch increments
# Now y._impl.values_epoch < GRAPH.epoch (stale!)
# y._impl._get_valid_values() returns []
```

The `values_epoch` field detects stale values. `_get_valid_values()` returns empty list if epoch doesn't match.

## Shape Properties

TensorImpl provides multiple shape views:

| Property | Description | Example |
|----------|-------------|---------|
| `physical_local_shape(shard_idx)` | Per-shard storage shape (includes batch_dims) | `[B, M/2, N]` |
| `logical_local_shape(shard_idx)` | Per-shard shape (excludes batch_dims) | `[M/2, N]` |
| `physical_global_shape` | Full storage shape (reconstructed from sharding) | `[B, M, N]` |
| `global_shape` | User-facing logical shape | `[M, N]` |

## Navigation

From any tensor, you can navigate the computation graph:

```python
# Get parent operation
tensor._impl.op  # → Operation that created this

# Get input tensors
tensor._impl.parents  # → list[TensorImpl]

# Get kwargs used
tensor._impl.op_kwargs  # → dict (original kwargs!)

# Check if leaf (no parents)
tensor._impl.is_leaf  # → bool
```

## Component Map

| File | Purpose | Key Exports |
|------|---------|-------------|
| [api.py](api.py) | User-facing Tensor class | `Tensor`, factory methods (`zeros`, `ones`, etc.) |
| [impl.py](impl.py) | Internal TensorImpl state | `TensorImpl` |

## Maintenance Guide

> **AI Agents - Critical Rules**:
> 1. **output_refs sharing**: Multi-output ops must share the same OpNode instance
> 2. **values_epoch**: Always check/update when manipulating `_values`
> 3. **Shapes**: Remember `batch_dims` offset when computing shapes
> 4. **Weak refs**: TensorImpl is weakly referenced by OpNode; can be GC'd
