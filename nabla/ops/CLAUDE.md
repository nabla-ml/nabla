# Ops Module: The Operation ABC Pattern

## Philosophy

Operations are **singleton objects** inheriting from `Operation`. This enables:
- Fast identity checks for autodiff graph walking
- Memory efficiency (stateless, one instance per type)
- Global rule registration (VJP/JVP/sharding)

---

## The Operation Hierarchy

### `Operation` (Base)

Defines: `name`, `maxpr()`, `communication_cost()`, optional `vjp_rule()`, `jvp_rule()`, `sharding_rule()`

**The `__call__` method orchestrates everything**:
1. Extract tensors from pytree inputs
2. Check for sharded inputs → route to SPMD path
3. Convert to TensorValue (MAX graph nodes)
4. Call `maxpr()` to build graph
5. Create TensorImpl for outputs
6. Propagate metadata (batch_dims, traced, sharding)
7. Create OutputRefs if tracing

### Specialized ABCs

| ABC | Purpose |
|-----|---------|
| `BinaryOperation` | Two inputs with batch-aware broadcasting |
| `UnaryOperation` | Single input, no broadcasting |
| `ReduceOperation` | Logical→physical axis translation |
| `LogicalShapeOperation` | Shape ops respecting batch_dims |

---

## Physical vs Logical Operations

**Logical** (user-facing): Work with logical shapes, translate internally

**Physical** (in `_physical.py`): Manipulate batch_dims counter, internal plumbing for vmap

When tensor has `batch_dims=2`:
- Physical: `(B1, B2, H, W)`
- Logical: `(H, W)` (what user sees)

User `reduce_sum(x, axis=0)` → physical_axis = batch_dims + 0 = 2

---

## Batch Dims Propagation

| Operation Type | Rule |
|----------------|------|
| Binary | `max(x.batch_dims, y.batch_dims)` |
| Unary | Preserve input's batch_dims |
| Reduction | Preserve (reduce over logical dims) |
| View | Preserve (reshape logical portion) |

---

## Sharding Integration

### SPMD Path

When `_has_sharded_inputs(args)` is True:
1. `_get_mesh_from_args()` extracts DeviceMesh
2. `_infer_output_sharding()` computes output spec
3. `_call_spmd()` runs per-shard execution:
   - Slice inputs per their sharding
   - Call `maxpr()` on shard data
   - Collect results into multi-value output

### Operation Sharding Rules

Override `sharding_rule()` for custom propagation:

```python
def sharding_rule(self, input_shapes, output_shapes, **kwargs):
    from nabla.sharding.propagation import matmul_template
    return matmul_template(batch_dims=0).instantiate(
        input_shapes, output_shapes
    )
```

Default: `elementwise_template` (all dims share factors)

---

## Multi-Output Operations

Multiple outputs share one `OutputRefs`:
- Single operation, single VJP call
- Each output has unique `output_idx`
- Pytree integration for arbitrary return structures

---

## Extension Points

### Adding New Operation

1. Subclass appropriate ABC
2. Implement `name` property and `maxpr()`
3. Optional: `vjp_rule()`, `jvp_rule()`, `sharding_rule()`
4. Create singleton and public function

Base class handles all metadata propagation automatically.

---

## File Organization

| File | Contents |
|------|----------|
| `operation.py` | Base classes, BinaryOperation |
| `binary.py` | Add, sub, mul, matmul |
| `unary.py` | Activations, math ops |
| `creation.py` | zeros, ones, arange |
| `reduction.py` | sum, mean (with axis) |
| `view.py` | reshape, transpose |
| `multi_output.py` | split, unbind |
| `_physical.py` | vmap internals (not user-facing) |
| `communication.py` | shard, collectives, **cost model logic** |
