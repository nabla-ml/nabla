# Vectorization

## `vmap`

```python
def vmap(func: 'Callable[..., T] | None' = None, in_axes: 'AxisSpec' = 0, out_axes: 'AxisSpec' = 0, axis_size: 'int | None' = None, spmd_axis_name: 'str | None' = None, mesh: 'DeviceMesh | None' = None) -> 'Callable[..., T]':
```
Vectorize *func* over a batch dimension (JAX-compatible API).

Transforms *func* so it runs in parallel over an extra leading batch
dimension, without changing the logic of *func* itself. Supports
nested ``vmap`` calls and composes with other transforms.

**Parameters**

- **`func`** – Function to vectorize. Can be used as a decorator
(called with no positional arguments).
- **`in_axes`** – Which axes of the inputs to batch over. An integer
applies the same axis to all inputs; a list/tuple can
specify per-input axes; ``None`` means an input is not
batched (broadcast). Default: ``0``.
- **`out_axes`** – Where to place the output batch dimensions. Default: ``0``.
- **`axis_size`** – Explicit batch size, required only when all inputs
are not batched (``in_axes`` is all-``None``).
- **`spmd_axis_name`** – When set, shard the batch axis across the mesh
dimension with this name (SPMD mode).
- **`mesh`** – Device mesh for SPMD batching.

**Returns**

A wrapped function with the same signature as *func* that
accepts an extra batch dimension on each (batched) input.


---
