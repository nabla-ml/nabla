# Vectorization

## `vmap`

```python
def vmap(func: 'Callable[..., T] | None' = None, in_axes: 'AxisSpec' = 0, out_axes: 'AxisSpec' = 0, axis_size: 'int | None' = None, spmd_axis_name: 'str | None' = None, mesh: "'DeviceMesh | None'" = None) -> 'Callable[..., T]':
```
Vectorize *func* over batch dimensions (JAX-compatible API).


---
