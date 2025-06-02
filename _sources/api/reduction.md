# Reduction Operations

Operations that reduce arrays along specified dimensions.

```{toctree}
:maxdepth: 1
:caption: Functions

reduction_sum
reduction_sum_batch_dims
```

## Quick Reference

### {doc}`sum <reduction_sum>`

```python
nabla.sum(arg: 'Array', axes: 'int | list[int] | tuple[int, ...] | None' = None, keep_dims: 'bool' = False) -> 'Array'
```

Sum of array elements over given axes.

### {doc}`sum_batch_dims <reduction_sum_batch_dims>`

```python
nabla.sum_batch_dims(arg: 'Array', axes: 'int | list[int] | tuple[int, ...] | None' = None, keep_dims: 'bool' = False) -> 'Array'
```

Nabla operation: `sum_batch_dims`

