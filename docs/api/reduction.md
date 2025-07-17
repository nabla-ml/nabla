# Reduction Operations

Operations that reduce arrays along specified dimensions.

```{toctree}
:maxdepth: 1
:caption: Functions

reduction_argmax
reduction_max
reduction_mean
reduction_sum
reduction_sum_batch_dims
```

## Quick Reference

### {doc}`argmax <reduction_argmax>`

```python
nabla.argmax(arg: 'Array', axes: 'int | None' = None, keep_dims: 'bool' = False) -> 'Array'
```

Nabla operation: `argmax`

### {doc}`max <reduction_max>`

```python
nabla.max(arg: 'Array', axes: 'int | list[int] | tuple[int, ...] | None' = None, keep_dims: 'bool' = False) -> 'Array'
```

Maximum of array elements over given axes.

### {doc}`mean <reduction_mean>`

```python
nabla.mean(arg: 'Array', axes: 'int | list[int] | tuple[int, ...] | None' = None, keep_dims: 'bool' = False) -> 'Array'
```

Arithmetic mean of array elements over given axes.

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

