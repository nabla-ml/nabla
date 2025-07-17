# sum

## Signature

```python
nabla.sum(arg: 'Array', axes: 'int | list[int] | tuple[int, ...] | None' = None, keep_dims: 'bool' = False) -> 'Array'
```

## Description

sum array elements over given axes.

## Examples

```python
import nabla as nb

# Sum along axes
x = nb.array([[1, 2], [3, 4]])
result = nb.sum(x)  # Sum all elements
print(result)  # 10

result_axis0 = nb.sum(x, axis=0)  # Sum along rows
print(result_axis0)  # [4, 6]
```

