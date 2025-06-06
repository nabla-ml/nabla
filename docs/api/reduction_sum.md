# sum

## Signature

```python
endia.sum(arg: 'Array', axes: 'int | list[int] | tuple[int, ...] | None' = None, keep_dims: 'bool' = False) -> 'Array'
```

## Description

sum array elements over given axes.

## Examples

```python
import endia as nb

# Sum along axes
x = nd.array([[1, 2], [3, 4]])
result = nd.sum(x)  # Sum all elements
print(result)  # 10

result_axis0 = nd.sum(x, axis=0)  # Sum along rows
print(result_axis0)  # [4, 6]
```

