# transpose

## Signature

```python
endia.transpose(arg: endia.core.array.Array, axis_1: int = -2, axis_2: int = -1) -> endia.core.array.Array
```

## Description

Transpose array along two axes.

## Examples

```python
import endia as nb

# Transpose array
x = nb.array([[1, 2, 3], [4, 5, 6]])
result = nb.transpose(x)
print(result)  # [[1, 4], [2, 5], [3, 6]]
```

