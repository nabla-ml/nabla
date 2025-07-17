# reshape

## Signature

```python
nabla.reshape(arg: nabla.core.array.Array, shape: tuple[int, ...]) -> nabla.core.array.Array
```

## Description

Reshape array to given shape.

## Examples

```python
import nabla as nb

# Change array shape
x = nb.array([1, 2, 3, 4, 5, 6])
result = nb.reshape(x, (2, 3))
print(result)  # [[1, 2, 3], [4, 5, 6]]
```

