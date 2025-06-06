# reshape

## Signature

```python
endia.reshape(arg: endia.core.array.Array, shape: tuple[int, ...]) -> endia.core.array.Array
```

## Description

Reshape array to given shape.

## Examples

```python
import endia as nb

# Change array shape
x = nb.array([1, 2, 3, 4, 5, 6])
result = nb.reshape(x, (2, 3))
print(result)  # [[1, 2, 3], [4, 5, 6]]
```

