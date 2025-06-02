# cos

## Signature

```python
nabla.cos(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

## Description

Element-wise cosine.

## Examples

```python
import nabla as nb

# Basic trigonometric function
x = nb.array([0, np.pi/2, np.pi])
result = nb.cos(x)
print(result)  # [1, 0, -1] (approximately)
```

## See Also

- {doc}`sin <unary_sin>` - Sine function
- {doc}`exp <unary_exp>` - Exponential function

