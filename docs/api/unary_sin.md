# sin

## Signature

```python
endia.sin(arg: endia.core.array.Array, dtype: max._core.dtype.DType | None = None) -> endia.core.array.Array
```

## Description

Element-wise sine.

## Examples

```python
import endia as nb

# Basic trigonometric function
x = nd.array([0, np.pi/2, np.pi])
result = nd.sin(x)
print(result)  # [0, 1, 0] (approximately)
```

## See Also

- {doc}`cos <unary_cos>` - Cosine function
- {doc}`exp <unary_exp>` - Exponential function

