# cos

## Signature

```python
endia.cos(arg: endia.core.array.Array) -> endia.core.array.Array
```

## Description

Element-wise cosine.

## Examples

```python
import endia as nd

# Basic trigonometric function
x = nd.array([0, np.pi/2, np.pi])
result = nd.cos(x)
print(result)  # [1, 0, -1] (approximately)
```

## See Also

- {doc}`sin <unary_sin>` - Sine function
- {doc}`exp <unary_exp>` - Exponential function

