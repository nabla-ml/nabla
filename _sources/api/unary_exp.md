# exp

## Signature

```python
nabla.exp(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

## Description

Element-wise exponential function.

## Examples

```python
import nabla as nb

# Exponential function
x = nb.array([0, 1, 2])
result = nb.exp(x)
print(result)  # [1, e, e^2] (approximately)
```

## See Also

- {doc}`log <unary_log>` - Natural logarithm
- {doc}`sin <unary_sin>`, {doc}`cos <unary_cos>` - Trigonometric functions

