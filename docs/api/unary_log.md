# log

## Signature

```python
nabla.log(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

## Description

Element-wise natural logarithm.

## Examples

```python
import nabla as nb

# Natural logarithm
x = nb.array([1, np.e, np.e**2])
result = nb.log(x)
print(result)  # [0, 1, 2] (approximately)
```

## See Also

- {doc}`exp <unary_exp>` - Exponential function

