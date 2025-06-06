# log

## Signature

```python
endia.log(arg: endia.core.array.Array) -> endia.core.array.Array
```

## Description

Element-wise natural logarithm.

## Examples

```python
import endia as nb

# Natural logarithm
x = nb.array([1, np.e, np.e**2])
result = nb.log(x)
print(result)  # [0, 1, 2] (approximately)
```

## See Also

- {doc}`exp <unary_exp>` - Exponential function

