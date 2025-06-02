# mul

## Signature

```python
nabla.mul(arg0, arg1) -> 'Array'
```

## Description

Element-wise multiplication of two arrays or array and scalar.

## Examples

```python
import nabla as nb

# Element-wise multiplication
a = nb.array([1, 2, 3])
b = nb.array([4, 5, 6])
result = nb.mul(a, b)
print(result)  # [4, 10, 18]
```

## See Also

- {doc}`add <binary_add>` - Addition
- {doc}`div <binary_div>` - Division
- {doc}`pow <binary_pow>` - Exponentiation

