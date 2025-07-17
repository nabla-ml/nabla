# add

## Signature

```python
nabla.add(arg0, arg1) -> 'Array'
```

## Description

Element-wise addition of two arrays or array and scalar.

## Examples

```python
import nabla as nb

# Element-wise addition
a = nb.array([1, 2, 3])
b = nb.array([4, 5, 6])
result = nb.add(a, b)
print(result)  # [5, 7, 9]
```

## See Also

- {doc}`sub <binary_sub>` - Subtraction
- {doc}`mul <binary_mul>` - Multiplication
- {doc}`div <binary_div>` - Division

