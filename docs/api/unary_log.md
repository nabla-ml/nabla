# log

## Signature

```python
nabla.log(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

## Description

Computes the element-wise natural logarithm (base e).

This function calculates `log(x)` for each element `x` in the input array.
For numerical stability with non-positive inputs, a small epsilon is
added to ensure the input to the logarithm is positive.

Parameters
----------
arg : Array
The input array. Values should be positive.

Returns
-------
Array
An array containing the natural logarithm of each element.

Examples
--------
>>> import nabla as nb
>>> x = nb.array([1.0, 2.71828, 10.0])
>>> nb.log(x)
Array([0.       , 0.9999993, 2.3025851], dtype=float32)

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

