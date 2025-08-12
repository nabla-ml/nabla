# cos

## Signature

```python
nabla.cos(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

## Description

Computes the element-wise cosine of an array.

Parameters
----------
arg : Array
The input array. Input is expected to be in radians.

Returns
-------
Array
An array containing the cosine of each element in the input.

Examples
--------
>>> import nabla as nb
>>> x = nb.array([0, 1.5707963, 3.1415926])
>>> nb.cos(x)
Array([ 1.000000e+00, -4.371139e-08, -1.000000e+00], dtype=float32)

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

