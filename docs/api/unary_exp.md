# exp

## Signature

```python
nabla.exp(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

## Description

Computes the element-wise exponential function (e^x).

This function calculates the base-e exponential of each element in the
input array.

Parameters
----------
arg : Array
The input array.

Returns
-------
Array
An array containing the exponential of each element.

Examples
--------
>>> import nabla as nb
>>> x = nb.array([0.0, 1.0, 2.0])
>>> nb.exp(x)
Array([1.       , 2.7182817, 7.389056 ], dtype=float32)

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

