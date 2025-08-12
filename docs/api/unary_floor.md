# floor

## Signature

```python
nabla.floor(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

## Description

Computes the element-wise floor of an array.

The floor of a scalar `x` is the largest integer `i` such that `i <= x`.
This function is not differentiable and its gradient is zero everywhere.

Parameters
----------
arg : Array
The input array.

Returns
-------
Array
An array containing the floor of each element.

Examples
--------
>>> import nabla as nb
>>> x = nb.array([-1.7, -0.2, 0.2, 1.7])
>>> nb.floor(x)
Array([-2., -1.,  0.,  1.], dtype=float32)

