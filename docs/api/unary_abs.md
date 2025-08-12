# abs

## Signature

```python
nabla.abs(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

## Description

Computes the element-wise absolute value of an array.

Parameters
----------
arg : Array
The input array.

Returns
-------
Array
An array containing the absolute value of each element.

Examples
--------
>>> import nabla as nb
>>> x = nb.array([-1.5, 0.0, 2.5])
>>> nb.abs(x)
Array([1.5, 0. , 2.5], dtype=float32)

