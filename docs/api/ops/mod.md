# mod

## Signature

```python
nabla.mod(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor'
```

**Source**: `nabla.ops.binary`

Computes the element-wise remainder of division.

This function calculates the remainder of `x / y` element-wise. The
sign of the result follows the sign of the divisor `y`. It provides the
implementation of the `%` operator for Nabla tensors.

Parameters
----------
x : Tensor | float | int
    The dividend tensor or scalar.
y : Tensor | float | int
    The divisor tensor or scalar. Must be broadcastable to the same shape
    as `x`.

Returns
-------
Tensor
    An tensor containing the element-wise remainder.

Examples
--------

.. code-block:: python

    >>> import nabla as nb
    >>> x = nb.tensor([10, -10, 9])
    >>> y = nb.tensor([3, 3, -3])
    >>> nb.mod(x, y)
    Tensor([ 1,  2, -0], dtype=int32)


.. code-block:: python

    >>> x % y
    Tensor([ 1,  2, -0], dtype=int32)

