# equal

## Signature

```python
nabla.equal(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor'
```

**Source**: `nabla.ops.binary`

Performs element-wise comparison `x == y`.

This function compares two tensors element-wise, returning a boolean tensor
indicating where elements of `x` are equal to elements of `y`. It
supports broadcasting and provides the implementation of the `==` operator
for Nabla tensors.

Parameters
----------
x : Tensor | float | int
    The first input tensor or scalar.
y : Tensor | float | int
    The second input tensor or scalar. Must be broadcastable to the same
    shape as `x`.

Returns
-------
Tensor
    A boolean tensor containing the result of the element-wise equality
    comparison.

Examples
--------

.. code-block:: python

    >>> import nabla as nb
    >>> x = nb.tensor([1, 2, 3])
    >>> y = nb.tensor([1, 5, 3])
    >>> nb.equal(x, y)
    Tensor([ True, False,  True], dtype=bool)


.. code-block:: python

    >>> x == y
    Tensor([ True, False,  True], dtype=bool)

