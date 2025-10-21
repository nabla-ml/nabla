# log

## Signature

```python
nabla.log(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.unary`

Computes the element-wise natural logarithm (base e).

This function calculates `log(x)` for each element `x` in the input tensor.
For numerical stability with non-positive inputs, a small epsilon is
added to ensure the input to the logarithm is positive.

Parameters
----------
arg : Tensor
    The input tensor. Values should be positive.

Returns
-------
Tensor
    An tensor containing the natural logarithm of each element.

Examples
--------

.. code-block:: python

    >>> import nabla as nb
    >>> x = nb.tensor([1.0, 2.71828, 10.0])
    >>> nb.log(x)
    Tensor([0.       , 0.9999993, 2.3025851], dtype=float32)

