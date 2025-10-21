# exp

## Signature

```python
nabla.exp(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.unary`

Computes the element-wise exponential function (e^x).

This function calculates the base-e exponential of each element in the
input tensor.

Parameters
----------
arg : Tensor
    The input tensor.

Returns
-------
Tensor
    An tensor containing the exponential of each element.

Examples
--------

.. code-block:: python

    >>> import nabla as nb
    >>> x = nb.tensor([0.0, 1.0, 2.0])
    >>> nb.exp(x)
    Tensor([1.       , 2.7182817, 7.389056 ], dtype=float32)

