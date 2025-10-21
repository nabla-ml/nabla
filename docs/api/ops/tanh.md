# tanh

## Signature

```python
nabla.tanh(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.unary`

Computes the element-wise hyperbolic tangent of an tensor.

The tanh function is a common activation function in neural networks,
squashing values to the range `[-1, 1]`.

Parameters
----------
arg : Tensor
    The input tensor.

Returns
-------
Tensor
    An tensor containing the hyperbolic tangent of each element.

Examples
--------

.. code-block:: python

    >>> import nabla as nb
    >>> x = nb.tensor([-1.0, 0.0, 1.0, 20.0])
    >>> nb.tanh(x)
    Tensor([-0.7615942,  0.       ,  0.7615942,  1.       ], dtype=float32)

