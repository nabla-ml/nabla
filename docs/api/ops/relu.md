# relu

## Signature

```python
nabla.relu(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.unary`

Computes the element-wise Rectified Linear Unit (ReLU) function.

The ReLU function is defined as `max(0, x)`. It is a widely used
activation function in neural networks.

Parameters
----------
arg : Tensor
    The input tensor.

Returns
-------
Tensor
    An tensor containing the result of the ReLU operation.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([-2.0, -0.5, 0.0, 1.0, 2.0])
>>> nb.relu(x)
Tensor([0., 0., 0., 1., 2.], dtype=float32)

