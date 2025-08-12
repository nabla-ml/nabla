# relu

## Signature

```python
nabla.relu(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

## Description

Computes the element-wise Rectified Linear Unit (ReLU) function.

The ReLU function is defined as `max(0, x)`. It is a widely used
activation function in neural networks.

Parameters
----------
arg : Array
The input array.

Returns
-------
Array
An array containing the result of the ReLU operation.

Examples
--------
>>> import nabla as nb
>>> x = nb.array([-2.0, -0.5, 0.0, 1.0, 2.0])
>>> nb.relu(x)
Array([0., 0., 0., 1., 2.], dtype=float32)

