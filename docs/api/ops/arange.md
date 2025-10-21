# arange

## Signature

```python
nabla.arange(start: 'int | float', stop: 'int | float | None' = None, step: 'int | float | None' = None, dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), traced: 'bool' = False, batch_dims: 'Shape' = ()) -> 'Tensor'
```

**Source**: `nabla.ops.creation`

Returns evenly spaced values within a given interval.

Values are generated within the half-open interval `[start, stop)`.
In other words, the interval includes `start` but excludes `stop`.
This function follows the JAX/NumPy `arange` API.

Parameters
----------
start : int | float
    Start of interval. If `stop` is None, `start` is treated as `stop`
    and the starting value is 0.
stop : int | float, optional
    End of interval. The interval does not include this value.
    Defaults to None.
step : int | float, optional
    Spacing between values. The default step size is 1.
dtype : DType, optional
    The data type of the output tensor. Defaults to DType.float32.
device : Device, optional
    The device to place the tensor on. Defaults to the CPU.
traced : bool, optional
    Whether the operation should be traced in the graph. Defaults to False.
batch_dims : Shape, optional
    Specifies leading dimensions to be treated as batch dimensions.
    Defaults to an empty tuple.

Returns
-------
Tensor
    A 1D tensor of evenly spaced values.

Examples
--------

.. code-block:: python

    >>> import nabla as nb
    >>> # nb.arange(stop)
    >>> nb.arange(5)
    Tensor([0., 1., 2., 3., 4.], dtype=float32)
    <BLANKLINE>
    >>> # nb.arange(start, stop)
    >>> nb.arange(5, 10)
    Tensor([5., 6., 7., 8., 9.], dtype=float32)
    <BLANKLINE>
    >>> # nb.arange(start, stop, step)
    >>> nb.arange(10, 20, 2, dtype=nb.DType.int32)
    Tensor([10, 12, 14, 16, 18], dtype=int32)

