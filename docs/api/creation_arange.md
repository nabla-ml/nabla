# arange

## Signature

```python
nabla.arange(start: 'int | float', stop: 'int | float | None' = None, step: 'int | float | None' = None, dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), traced: 'bool' = False, batch_dims: 'Shape' = ()) -> 'Array'
```

## Description

Return evenly spaced values within a given interval.

This function follows the JAX/NumPy `arange` API.


## Parameters

start: Start of interval. The interval includes this value.
stop: End of interval. The interval does not include this value. If None,
the range is `[0, start)`.
step: Spacing between values. The default step size is 1.
dtype: The data type of the output array.
device: The device to place the array on.
traced: Whether the operation should be traced in the graph.


## Returns

A 1D array of evenly spaced values.

