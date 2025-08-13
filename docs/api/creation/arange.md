# arange

## Signature

```python
nabla.arange(start: 'int | float', stop: 'int | float | None', step: 'int | float | None', dtype: 'DType', device: 'Device', traced: 'bool', batch_dims: 'Shape') -> 'Array'
```

## Description

Returns evenly spaced values within a given interval.

Values are generated within the half-open interval `[start, stop)`.
In other words, the interval includes `start` but excludes `stop`.
This function follows the JAX/NumPy `arange` API.

## Parameters

- **`start`** (`int | float`): Start of interval. If `stop` is None, `start` is treated as `stop` and the starting value is 0.

- **`stop`** (`int | float, optional`): End of interval. The interval does not include this value. Defaults to None.

- **`step`** (`int | float, optional`): Spacing between values. The default step size is 1.

- **`dtype`** (`DType, optional`): The data type of the output array. Defaults to DType.float32.

- **`device`** (`Device, optional`): The device to place the array on. Defaults to the CPU.

- **`traced`** (`bool, optional`): Whether the operation should be traced in the graph. Defaults to False.

- **`batch_dims`** (`Shape, optional`): Specifies leading dimensions to be treated as batch dimensions. Defaults to an empty tuple.

## Returns

- `Array`: A 1D array of evenly spaced values.

## Examples

```pycon
>>> import nabla as nb
>>> # nb.arange(stop)
>>> nb.arange(5)
Array([0., 1., 2., 3., 4.], dtype=float32)

>>> # nb.arange(start, stop)
>>> nb.arange(5, 10)
Array([5., 6., 7., 8., 9.], dtype=float32)

>>> # nb.arange(start, stop, step)
>>> nb.arange(10, 20, 2, dtype=nb.DType.int32)
Array([10, 12, 14, 16, 18], dtype=int32)
```
