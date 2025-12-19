# Creation Ops

## `zeros`

```python
def zeros(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Creates an tensor of a given shape filled with zeros.

Parameters
----------
shape : Shape
    The shape of the new tensor, e.g., `(2, 3)` or `(5,)`.
dtype : DType, optional
    The desired data type for the tensor. Defaults to DType.float32.
device : Device, optional
    The device to place the tensor on. Defaults to the CPU.
batch_dims : Shape, optional
    Specifies leading dimensions to be treated as batch dimensions.
    Defaults to an empty tuple.
traced : bool, optional
    Whether the operation should be traced in the graph. Defaults to False.

Returns
-------
Tensor
    An tensor of the specified shape and dtype, filled with zeros.

Examples
--------
>>> import nabla as nb
>>> # Create a 2x3 matrix of zeros
>>> nb.zeros((2, 3), dtype=nb.DType.int32)
Tensor([[0, 0, 0],
       [0, 0, 0]], dtype=int32)

---
## `ones`

```python
def ones(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Creates an tensor of a given shape filled with ones.

Parameters
----------
shape : Shape
    The shape of the new tensor, e.g., `(2, 3)` or `(5,)`.
dtype : DType, optional
    The desired data type for the tensor. Defaults to DType.float32.
device : Device, optional
    The device to place the tensor on. Defaults to the CPU.
batch_dims : Shape, optional
    Specifies leading dimensions to be treated as batch dimensions.
    Defaults to an empty tuple.
traced : bool, optional
    Whether the operation should be traced in the graph. Defaults to False.

Returns
-------
Tensor
    An tensor of the specified shape and dtype, filled with ones.

Examples
--------
>>> import nabla as nb
>>> # Create a vector of ones
>>> nb.ones((4,), dtype=nb.DType.float32)
Tensor([1., 1., 1., 1.], dtype=float32)

---
## `rand`

```python
def rand(shape: 'Shape', dtype: 'DType' = float32, lower: 'float' = 0.0, upper: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Creates an tensor with uniformly distributed random values.

The values are drawn from a continuous uniform distribution over the
interval `[lower, upper)`.

Parameters
----------
shape : Shape
    The shape of the output tensor.
dtype : DType, optional
    The desired data type for the tensor. Defaults to DType.float32.
lower : float, optional
    The lower boundary of the output interval. Defaults to 0.0.
upper : float, optional
    The upper boundary of the output interval. Defaults to 1.0.
device : Device, optional
    The device to place the tensor on. Defaults to the CPU.
seed : int, optional
    The seed for the random number generator. Defaults to 0.
batch_dims : Shape, optional
    Specifies leading dimensions to be treated as batch dimensions.
    Defaults to an empty tuple.
traced : bool, optional
    Whether the operation should be traced in the graph. Defaults to False.

Returns
-------
Tensor
    An tensor of the specified shape filled with random values.

---
## `randn`

```python
def randn(shape: 'Shape', dtype: 'DType' = float32, mean: 'float' = 0.0, std: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Creates an tensor with normally distributed random values.

The values are drawn from a normal (Gaussian) distribution with the
specified mean and standard deviation.

Parameters
----------
shape : Shape
    The shape of the output tensor.
dtype : DType, optional
    The desired data type for the tensor. Defaults to DType.float32.
mean : float, optional
    The mean of the normal distribution. Defaults to 0.0.
std : float, optional
    The standard deviation of the normal distribution. Defaults to 1.0.
device : Device, optional
    The device to place the tensor on. Defaults to the CPU.
seed : int, optional
    The seed for the random number generator for reproducibility.
    Defaults to 0.
batch_dims : Shape, optional
    Specifies leading dimensions to be treated as batch dimensions.
    Defaults to an empty tuple.
traced : bool, optional
    Whether the operation should be traced in the graph. Defaults to False.

Returns
-------
Tensor
    An tensor of the specified shape filled with random values.

---
## `arange`

```python
def arange(start: 'int | float', stop: 'int | float | None' = None, step: 'int | float | None' = None, dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), traced: 'bool' = False, batch_dims: 'Shape' = ()) -> 'Tensor':
```
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

---
## `full_like`

```python
def full_like(template: 'Tensor', fill_value: 'float') -> 'Tensor':
```
Creates a filled tensor with the same properties as a template tensor.

The new tensor will have the same shape, dtype, device, and batch
dimensions as the template tensor, filled with `fill_value`.

Parameters
----------
template : Tensor
    The template tensor to match properties from.
fill_value : float
    The value to fill the new tensor with.

Returns
-------
Tensor
    A new tensor filled with `fill_value` and with the same properties
    as the template.

Examples
--------
>>> import nabla as nb
>>> x = nb.zeros((2, 2))
>>> nb.full_like(x, 7.0)
Tensor([[7., 7.],
       [7., 7.]], dtype=float32)

---
## `ones_like`

```python
def ones_like(template: 'Tensor') -> 'Tensor':
```
Creates an tensor of ones with the same properties as a template tensor.

The new tensor will have the same shape, dtype, device, and batch
dimensions as the template tensor.

Parameters
----------
template : Tensor
    The template tensor to match properties from.

Returns
-------
Tensor
    A new tensor of ones with the same properties as the template.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([[1., 2.], [3., 4.]])
>>> nb.ones_like(x)
Tensor([[1., 1.],
       [1., 1.]], dtype=float32)

---
## `zeros_like`

```python
def zeros_like(template: 'Tensor') -> 'Tensor':
```
Creates an tensor of zeros with the same properties as a template tensor.

The new tensor will have the same shape, dtype, device, and batch
dimensions as the template tensor.

Parameters
----------
template : Tensor
    The template tensor to match properties from.

Returns
-------
Tensor
    A new tensor of zeros with the same properties as the template.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([[1, 2], [3, 4]], dtype=nb.DType.int32)
>>> nb.zeros_like(x)
Tensor([[0, 0],
       [0, 0]], dtype=int32)

---
