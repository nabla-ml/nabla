# Creation Operations

## `tensor`

```python
def tensor(data: 'list | np.ndarray | float | int', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Creates an tensor from a Python list, NumPy tensor, or scalar.

This function is the primary way to create a Nabla tensor from existing
data. It converts the input data into a Nabla tensor on the specified
device and with the given data type.

**Parameters**

- **`data`** : `list | np.ndarray | float | int` – The input data to convert to an tensor.
- **`dtype`** : `DType`, optional, default: `DType.float32` – The desired data type for the tensor. Defaults to DType.float32.
- **`device`** : `Device`, optional, default: `s` – The computational device where the tensor will be stored. Defaults
to the CPU.
- **`batch_dims`** : `Shape`, optional, default: `an` – Specifies leading dimensions to be treated as batch dimensions.
Defaults to an empty tuple.
- **`traced`** : `bool`, optional, default: `False` – Whether the operation should be traced in the graph. Defaults to False.

**Returns**

`Tensor` – A new Nabla tensor containing the provided data.

**Examples**

```python
>>> import nabla as nb
>>> import numpy as np
>>> # Create from a Python list
>>> nb.tensor([1, 2, 3])
Tensor([1, 2, 3], dtype=int32)
<BLANKLINE>
>>> # Create from a NumPy tensor
>>> np_arr = np.array([[4.0, 5.0], [6.0, 7.0]])
>>> nb.tensor(np_arr)
Tensor([[4., 5.],
       [6., 7.]], dtype=float32)
<BLANKLINE>
>>> # Create a scalar tensor
>>> nb.tensor(100, dtype=nb.DType.int64)
Tensor(100, dtype=int64)
```

---
## `zeros`

```python
def zeros(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Creates an tensor of a given shape filled with zeros.

**Parameters**

- **`shape`** : `Shape` – The shape of the new tensor, e.g., `(2, 3)` or `(5,)`.
- **`dtype`** : `DType`, optional, default: `DType.float32` – The desired data type for the tensor. Defaults to DType.float32.
- **`device`** : `Device`, optional, default: `the` – The device to place the tensor on. Defaults to the CPU.
- **`batch_dims`** : `Shape`, optional, default: `an` – Specifies leading dimensions to be treated as batch dimensions.
Defaults to an empty tuple.
- **`traced`** : `bool`, optional, default: `False` – Whether the operation should be traced in the graph. Defaults to False.

**Returns**

`Tensor` – An tensor of the specified shape and dtype, filled with zeros.

**Examples**

```python
>>> import nabla as nb
>>> # Create a 2x3 matrix of zeros
>>> nb.zeros((2, 3), dtype=nb.DType.int32)
Tensor([[0, 0, 0],
       [0, 0, 0]], dtype=int32)
```

---
## `ones`

```python
def ones(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Creates an tensor of a given shape filled with ones.

**Parameters**

- **`shape`** : `Shape` – The shape of the new tensor, e.g., `(2, 3)` or `(5,)`.
- **`dtype`** : `DType`, optional, default: `DType.float32` – The desired data type for the tensor. Defaults to DType.float32.
- **`device`** : `Device`, optional, default: `the` – The device to place the tensor on. Defaults to the CPU.
- **`batch_dims`** : `Shape`, optional, default: `an` – Specifies leading dimensions to be treated as batch dimensions.
Defaults to an empty tuple.
- **`traced`** : `bool`, optional, default: `False` – Whether the operation should be traced in the graph. Defaults to False.

**Returns**

`Tensor` – An tensor of the specified shape and dtype, filled with ones.

**Examples**

```python
>>> import nabla as nb
>>> # Create a vector of ones
>>> nb.ones((4,), dtype=nb.DType.float32)
Tensor([1., 1., 1., 1.], dtype=float32)
```

---
## `rand`

```python
def rand(shape: 'Shape', dtype: 'DType' = float32, lower: 'float' = 0.0, upper: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Creates an tensor with uniformly distributed random values.

The values are drawn from a continuous uniform distribution over the
interval `[lower, upper)`.

**Parameters**

- **`shape`** : `Shape` – The shape of the output tensor.
- **`dtype`** : `DType`, optional, default: `DType.float32` – The desired data type for the tensor. Defaults to DType.float32.
- **`lower`** : `float`, optional, default: `0.0` – The lower boundary of the output interval. Defaults to 0.0.
- **`upper`** : `float`, optional, default: `1.0` – The upper boundary of the output interval. Defaults to 1.0.
- **`device`** : `Device`, optional, default: `the` – The device to place the tensor on. Defaults to the CPU.
- **`seed`** : `int`, optional, default: `0` – The seed for the random number generator. Defaults to 0.
- **`batch_dims`** : `Shape`, optional, default: `an` – Specifies leading dimensions to be treated as batch dimensions.
Defaults to an empty tuple.
- **`traced`** : `bool`, optional, default: `False` – Whether the operation should be traced in the graph. Defaults to False.

**Returns**

`Tensor` – An tensor of the specified shape filled with random values.


---
## `randn`

```python
def randn(shape: 'Shape', dtype: 'DType' = float32, mean: 'float' = 0.0, std: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Creates an tensor with normally distributed random values.

The values are drawn from a normal (Gaussian) distribution with the
specified mean and standard deviation.

**Parameters**

- **`shape`** : `Shape` – The shape of the output tensor.
- **`dtype`** : `DType`, optional, default: `DType.float32` – The desired data type for the tensor. Defaults to DType.float32.
- **`mean`** : `float`, optional, default: `0.0` – The mean of the normal distribution. Defaults to 0.0.
- **`std`** : `float`, optional, default: `1.0` – The standard deviation of the normal distribution. Defaults to 1.0.
- **`device`** : `Device`, optional, default: `the` – The device to place the tensor on. Defaults to the CPU.
- **`seed`** : `int`, optional, default: `0` – The seed for the random number generator for reproducibility.
Defaults to 0.
- **`batch_dims`** : `Shape`, optional, default: `an` – Specifies leading dimensions to be treated as batch dimensions.
Defaults to an empty tuple.
- **`traced`** : `bool`, optional, default: `False` – Whether the operation should be traced in the graph. Defaults to False.

**Returns**

`Tensor` – An tensor of the specified shape filled with random values.


---
## `arange`

```python
def arange(start: 'int | float', stop: 'int | float | None' = None, step: 'int | float | None' = None, dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), traced: 'bool' = False, batch_dims: 'Shape' = ()) -> 'Tensor':
```
Returns evenly spaced values within a given interval.

Values are generated within the half-open interval `[start, stop)`.
In other words, the interval includes `start` but excludes `stop`.
This function follows the JAX/NumPy `arange` API.

**Parameters**

- **`start`** : `int | float` – Start of interval. If `stop` is None, `start` is treated as `stop`
and the starting value is 0.
- **`stop`** : `int | float`, optional, default: `None` – End of interval. The interval does not include this value.
Defaults to None.
- **`step`** : `int | float`, optional, default: `step` – Spacing between values. The default step size is 1.
- **`dtype`** : `DType`, optional, default: `DType.float32` – The data type of the output tensor. Defaults to DType.float32.
- **`device`** : `Device`, optional, default: `the` – The device to place the tensor on. Defaults to the CPU.
- **`traced`** : `bool`, optional, default: `False` – Whether the operation should be traced in the graph. Defaults to False.
- **`batch_dims`** : `Shape`, optional, default: `an` – Specifies leading dimensions to be treated as batch dimensions.
Defaults to an empty tuple.

**Returns**

`Tensor` – A 1D tensor of evenly spaced values.

**Examples**

```python
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
```

---
## `zeros_like`

```python
def zeros_like(template: 'Tensor') -> 'Tensor':
```
Creates an tensor of zeros with the same properties as a template tensor.

The new tensor will have the same shape, dtype, device, and batch
dimensions as the template tensor.

**Parameters**

- **`template`** : `Tensor` – The template tensor to match properties from.

**Returns**

`Tensor` – A new tensor of zeros with the same properties as the template.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([[1, 2], [3, 4]], dtype=nb.DType.int32)
>>> nb.zeros_like(x)
Tensor([[0, 0],
       [0, 0]], dtype=int32)
```

---
## `ones_like`

```python
def ones_like(template: 'Tensor') -> 'Tensor':
```
Creates an tensor of ones with the same properties as a template tensor.

The new tensor will have the same shape, dtype, device, and batch
dimensions as the template tensor.

**Parameters**

- **`template`** : `Tensor` – The template tensor to match properties from.

**Returns**

`Tensor` – A new tensor of ones with the same properties as the template.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([[1., 2.], [3., 4.]])
>>> nb.ones_like(x)
Tensor([[1., 1.],
       [1., 1.]], dtype=float32)
```

---
## `rand_like`

```python
def rand_like(template: 'Tensor', lower: 'float' = 0.0, upper: 'float' = 1.0, seed: 'int' = 0) -> 'Tensor':
```
Creates an tensor with uniformly distributed random values like a template.

The new tensor will have the same shape, dtype, device, and batch
dimensions as the template tensor.

**Parameters**

- **`template`** : `Tensor` – The template tensor to match properties from.
- **`lower`** : `float`, optional, default: `0.0` – The lower boundary of the output interval. Defaults to 0.0.
- **`upper`** : `float`, optional, default: `1.0` – The upper boundary of the output interval. Defaults to 1.0.
- **`seed`** : `int`, optional, default: `0` – The seed for the random number generator. Defaults to 0.

**Returns**

`Tensor` – A new tensor with the same properties as the template, filled with
uniformly distributed random values.


---
## `randn_like`

```python
def randn_like(template: 'Tensor', mean: 'float' = 0.0, std: 'float' = 1.0, seed: 'int' = 0) -> 'Tensor':
```
Creates an tensor with normally distributed random values like a template.

The new tensor will have the same shape, dtype, device, and batch
dimensions as the template tensor.

**Parameters**

- **`template`** : `Tensor` – The template tensor to match properties from.
- **`mean`** : `float`, optional, default: `0.0` – The mean of the normal distribution. Defaults to 0.0.
- **`std`** : `float`, optional, default: `1.0` – The standard deviation of the normal distribution. Defaults to 1.0.
- **`seed`** : `int`, optional, default: `0` – The seed for the random number generator. Defaults to 0.

**Returns**

`Tensor` – A new tensor with the same properties as the template, filled with
normally distributed random values.


---
## `full_like`

```python
def full_like(template: 'Tensor', fill_value: 'float') -> 'Tensor':
```
Creates a filled tensor with the same properties as a template tensor.

The new tensor will have the same shape, dtype, device, and batch
dimensions as the template tensor, filled with `fill_value`.

**Parameters**

- **`template`** : `Tensor` – The template tensor to match properties from.
- **`fill_value`** : `float` – The value to fill the new tensor with.

**Returns**

`Tensor` – A new tensor filled with `fill_value` and with the same properties
as the template.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.zeros((2, 2))
>>> nb.full_like(x, 7.0)
Tensor([[7., 7.],
       [7., 7.]], dtype=float32)
```

---
## `triu`

```python
def triu(x: 'Tensor', k: 'int' = 0) -> 'Tensor':
```
Returns the upper triangular part of a matrix or batch of matrices.

The elements below the k-th diagonal are zeroed out. The input is
expected to be at least 2-dimensional.

**Parameters**

- **`x`** : `Tensor` – Input tensor with shape (..., M, N).
- **`k`** : `int`, optional, default: `0` – Diagonal offset. `k = 0` is the main diagonal. `k > 0` is above the
main diagonal, and `k < 0` is below the main diagonal. Defaults to 0.

**Returns**

`Tensor` – An tensor with the lower triangular part zeroed out, with the same
shape and dtype as `x`.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.ndarange((3, 3), dtype=nb.DType.int32)
>>> x
Tensor([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]], dtype=int32)
<BLANKLINE>
>>> # Upper triangle with the main diagonal
>>> nb.triu(x, k=0)
Tensor([[0, 1, 2],
       [0, 4, 5],
       [0, 0, 8]], dtype=int32)
<BLANKLINE>
>>> # Upper triangle above the main diagonal
>>> nb.triu(x, k=1)
Tensor([[0, 1, 2],
       [0, 0, 5],
       [0, 0, 0]], dtype=int32)
```

---
## `glorot_uniform`

```python
def glorot_uniform(shape: 'Shape', dtype: 'DType' = float32, gain: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Fills an tensor with values according to the Glorot uniform initializer.

This is an alias for `xavier_uniform`. It samples from a uniform
distribution U(-a, a) where a = sqrt(6 / (fan_in + fan_out)).

**Parameters**

- **`shape`** : `Shape` – The shape of the output tensor. Must be at least 2D.
- **`dtype`** : `DType`, optional, default: `DType.float32` – The desired data type for the tensor. Defaults to DType.float32.
- **`gain`** : `float`, optional, default: `1.0` – An optional scaling factor. Defaults to 1.0.
- **`device`** : `Device`, optional, default: `the` – The device to place the tensor on. Defaults to the CPU.
- **`seed`** : `int`, optional, default: `0` – The seed for the random number generator. Defaults to 0.
- **`batch_dims`** : `Shape`, optional, default: `an` – Specifies leading dimensions to be treated as batch dimensions.
Defaults to an empty tuple.
- **`traced`** : `bool`, optional, default: `False` – Whether the operation should be traced in the graph. Defaults to False.

**Returns**

`Tensor` – An tensor initialized with the Glorot uniform distribution.


---
## `xavier_uniform`

```python
def xavier_uniform(shape: 'Shape', dtype: 'DType' = float32, gain: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Fills an tensor with values according to the Xavier uniform initializer.

Also known as Glorot uniform initialization, this method is designed to
keep the variance of activations the same across every layer in a network.
It samples from a uniform distribution U(-a, a) where
a = gain * sqrt(6 / (fan_in + fan_out)).

**Parameters**

- **`shape`** : `Shape` – The shape of the output tensor. Must be at least 2D.
- **`dtype`** : `DType`, optional, default: `DType.float32` – The desired data type for the tensor. Defaults to DType.float32.
- **`gain`** : `float`, optional, default: `1.0` – An optional scaling factor. Defaults to 1.0.
- **`device`** : `Device`, optional, default: `the` – The device to place the tensor on. Defaults to the CPU.
- **`seed`** : `int`, optional, default: `0` – The seed for the random number generator. Defaults to 0.
- **`batch_dims`** : `Shape`, optional, default: `an` – Specifies leading dimensions to be treated as batch dimensions.
Defaults to an empty tuple.
- **`traced`** : `bool`, optional, default: `False` – Whether the operation should be traced in the graph. Defaults to False.

**Returns**

`Tensor` – An tensor initialized with the Xavier uniform distribution.


---
## `xavier_normal`

```python
def xavier_normal(shape: 'Shape', dtype: 'DType' = float32, gain: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Fills an tensor with values according to the Xavier normal initializer.

Also known as Glorot normal initialization. It samples from a normal
distribution N(0, std^2) where std = gain * sqrt(2 / (fan_in + fan_out)).

**Parameters**

- **`shape`** : `Shape` – The shape of the output tensor. Must be at least 2D.
- **`dtype`** : `DType`, optional, default: `DType.float32` – The desired data type for the tensor. Defaults to DType.float32.
- **`gain`** : `float`, optional, default: `1.0` – An optional scaling factor. Defaults to 1.0.
- **`device`** : `Device`, optional, default: `the` – The device to place the tensor on. Defaults to the CPU.
- **`seed`** : `int`, optional, default: `0` – The seed for the random number generator. Defaults to 0.
- **`batch_dims`** : `Shape`, optional, default: `an` – Specifies leading dimensions to be treated as batch dimensions.
Defaults to an empty tuple.
- **`traced`** : `bool`, optional, default: `False` – Whether the operation should be traced in the graph. Defaults to False.

**Returns**

`Tensor` – An tensor initialized with the Xavier normal distribution.


---
## `he_uniform`

```python
def he_uniform(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Fills an tensor with values according to the He uniform initializer.

This method is designed for layers with ReLU activations. It samples from
a uniform distribution U(-a, a) where a = sqrt(6 / fan_in).

**Parameters**

- **`shape`** : `Shape` – The shape of the output tensor. Must be at least 2D.
- **`dtype`** : `DType`, optional, default: `DType.float32` – The desired data type for the tensor. Defaults to DType.float32.
- **`device`** : `Device`, optional, default: `the` – The device to place the tensor on. Defaults to the CPU.
- **`seed`** : `int`, optional, default: `0` – The seed for the random number generator. Defaults to 0.
- **`batch_dims`** : `Shape`, optional, default: `an` – Specifies leading dimensions to be treated as batch dimensions.
Defaults to an empty tuple.
- **`traced`** : `bool`, optional, default: `False` – Whether the operation should be traced in the graph. Defaults to False.

**Returns**

`Tensor` – An tensor initialized with the He uniform distribution.


---
## `he_normal`

```python
def he_normal(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Fills an tensor with values according to the He normal initializer.

This method is designed for layers with ReLU activations. It samples from
a normal distribution N(0, std^2) where std = sqrt(2 / fan_in).

**Parameters**

- **`shape`** : `Shape` – The shape of the output tensor. Must be at least 2D.
- **`dtype`** : `DType`, optional, default: `DType.float32` – The desired data type for the tensor. Defaults to DType.float32.
- **`device`** : `Device`, optional, default: `the` – The device to place the tensor on. Defaults to the CPU.
- **`seed`** : `int`, optional, default: `0` – The seed for the random number generator. Defaults to 0.
- **`batch_dims`** : `Shape`, optional, default: `an` – Specifies leading dimensions to be treated as batch dimensions.
Defaults to an empty tuple.
- **`traced`** : `bool`, optional, default: `False` – Whether the operation should be traced in the graph. Defaults to False.

**Returns**

`Tensor` – An tensor initialized with the He normal distribution.


---
## `lecun_uniform`

```python
def lecun_uniform(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Fills an tensor with values according to the LeCun uniform initializer.

This method is often used for layers with SELU activations. It samples from
a uniform distribution U(-a, a) where a = sqrt(3 / fan_in).

**Parameters**

- **`shape`** : `Shape` – The shape of the output tensor. Must be at least 2D.
- **`dtype`** : `DType`, optional, default: `DType.float32` – The desired data type for the tensor. Defaults to DType.float32.
- **`device`** : `Device`, optional, default: `the` – The device to place the tensor on. Defaults to the CPU.
- **`seed`** : `int`, optional, default: `0` – The seed for the random number generator. Defaults to 0.
- **`batch_dims`** : `Shape`, optional, default: `an` – Specifies leading dimensions to be treated as batch dimensions.
Defaults to an empty tuple.
- **`traced`** : `bool`, optional, default: `False` – Whether the operation should be traced in the graph. Defaults to False.

**Returns**

`Tensor` – An tensor initialized with the LeCun uniform distribution.


---
## `lecun_normal`

```python
def lecun_normal(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Fills an tensor with values according to the LeCun normal initializer.

This method is often used for layers with SELU activations. It samples from
a normal distribution N(0, std^2) where std = sqrt(1 / fan_in).

**Parameters**

- **`shape`** : `Shape` – The shape of the output tensor. Must be at least 2D.
- **`dtype`** : `DType`, optional, default: `DType.float32` – The desired data type for the tensor. Defaults to DType.float32.
- **`device`** : `Device`, optional, default: `the` – The device to place the tensor on. Defaults to the CPU.
- **`seed`** : `int`, optional, default: `0` – The seed for the random number generator. Defaults to 0.
- **`batch_dims`** : `Shape`, optional, default: `an` – Specifies leading dimensions to be treated as batch dimensions.
Defaults to an empty tuple.
- **`traced`** : `bool`, optional, default: `False` – Whether the operation should be traced in the graph. Defaults to False.

**Returns**

`Tensor` – An tensor initialized with the LeCun normal distribution.


---
## `ndarange`

```python
def ndarange(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor':
```
Creates an tensor of a given shape with sequential values.

The tensor is filled with values from 0 to N-1, where N is the total
number of elements (the product of the shape dimensions).

**Parameters**

- **`shape`** : `Shape` – The shape of the output tensor.
- **`dtype`** : `DType`, optional, default: `DType.float32` – The desired data type for the tensor. Defaults to DType.float32.
- **`device`** : `Device`, optional, default: `the` – The device to place the tensor on. Defaults to the CPU.
- **`batch_dims`** : `Shape`, optional, default: `an` – Specifies leading dimensions to be treated as batch dimensions.
Defaults to an empty tuple.
- **`traced`** : `bool`, optional, default: `False` – Whether the operation should be traced in the graph. Defaults to False.

**Returns**

`Tensor` – An tensor of the specified shape containing values from 0 to N-1.

**Examples**

```python
>>> import nabla as nb
>>> nb.ndarange((2, 3), dtype=nb.DType.int32)
Tensor([[0, 1, 2],
       [3, 4, 5]], dtype=int32)
```

---
## `ndarange_like`

```python
def ndarange_like(template: 'Tensor') -> 'Tensor':
```
Creates an tensor with sequential values like a template tensor.

The new tensor will have the same shape, dtype, device, and batch
dimensions as the template tensor. It is filled with values from 0 to
N-1, where N is the total number of elements.

**Parameters**

- **`template`** : `Tensor` – The template tensor to match properties from.

**Returns**

`Tensor` – A new tensor with the same properties as the template, filled with
sequential values.

**Examples**

```python
>>> import nabla as nb
>>> template = nb.zeros((2, 2), dtype=nb.DType.int32)
>>> nb.ndarange_like(template)
Tensor([[0, 1],
       [2, 3]], dtype=int32)
```

---
