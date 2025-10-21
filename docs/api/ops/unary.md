# Unary Operations

## `negate`

```python
def negate(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise numerical negative of an tensor.

This function returns a new tensor with each element being the negation
of the corresponding element in the input tensor. It also provides the
implementation for the unary `-` operator on Nabla tensors.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.

**Returns**

`Tensor` – An tensor containing the negated elements.

**Examples**

--
```python
>>> import nabla as nb
>>> x = nb.tensor([1, -2, 3.5])
>>> nb.negate(x)
Tensor([-1.,  2., -3.5], dtype=float32)
```

Using the `-` operator:
```python
>>> -x
Tensor([-1.,  2., -3.5], dtype=float32)
```

---
## `abs`

```python
def abs(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise absolute value of an tensor.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.

**Returns**

`Tensor` – An tensor containing the absolute value of each element.

**Examples**

--
```python
>>> import nabla as nb
>>> x = nb.tensor([-1.5, 0.0, 2.5])
>>> nb.abs(x)
Tensor([1.5, 0. , 2.5], dtype=float32)
```

---
## `exp`

```python
def exp(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise exponential function (e^x).

This function calculates the base-e exponential of each element in the
input tensor.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.

**Returns**

`Tensor` – An tensor containing the exponential of each element.

**Examples**

--
```python
>>> import nabla as nb
>>> x = nb.tensor([0.0, 1.0, 2.0])
>>> nb.exp(x)
Tensor([1.       , 2.7182817, 7.389056 ], dtype=float32)
```

---
## `log`

```python
def log(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise natural logarithm (base e).

This function calculates `log(x)` for each element `x` in the input tensor.
For numerical stability with non-positive inputs, a small epsilon is
added to ensure the input to the logarithm is positive.

**Parameters**

- **`arg`** : `Tensor` – The input tensor. Values should be positive.

**Returns**

`Tensor` – An tensor containing the natural logarithm of each element.

**Examples**

--
```python
>>> import nabla as nb
>>> x = nb.tensor([1.0, 2.71828, 10.0])
>>> nb.log(x)
Tensor([0.       , 0.9999993, 2.3025851], dtype=float32)
```

---
## `sqrt`

```python
def sqrt(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise non-negative square root of an tensor.

This function is implemented as `nabla.pow(arg, 0.5)` to ensure it is
compatible with the automatic differentiation system.

**Parameters**

- **`arg`** : `Tensor` – The input tensor. All elements must be non-negative.

**Returns**

`Tensor` – An tensor containing the square root of each element.

**Examples**

--
```python
>>> import nabla as nb
>>> x = nb.tensor([0.0, 4.0, 9.0])
>>> nb.sqrt(x)
Tensor([0., 2., 3.], dtype=float32)
```

---
## `sin`

```python
def sin(arg: nabla.core.tensor.Tensor, dtype: max._core.dtype.DType | None = None) -> nabla.core.tensor.Tensor:
```
Computes the element-wise sine of an tensor.

**Parameters**

- **`arg`** : `Tensor` – The input tensor. Input is expected to be in radians.
- **`dtype`** : `DType | None`, optional – If provided, the output tensor will be cast to this data type.

**Returns**

`Tensor` – An tensor containing the sine of each element in the input.

**Examples**

--
```python
>>> import nabla as nb
>>> x = nb.tensor([0, 1.5707963, 3.1415926])
>>> nb.sin(x)
Tensor([0.0000000e+00, 1.0000000e+00, -8.7422780e-08], dtype=float32)
```

---
## `cos`

```python
def cos(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise cosine of an tensor.

**Parameters**

- **`arg`** : `Tensor` – The input tensor. Input is expected to be in radians.

**Returns**

`Tensor` – An tensor containing the cosine of each element in the input.

**Examples**

--
```python
>>> import nabla as nb
>>> x = nb.tensor([0, 1.5707963, 3.1415926])
>>> nb.cos(x)
Tensor([ 1.000000e+00, -4.371139e-08, -1.000000e+00], dtype=float32)
```

---
## `tanh`

```python
def tanh(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise hyperbolic tangent of an tensor.

The tanh function is a common activation function in neural networks,
squashing values to the range `[-1, 1]`.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.

**Returns**

`Tensor` – An tensor containing the hyperbolic tangent of each element.

**Examples**

--
```python
>>> import nabla as nb
>>> x = nb.tensor([-1.0, 0.0, 1.0, 20.0])
>>> nb.tanh(x)
Tensor([-0.7615942,  0.       ,  0.7615942,  1.       ], dtype=float32)
```

---
## `relu`

```python
def relu(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise Rectified Linear Unit (ReLU) function.

The ReLU function is defined as `max(0, x)`. It is a widely used
activation function in neural networks.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.

**Returns**

`Tensor` – An tensor containing the result of the ReLU operation.

**Examples**

--
```python
>>> import nabla as nb
>>> x = nb.tensor([-2.0, -0.5, 0.0, 1.0, 2.0])
>>> nb.relu(x)
Tensor([0., 0., 0., 1., 2.], dtype=float32)
```

---
## `sigmoid`

```python
def sigmoid(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise sigmoid function.

The sigmoid function, defined as `1 / (1 + exp(-x))`, is a common
activation function that squashes values to the range `(0, 1)`.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.

**Returns**

`Tensor` – An tensor containing the sigmoid of each element.

**Examples**

--
```python
>>> import nabla as nb
>>> x = nb.tensor([-1.0, 0.0, 1.0, 20.0])
>>> nb.sigmoid(x)
Tensor([0.26894143, 0.5       , 0.7310586 , 1.        ], dtype=float32)
```

---
## `floor`

```python
def floor(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise floor of an tensor.

The floor of a scalar `x` is the largest integer `i` such that `i <= x`.
This function is not differentiable and its gradient is zero everywhere.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.

**Returns**

`Tensor` – An tensor containing the floor of each element.

**Examples**

--
```python
>>> import nabla as nb
>>> x = nb.tensor([-1.7, -0.2, 0.2, 1.7])
>>> nb.floor(x)
Tensor([-2., -1.,  0.,  1.], dtype=float32)
```

---
## `cast`

```python
def cast(arg: nabla.core.tensor.Tensor, dtype: max._core.dtype.DType) -> nabla.core.tensor.Tensor:
```
Casts an tensor to a specified data type.

This function creates a new tensor with the same shape as the input but
with the specified data type (`dtype`).

**Parameters**

- **`arg`** : `Tensor` – The input tensor to be cast.
- **`dtype`** : `DType` – The target Nabla data type (e.g., `nb.float32`, `nb.int32`).

**Returns**

`Tensor` – A new tensor with the elements cast to the specified `dtype`.

**Examples**

--
```python
>>> import nabla as nb
>>> x = nb.tensor([1, 2, 3])
>>> x.dtype
int32
>>> y = nb.cast(x, nb.float32)
>>> y
Tensor([1., 2., 3.], dtype=float32)
```

---
## `logical_not`

```python
def logical_not(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise logical NOT of a boolean tensor.

This function inverts the boolean value of each element in the input tensor.
Input tensors of non-boolean types will be cast to boolean first.

**Parameters**

- **`arg`** : `Tensor` – The input boolean tensor.

**Returns**

`Tensor` – A boolean tensor containing the inverted values.

**Examples**

--
```python
>>> import nabla as nb
>>> x = nb.tensor([True, False, True])
>>> nb.logical_not(x)
Tensor([False,  True, False], dtype=bool)
```

---
## `transfer_to`

```python
def transfer_to(arg: nabla.core.tensor.Tensor, device: max._core.driver.Device) -> nabla.core.tensor.Tensor:
```
Transfers an tensor to a different compute device.

This function moves the data of a Nabla tensor to the specified device
(e.g., from CPU to GPU). If the tensor is already on the target device,
it is returned unchanged.

**Parameters**

- **`arg`** : `Tensor` – The input tensor to transfer.
- **`device`** : `Device` – The target device instance (e.g., `nb.Device.cpu()`, `nb.Device.gpu()`).

**Returns**

`Tensor` – A new tensor residing on the target device.


---
## `incr_batch_dim_ctr`

```python
def incr_batch_dim_ctr(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Moves the leading axis from `shape` to `batch_dims`. (Internal use)

This is an internal-use function primarily for developing function
transformations like `vmap`. It re-interprets the first dimension of the
tensor's logical shape as a new batch dimension.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.

**Returns**

`Tensor` – A new tensor with an additional batch dimension.


---
## `decr_batch_dim_ctr`

```python
def decr_batch_dim_ctr(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Moves the last `batch_dim` to be the leading axis of `shape`. (Internal use)

This is an internal-use function primarily for developing function
transformations like `vmap`. It re-interprets the last batch dimension
as the new first dimension of the tensor's logical shape.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.

**Returns**

`Tensor` – A new tensor with one fewer batch dimension.


---
