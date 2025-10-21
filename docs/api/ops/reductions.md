# Reduction Ops

## `sum`

```python
def sum(arg: 'Tensor', axes: 'int | list[int] | tuple[int, ...] | None' = None, keep_dims: 'bool' = False) -> 'Tensor':
```
Calculates the sum of tensor elements over given axes.

This function reduces an tensor by summing its elements along the
specified axes. If no axes are provided, the sum of all elements in the
tensor is calculated.

**Parameters**

- **`arg`** : `Tensor` – The input tensor to be summed.
- **`axes`** : `int | list[int] | tuple[int, ...] | None`, optional – The axis or axes along which to perform the sum. If None (the
default), the sum is performed over all axes, resulting in a scalar
tensor.
- **`keep_dims`** : `bool`, optional, default: `False` – If True, the axes which are reduced are left in the result as
dimensions with size one. This allows the result to broadcast
correctly against the original tensor. Defaults to False.

**Returns**

`Tensor` – An tensor containing the summed values.

**Examples**


Sum all elements:
```python
>>> import nabla as nb
>>> x = nb.tensor([[1, 2, 3], [4, 5, 6]])
```

Tensor([21], dtype=int32)

Sum along an axis:
```python
>>> nb.sum(x)
```

Tensor([5, 7, 9], dtype=int32)

Sum along an axis and keep dimensions:
```python
>>> nb.sum(x, axes=0)
```

Tensor([[ 6],
       [15]], dtype=int32)
```python
>>> nb.sum(x, axes=1, keep_dims=True)
```


---
## `mean`

```python
def mean(arg: 'Tensor', axes: 'int | list[int] | tuple[int, ...] | None' = None, keep_dims: 'bool' = False) -> 'Tensor':
```
Computes the arithmetic mean of tensor elements over given axes.

This function calculates the average of an tensor's elements along the
specified axes. If no axes are provided, the mean of all elements in the
tensor is calculated.

**Parameters**

- **`arg`** : `Tensor` – The input tensor for which to compute the mean.
- **`axes`** : `int | list[int] | tuple[int, ...] | None`, optional – The axis or axes along which to compute the mean. If None (the default),
the mean is computed over all axes, resulting in a scalar tensor.
- **`keep_dims`** : `bool`, optional, default: `False` – If True, the axes which are reduced are left in the result as
dimensions with size one. This allows the result to broadcast
correctly against the original tensor. Defaults to False.

**Returns**

`Tensor` – An tensor containing the mean values, typically of a floating-point dtype.

**Examples**


Compute the mean of all elements:
```python
>>> import nabla as nb
>>> x = nb.tensor([[1, 2, 3], [4, 5, 6]])
```

Tensor([3.5], dtype=float32)

Compute the mean along an axis:
```python
>>> nb.mean(x)
```

Tensor([2.5, 3.5, 4.5], dtype=float32)

Compute the mean along an axis and keep dimensions:
```python
>>> nb.mean(x, axes=0)
```

Tensor([[2.],
       [5.]], dtype=float32)
```python
>>> nb.mean(x, axes=1, keep_dims=True)
```


---
## `max`

```python
def max(arg: 'Tensor', axes: 'int | list[int] | tuple[int, ...] | None' = None, keep_dims: 'bool' = False) -> 'Tensor':
```
Finds the maximum value of tensor elements over given axes.

This function reduces an tensor by finding the maximum element along the
specified axes. If no axes are provided, the maximum of all elements in the
tensor is returned.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.
- **`axes`** : `int | list[int] | tuple[int, ...] | None`, optional – The axis or axes along which to find the maximum. If None (the
default), the maximum is found over all axes, resulting in a scalar
tensor.
- **`keep_dims`** : `bool`, optional, default: `False` – If True, the axes which are reduced are left in the result as
dimensions with size one. This allows the result to broadcast
correctly against the original tensor. Defaults to False.

**Returns**

`Tensor` – An tensor containing the maximum values.

**Examples**


Find the maximum of all elements:
```python
>>> import nabla as nb
>>> x = nb.tensor([[1, 5, 2], [4, 3, 6]])
```

Tensor([6], dtype=int32)

Find the maximum along an axis:
```python
>>> nb.max(x)
```

Tensor([5, 6], dtype=int32)

Find the maximum along an axis and keep dimensions:
```python
>>> nb.max(x, axes=1)
```

Tensor([[4, 5, 6]], dtype=int32)
```python
>>> nb.max(x, axes=0, keep_dims=True)
```


---
## `argmax`

```python
def argmax(arg: 'Tensor', axes: 'int | None' = None, keep_dims: 'bool' = False) -> 'Tensor':
```
Finds the indices of maximum tensor elements over a given axis.

This function returns the indices of the maximum values along an axis. If
multiple occurrences of the maximum value exist, the index of the first
occurrence is returned.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.
- **`axes`** : `int | None`, optional – The axis along which to find the indices of the maximum values. If
None (the default), the tensor is flattened before finding the index
of the overall maximum value.
- **`keep_dims`** : `bool`, optional, default: `False` – If True, the axis which is reduced is left in the result as a
dimension with size one. This is not supported when `axes` is None.
Defaults to False.

**Returns**

`Tensor` – An tensor of `int64` integers containing the indices of the maximum
elements.

**Examples**

Tensor(1, dtype=int64)

```python
>>> import nabla as nb
>>> x = nb.tensor([1, 5, 2, 5])
>>> nb.argmax(x)
```

Tensor([1, 2], dtype=int64)

```python
>>> y = nb.tensor([[1, 5, 2], [4, 3, 6]])
>>> nb.argmax(y, axes=1)
```

Tensor([[1, 0, 1]], dtype=int64)
```python
>>> nb.argmax(y, axes=0, keep_dims=True)
```


---
