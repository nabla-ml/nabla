# mean

## Signature

```python
nabla.mean(arg: 'Tensor', axes: 'int | list[int] | tuple[int, ...] | None' = None, keep_dims: 'bool' = False) -> 'Tensor'
```

**Source**: `nabla.ops.reduce`

## Description

Computes the arithmetic mean of tensor elements over given axes.

This function calculates the average of an tensor's elements along the
specified axes. If no axes are provided, the mean of all elements in the
tensor is calculated.

## Parameters

- **`arg`** (`Tensor`): The input tensor for which to compute the mean.

- **`axes`** (`int | list[int] | tuple[int, ...] | None, optional`): The axis or axes along which to compute the mean. If None (the default), the mean is computed over all axes, resulting in a scalar tensor.

- **`keep_dims`** (`bool, optional`): If True, the axes which are reduced are left in the result as dimensions with size one. This allows the result to broadcast correctly against the original tensor. Defaults to False.

## Returns

Tensor
    An tensor containing the mean values, typically of a floating-point dtype.

## Examples

```python
import nabla as nb
x = nb.tensor([[1, 2, 3], [4, 5, 6]])
```

Compute the mean of all elements:

```python
nb.mean(x)
```

Compute the mean along an axis:

```python
nb.mean(x, axes=0)
```

Compute the mean along an axis and keep dimensions:

```python
nb.mean(x, axes=1, keep_dims=True)
```
