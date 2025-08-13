# logsumexp

## Signature

```python
nabla.logsumexp(arg: 'Array', axis: 'int | None', keep_dims: 'bool') -> 'Array'
```

## Description

Computes the log of the sum of exponentials of input elements.

This function computes `log(sum(exp(x)))` in a numerically stable way by using
the identity: `logsumexp(x) = max(x) + log(sum(exp(x - max(x))))`. This
avoids overflow errors that can occur when `exp(x)` is very large.

## Parameters

- **`arg`** (`Array`): The input array.

- **`axis`** (`int | None, optional`): The axis or axes along which to compute the `logsumexp`. If None (the default), the operation is performed over all elements of the array.

- **`keep_dims`** (`bool, optional`): If True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array. Defaults to False.

## Returns

- `Array`: An array containing the result of the `logsumexp` operation.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.array([1.0, 2.0, 3.0])
>>> nb.logsumexp(x)
Array([3.407606], dtype=float32)

>>> data = nb.array([[1, 2, 3], [4, 5, 6]])
>>> nb.logsumexp(data, axis=1)
Array([3.407606, 6.407606], dtype=float32)
```
