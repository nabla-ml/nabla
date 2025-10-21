# floordiv

## Signature

```python
nabla.floordiv(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor'
```

**Source**: `nabla.ops.binary`

## Description

Performs element-wise floor division on two tensors.

Floor division is equivalent to `floor(x / y)`, rounding the result
towards negative infinity. This matches the behavior of Python's `//`
operator, which this function implements for Nabla tensors.

## Parameters

- **`x`** (`Tensor | float | int`): The first input tensor or scalar (the dividend).

- **`y`** (`Tensor | float | int`): The second input tensor or scalar (the divisor). Must be broadcastable to the same shape as `x`.

## Returns

- `Tensor`: An tensor containing the result of the element-wise floor division.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.tensor([10, -10, 9])
>>> y = nb.tensor([3, 3, 3])
>>> nb.floordiv(x, y)
Tensor([ 3, -4,  3], dtype=int32)

>>> x // y
Tensor([ 3, -4,  3], dtype=int32)
```
