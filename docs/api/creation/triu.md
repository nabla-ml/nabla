# triu

## Signature

```python
nabla.triu(x: 'Tensor', k: 'int') -> 'Tensor'
```

## Description

Returns the upper triangular part of a matrix or batch of matrices.

The elements below the k-th diagonal are zeroed out. The input is
expected to be at least 2-dimensional.

## Parameters

- **`x`** (`Tensor`): Input tensor with shape (..., M, N).

- **`k`** (`int, optional`): Diagonal offset. `k = 0` is the main diagonal. `k > 0` is above the main diagonal, and `k < 0` is below the main diagonal. Defaults to 0.

## Returns

- `Tensor`: An tensor with the lower triangular part zeroed out, with the same shape and dtype as `x`.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.ndarange((3, 3), dtype=nb.DType.int32)
>>> x
Tensor([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]], dtype=int32)

>>> # Upper triangle with the main diagonal
>>> nb.triu(x, k=0)
Tensor([[0, 1, 2],
       [0, 4, 5],
       [0, 0, 8]], dtype=int32)

>>> # Upper triangle above the main diagonal
>>> nb.triu(x, k=1)
Tensor([[0, 1, 2],
       [0, 0, 5],
       [0, 0, 0]], dtype=int32)
```
