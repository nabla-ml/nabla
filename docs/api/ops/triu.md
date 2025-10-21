# triu

## Signature

```python
nabla.triu(x: 'Tensor', k: 'int' = 0) -> 'Tensor'
```

**Source**: `nabla.ops.creation`

## Description

Returns the upper triangular part of a matrix or batch of matrices.

The elements below the k-th diagonal are zeroed out. The input is
expected to be at least 2-dimensional.

## Parameters

- **`x`** (`Tensor`): Input tensor with shape (..., M, N).

- **`k`** (`int, optional`): Diagonal offset. `k = 0` is the main diagonal. `k > 0` is above the main diagonal, and `k < 0` is below the main diagonal. Defaults to 0.

## Returns

Tensor
    An tensor with the lower triangular part zeroed out, with the same
    shape and dtype as `x`.

## Examples

```python
import nabla as nb
x = nb.ndarange((3, 3), dtype=nb.DType.int32)
x
```

```python
# Upper triangle with the main diagonal
nb.triu(x, k=0)
```

```python
# Upper triangle above the main diagonal
nb.triu(x, k=1)
```
