# matmul

## Signature

```python
nabla.matmul(arg0: nabla.core.tensor.Tensor | float | int, arg1: nabla.core.tensor.Tensor | float | int) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.linalg`

## Description

Performs matrix multiplication on two tensors.

This function follows the semantics of `numpy.matmul`, supporting
multiplication of 1D vectors, 2D matrices, and stacks of matrices.

- If both arguments are 1D tensors of size `N`, it computes the inner
  (dot) product and returns a scalar-like tensor.
- If one argument is a 2D tensor (M, K) and the other is a 1D tensor (K),
  it promotes the vector to a matrix (1, K) or (K, 1) for the
  multiplication, then squeezes the result back to a 1D tensor.
- If both arguments are 2D tensors, `(M, K) @ (K, N)`, it performs standard
  matrix multiplication, resulting in an tensor of shape `(M, N)`.
- If either argument has more than 2 dimensions, it is treated as a stack
  of matrices residing in the last two dimensions and is broadcast accordingly.

## Parameters

- **`arg0`** (`Tensor | float | int`): The first input tensor.

- **`arg1`** (`Tensor | float | int`): The second input tensor.

## Returns

- `Tensor`: The result of the matrix multiplication.

## Examples

```pycon
>>> import nabla as nb
>>> # Vector-vector product (dot product)
>>> v1 = nb.tensor([1, 2, 3])
>>> v2 = nb.tensor([4, 5, 6])
>>> nb.matmul(v1, v2)
Tensor([32], dtype=int32)

>>> # Matrix-vector product
>>> M = nb.tensor([[1, 2], [3, 4]])
>>> v = nb.tensor([5, 6])
>>> nb.matmul(M, v)
Tensor([17, 39], dtype=int32)

>>> # Batched matrix-matrix product
>>> M1 = nb.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) # Shape (2, 2, 2)
>>> M2 = nb.tensor([[[9, 1], [2, 3]], [[4, 5], [6, 7]]]) # Shape (2, 2, 2)
>>> nb.matmul(M1, M2)
Tensor([[[ 13,   7],
        [ 35,  15]],

       [[ 56,  47],
        [ 76,  67]]], dtype=int32)
```
