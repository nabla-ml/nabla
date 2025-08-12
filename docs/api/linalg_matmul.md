# matmul

## Signature

```python
nabla.matmul(arg0: nabla.core.array.Array | float | int, arg1: nabla.core.array.Array | float | int) -> nabla.core.array.Array
```

## Description

Performs matrix multiplication on two arrays.

This function follows the semantics of `numpy.matmul`, supporting
multiplication of 1D vectors, 2D matrices, and stacks of matrices.

- If both arguments are 1D arrays of size `N`, it computes the inner
(dot) product and returns a scalar-like array.
- If one argument is a 2D array (M, K) and the other is a 1D array (K),
it promotes the vector to a matrix (1, K) or (K, 1) for the
multiplication, then squeezes the result back to a 1D array.
- If both arguments are 2D arrays, `(M, K) @ (K, N)`, it performs standard
matrix multiplication, resulting in an array of shape `(M, N)`.
- If either argument has more than 2 dimensions, it is treated as a stack
of matrices residing in the last two dimensions and is broadcast accordingly.

Parameters
----------
arg0 : Array | float | int
The first input array.
arg1 : Array | float | int
The second input array.

Returns
-------
Array
The result of the matrix multiplication.

Examples
--------
>>> import nabla as nb
>>> # Vector-vector product (dot product)
>>> v1 = nb.array([1, 2, 3])
>>> v2 = nb.array([4, 5, 6])
>>> nb.matmul(v1, v2)
Array([32], dtype=int32)

>>> # Matrix-vector product
>>> M = nb.array([[1, 2], [3, 4]])
>>> v = nb.array([5, 6])
>>> nb.matmul(M, v)
Array([17, 39], dtype=int32)

>>> # Batched matrix-matrix product
>>> M1 = nb.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) # Shape (2, 2, 2)
>>> M2 = nb.array([[[9, 1], [2, 3]], [[4, 5], [6, 7]]]) # Shape (2, 2, 2)
>>> nb.matmul(M1, M2)
Array([[[ 13,   7],
[ 35,  15]],
<BLANKLINE>
[[ 56,  47],
[ 76,  67]]], dtype=int32)

## Examples

```python
import nabla as nb

# Matrix multiplication
A = nb.array([[1, 2], [3, 4]])
B = nb.array([[5, 6], [7, 8]])
result = nb.matmul(A, B)
print(result)  # [[19, 22], [43, 50]]
```

