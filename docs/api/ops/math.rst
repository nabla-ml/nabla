Mathematical Operations
=======================

.. currentmodule:: nabla

add
---

Description
-----------

Adds two tensors element-wise.

This function performs element-wise addition on two tensors. It supports
broadcasting, allowing tensors of different shapes to be combined as long
as their shapes are compatible. This function also provides the
implementation of the `+` operator for Nabla tensors.

Parameters
----------
x : Tensor | float | int
    The first input tensor or scalar.
y : Tensor | float | int
    The second input tensor or scalar. Must be broadcastable to the same
    shape as `x`.

Returns
-------
Tensor
    An tensor containing the result of the element-wise addition.

Examples
--------
Calling `add` explicitly:

.. code-block:: python

    >>> import nabla as nb
    >>> x = nb.tensor([1, 2, 3])
    >>> y = nb.tensor([4, 5, 6])
    >>> nb.add(x, y)

Tensor([5, 7, 9], dtype=int32)

Calling `add` via the `+` operator:

.. code-block:: python

    >>> x + y

Tensor([5, 7, 9], dtype=int32)

Broadcasting a scalar:

.. code-block:: python

    >>> x + 10

Tensor([11, 12, 13], dtype=int32)

.. autofunction:: nabla.add

matmul
------

Description
-----------

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

Parameters
----------
arg0 : Tensor | float | int
    The first input tensor.
arg1 : Tensor | float | int
    The second input tensor.

Returns
-------
Tensor
    The result of the matrix multiplication.

Examples
--------
.. code-block:: python

    >>> import nabla as nb
    >>> # Vector-vector product (dot product)
    >>> v1 = nb.tensor([1, 2, 3])
    >>> v2 = nb.tensor([4, 5, 6])
    >>> nb.matmul(v1, v2)

Tensor([32], dtype=int32)

.. code-block:: python

    >>> # Matrix-vector product
    >>> M = nb.tensor([[1, 2], [3, 4]])
    >>> v = nb.tensor([5, 6])
    >>> nb.matmul(M, v)

Tensor([17, 39], dtype=int32)

.. code-block:: python

    >>> # Batched matrix-matrix product
    >>> M1 = nb.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) # Shape (2, 2, 2)
    >>> M2 = nb.tensor([[[9, 1], [2, 3]], [[4, 5], [6, 7]]]) # Shape (2, 2, 2)
    >>> nb.matmul(M1, M2)

Tensor([[[ 13,   7],
        [ 35,  15]],
<BLANKLINE>
       [[ 56,  47],
        [ 76,  67]]], dtype=int32)

.. autofunction:: nabla.matmul

sum
---

Description
-----------

Calculates the sum of tensor elements over given axes.

This function reduces an tensor by summing its elements along the
specified axes. If no axes are provided, the sum of all elements in the
tensor is calculated.

Parameters
----------
arg : Tensor
    The input tensor to be summed.
axes : int | list[int] | tuple[int, ...] | None, optional
    The axis or axes along which to perform the sum. If None (the
    default), the sum is performed over all axes, resulting in a scalar
    tensor.
keep_dims : bool, optional
    If True, the axes which are reduced are left in the result as
    dimensions with size one. This allows the result to broadcast
    correctly against the original tensor. Defaults to False.

Returns
-------
Tensor
    An tensor containing the summed values.

Examples
--------
.. code-block:: python

    >>> import nabla as nb
    >>> x = nb.tensor([[1, 2, 3], [4, 5, 6]])


Sum all elements:
.. code-block:: python

    >>> nb.sum(x)

Tensor([21], dtype=int32)

Sum along an axis:
.. code-block:: python

    >>> nb.sum(x, axes=0)

Tensor([5, 7, 9], dtype=int32)

Sum along an axis and keep dimensions:
.. code-block:: python

    >>> nb.sum(x, axes=1, keep_dims=True)

Tensor([[ 6],
       [15]], dtype=int32)

.. autofunction:: nabla.sum
