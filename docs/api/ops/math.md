# Math Ops

## `add`

```python
def add(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor':
```
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

>>> import nabla as nb
>>> x = nb.tensor([1, 2, 3])
>>> y = nb.tensor([4, 5, 6])
>>> nb.add(x, y)
Tensor([5, 7, 9], dtype=int32)

Calling `add` via the `+` operator:

>>> x + y
Tensor([5, 7, 9], dtype=int32)

Broadcasting a scalar:

>>> x + 10
Tensor([11, 12, 13], dtype=int32)

---
## `sub`

```python
def sub(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor':
```
Subtracts two tensors element-wise.

This function performs element-wise subtraction on two tensors. It supports
broadcasting, allowing tensors of different shapes to be combined as long
as their shapes are compatible. This function also provides the
implementation of the `-` operator for Nabla tensors.

Parameters
----------
x : Tensor | float | int
    The first input tensor or scalar (the minuend).
y : Tensor | float | int
    The second input tensor or scalar (the subtrahend). Must be
    broadcastable to the same shape as `x`.

Returns
-------
Tensor
    An tensor containing the result of the element-wise subtraction.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([10, 20, 30])
>>> y = nb.tensor([1, 2, 3])
>>> nb.sub(x, y)
Tensor([ 9, 18, 27], dtype=int32)

>>> x - y
Tensor([ 9, 18, 27], dtype=int32)

---
## `mul`

```python
def mul(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor':
```
Multiplies two tensors element-wise.

This function performs element-wise multiplication on two tensors. It
supports broadcasting, allowing tensors of different shapes to be combined
as long as their shapes are compatible. This function also provides the
implementation of the `*` operator for Nabla tensors.

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
    An tensor containing the result of the element-wise multiplication.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([1, 2, 3])
>>> y = nb.tensor([4, 5, 6])
>>> nb.mul(x, y)
Tensor([4, 10, 18], dtype=int32)

>>> x * y
Tensor([4, 10, 18], dtype=int32)

---
## `div`

```python
def div(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor':
```
Divides two tensors element-wise.

This function performs element-wise (true) division on two tensors. It
supports broadcasting, allowing tensors of different shapes to be combined
as long as their shapes are compatible. This function also provides the
implementation of the `/` operator for Nabla tensors.

Parameters
----------
x : Tensor | float | int
    The first input tensor or scalar (the dividend).
y : Tensor | float | int
    The second input tensor or scalar (the divisor). Must be broadcastable
    to the same shape as `x`.

Returns
-------
Tensor
    An tensor containing the result of the element-wise division. The
    result is typically a floating-point tensor.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([10, 20, 30])
>>> y = nb.tensor([2, 5, 10])
>>> nb.div(x, y)
Tensor([5., 4., 3.], dtype=float32)

>>> x / y
Tensor([5., 4., 3.], dtype=float32)

---
## `pow`

```python
def pow(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor':
```
Computes `x` raised to the power of `y` element-wise.

This function calculates `x ** y` for each element in the input tensors.
It supports broadcasting and provides the implementation of the `**`
operator for Nabla tensors.

Parameters
----------
x : Tensor | float | int
    The base tensor or scalar.
y : Tensor | float | int
    The exponent tensor or scalar. Must be broadcastable to the same shape
    as `x`.

Returns
-------
Tensor
    An tensor containing the result of the element-wise power operation.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([1, 2, 3])
>>> y = nb.tensor([2, 3, 2])
>>> nb.pow(x, y)
Tensor([1, 8, 9], dtype=int32)

>>> x ** y
Tensor([1, 8, 9], dtype=int32)

---
## `negate`

```python
def negate(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise numerical negative of an tensor.

This function returns a new tensor with each element being the negation
of the corresponding element in the input tensor. It also provides the
implementation for the unary `-` operator on Nabla tensors.

Parameters
----------
arg : Tensor
    The input tensor.

Returns
-------
Tensor
    An tensor containing the negated elements.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([1, -2, 3.5])
>>> nb.negate(x)
Tensor([-1.,  2., -3.5], dtype=float32)

Using the `-` operator:
>>> -x
Tensor([-1.,  2., -3.5], dtype=float32)

---
## `exp`

```python
def exp(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise exponential function (e^x).

This function calculates the base-e exponential of each element in the
input tensor.

Parameters
----------
arg : Tensor
    The input tensor.

Returns
-------
Tensor
    An tensor containing the exponential of each element.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([0.0, 1.0, 2.0])
>>> nb.exp(x)
Tensor([1.       , 2.7182817, 7.389056 ], dtype=float32)

---
## `log`

```python
def log(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise natural logarithm (base e).

This function calculates `log(x)` for each element `x` in the input tensor.
For numerical stability with non-positive inputs, a small epsilon is
added to ensure the input to the logarithm is positive.

Parameters
----------
arg : Tensor
    The input tensor. Values should be positive.

Returns
-------
Tensor
    An tensor containing the natural logarithm of each element.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([1.0, 2.71828, 10.0])
>>> nb.log(x)
Tensor([0.       , 0.9999993, 2.3025851], dtype=float32)

---
## `sqrt`

```python
def sqrt(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Computes the element-wise non-negative square root of an tensor.

This function is implemented as `nabla.pow(arg, 0.5)` to ensure it is
compatible with the automatic differentiation system.

Parameters
----------
arg : Tensor
    The input tensor. All elements must be non-negative.

Returns
-------
Tensor
    An tensor containing the square root of each element.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([0.0, 4.0, 9.0])
>>> nb.sqrt(x)
Tensor([0., 2., 3.], dtype=float32)

---
## `matmul`

```python
def matmul(arg0: nabla.core.tensor.Tensor | float | int, arg1: nabla.core.tensor.Tensor | float | int) -> nabla.core.tensor.Tensor:
```
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
<BLANKLINE>
       [[ 56,  47],
        [ 76,  67]]], dtype=int32)

---
