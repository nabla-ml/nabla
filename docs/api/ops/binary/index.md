# Binary Operations

## `add`

```python
def add(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor':
```
Adds two tensors element-wise.

This function performs element-wise addition on two tensors. It supports
broadcasting, allowing tensors of different shapes to be combined as long
as their shapes are compatible. This function also provides the
implementation of the `+` operator for Nabla tensors.

**Parameters**

- **`x`** : `Tensor | float | int` – The first input tensor or scalar.
- **`y`** : `Tensor | float | int` – The second input tensor or scalar. Must be broadcastable to the same
shape as `x`.

**Returns**

`Tensor` – An tensor containing the result of the element-wise addition.

**Examples**

Calling `add` explicitly:

```python
>>> import nabla as nb
>>> x = nb.tensor([1, 2, 3])
>>> y = nb.tensor([4, 5, 6])
>>> nb.add(x, y)
Tensor([5, 7, 9], dtype=int32)
```

Calling `add` via the `+` operator:

```python
>>> x + y
Tensor([5, 7, 9], dtype=int32)
```

Broadcasting a scalar:

```python
>>> x + 10
Tensor([11, 12, 13], dtype=int32)
```

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

**Parameters**

- **`x`** : `Tensor | float | int` – The first input tensor or scalar (the minuend).
- **`y`** : `Tensor | float | int` – The second input tensor or scalar (the subtrahend). Must be
broadcastable to the same shape as `x`.

**Returns**

`Tensor` – An tensor containing the result of the element-wise subtraction.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([10, 20, 30])
>>> y = nb.tensor([1, 2, 3])
>>> nb.sub(x, y)
Tensor([ 9, 18, 27], dtype=int32)
```

```python
>>> x - y
Tensor([ 9, 18, 27], dtype=int32)
```

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

**Parameters**

- **`x`** : `Tensor | float | int` – The first input tensor or scalar.
- **`y`** : `Tensor | float | int` – The second input tensor or scalar. Must be broadcastable to the same
shape as `x`.

**Returns**

`Tensor` – An tensor containing the result of the element-wise multiplication.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([1, 2, 3])
>>> y = nb.tensor([4, 5, 6])
>>> nb.mul(x, y)
Tensor([4, 10, 18], dtype=int32)
```

```python
>>> x * y
Tensor([4, 10, 18], dtype=int32)
```

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

**Parameters**

- **`x`** : `Tensor | float | int` – The first input tensor or scalar (the dividend).
- **`y`** : `Tensor | float | int` – The second input tensor or scalar (the divisor). Must be broadcastable
to the same shape as `x`.

**Returns**

`Tensor` – An tensor containing the result of the element-wise division. The
result is typically a floating-point tensor.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([10, 20, 30])
>>> y = nb.tensor([2, 5, 10])
>>> nb.div(x, y)
Tensor([5., 4., 3.], dtype=float32)
```

```python
>>> x / y
Tensor([5., 4., 3.], dtype=float32)
```

---
## `pow`

```python
def pow(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor':
```
Computes `x` raised to the power of `y` element-wise.

This function calculates `x ** y` for each element in the input tensors.
It supports broadcasting and provides the implementation of the `**`
operator for Nabla tensors.

**Parameters**

- **`x`** : `Tensor | float | int` – The base tensor or scalar.
- **`y`** : `Tensor | float | int` – The exponent tensor or scalar. Must be broadcastable to the same shape
as `x`.

**Returns**

`Tensor` – An tensor containing the result of the element-wise power operation.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([1, 2, 3])
>>> y = nb.tensor([2, 3, 2])
>>> nb.pow(x, y)
Tensor([1, 8, 9], dtype=int32)
```

```python
>>> x ** y
Tensor([1, 8, 9], dtype=int32)
```

---
## `equal`

```python
def equal(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor':
```
Performs element-wise comparison `x == y`.

This function compares two tensors element-wise, returning a boolean tensor
indicating where elements of `x` are equal to elements of `y`. It
supports broadcasting and provides the implementation of the `==` operator
for Nabla tensors.

**Parameters**

- **`x`** : `Tensor | float | int` – The first input tensor or scalar.
- **`y`** : `Tensor | float | int` – The second input tensor or scalar. Must be broadcastable to the same
shape as `x`.

**Returns**

`Tensor` – A boolean tensor containing the result of the element-wise equality
comparison.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([1, 2, 3])
>>> y = nb.tensor([1, 5, 3])
>>> nb.equal(x, y)
Tensor([ True, False,  True], dtype=bool)
```

```python
>>> x == y
Tensor([ True, False,  True], dtype=bool)
```

---
## `not_equal`

```python
def not_equal(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor':
```
Performs element-wise comparison `x != y`.

This function compares two tensors element-wise, returning a boolean tensor
indicating where elements of `x` are not equal to elements of `y`. It
supports broadcasting and provides the implementation of the `!=` operator
for Nabla tensors.

**Parameters**

- **`x`** : `Tensor | float | int` – The first input tensor or scalar.
- **`y`** : `Tensor | float | int` – The second input tensor or scalar. Must be broadcastable to the same
shape as `x`.

**Returns**

`Tensor` – A boolean tensor containing the result of the element-wise inequality
comparison.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([1, 2, 3])
>>> y = nb.tensor([1, 5, 3])
>>> nb.not_equal(x, y)
Tensor([False,  True, False], dtype=bool)
```

```python
>>> x != y
Tensor([False,  True, False], dtype=bool)
```

---
## `greater_equal`

```python
def greater_equal(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor':
```
Performs element-wise comparison `x >= y`.

This function compares two tensors element-wise, returning a boolean tensor
indicating where elements of `x` are greater than or equal to elements
of `y`. It supports broadcasting and provides the implementation of the
`>=` operator for Nabla tensors.

**Parameters**

- **`x`** : `Tensor | float | int` – The first input tensor or scalar.
- **`y`** : `Tensor | float | int` – The second input tensor or scalar. Must be broadcastable to the same
shape as `x`.

**Returns**

`Tensor` – A boolean tensor containing the result of the element-wise comparison.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([1, 5, 3])
>>> y = nb.tensor([2, 5, 1])
>>> nb.greater_equal(x, y)
Tensor([False,  True,  True], dtype=bool)
```

```python
>>> x >= y
Tensor([False,  True,  True], dtype=bool)
```

---
## `maximum`

```python
def maximum(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor':
```
Computes the element-wise maximum of two tensors.

This function compares two tensors element-wise and returns a new tensor
containing the larger of the two elements at each position. It supports
broadcasting.

**Parameters**

- **`x`** : `Tensor | float | int` – The first input tensor or scalar.
- **`y`** : `Tensor | float | int` – The second input tensor or scalar. Must be broadcastable to the same
shape as `x`.

**Returns**

`Tensor` – An tensor containing the element-wise maximum of `x` and `y`.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([1, 5, 2])
>>> y = nb.tensor([2, 3, 6])
>>> nb.maximum(x, y)
Tensor([2, 5, 6], dtype=int32)
```

---
## `minimum`

```python
def minimum(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor':
```
Computes the element-wise minimum of two tensors.

This function compares two tensors element-wise and returns a new tensor
containing the smaller of the two elements at each position. It supports
broadcasting.

**Parameters**

- **`x`** : `Tensor | float | int` – The first input tensor or scalar.
- **`y`** : `Tensor | float | int` – The second input tensor or scalar. Must be broadcastable to the same
shape as `x`.

**Returns**

`Tensor` – An tensor containing the element-wise minimum of `x` and `y`.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([1, 5, 2])
>>> y = nb.tensor([2, 3, 6])
>>> nb.minimum(x, y)
Tensor([1, 3, 2], dtype=int32)
```

---
## `mod`

```python
def mod(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor':
```
Computes the element-wise remainder of division.

This function calculates the remainder of `x / y` element-wise. The
sign of the result follows the sign of the divisor `y`. It provides the
implementation of the `%` operator for Nabla tensors.

**Parameters**

- **`x`** : `Tensor | float | int` – The dividend tensor or scalar.
- **`y`** : `Tensor | float | int` – The divisor tensor or scalar. Must be broadcastable to the same shape
as `x`.

**Returns**

`Tensor` – An tensor containing the element-wise remainder.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([10, -10, 9])
>>> y = nb.tensor([3, 3, -3])
>>> nb.mod(x, y)
Tensor([ 1,  2, -0], dtype=int32)
```

```python
>>> x % y
Tensor([ 1,  2, -0], dtype=int32)
```

---
## `floordiv`

```python
def floordiv(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor':
```
Performs element-wise floor division on two tensors.

Floor division is equivalent to `floor(x / y)`, rounding the result
towards negative infinity. This matches the behavior of Python's `//`
operator, which this function implements for Nabla tensors.

**Parameters**

- **`x`** : `Tensor | float | int` – The first input tensor or scalar (the dividend).
- **`y`** : `Tensor | float | int` – The second input tensor or scalar (the divisor). Must be broadcastable
to the same shape as `x`.

**Returns**

`Tensor` – An tensor containing the result of the element-wise floor division.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([10, -10, 9])
>>> y = nb.tensor([3, 3, 3])
>>> nb.floordiv(x, y)
Tensor([ 3, -4,  3], dtype=int32)
```

```python
>>> x // y
Tensor([ 3, -4,  3], dtype=int32)
```

---
