# Binary

## `add`

```python
def add(x: 'Tensor', y: 'Tensor | float | int') -> 'Tensor':
```
Add *x* and *y* element-wise, with broadcasting.


---
## `sub`

```python
def sub(x: 'Tensor', y: 'Tensor | float | int') -> 'Tensor':
```
Subtract *y* from *x* element-wise, with broadcasting.


---
## `mul`

```python
def mul(x: 'Tensor', y: 'Tensor | float | int') -> 'Tensor':
```
Multiply *x* and *y* element-wise, with broadcasting.


---
## `div`

```python
def div(x: 'Tensor', y: 'Tensor | float | int') -> 'Tensor':
```
Divide *x* by *y* element-wise, with broadcasting.


---
## `matmul`

```python
def matmul(x: 'Tensor', y: 'Tensor') -> 'Tensor':
```
Matrix multiplication of *x* and *y*.

Supports batched inputs with arbitrary-rank batch prefixes.
1-D inputs are automatically promoted: a vector of shape ``(N,)`` becomes
``(1, N)`` or ``(N, 1)`` for the left/right operand respectively, and the
added dimension is squeezed from the result.

**Parameters**

- **`x`** – Tensor of shape ``(*, M, K)`` or ``(K,)``.
- **`y`** – Tensor of shape ``(*, K, N)`` or ``(K,)``.

**Returns**

 – Tensor of shape ``(*, M, N)`` (or scalar for vector × vector).


---
## `mod`

```python
def mod(x: 'Tensor', y: 'Tensor | float | int') -> 'Tensor':
```
Compute the element-wise remainder ``x % y``, with broadcasting.


---
## `pow`

```python
def pow(x: 'Tensor', y: 'Tensor | float | int') -> 'Tensor':
```
Compute the element-wise power ``x ** y``, with broadcasting.


---
## `outer`

```python
def outer(x: 'Tensor', y: 'Tensor') -> 'Tensor':
```
Compute the outer product of vectors *x* and *y*.

For 1-D inputs of shapes ``(M,)`` and ``(N,)``, returns an ``(M, N)``
matrix. Under ``vmap``, operates on the non-batched trailing dimensions.

**Parameters**

- **`x`** – Vector of shape ``(M,)``.
- **`y`** – Vector of shape ``(N,)``.

**Returns**

 – Tensor of shape ``(M, N)``.


---
