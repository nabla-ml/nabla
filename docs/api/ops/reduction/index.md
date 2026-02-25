# Reduction

## `reduce_sum`

```python
def reduce_sum(x: 'Tensor', *, axis: 'int | tuple[int, ...] | list[int] | None' = None, keepdims: 'bool' = False) -> 'Tensor':
```
Sum elements of *x* over the given axis (or axes).

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Axis or axes to reduce. ``None`` reduces over all elements.
- **`keepdims`** – If ``True``, the reduced axes are kept as size-1 dimensions.

**Returns**

Reduced tensor. When *axis* is ``None`` and *keepdims* is ``False``,
a scalar tensor is returned.


---
## `reduce_max`

```python
def reduce_max(x: 'Tensor', *, axis: 'int | tuple[int, ...] | list[int] | None' = None, keepdims: 'bool' = False) -> 'Tensor':
```
Return the maximum value of *x* along the given axis (or axes).

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Axis or axes to reduce. ``None`` reduces over all elements.
- **`keepdims`** – If ``True``, the reduced axes are kept as size-1 dimensions.

**Returns**

Tensor with maximum values.


---
## `reduce_min`

```python
def reduce_min(x: 'Tensor', *, axis: 'int | tuple[int, ...] | list[int] | None' = None, keepdims: 'bool' = False) -> 'Tensor':
```
Return the minimum value of *x* along the given axis (or axes).

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Axis or axes to reduce. ``None`` reduces over all elements.
- **`keepdims`** – If ``True``, the reduced axes are kept as size-1 dimensions.

**Returns**

Tensor with minimum values.


---
## `sum`

```python
def sum(x: 'Tensor', *, axis: 'int | tuple[int, ...] | list[int] | None' = None, keepdims: 'bool' = False) -> 'Tensor':
```
Sum elements of *x* over the given axis (or axes).

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Axis or axes to reduce. ``None`` reduces over all elements.
- **`keepdims`** – If ``True``, the reduced axes are kept as size-1 dimensions.

**Returns**

Reduced tensor. When *axis* is ``None`` and *keepdims* is ``False``,
a scalar tensor is returned.


---
## `max`

```python
def max(x: 'Tensor', *, axis: 'int | tuple[int, ...] | list[int] | None' = None, keepdims: 'bool' = False) -> 'Tensor':
```
Return the maximum value of *x* along the given axis (or axes).

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Axis or axes to reduce. ``None`` reduces over all elements.
- **`keepdims`** – If ``True``, the reduced axes are kept as size-1 dimensions.

**Returns**

Tensor with maximum values.


---
## `min`

```python
def min(x: 'Tensor', *, axis: 'int | tuple[int, ...] | list[int] | None' = None, keepdims: 'bool' = False) -> 'Tensor':
```
Return the minimum value of *x* along the given axis (or axes).

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Axis or axes to reduce. ``None`` reduces over all elements.
- **`keepdims`** – If ``True``, the reduced axes are kept as size-1 dimensions.

**Returns**

Tensor with minimum values.


---
## `mean`

```python
def mean(x: 'Tensor', *, axis: 'int | tuple[int, ...] | list[int] | None' = None, keepdims: 'bool' = False) -> 'Tensor':
```
Compute the arithmetic mean of *x* along the given axis (or axes).

Internally implemented as ``sum(x) / n`` where *n* is the product of
the reduced axis sizes. This ensures correct results across sharded tensors.

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Axis or axes to reduce. ``None`` averages over all elements.
- **`keepdims`** – If ``True``, the reduced axes are kept as size-1 dimensions.

**Returns**

Tensor with the mean values.


---
## `argmax`

```python
def argmax(x: 'Tensor', axis: 'int' = -1, keepdims: 'bool' = False) -> 'Tensor':
```
Return the indices of the maximum values along *axis*.

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Axis along which to find the maximum. Default: ``-1``.
- **`keepdims`** – If ``True``, the reduced axis is kept as a size-1 dimension.

**Returns**

Integer tensor of dtype ``int64`` with the argmax indices.


---
## `argmin`

```python
def argmin(x: 'Tensor', axis: 'int' = -1, keepdims: 'bool' = False) -> 'Tensor':
```
Return the indices of the minimum values along *axis*.

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Axis along which to find the minimum. Default: ``-1``.
- **`keepdims`** – If ``True``, the reduced axis is kept as a size-1 dimension.

**Returns**

Integer tensor of dtype ``int64`` with the argmin indices.


---
## `cumsum`

```python
def cumsum(x: 'Tensor', axis: 'int' = -1, exclusive: 'bool' = False, reverse: 'bool' = False) -> 'Tensor':
```
Compute the cumulative sum of *x* along *axis*.

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Axis along which to accumulate. Default: ``-1``.
- **`exclusive`** – If ``True``, each element is the sum of all *preceding*
elements (the first output element is ``0``).
- **`reverse`** – If ``True``, accumulate from right to left.

**Returns**

Tensor of the same shape as *x* with cumulative sums.


---
## `reduce_sum_physical`

```python
def reduce_sum_physical(x: 'Tensor', axis: 'int', keepdims: 'bool' = False) -> 'Tensor':
```
Sum along *axis* in the physical (sharded) tensor representation.

Unlike :func:`reduce_sum`, this operates directly on the physical shape
(including batch dimensions added by ``vmap``). It is used internally by
transforms that need fine-grained control over the reduction axis.

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Physical axis index to reduce along.
- **`keepdims`** – If ``True``, the reduced axis is kept as size 1.

**Returns**

Physically-reduced tensor.


---
## `mean_physical`

```python
def mean_physical(x: 'Tensor', axis: 'int', keepdims: 'bool' = False) -> 'Tensor':
```
Compute the mean along *axis* in the physical (sharded) tensor representation.

Analogous to :func:`reduce_sum_physical` but divides by the axis size.
Used internally by transforms operating on the physical layout.

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Physical axis index to reduce along.
- **`keepdims`** – If ``True``, the reduced axis is kept as size 1.

**Returns**

Physically-averaged tensor.


---
