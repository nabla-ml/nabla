# View & Shape

## `reshape`

```python
def reshape(x: 'Tensor', shape: 'tuple[int, ...]') -> 'Tensor':
```
Return a tensor with the same data as *x* reshaped to *shape*.

The total number of elements must be preserved. A single ``-1``
in *shape* is automatically inferred.

**Parameters**

- **`x`** – Input tensor.
- **`shape`** – Target shape.

**Returns**

 – Tensor with the same data and new shape.


---
## `transpose`

```python
def transpose(x: 'Tensor', axis1: 'int', axis2: 'int') -> 'Tensor':
```
Swap (transpose) two dimensions of *x*.

**Parameters**

- **`x`** – Input tensor.
- **`axis1`** – First axis. Supports negative indexing.
- **`axis2`** – Second axis. Supports negative indexing.

**Returns**

 – View with *axis1* and *axis2* swapped.


---
## `permute`

```python
def permute(x: 'Tensor', order: 'tuple[int, ...]') -> 'Tensor':
```
Reorder the dimensions of *x* according to *order*.

**Parameters**

- **`x`** – Input tensor of rank ``N``.
- **`order`** – A permutation of ``(0, 1, ..., N-1)`` giving the new dimension
ordering. Equivalent to NumPy's ``transpose(axes=order)``.

**Returns**

 – Tensor with dimensions reordered as specified.


---
## `unsqueeze`

```python
def unsqueeze(x: 'Tensor', axis: 'int' = 0) -> 'Tensor':
```
Insert a size-1 dimension at *axis* into *x*'s shape.

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Position at which to insert the new dimension.
Supports negative indexing.

**Returns**

 – Tensor with one additional dimension of size 1.


---
## `squeeze`

```python
def squeeze(x: 'Tensor', axis: 'int' = 0) -> 'Tensor':
```
Remove the size-1 dimension at *axis* from *x*'s shape.

**Parameters**

- **`x`** – Input tensor. The dimension at *axis* must be 1.
- **`axis`** – Dimension to remove. Supports negative indexing.
Pass ``None`` to squeeze all size-1 dimensions.

**Returns**

 – Tensor with the specified dimension removed.


---
## `flatten`

```python
def flatten(x: 'Tensor', start_dim: 'int' = 0, end_dim: 'int' = -1) -> 'Tensor':
```
Flatten a contiguous range of dimensions into one.

**Parameters**

- **`x`** – Input tensor.
- **`start_dim`** – First dimension to flatten (inclusive). Default: ``0``.
- **`end_dim`** – Last dimension to flatten (inclusive). Default: ``-1``
(last dimension).

**Returns**

 – Tensor with dimensions ``start_dim`` through ``end_dim`` collapsed
into a single dimension.


---
## `broadcast_to`

```python
def broadcast_to(x: 'Tensor', shape: 'tuple[int, ...]') -> 'Tensor':
```
Broadcast *x* to a new *shape*.

Leading dimensions are added as needed (NumPy-style broadcasting).
Size-1 dimensions in *x* are expanded to match *shape*.

**Parameters**

- **`x`** – Input tensor.
- **`shape`** – Target output shape. Must be broadcast-compatible with ``x.shape``.

**Returns**

 – Tensor of the given *shape* sharing data with *x* where possible.


---
## `concatenate`

```python
def concatenate(tensors: 'Sequence[Tensor]', axis: 'int' = 0) -> 'Tensor':
```
Concatenate a sequence of tensors along an existing *axis*.

**Parameters**

- **`tensors`** – Non-empty sequence of tensors with the same shape except
along *axis*.
- **`axis`** – Axis along which to concatenate. Default: ``0``.

**Returns**

 – Tensor whose *axis* dimension is the sum of the inputs'.


---
## `stack`

```python
def stack(tensors: 'list[Tensor]', axis: 'int' = 0) -> 'Tensor':
```
Stack a sequence of tensors along a **new** *axis*.

All tensors must have the same shape. The result has one more dimension
than the inputs.

**Parameters**

- **`tensors`** – List of tensors with identical shapes.
- **`axis`** – Position of the new dimension in the output. Default: ``0``.

**Returns**

 – Tensor of shape ``tensors[0].shape[:axis] + (N,) + tensors[0].shape[axis:]``
where *N* is the number of input tensors.


---
## `gather`

```python
def gather(x: 'Tensor', indices: 'Tensor', axis: 'int' = 0) -> 'Tensor':
```
Gather elements from x along axis using indices.


---
## `scatter`

```python
def scatter(x: 'Tensor', indices: 'Tensor', updates: 'Tensor', axis: 'int' = 0) -> 'Tensor':
```
Scatter updates into x at indices along axis.


---
## `slice_tensor`

```python
def slice_tensor(x: 'Tensor', start: 'Any', size: 'Any') -> 'Tensor':
```
Extract a rectangular slice from *x*.

**Parameters**

- **`x`** – Input tensor.
- **`start`** – Sequence of per-dimension start indices (supports negative).
- **`size`** – Sequence of per-dimension slice sizes.

**Returns**

 – Tensor of shape *size* containing the requested slice.


---
## `slice_update`

```python
def slice_update(x: 'Tensor', update: 'Tensor', start: 'Any', size: 'Any') -> 'Tensor':
```
Return *x* with a rectangular region replaced by *update*.

This is a functional (out-of-place) operation. The original *x* is not
modified. Supports autograd.

**Parameters**

- **`x`** – Base tensor to update.
- **`update`** – Values to write into *x*. Must have shape *size*.
- **`start`** – Sequence of per-dimension start indices.
- **`size`** – Sequence of per-dimension region sizes.

**Returns**

 – New tensor equal to *x* except at the specified slice.


---
## `moveaxis`

```python
def moveaxis(x: 'Tensor', source: 'int', destination: 'int') -> 'Tensor':
```
Move axis *source* to position *destination*.

**Parameters**

- **`x`** – Input tensor.
- **`source`** – Original axis position. Supports negative indexing.
- **`destination`** – Target axis position. Supports negative indexing.

**Returns**

 – Tensor with the axis at *source* moved to *destination*.


---
## `swap_axes`

```python
def swap_axes(x: 'Tensor', axis1: 'int', axis2: 'int') -> 'Tensor':
```
Swap (transpose) two dimensions of *x*.

**Parameters**

- **`x`** – Input tensor.
- **`axis1`** – First axis. Supports negative indexing.
- **`axis2`** – Second axis. Supports negative indexing.

**Returns**

 – View with *axis1* and *axis2* swapped.


---
## `flip`

```python
def flip(x: 'Tensor', axis: 'int') -> 'Tensor':
```
Reverse the elements of *x* along the specified axis.

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – The axis along which to reverse. Supports negative indexing.

**Returns**

 – Tensor with elements reversed along *axis*. Shape is unchanged.


---
## `pad`

```python
def pad(x: 'Tensor', paddings: 'list[tuple[int, int]]' = None, mode: 'str' = 'constant', value: 'float' = 0.0, **kwargs) -> 'Tensor':
```
Pad a tensor with a constant value (or a specific padding mode).

**Parameters**

- **`x`** – Input tensor.
- **`paddings`** – List of ``(before, after)`` tuples, one per logical dimension.
Also accepted via the ``pad_width`` keyword alias.
- **`mode`** – Padding mode. Currently only ``"constant"`` is supported.
- **`value`** – Fill value for constant padding. Default: ``0.0``.

**Returns**

 – Padded tensor. Each dimension *i* grows by
``paddings[i][0] + paddings[i][1]`` elements.


---
## `rebind`

```python
def rebind(x: 'Tensor', shape: 'tuple[int, ...]', **kwargs) -> 'Tensor':
```
Rebind a tensor to a new symbolic shape without changing the data.

Used to introduce or update shape constraints known at compile time.
Has no gradient — the cotangent is passed through unchanged.

**Parameters**

- **`x`** – Input tensor.
- **`shape`** – New shape annotation (can include symbolic dimensions).

**Returns**

 – Tensor with updated shape metadata.


---
## `as_interleaved_complex`

```python
def as_interleaved_complex(x: 'Tensor') -> 'Tensor':
```

---
## `view_as_real_interleaved`

```python
def view_as_real_interleaved(x: 'Tensor') -> 'Tensor':
```

---
## `broadcast_to_physical`

```python
def broadcast_to_physical(x: 'Tensor', shape: 'tuple[int, ...]') -> 'Tensor':
```
Broadcast *x* to *shape* in the physical tensor layout.

Unlike :func:`broadcast_to`, this operates on the physical shape
(including batch dimensions added by ``vmap``). Used internally by
transforms and physical gradient rules.

**Parameters**

- **`x`** – Input tensor.
- **`shape`** – Target **physical** shape.

**Returns**

 – Tensor broadcast to the given physical shape.


---
## `squeeze_physical`

```python
def squeeze_physical(x: 'Tensor', axis: 'int' = 0) -> 'Tensor':
```
Remove the size-1 dimension at *axis* in the **physical** tensor layout.

Counterpart to :func:`unsqueeze_physical`. Used internally by transforms.


---
## `unsqueeze_physical`

```python
def unsqueeze_physical(x: 'Tensor', axis: 'int' = 0) -> 'Tensor':
```
Insert a size-1 dimension at *axis* in the **physical** tensor layout.

Unlike :func:`unsqueeze`, this operates on the physical shape (which
includes batch dimensions added by ``vmap``). Used internally by
transforms that manipulate the physical layout directly.


---
## `incr_batch_dims`

```python
def incr_batch_dims(x: 'Tensor') -> 'Tensor':
```
Increment batch_dims counter (first physical dim becomes batch dim).


---
## `decr_batch_dims`

```python
def decr_batch_dims(x: 'Tensor') -> 'Tensor':
```
Decrement batch_dims counter (first batch dim becomes logical dim).


---
## `move_axis_to_batch_dims`

```python
def move_axis_to_batch_dims(x: 'Tensor', axis: 'int') -> 'Tensor':
```
Move a logical axis into the batch dimensions (3 ops: calc + moveaxis_physical + incr).


---
## `move_axis_from_batch_dims`

```python
def move_axis_from_batch_dims(x: 'Tensor', batch_axis: 'int' = 0, logical_destination: 'int' = 0) -> 'Tensor':
```
Move a batch dimension to logical axis (3 ops: calc + moveaxis_physical + decr).


---
