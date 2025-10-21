# View & Manipulation Operations

## `reshape`

```python
def reshape(arg: nabla.core.tensor.Tensor, shape: tuple[int, ...]) -> nabla.core.tensor.Tensor:
```
Reshape tensor to given shape.


---
## `transpose`

```python
def transpose(arg: nabla.core.tensor.Tensor, axis_1: int = -2, axis_2: int = -1) -> nabla.core.tensor.Tensor:
```
Transpose tensor along two axes.


---
## `permute`

```python
def permute(input_tensor: nabla.core.tensor.Tensor, axes: tuple[int, ...]) -> nabla.core.tensor.Tensor:
```
Permute (reorder) the dimensions of a tensor.

**Examples**

```python
>>> x = nb.ones((2, 3, 4))  # shape (2, 3, 4)
>>> y = permute(x, (2, 0, 1))  # shape (4, 2, 3)
>>> # Dimension 2 -> position 0, dimension 0 -> position 1, dimension 1 -> position 2
```

---
## `concatenate`

```python
def concatenate(args: list[nabla.core.tensor.Tensor], axis: int = 0) -> nabla.core.tensor.Tensor:
```
Concatenate tensors along an existing axis.

Parameters
----------
    args: List of tensors to concatenate
    axis: Axis along which to concatenate tensors (default: 0)

Returns
-------
    Concatenated tensor


---
## `stack`

```python
def stack(tensors: list[nabla.core.tensor.Tensor], axis: int = 0) -> nabla.core.tensor.Tensor:
```
Stack tensors along a new axis.

Parameters
----------
    tensors: List of tensors to stack
    axis: Axis along which to stack the tensors (default: 0)

Returns
-------
    Stacked tensor


---
## `split`

```python
def split(arg: nabla.core.tensor.Tensor, sizes: list[int], axis: int = 0) -> list[nabla.core.tensor.Tensor]:
```
Split an tensor into multiple sub-tensors along a specified axis.

Parameters
----------
    arg: Input tensor to split
    sizes: List of sizes for each split along the specified axis
    axis: Axis along which to split the tensor (default: 0)
Returns
-------
    List of sub-tensors resulting from the split


---
## `squeeze`

```python
def squeeze(arg: nabla.core.tensor.Tensor, axes: list[int] | None = None) -> nabla.core.tensor.Tensor:
```
Squeeze tensor by removing dimensions of size 1.


---
## `unsqueeze`

```python
def unsqueeze(arg: nabla.core.tensor.Tensor, axes: list[int] | None = None) -> nabla.core.tensor.Tensor:
```
Unsqueeze tensor by adding dimensions of size 1.


---
## `pad`

```python
def pad(arg: nabla.core.tensor.Tensor, slices: list[slice], target_shape: tuple[int, ...]) -> nabla.core.tensor.Tensor:
```
Place a smaller tensor into a larger zero-filled tensor at the location specified by slices.

This is the inverse operation of tensor slicing - given slices, a small tensor, and target shape,
it creates a larger tensor where the small tensor is placed at the sliced location
and everything else is zero.

Parameters
----------
    arg: Input tensor (the smaller tensor to be placed)
    slices: List of slice objects defining where to place the tensor
    target_shape: The shape of the output tensor

Returns
-------
    Larger tensor with input placed at sliced location, zeros elsewhere


---
## `broadcast_to`

```python
def broadcast_to(arg: nabla.core.tensor.Tensor, shape: tuple[int, ...]) -> nabla.core.tensor.Tensor:
```
Broadcast tensor to target shape.


---
## `tensor_slice`

```python
def tensor_slice(arg: nabla.core.tensor.Tensor, slices: list[slice], squeeze_axes: list[int] | None = None) -> nabla.core.tensor.Tensor:
```
Slice an tensor along specified dimensions.

Parameters
----------
    arg: Input tensor to slice
    slices: List of slice objects defining the slicing for each dimension
    squeeze_axes: List of axes that should be squeezed (for JAX compatibility)

Returns
-------
    Sliced tensor


---
## `shallow_copy`

```python
def shallow_copy(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Create a shallow copy of the tensor.


---
## `transpose_batch_dims`

```python
def transpose_batch_dims(arg: nabla.core.tensor.Tensor, axis_1: int = -2, axis_2: int = -1) -> nabla.core.tensor.Tensor:
```
Transpose batch dimensions along two axes.

This operation swaps two axes in the batch_dims of an Tensor, similar to how
regular transpose works on shape dimensions. The shape dimensions remain unchanged.

**Examples**

```python
>>> import nabla as nb
>>> # Tensor with batch_dims=(2, 3, 4) and shape=(5, 6)
>>> x = nb.ones((5, 6))
>>> x.batch_dims = (2, 3, 4)  # Simulated for example
>>> y = transpose_batch_dims(x, -3, -1)  # Swap first and last batch dims
>>> # Result has batch_dims=(4, 3, 2) and shape=(5, 6)
```

---
## `permute_batch_dims`

```python
def permute_batch_dims(input_tensor: nabla.core.tensor.Tensor, axes: tuple[int, ...]) -> nabla.core.tensor.Tensor:
```
Permute (reorder) the batch dimensions of an tensor.

This operation reorders the batch_dims of an Tensor according to the given axes,
similar to how regular permute works on shape dimensions. The shape dimensions
remain unchanged.

**Examples**

```python
>>> import nabla as nb
>>> # Tensor with batch_dims=(2, 3, 4) and shape=(5, 6)
>>> x = nb.ones((5, 6))
>>> x.batch_dims = (2, 3, 4)  # Simulated for example
>>> y = permute_batch_dims(x, (-1, -3, -2))  # Reorder as (4, 2, 3)
>>> # Result has batch_dims=(4, 2, 3) and shape=(5, 6)
```

---
## `broadcast_batch_dims`

```python
def broadcast_batch_dims(arg: nabla.core.tensor.Tensor, batch_dims: tuple[int, ...]) -> nabla.core.tensor.Tensor:
```
Broadcast tensor to target batch_dims.


---
## `squeeze_batch_dims`

```python
def squeeze_batch_dims(arg: nabla.core.tensor.Tensor, axes: list[int] | None = None) -> nabla.core.tensor.Tensor:
```
Squeeze tensor by removing batch dimensions of size 1.

Parameters
----------
    arg: Input tensor
    axes: List of batch dimension axes to squeeze. If None, returns tensor unchanged.

Returns
-------
    Tensor with specified batch dimensions of size 1 removed


---
## `unsqueeze_batch_dims`

```python
def unsqueeze_batch_dims(arg: nabla.core.tensor.Tensor, axes: list[int] | None = None) -> nabla.core.tensor.Tensor:
```
Unsqueeze tensor by adding batch dimensions of size 1.

Parameters
----------
    arg: Input tensor
    axes: List of positions where to insert batch dimensions of size 1.
          If None, returns tensor unchanged.

Returns
-------
    Tensor with batch dimensions of size 1 added at specified positions


---
