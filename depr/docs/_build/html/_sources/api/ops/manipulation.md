# Manipulation Ops

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
