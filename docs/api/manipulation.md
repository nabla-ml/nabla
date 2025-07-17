# Array Manipulation

Operations for reshaping, indexing, and manipulating array structure.

```{toctree}
:maxdepth: 1
:caption: Functions

manipulation_array_slice
manipulation_broadcast_batch_dims
manipulation_broadcast_to
manipulation_concatenate
manipulation_move_axis_from_front
manipulation_move_axis_from_front_of_batch_dims
manipulation_move_axis_to_front
manipulation_move_axis_to_front_of_batch_dims
manipulation_pad
manipulation_permute
manipulation_permute_batch_dims
manipulation_reshape
manipulation_shallow_copy
manipulation_squeeze
manipulation_squeeze_batch_dims
manipulation_stack
manipulation_transpose
manipulation_unsqueeze
manipulation_unsqueeze_batch_dims
```

## Quick Reference

### {doc}`array_slice <manipulation_array_slice>`

```python
nabla.array_slice(arg: nabla.core.array.Array, slices: list[slice], squeeze_axes: list[int] | None = None) -> nabla.core.array.Array
```

Nabla operation: `array_slice`

### {doc}`broadcast_batch_dims <manipulation_broadcast_batch_dims>`

```python
nabla.broadcast_batch_dims(arg: nabla.core.array.Array, batch_dims: tuple[int, ...]) -> nabla.core.array.Array
```

Nabla operation: `broadcast_batch_dims`

### {doc}`broadcast_to <manipulation_broadcast_to>`

```python
nabla.broadcast_to(arg: nabla.core.array.Array, shape: tuple[int, ...]) -> nabla.core.array.Array
```

Nabla operation: `broadcast_to`

### {doc}`concatenate <manipulation_concatenate>`

```python
nabla.concatenate(args: list[nabla.core.array.Array], axis: int = 0) -> nabla.core.array.Array
```

Nabla operation: `concatenate`

### {doc}`move_axis_from_front <manipulation_move_axis_from_front>`

```python
nabla.move_axis_from_front(input_array: nabla.core.array.Array, target_axis: int) -> nabla.core.array.Array
```

Nabla operation: `move_axis_from_front`

### {doc}`move_axis_from_front_of_batch_dims <manipulation_move_axis_from_front_of_batch_dims>`

```python
nabla.move_axis_from_front_of_batch_dims(input_array: nabla.core.array.Array, target_axis: int) -> nabla.core.array.Array
```

Nabla operation: `move_axis_from_front_of_batch_dims`

### {doc}`move_axis_to_front <manipulation_move_axis_to_front>`

```python
nabla.move_axis_to_front(input_array: nabla.core.array.Array, axis: int) -> nabla.core.array.Array
```

Nabla operation: `move_axis_to_front`

### {doc}`move_axis_to_front_of_batch_dims <manipulation_move_axis_to_front_of_batch_dims>`

```python
nabla.move_axis_to_front_of_batch_dims(input_array: nabla.core.array.Array, axis: int) -> nabla.core.array.Array
```

Nabla operation: `move_axis_to_front_of_batch_dims`

### {doc}`pad <manipulation_pad>`

```python
nabla.pad(arg: nabla.core.array.Array, slices: list[slice], target_shape: tuple[int, ...]) -> nabla.core.array.Array
```

Nabla operation: `pad`

### {doc}`permute <manipulation_permute>`

```python
nabla.permute(input_array: nabla.core.array.Array, axes: tuple[int, ...]) -> nabla.core.array.Array
```

Nabla operation: `permute`

### {doc}`permute_batch_dims <manipulation_permute_batch_dims>`

```python
nabla.permute_batch_dims(input_array: nabla.core.array.Array, axes: tuple[int, ...]) -> nabla.core.array.Array
```

Nabla operation: `permute_batch_dims`

### {doc}`reshape <manipulation_reshape>`

```python
nabla.reshape(arg: nabla.core.array.Array, shape: tuple[int, ...]) -> nabla.core.array.Array
```

Change the shape of an array without changing its data.

### {doc}`shallow_copy <manipulation_shallow_copy>`

```python
nabla.shallow_copy(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Nabla operation: `shallow_copy`

### {doc}`squeeze <manipulation_squeeze>`

```python
nabla.squeeze(arg: nabla.core.array.Array, axes: list[int] | None = None) -> nabla.core.array.Array
```

Remove single-dimensional entries from array shape.

### {doc}`squeeze_batch_dims <manipulation_squeeze_batch_dims>`

```python
nabla.squeeze_batch_dims(arg: nabla.core.array.Array, axes: list[int] | None = None) -> nabla.core.array.Array
```

Nabla operation: `squeeze_batch_dims`

### {doc}`stack <manipulation_stack>`

```python
nabla.stack(arrays: list[nabla.core.array.Array], axis: int = 0) -> nabla.core.array.Array
```

Nabla operation: `stack`

### {doc}`transpose <manipulation_transpose>`

```python
nabla.transpose(arg: nabla.core.array.Array, axis_1: int = -2, axis_2: int = -1) -> nabla.core.array.Array
```

Permute the dimensions of an array.

### {doc}`unsqueeze <manipulation_unsqueeze>`

```python
nabla.unsqueeze(arg: nabla.core.array.Array, axes: list[int] | None = None) -> nabla.core.array.Array
```

Add single-dimensional entries to array shape.

### {doc}`unsqueeze_batch_dims <manipulation_unsqueeze_batch_dims>`

```python
nabla.unsqueeze_batch_dims(arg: nabla.core.array.Array, axes: list[int] | None = None) -> nabla.core.array.Array
```

Nabla operation: `unsqueeze_batch_dims`

