# Array Manipulation

Operations for reshaping, indexing, and manipulating array structure.

```{toctree}
:maxdepth: 1
:caption: Functions

manipulation_broadcast_batch_dims
manipulation_broadcast_to
manipulation_reshape
manipulation_shallow_copy
manipulation_squeeze
manipulation_transpose
manipulation_unsqueeze
```

## Quick Reference

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
nabla.squeeze(arg: nabla.core.array.Array, axes: list[int] = None) -> nabla.core.array.Array
```

Remove single-dimensional entries from array shape.

### {doc}`transpose <manipulation_transpose>`

```python
nabla.transpose(arg: nabla.core.array.Array, axis_1: int = -2, axis_2: int = -1) -> nabla.core.array.Array
```

Permute the dimensions of an array.

### {doc}`unsqueeze <manipulation_unsqueeze>`

```python
nabla.unsqueeze(arg: nabla.core.array.Array, axes: list[int] = None) -> nabla.core.array.Array
```

Add single-dimensional entries to array shape.

