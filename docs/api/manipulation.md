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
endia.broadcast_batch_dims(arg: endia.core.array.Array, batch_dims: tuple[int, ...]) -> endia.core.array.Array
```

Endia operation: `broadcast_batch_dims`

### {doc}`broadcast_to <manipulation_broadcast_to>`

```python
endia.broadcast_to(arg: endia.core.array.Array, shape: tuple[int, ...]) -> endia.core.array.Array
```

Endia operation: `broadcast_to`

### {doc}`reshape <manipulation_reshape>`

```python
endia.reshape(arg: endia.core.array.Array, shape: tuple[int, ...]) -> endia.core.array.Array
```

Change the shape of an array without changing its data.

### {doc}`shallow_copy <manipulation_shallow_copy>`

```python
endia.shallow_copy(arg: endia.core.array.Array) -> endia.core.array.Array
```

Endia operation: `shallow_copy`

### {doc}`squeeze <manipulation_squeeze>`

```python
endia.squeeze(arg: endia.core.array.Array, axes: list[int] = None) -> endia.core.array.Array
```

Remove single-dimensional entries from array shape.

### {doc}`transpose <manipulation_transpose>`

```python
endia.transpose(arg: endia.core.array.Array, axis_1: int = -2, axis_2: int = -1) -> endia.core.array.Array
```

Permute the dimensions of an array.

### {doc}`unsqueeze <manipulation_unsqueeze>`

```python
endia.unsqueeze(arg: endia.core.array.Array, axes: list[int] = None) -> endia.core.array.Array
```

Add single-dimensional entries to array shape.

