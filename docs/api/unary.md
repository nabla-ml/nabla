# Unary Operations

Element-wise unary operations that operate on a single array.

```{toctree}
:maxdepth: 1

unary_cast
unary_cos
unary_decr_batch_dim_ctr
unary_exp
unary_incr_batch_dim_ctr
unary_log
unary_negate
unary_relu
unary_sin
unary_sqrt
```

## Quick Reference

### `cast`

```python
nabla.cast(arg: nabla.core.array.Array, dtype: max._core.dtype.DType) -> nabla.core.array.Array
```

Cast array elements to a different data type.

### `cos`

```python
nabla.cos(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Element-wise cosine function.

### `decr_batch_dim_ctr`

```python
nabla.decr_batch_dim_ctr(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Nabla operation: `decr_batch_dim_ctr`

### `exp`

```python
nabla.exp(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Element-wise exponential function.

### `incr_batch_dim_ctr`

```python
nabla.incr_batch_dim_ctr(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Nabla operation: `incr_batch_dim_ctr`

### `log`

```python
nabla.log(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Element-wise natural logarithm.

### `negate`

```python
nabla.negate(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Element-wise negation.

### `relu`

```python
nabla.relu(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Element-wise rectified linear unit activation.

### `sin`

```python
nabla.sin(arg: nabla.core.array.Array, dtype: max._core.dtype.DType | None = None) -> nabla.core.array.Array
```

Element-wise sine function.

### `sqrt`

```python
nabla.sqrt(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Nabla operation: `sqrt`

