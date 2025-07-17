# Unary Operations

Element-wise unary operations that operate on a single array.

```{toctree}
:maxdepth: 1
:caption: Functions

unary_abs
unary_cast
unary_cos
unary_decr_batch_dim_ctr
unary_exp
unary_floor
unary_incr_batch_dim_ctr
unary_log
unary_logical_not
unary_negate
unary_relu
unary_sigmoid
unary_sin
unary_sqrt
unary_tanh
unary_transfer_to
```

## Quick Reference

### {doc}`abs <unary_abs>`

```python
nabla.abs(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Nabla operation: `abs`

### {doc}`cast <unary_cast>`

```python
nabla.cast(arg: nabla.core.array.Array, dtype: max._core.dtype.DType) -> nabla.core.array.Array
```

Cast array elements to a different data type.

### {doc}`cos <unary_cos>`

```python
nabla.cos(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Element-wise cosine function.

### {doc}`decr_batch_dim_ctr <unary_decr_batch_dim_ctr>`

```python
nabla.decr_batch_dim_ctr(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Nabla operation: `decr_batch_dim_ctr`

### {doc}`exp <unary_exp>`

```python
nabla.exp(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Element-wise exponential function.

### {doc}`floor <unary_floor>`

```python
nabla.floor(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Nabla operation: `floor`

### {doc}`incr_batch_dim_ctr <unary_incr_batch_dim_ctr>`

```python
nabla.incr_batch_dim_ctr(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Nabla operation: `incr_batch_dim_ctr`

### {doc}`log <unary_log>`

```python
nabla.log(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Element-wise natural logarithm.

### {doc}`logical_not <unary_logical_not>`

```python
nabla.logical_not(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Nabla operation: `logical_not`

### {doc}`negate <unary_negate>`

```python
nabla.negate(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Element-wise negation.

### {doc}`relu <unary_relu>`

```python
nabla.relu(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Element-wise rectified linear unit activation.

### {doc}`sigmoid <unary_sigmoid>`

```python
nabla.sigmoid(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Nabla operation: `sigmoid`

### {doc}`sin <unary_sin>`

```python
nabla.sin(arg: nabla.core.array.Array, dtype: max._core.dtype.DType | None = None) -> nabla.core.array.Array
```

Element-wise sine function.

### {doc}`sqrt <unary_sqrt>`

```python
nabla.sqrt(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Nabla operation: `sqrt`

### {doc}`tanh <unary_tanh>`

```python
nabla.tanh(arg: nabla.core.array.Array) -> nabla.core.array.Array
```

Nabla operation: `tanh`

### {doc}`transfer_to <unary_transfer_to>`

```python
nabla.transfer_to(arg: nabla.core.array.Array, device: max._core.driver.Device) -> nabla.core.array.Array
```

Nabla operation: `transfer_to`

