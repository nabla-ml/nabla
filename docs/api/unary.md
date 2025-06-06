# Unary Operations

Element-wise unary operations that operate on a single array.

```{toctree}
:maxdepth: 1
:caption: Functions

unary_cast
unary_cos
unary_decr_batch_dim_ctr
unary_exp
unary_incr_batch_dim_ctr
unary_log
unary_negate
unary_relu
unary_sin
```

## Quick Reference

### {doc}`cast <unary_cast>`

```python
endia.cast(arg: endia.core.array.Array, dtype: max._core.dtype.DType) -> endia.core.array.Array
```

Cast array elements to a different data type.

### {doc}`cos <unary_cos>`

```python
endia.cos(arg: endia.core.array.Array) -> endia.core.array.Array
```

Element-wise cosine function.

### {doc}`decr_batch_dim_ctr <unary_decr_batch_dim_ctr>`

```python
endia.decr_batch_dim_ctr(arg: endia.core.array.Array) -> endia.core.array.Array
```

Endia operation: `decr_batch_dim_ctr`

### {doc}`exp <unary_exp>`

```python
endia.exp(arg: endia.core.array.Array) -> endia.core.array.Array
```

Element-wise exponential function.

### {doc}`incr_batch_dim_ctr <unary_incr_batch_dim_ctr>`

```python
endia.incr_batch_dim_ctr(arg: endia.core.array.Array) -> endia.core.array.Array
```

Endia operation: `incr_batch_dim_ctr`

### {doc}`log <unary_log>`

```python
endia.log(arg: endia.core.array.Array) -> endia.core.array.Array
```

Element-wise natural logarithm.

### {doc}`negate <unary_negate>`

```python
endia.negate(arg: endia.core.array.Array) -> endia.core.array.Array
```

Element-wise negation.

### {doc}`relu <unary_relu>`

```python
endia.relu(arg: endia.core.array.Array) -> endia.core.array.Array
```

Element-wise rectified linear unit activation.

### {doc}`sin <unary_sin>`

```python
endia.sin(arg: endia.core.array.Array, dtype: max._core.dtype.DType | None = None) -> endia.core.array.Array
```

Element-wise sine function.

