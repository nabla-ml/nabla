# Linear Algebra

Linear algebra operations including matrix multiplication and decompositions.

```{toctree}
:maxdepth: 1

linalg_conv2d
linalg_conv2d_transpose
linalg_matmul
```

## Quick Reference

### `conv2d`

```python
nabla.conv2d(input_arr: nabla.core.array.Array, filter_arr: nabla.core.array.Array, stride=(1, 1), dilation=(1, 1), padding=0, groups=1) -> nabla.core.array.Array
```

Nabla operation: `conv2d`

### `conv2d_transpose`

```python
nabla.conv2d_transpose(input_arr: nabla.core.array.Array, filter_arr: nabla.core.array.Array, stride=(1, 1), dilation=(1, 1), padding=0, output_padding=0, groups=1) -> nabla.core.array.Array
```

Nabla operation: `conv2d_transpose`

### `matmul`

```python
nabla.matmul(arg0, arg1) -> nabla.core.array.Array
```

Matrix multiplication.

