# matmul

## Signature

```python
nabla.matmul(arg0: nabla.core.array.Array, arg1: nabla.core.array.Array) -> nabla.core.array.Array
```

## Description

Matrix multiplication with broadcasting support.

## Examples

```python
import nabla as nb

# Matrix multiplication
A = nb.array([[1, 2], [3, 4]])
B = nb.array([[5, 6], [7, 8]])
result = nb.matmul(A, B)
print(result)  # [[19, 22], [43, 50]]
```

