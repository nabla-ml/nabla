# matmul

## Signature

```python
endia.matmul(arg0: endia.core.array.Array, arg1: endia.core.array.Array) -> endia.core.array.Array
```

## Description

Matrix multiplication with broadcasting support.

## Examples

```python
import endia as nd

# Matrix multiplication
A = nd.array([[1, 2], [3, 4]])
B = nd.array([[5, 6], [7, 8]])
result = nd.matmul(A, B)
print(result)  # [[19, 22], [43, 50]]
```

