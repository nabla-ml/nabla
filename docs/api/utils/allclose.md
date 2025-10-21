# allclose

## Signature

```python
nabla.allclose(a: Union[nabla.core.tensor.Tensor, numpy.ndarray, float, int], b: Union[nabla.core.tensor.Tensor, numpy.ndarray, float, int], rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> bool
```

**Source**: `nabla.utils.testing`

Returns True if two tensors are element-wise equal within a tolerance.

This function automatically converts Nabla Tensors to numpy tensors using
.to_numpy() before comparison, providing a convenient way to compare
Nabla tensors with each other or with numpy tensors/scalars.

Args:
    a: Input tensor or scalar
    b: Input tensor or scalar
    rtol: Relative tolerance parameter
    atol: Absolute tolerance parameter
    equal_nan: Whether to compare NaN's as equal

Returns:
    bool: True if the tensors are equal within the given tolerance

Examples:
```python
import nabla as nb
a = nb.tensor([1.0, 2.0, 3.0])
b = nb.tensor([1.0, 2.0, 3.000001])
nb.allclose(a, b)
```
    True
```python
nb.allclose(a, np.array([1.0, 2.0, 3.0]))
```
    True

