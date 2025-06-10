# array

## Signature

```python
nabla.array(data: 'list | np.ndarray | float | int', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = ()) -> 'Array'
```

## Description

Create an array from Python list, numpy array, or scalar value.

## Examples

```python
import nabla as nb

# Create array from data
data = [[1, 2, 3], [4, 5, 6]]
result = nb.array(data)
print(result)  # [[1, 2, 3], [4, 5, 6]]
```

