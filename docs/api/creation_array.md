# array

## Signature

```python
nabla.array(data: 'list | np.ndarray', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0)) -> 'Array'
```

## Description

Create an array from Python list or numpy array.

## Examples

```python
import nabla as nb

# Create array from data
data = [[1, 2, 3], [4, 5, 6]]
result = nb.array(data)
print(result)  # [[1, 2, 3], [4, 5, 6]]
```

