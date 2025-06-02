# zeros

## Signature

```python
nabla.zeros(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0)) -> 'Array'
```

## Description

Create an array filled with zeros.

## Examples

```python
import nabla as nb

# Create array of zeros
result = nb.zeros((2, 3))
print(result)  # [[0, 0, 0], [0, 0, 0]]
```

