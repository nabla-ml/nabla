# ones

## Signature

```python
endia.ones(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0)) -> 'Array'
```

## Description

Create an array filled with ones.

## Examples

```python
import endia as nb

# Create array of ones
result = nd.ones((2, 3))
print(result)  # [[1, 1, 1], [1, 1, 1]]
```

