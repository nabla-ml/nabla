# lecun_uniform

## Signature

```python
nabla.lecun_uniform(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = ()) -> 'Array'
```

## Description

LeCun uniform initialization for SELU activations.

Samples from uniform distribution U(-a, a) where a = sqrt(3 / fan_in)

