# lecun_normal

## Signature

```python
nabla.lecun_normal(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = ()) -> 'Array'
```

## Description

LeCun normal initialization for SELU activations.

Samples from normal distribution N(0, std²) where std = sqrt(1 / fan_in)

