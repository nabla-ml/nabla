# he_normal

## Signature

```python
nabla.he_normal(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Array'
```

## Description

He normal initialization for ReLU activations.

Samples from normal distribution N(0, std²) where std = sqrt(2 / fan_in)

