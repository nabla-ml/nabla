# he_uniform

## Signature

```python
nabla.he_uniform(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Array'
```

## Description

He uniform initialization for ReLU activations.

Samples from uniform distribution U(-a, a) where a = sqrt(6 / fan_in)

