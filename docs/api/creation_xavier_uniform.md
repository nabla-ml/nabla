# xavier_uniform

## Signature

```python
nabla.xavier_uniform(shape: 'Shape', dtype: 'DType' = float32, gain: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = ()) -> 'Array'
```

## Description

Xavier/Glorot uniform initialization for sigmoid/tanh activations.

Samples from uniform distribution U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))

